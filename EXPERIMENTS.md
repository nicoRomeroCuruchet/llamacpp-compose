# EXPERIMENTS

A lab notebook for the tuning done on this server: what was changed, what was measured,
and what the number turned out to mean. `README.md` documents the configuration that came
out of it; this file documents how it was arrived at, including the measurements that
argued *against* a change.

Every number here was measured on the machine described below. None of it is quoted from a
model card or a benchmark elsewhere.

**Three models and two cards share this notebook.** Experiments 1-9 are Qwen3.8-27B, a
dense model, on an RTX 3090. Experiment 10 is Ornith 1.0 35B on an RTX 4090 - and finds
that a 3090 conclusion about batch size does not hold there. Experiment 11 is Ornith 1.5
35B-A3B, a mixture of experts, back on the 3090 - and reverses three more. Read 10 and 11
before carrying any tuning decision from here to a new model or a new card.

---

## 0. Fixed conditions

Unless an experiment says otherwise, all of it ran under these conditions. **Experiments 10
and 11 do say otherwise** and declare their own.

| | |
|---|---|
| Host | `udesa` — RTX 3090, 24,576 MiB VRAM, 27 GiB system RAM, 31 GiB swap |
| Driver | 595.84 |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda`, build 10335 |
| Model | `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17,923,394,624 B (17,092 MiB) |
| Architecture | `qwen35`, 65 blocks, hybrid attention — see experiment 6 |
| Server | `-ngl all`, `-np 1`, `-fa on`, `-ctk q8_0 -ctv q8_0`, `-c 65536` |
| Sampling | `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0` (Qwen3 card) |

### Where the numbers come from

Three sources, and they do not measure the same thing:

| Source | What it gives | Caveat |
|---|---|---|
| `docker compose logs \| grep print_timing` | per-request prefill and decode rates, draft acceptance | one line per completed request; nothing while a request is in flight |
| `GET /metrics` | cumulative counters since the container started | totals, not rates — divide two counters yourself |
| `nvidia-smi` | VRAM actually resident | includes the CUDA context; says nothing about *what* is holding it |

Two traps that cost time here and are worth writing down:

- **`prompt_tokens_total` in `/metrics` excludes cached tokens.** The cached ones are in
  `prompt_tokens_cached_total`. Dividing `prompt_tokens_total / prompt_seconds_total` gives
  the rate of *work actually done*, which is the honest prefill number; dividing the sum of
  both by the same seconds gives the rate *the client perceives*, which is higher and is a
  different claim. Both appear below, labelled.
- **Prefill t/s on a short prompt is meaningless.** Task 14 below reports 37 t/s for an
  11-token prompt — that is fixed per-request overhead divided by 11, not a throughput.
  Only read prefill rates off prompts of a few hundred tokens or more.

---

## 1. Speculative decoding: which drafter

**Question.** The `.gguf` carries MTP heads (`blk.64.nextn.*`). Do they beat the n-gram
drafter, and by how much?

**Method.** 700-token generation from a short coding prompt, one variable at a time,
container restarted between runs. Decode rate and acceptance read from `print_timing`,
VRAM from `nvidia-smi` after the model settled.

| `SPEC_TYPE` | `N_MAX` | `P_MIN` | decode | acceptance | VRAM |
|---|---|---|---|---|---|
| `ngram-simple` | — | — | 40.8 t/s | 4–37% | 19.1 GB |
| `draft-mtp` | 4 | 0.7 | 50.9 t/s | 78–100% | 20.5 GB |
| `draft-mtp` | 8 | 0.5 | 52.0 t/s | 50% | 21.1 GB |
| **`draft-mtp`** | **8** | **0.7** | **53.5 t/s** | **62%** | **21.1 GB** |

**Result.** `draft-mtp` at `N_MAX=8`, `P_MIN=0.7` — **+31% decode** over `ngram-simple`.

**What it cost.** ~2.0 GB of VRAM, which is not the drafter's weights — those are in the
file either way — but the extra KV and compute state for evaluating a draft branch.

**Negative results, which are the useful part:**

- **Loosening `P_MIN` from 0.7 to 0.5 makes it worse** (52.0 vs 53.5 t/s). It drafts more
  aggressively, acceptance falls from 62% to 50%, and the rejected drafts cost more compute
  than the extra accepted ones save. The confidence floor is doing real work; it is not a
  formality.
- **Raising `N_MAX` past 8 was not tested, deliberately.** Mean accepted length in that run
  was 3.5 tokens, so a budget of 8 already goes unspent. Testing 12 or 16 would measure
  nothing.
- `ngram-simple` drafts by replaying n-grams it has already emitted. It does well on
  repetitive edits and badly on prose — the 4–37% spread is that, not noise.

**Why this knob is different from the others.** Speculation is **lossless**. Rejected
drafts are discarded and the emitted text is byte-identical to what the model would have
produced unaided. Only time is at stake. `KV_TYPE` and the sampling values do not have that
property.

---

## 2. Prefill batch size

**Question.** `-b 512 -ub 256` was inherited from an earlier config. Is it costing prefill?

**Change.** `BATCH 512 → 2048`, `UBATCH 256 → 512` (llama.cpp's own defaults).

**Result.** Prefill on large prompts now runs at **900–1,120 t/s** (tasks 121 and 209 in
experiment 5). The previous values were throttling it: the 3090 was being handed work in
pieces too small to saturate.

**Caveat, stated plainly.** There is no clean before/after pair for this one — it was
changed in the same pass as `CACHE_RAM` while chasing a slow first token, and only the
after-state was measured properly. The direction is not in doubt (`-ub 256` under-feeds an
Ampere card), but "+X%" would be a number I do not have. Anyone wanting it should run the
sweep.

---

## 3. Prompt cache in host RAM

**Question.** The log showed `making room for prompt cache entry` — the cache was evicting.
What does that cost?

**Observation.** An evicted 48k-token context has to be prefilled from scratch on the next
turn: **35 seconds before the first token**. In an agent loop, where each turn resends a
long and mostly unchanged conversation, this is the single worst latency in the system.

**Change.** `CACHE_RAM 8192 → 10240` MiB.

**Result.** Measured over the live agent workload (experiment 4): **71.2% of prompt tokens
served from cache**, 55,935 of 78,514. Slot reuse shows up in the logs as
`selected slot by LCP similarity` with a high `f_sim_best`.

**Ceiling.** This is **host RAM, not VRAM** — a distinct budget from everything else in
this file. `free -g` on this box reports 27 GiB total with 22 GiB available. 10 GiB is
close to the sensible limit; raising it further trades against whatever else runs here, and
this is a shared machine. Check `free -g` before touching it.

---

## 4. Real throughput under an agent workload

**Question.** The experiment-1 numbers come from a synthetic prompt. What does the server
actually do when a coding agent drives it?

**Method.** An isolated Claude Code + OMP box (`agentic-box`, running under the `agentic`
user on a different host) pointed at this server over the tailnet, then used normally.
Counters read from `/metrics`; the traffic is genuine agent work, not a script.

| Metric | Value |
|---|---|
| Prompt tokens processed | 22,579 |
| Prompt tokens served from cache | 55,935 (**71.2%** of 78,514) |
| Prefill time | 22.84 s → **988 t/s** of work actually done |
| Tokens generated | 2,436 |
| Generation time | 38.52 s → **63.2 t/s** aggregate |
| Decode steps (`n_decode_total`) | 769 → **3.17 tokens emitted per forward pass** |
| Draft acceptance | 1,717 / 2,202 = **78.0%** |
| Drafts issued | 501 → 4.4 drafted, 3.4 accepted per round |
| Peak context reached | 22,564 of 65,536 tokens (**34%**) |
| Requests truncated | **0** |

**Result.** **63.2 t/s** aggregate decode against the **53.5 t/s** of the synthetic bench —
real agent traffic is *faster*, not slower. The reason is acceptance: 78.0% here against
62% in the bench. Agent output is repetitive in exactly the way MTP drafting exploits —
code, tool calls, file paths, structured edits — so more of each draft survives.

The `n_decode_total` figure is the clearest single statement of what speculation buys:
**3.17 tokens per forward pass** instead of 1.

**The thing this measurement is not.** It ran over roughly one working session with a
single client and `-np 1`. It says nothing about concurrency; `requests_deferred` stayed at
0 throughout, meaning the queue was never tested.

---

## 5. Per-request breakdown: what actually predicts decode speed

The eight most recent completed requests, from `print_timing`:

| task | prefill tok | prefill t/s | gen tok | decode t/s | acceptance | mean len |
|---|---|---|---|---|---|---|
| 0 | 66 | 170 | 39 | 68.5 | 97% | 5.57 |
| 14 | 11 | 37 | 46 | 74.4 | 70% | 4.50 |
| 29 | 17 | 38 | 238 | 57.2 | 73% | 3.35 |
| 121 | 17,415 | 1,120 | 51 | 47.2 | 71% | 3.31 |
| 157 | 300 | 562 | 14 | 53.4 | 100% | 4.00 |
| 158 | 119 | 218 | 193 | **82.3** | 88% | 5.81 |
| 209 | 2,771 | 903 | 1,721 | 64.5 | 78% | 4.65 |
| 716 | 1,880 | 734 | 134 | **45.4** | 69% | 3.24 |

**Result.** Decode rate tracks **mean accepted draft length**, not context size. Sorted by
`mean len`, decode rises monotonically apart from two inversions:

```
mean len   3.24  3.31  3.35  4.00  4.50  4.65  5.57  5.81
decode     45.4  47.2  57.2  53.4  74.4  64.5  68.5  82.3
```

Roughly **+10 t/s per additional accepted token**, over a 45–82 t/s range — the spread
between the best and worst request is nearly a factor of two, and it is almost all
drafting.

**This corrects an earlier reading.** Watching task 209's rate fall as it generated
(78 → 80 → 68 → 65 → 64 t/s over 1,721 tokens) suggested KV growth was the cause. The table
says otherwise: task 29 ran at 57 t/s from a **17-token** context, and task 158 hit 82 t/s
at a similar size. Context length is confounded with acceptance in that sample and is not
the driver.

**Practical consequence.** Nothing here is worth tuning for. Acceptance is a property of
what the model is being asked to write, not a server setting. The observation's value is
diagnostic: **a decode rate near 45 t/s means acceptance dropped, not that the server
degraded.** Do not go looking for a configuration problem that is not there.

---

## 6. What the model actually is: reading the GGUF

**Question.** Arithmetic said the KV cache at `CTX=65536` should be ~8.7 GB. Measurement
said everything above the weights totalled ~4.0 GB. One of them was wrong.

**Method.** The server does not print KV cache size at this verbosity and `/props` carries
no architecture fields, so the `.gguf` header was parsed directly — metadata plus the
tensor index, without loading the model. The tool is committed as
**`scripts/gguf-info.py`** (standard library only):

```bash
python3 scripts/gguf-info.py ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
```

**Result. The model is hybrid**, and metadata says so outright:

```
qwen35.block_count             65
qwen35.full_attention_interval 4        <- one full-attention block in every four
qwen35.attention.head_count    24
qwen35.attention.head_count_kv 4
qwen35.attention.key_length    256
qwen35.attention.value_length  256
qwen35.context_length          262144
qwen35.ssm.state_size          128
qwen35.ssm.inner_size          6144
qwen35.nextn_predict_layers    1
```

Confirmed against the tensor index — the metadata could be aspirational, the tensors cannot
be:

| Blocks | Count | Which | Tensors |
|---|---|---|---|
| Full attention (carry KV) | 16 | 3, 7, 11 … 63 | `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_{q,k}_norm` |
| Linear attention / SSM | 48 | everything else | `ssm_a`, `ssm_conv1d`, `ssm_out`, `ssm_norm`, `attn_gate`, `attn_qkv` |
| MTP head (also carries KV) | 1 | 64 | `nextn.*` plus a full attention set |

**So 17 of 65 blocks store a KV cache, not 65.** The 48 SSM blocks hold a recurrent state
of **fixed** size — it does not grow with the context at all. That is the whole discrepancy:
the first estimate was off by a factor of four because it assumed a dense transformer.

**KV cache per token:**

```
(key_length + value_length) x head_count_kv = (256 + 256) x 4 = 2,048 values / token / layer
```

| `KV_TYPE` | bytes/value | per token | at 32k | at 64k | at 128k |
|---|---|---|---|---|---|
| `f16` | 2.0 | 68.0 KiB | 2,176 MiB | 4,352 MiB | 8,704 MiB |
| **`q8_0`** | **1.0625** | **36.1 KiB** | **1,156 MiB** | **2,312 MiB** | **4,624 MiB** |
| `q4_0` | 0.5625 | 19.1 KiB | 612 MiB | 1,224 MiB | 2,448 MiB |

(`q8_0` stores 32 values in 34 bytes: 32 int8 plus one f16 scale.)

**Cross-check against measurement.** 21,074 MiB resident − 17,092 MiB of weights =
**3,982 MiB** above the model. Of that, 2,312 MiB is KV at the current context, leaving
~1,670 MiB for compute buffers, the CUDA context, the SSM recurrent state (48 blocks ×
6,144 × 128 f32 ≈ 155 MiB with `-np 1`) and the draft branch. That is a plausible split;
the earlier 8.7 GB estimate was not.

**Uncertainty worth flagging.** Whether block 64's KV is allocated at the full `n_ctx` or
only for the short draft window is not determined here. If it is the latter, KV is
2,176 MiB rather than 2,312 and there is ~136 MiB more headroom than the tables below
assume. The tables take the conservative branch.

---

## 7. VRAM headroom and the real `CTX` ceiling

**Question.** Can `CTX` be raised, and how far before it stops fitting?

**Baseline, measured.** `21,074 MiB of 24,576` in use → **3,502 MiB free**. Weights are
17,092 MiB of that.

Each additional 32,768 tokens of context costs 1,156 MiB at `q8_0`:

| `CTX` | KV | Δ vs now | est. total | margin | |
|---|---|---|---|---|---|
| 32,768 | 1,156 MiB | −1,156 | 19,918 | 4,658 | |
| **65,536** | **2,312 MiB** | **—** | **21,074** | **3,502** | current |
| 98,304 | 3,468 MiB | +1,156 | 22,230 | 2,346 | fits |
| 131,072 | 4,624 MiB | +2,312 | 23,386 | 1,190 | fits, thin |
| 163,840 | 5,780 MiB | +3,468 | 24,542 | 34 | **no** |
| 262,144 | 9,248 MiB | +6,936 | 28,010 | −3,434 | **no** — the model's own maximum |

**Result. 131,072 is the practical ceiling** on this card, with ~1.2 GB of margin.
163,840 leaves 34 MiB, which is an OOM waiting for the first long prompt. The model's
declared 262,144 is out of reach in 24 GB at this quantization, by a wide margin.

**On raising `KV_TYPE` instead.** `f16` at the current 65,536 costs 4,352 MiB — 2,040 MiB
more than now, which fits with ~1.5 GB to spare. But **not both**: `f16` at 131,072 is
8,704 MiB and does not fit. The `q8_0` loss is negligible; spend the margin on context.

**Recommendation: do not raise it yet.** Peak context reached under real agent load is
**22,564 tokens — 34% of the window already allocated** — and `truncated = 0` on every
request. llama.cpp allocates the whole KV cache at load time, so raising `CTX` bills the
VRAM immediately whether the window is used or not, on a card that is shared with training
jobs. The trigger to revisit is `truncated = 1` appearing in the logs, not free VRAM
existing.

---

## 8. Not measured

Stated so nobody mistakes silence for a result:

- **A real `CTX` sweep.** Rows 98,304 and 131,072 above are arithmetic, not observations.
  Confirming them means restarting the server and reading `nvidia-smi` and time-to-first-
  token at each setting. Not run because it interrupts a live agent session.
- **Concurrency.** `-np 1` throughout; `requests_deferred` never left 0. Behaviour at
  `-np 2+` — where the KV cache is split per slot, changing every number in §7 — is unknown.
- **Quantization quality.** `UD-Q4_K_XL` was chosen on size. No `Q3` vs `Q4` vs `Q5`
  comparison was made on output quality, only on whether they fit (README §5).
- **`-b`/`-ub` before/after.** See the caveat in experiment 2.
- **KV cache size as reported by the server.** The startup log lines that would state it
  directly had already rotated out of `docker compose logs` by the time they were wanted.
  Raise verbosity and capture them on the next restart.

---

## 9. Configuration changelog

| Setting | From | To | Driven by |
|---|---|---|---|
| `SPEC_TYPE` | `ngram-simple` | `draft-mtp` | exp. 1 — +31% decode, lossless |
| `SPEC_N_MAX` | 4 | 8 | exp. 1 |
| `SPEC_P_MIN` | 0.5 | 0.7 | exp. 1 — 0.5 measured *worse* |
| `BATCH` / `UBATCH` | 512 / 256 | 2048 / 512 | exp. 2 — prefill was throttled |
| `CACHE_RAM` | 8192 | 10240 | exp. 3 — evictions cost 35 s to first token |
| `CTX` | 65536 | 65536 | exp. 7 — headroom exists, need does not |
| `KV_TYPE` | `q8_0` | `q8_0` | exp. 7 — `f16` fits but buys less than context would |

---
## 10. Ornith 1.0 35B on the RTX 4090: is the "optimized" config faster?

**Question.** `docs/profiles/ornith-35b.env` carries two values over from the 3090 work
without having measured them on the card that serves Ornith, and it flags an open one
("setting `SPEC_TYPE=none` and measuring the difference is worth doing; it may well be
faster"). This experiment answers all three on a 4090: does an "optimized" config beat the
deployment that is actually serving the model?

**Fixed conditions.** Two nodes with identical hardware, the same model file, the same
image — the only differences are the four config values under test:

| | Arm A (baseline, live) | Arm B (optimized) |
|---|---|---|
| Node | `a1554-ubu` | `a1553-ubu` |
| GPU | RTX 4090, 24,564 MiB | RTX 4090, 24,564 MiB |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda` | same |
| Model | `Ornith-1.0-35B-UD-Q4_K_XL.gguf`, 22.3 GB | same file, MD5 `02c7a761…` verified identical |
| `-b` / `-ub` | 512 / 256 | 256 / 256 |
| KV (`-ctk` / `-ctv`) | `q8_0` | `q4_0` |
| `--spec-type` | `ngram-simple` | `none` |
| shared | `-c 131072 -ngl all -np 1 -fa on --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0.0 --reasoning-format deepseek`, no `--cache-ram` | same |

Neither container exposes the four values on their own, so this A/B varies them **together** —
it answers "is B worth running", not "which knob was responsible". The shared values are
the ones from `docs/models.md` §6.

**Method.** Same prompt on both arms, one request per run, timings read from the API's
`timings` field (`prompt_per_second` / `predicted_per_second`). Because llama.cpp's prompt
cache turns a cache-miss into a near-instant `prompt_n=4` prefill, every prefill run used a
unique leading marker in the user content to force a clean, honest cache-miss measure
(`prompt_n` verified non-trivial). Workloads, in ascending order of representativeness:

- **1.2k short** coding prompt, 5 rounds each arm — reported for decode only; at 37 t/s in
  §0 the prefill number on a short prompt is meaningless.
- **16k synthetic** context, generated random text (not repeated), one run each arm.
- **128k synthetic** context, random text — with a caveat: the generator repeats ~20 words,
  which is pathological for the n-gram drafter and inflates Arm A's decode advantage. Read
  the 35.6k real number as the representative one.
- **35.6k real** the actual `llamacpp-compose` repo tree plus the analysis request that
  prompted all this, 3 rounds each arm.

Prefill after the first request hits the warm prompt cache only if the prompt is identical;
unique markers avoided that. Decode numbers came from short prompts (no cache interaction).

**Results.** Mean prefill / decode tokens per second, one client:

| Workload | Arm A (baseline) | Arm B (optimized) |
|---|---|---|
| 1.2k short — decode | 146 | 145 |
| 16k synthetic | 5,538 / 137 | 5,513 / 135 |
| 128k synthetic | 4,136 / 99 | 4,133 / 94 |
| **35.6k real** | **4,525 / 127** | **4,481 / 125** |

VRAM after the model settled (`nvidia-smi`, MiB of 24,564): **Arm A 23,117** (1,447 free),
**Arm B 22,463** (2,101 free) — a **~0.65 GB** saving from `q4_0` KV.

**What the three levers actually did:**

- **`--spec-type none` did not speed decode.** This answers the open question in
  `docs/profiles/ornith-35b.env` in the negative: removing the n-gram drafter was not
  faster in any workload; Arm A's baseline (with `ngram-simple`) was equal or, at longer
  context, up to ~5% ahead in decode. The profile's "it may well be faster" reads the wrong
  way — keep `ngram-simple`. Consistent with §3's measure of 1.07 tokens per forward pass.
- **KV `q8_0` → `q4_0` is throughput-neutral.** Prefill and decode were the same within
  noise on both arms. The cache is small (1,360 MiB at this context) and decode is dominated
  by weight reads, not KV reads. `q4_0` buys **+0.65 GB of VRAM headroom**, nothing in speed.
  It is the only lever that helps the "under 1.5 GB free" margin `docs/profiles/ornith-35b.env`
  calls out — at a cost of ~1–2% decode.
- **`-b 256` equal to `-b 512`.** The 4090 honestly prefills at 4,136–5,538 t/s even at
  `-b 512 -ub 256`. That directly contradicts the "512/256 throttles prefill" claim applied
  to this model in `docs/models.md` §6 — that was a 3090 observation (988 t/s) and does not
  hold on a 4090.

**Result. Arm A (the running deployment) is the optimal throughput config, and the
"optimized" Arm B adds nothing but VRAM headroom.** Prefill was identical within noise, and
Arm A held a small decode edge in every test. There is no reason to switch. The only change
ever worth considering is KV `q4_0` for +0.65 GB headroom at a 1–2% decode cost, which the
restraint about shared 24 GB cards may or may not justify.

**Not measured (stated so it is not mistaken for a result):** `-b 2048 -ub 512` (the 3090's
chosen values) was not run on the 4090 — the A/B is baseline-512/256 vs 256/256 only, so it
does not re-test the 3090 improvement. `--cache-ram` was absent on both arms; its case
stands on the 3090 latency result (§3), not here. No concurrency (`-np 1` throughout).

---

## 11. A third model, and a second card generation: Ornith 1.5 35B-A3B

Experiments 1–9 are Qwen3.8-27B and experiment 10 is Ornith 1.0 on a 4090. This one is
Ornith 1.5 35B-A3B, back on the 3090, run on 2026-08-20, and it is worth reading precisely
because **it reverses three of the conclusions above**. None of the three announces itself; each looks like the earlier
result still holds until it is measured.

### Fixed conditions

| | |
|---|---|
| Host | `udesa` — RTX 3090, 24,576 MiB VRAM, 27 GiB system RAM |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda`, pulled 2026-08-14 |
| Model | `Ornith-1.5-35B-Q4_K_M.gguf`, 21,713,462,848 B (20.2 GiB), `ornith-ai/Ornith-1.5-35B-A3B-GGUF` |
| Architecture | `qwen35moe` — MoE, 256 experts, 8 active (~3B active params), 41 blocks, 11 with KV, `head_count_kv 2` |
| Server | `-ngl all`, `-np 1`, `-fa on`, `-b 2048 -ub 512` |
| Sampling | `--temp 1.0 --top-p 0.95 --top-k 20` — read from the GGUF, see below |
| Workload | fixed ~3.5k-token prompt, `n_predict 256`, `cache_prompt false`, best decode rate of 3 runs |

Five batches, container recreated between every row, each batch ending in a control run
that repeats its first row. **The controls came back identical** — 161.00 vs 161.00 in
batch 1, 160.42 against a 161.00 baseline in batch 2, 162.63 vs 161.28 for `q8_0` at 262k.
The 3090 does not thermally drift over a benchmark block the way the laptop 3070 does, so
every difference below is an effect rather than noise.

### 11.1 Two things that must be read from the file, not inferred

`scripts/gguf-info.py` on the actual `.gguf`:

- **Sampling is in the metadata.** `general.sampling.temp 1.0 / top_k 20 / top_p 0.95`.
  The model card's Python examples say `temperature=0.6`, but every benchmark reported in
  that same card was run at 1.0. The file agrees with the benchmarks. Trust the file.
- **The MTP head survived quantization**: `blk.40.nextn.*` is present,
  `nextn_predict_layers 1`. This is the first GGUF in this repo's history where it did —
  the Unsloth builds of Ornith 1.0 dropped it. Which made the next experiment worth running
  and, as it turns out, worth writing down.

### 11.2 Speculative decoding — the `draft-mtp` result does not transfer

**Question.** Experiment 1 measured +31% from `draft-mtp` on Qwen3.8-27B. This model ships
the heads too. Same win?

| `SPEC_TYPE` | `N_MAX` | decode | acceptance | VRAM |
|---|---|---|---|---|
| **`none`** | — | **160.4 t/s** | — | **21,214 MiB** |
| `ngram-simple` | — | 158.0 t/s | 15% (37/240) | 21,214 MiB |
| `draft-mtp` | 8 | 122.0 t/s | 87% (173/198) | 22,480 MiB |
| `draft-mtp` | 3 | 129.5 t/s | 88% (142/162) | 22,166 MiB |

**Result. `draft-mtp` costs 24% of decode here.** Not a tuning failure: acceptance is 87–88%
in both runs, so the drafter is guessing right almost every time and still loses. Shortening
the draft from 8 to 3 recovers part of it and stays far below doing nothing at all.

**Why.** Speculation trades extra compute for fewer sequential steps, and only pays when a
token from the target model is expensive. This is a mixture of experts with ~3B active
parameters: a token is cheap, so the verification pass never amortizes. Qwen3.8-27B is
dense — every one of its 27B parameters runs per token — which is exactly the regime where
speculation wins.

**The rule to carry forward: judge `draft-mtp` by ACTIVE parameters, not by file size.** A
20 GiB MoE behaves, for this decision, like a small model. It also costs 1.3 GB of VRAM,
which at the context chosen below is VRAM that does not exist.

### 11.3 KV cache type — `f16` is faster than `q8_0`, not just more accurate

`.env.example` says `q8_0` "uses half of what f16 does, at negligible loss". On this model
the loss is negative — you pay speed for the privilege:

| `CTX` | `q8_0` | `f16` | Δ |
|---|---|---|---|
| 65,536 | 160.4 t/s | **168.0 t/s** | +4.7% |
| 131,072 | 161.5 t/s | **167.7 t/s** | +3.8% |

**Why.** The model barely has a KV cache: 11 of 41 blocks carry one, at `head_count_kv 2`.
Quantizing something that small saves little bandwidth, while the dequantization it adds to
every attention operation costs real time. `q8_0` earns its keep when the cache is large —
Qwen3.8-27B at 36.1 KiB/token — and stops earning it when the cache is small.

**`f16` is nonetheless not what this profile ships**, for the reason in 11.5: it does not
fit at the context that matters more.

### 11.4 Mixed K/V types — a silent 3.6x slowdown

**Question.** `docker-compose.yml` exposes `KV_TYPE_K` and `KV_TYPE_V` separately. K is the
more quantization-sensitive of the two, so an exact K with a quantized V looks like the
obvious way to buy back precision at half the VRAM. Does it work?

| K / V | `CTX` | prefill | decode | VRAM |
|---|---|---|---|---|
| `q8_0` / `q8_0` | 131,072 | 3,146 t/s | 161.5 t/s | 22,086 MiB |
| `f16` / `q8_0` | 131,072 | **241 t/s** | **44.5 t/s** | 22,322 MiB |
| `f16` / `q8_0` | 196,608 | **259 t/s** | **45.0 t/s** | 23,302 MiB |

**Result. No. Decode collapses to a quarter and prefill to a thirteenth.** The flash
attention kernel has no path for mismatched K and V types and falls back to a generic one.

**Is it the mismatch or the `f16`?** `.env.example` recommended a *different* mixed pair —
`q8_0` for K, `q4_0` for V — so generalizing from one combination would have been exactly
the error this notebook keeps warning about. Measured at `CTX=262144`, with both uniform
types as controls:

| K / V | prefill | decode | VRAM |
|---|---|---|---|
| `q8_0` / `q8_0` | 3,252 t/s | 161.4 t/s | 23,830 MiB |
| `q4_0` / `q4_0` | 3,258 t/s | 160.4 t/s | 22,550 MiB |
| `q8_0` / `q4_0` | **244 t/s** | **58.9 t/s** | 22,442 MiB |

**It is the mismatch.** Any uniform type runs at full speed, `q4_0` included; any mismatched
pair collapses. The quantization level is not the variable — whether K and V agree is.

Two consequences. First, the suggestion `.env.example` used to carry is a performance bug,
and has been replaced with this table. Second, `q4_0` uniform is a real option that nobody
had noticed: same speed as `q8_0` and **1,280 MiB cheaper** at 262k. It is not recommended
here only because its accuracy cost is unmeasured (11.7), but it is the lever to reach for
if headroom is ever needed.

**The part that matters: there is no warning.** `docker inspect` confirms the flags arrive
as `-ctk f16 -ctv q8_0 -fa on`; the log contains nothing about a fallback; the server is
healthy and answers correctly. Only the throughput says anything, and only if you are
measuring. Set `KV_TYPE` and leave `KV_TYPE_K`/`KV_TYPE_V` alone unless you are prepared to
benchmark the result.

### 11.5 The `CTX` ceiling, measured rather than estimated

VRAM in use at idle, `spec=none`, every cell an observation:

| `CTX` | `q8_0` | `f16` |
|---|---|---|
| 65,536 | 21,214 MiB | 21,702 MiB |
| 131,072 | 22,086 MiB | 23,046 MiB |
| 196,608 | 22,958 MiB | **OOM** — `failed to allocate compute pp buffers` |
| 262,144 | **23,830 MiB** | **OOM** — `failed to allocate buffer for kv cache` |

`q8_0` is exactly linear: **872 MiB per 65,536 tokens = 13.6 KiB/token**, extrapolating back
to ~20,342 MiB at zero context. `f16` costs 21.0 KiB/token and runs out between 131k and
196k.

**`gguf-info.py` computes 11.7 KiB/token — it undershoots by 17%**, because it models the
cache and not the context-dependent compute buffers. On the 9B it undershot by 29%. The sign
of the error has been the same every time; leave room for it.

**Context is free in speed.** 160.4 / 161.5 / 161.6 / 162.6 t/s at 65k / 131k / 196k / 262k.
It costs VRAM and nothing else, which makes the ceiling the only question worth asking.

### 11.6 Result

**`CTX=262144`, `KV_TYPE=q8_0`, `SPEC_TYPE=none`** — 23,830 MiB of 24,576, 162.6 t/s decode,
3,271 t/s prefill, and the model's full declared context window.

The one genuine trade-off in the whole exercise is this pair:

| | decode | context | KV | free VRAM |
|---|---|---|---|---|
| `q8_0` @ 262,144 | 162.6 t/s | **262,144** | quantized | 746 MiB |
| `f16` @ 131,072 | **167.7 t/s** | 131,072 | **exact** | 1,530 MiB |

3% of decode against twice the context and a quantized cache. **262k wins on a dedicated
card**; the `f16` row is the right answer on a shared one, where 746 MiB of headroom is an
OOM waiting for someone else's job. Everything else — the drafter, the mixed cache — is not
a trade-off but a mistake with a measurement attached.

The full configuration is `docs/profiles/ornith15-35b.env`.

### 11.7 Not measured

- **Output quality.** Every claim above is about speed and VRAM. That `q8_0` KV costs
  little accuracy is inherited belief, not something tested here, and it is the assumption
  the 262k recommendation rests on. The same gap is why `q4_0` uniform — same speed,
  1,280 MiB cheaper — is documented in 11.4 but not recommended.
- **Whether the mixed-cache collapse also hits the dense reference model.** 11.4 was
  measured on `qwen35moe` only. The `.env.example` suggestion it invalidates was written
  for Qwen3.8-27B, and confirming it there means swapping the served model, so it was not
  run. The warning is therefore scoped to what was observed.
- **Concurrency.** `-np 1` throughout, as in experiments 1–9.
- **`BATCH`/`UBATCH`.** Left at 2048/512 from the Qwen tuning. Prefill sat at ~3,270 t/s
  and was never the bottleneck, so they were not swept.
- **YaRN beyond 262,144.** The card claims ~1M tokens with a scaling factor of 4.0, but
  validates it under vLLM and SGLang. Untested on llama.cpp, and there is no VRAM for it.
