# EXPERIMENTS

A lab notebook for the tuning done on this server: what was changed, what was measured,
and what the number turned out to mean. `README.md` documents the configuration that came
out of it; this file documents how it was arrived at, including the measurements that
argued *against* a change.

Every number here was measured on the machine described below. None of it is quoted from a
model card or a benchmark elsewhere.

**Four models and three cards share this notebook.** Experiments 1-9 are Qwen3.8-27B, a
dense model, on an RTX 3090. Experiment 10 is Ornith 1.0 35B on an RTX 4090 - and finds
that a 3090 conclusion about batch size does not hold there. Experiment 11 is Ornith 1.5
35B-A3B, a mixture of experts, back on the 3090 - and reverses three more. Experiment 12
is an 8 GB laptop RTX 3070, where the constraint stops being throughput and becomes VRAM,
and where one of the two findings is that a whole sweep measured the GPU's temperature
instead of its subject. Read 10, 11 and 12 before carrying any tuning decision from here
to a new model or a new card. Experiment 13 puts the section 11 model on the section 12
card - 20.2 GiB of weights on 8 GiB of VRAM - and finds that when the card is the binding
constraint, every feature that costs VRAM is charged twice.

---

## 0. Fixed conditions

Unless an experiment says otherwise, all of it ran under these conditions. **Experiments 10
and 11 do say otherwise** and declare their own.

| | |
|---|---|
| Host | `udesa` — RTX 3090, 24,576 MiB VRAM, 27 GiB system RAM, 31 GiB swap |
| Driver | 595.84 |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda`, build 10335 — **see 1.1: a later build changed the absolute numbers by ~28%** |
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

### 1.1 The absolute numbers above are stale; the ranking is not

Re-measured 2026-08-21 on the same host, same model file, same `.env`, with the only
change being that the image had been re-pulled in the meantime — **build 10335 → 10423**.
Three runs of ~700 tokens, `draft-mtp` at `N_MAX=8`, `P_MIN=0.7`:

| | build 10335 | build 10423 |
|---|---|---|
| decode | 53.5 t/s | **62.8 / 68.8 / 73.4 t/s** |
| acceptance | 62% | 63–79% |

**About +28% for pulling the image.** Nothing in this repository changed.

Two things follow. **The comparisons in this notebook stay valid and its absolute numbers
do not** — every row of a given table was measured against its own controls on one build,
so `draft-mtp` beating `ngram-simple` by 31% is a result, while "53.5 t/s" is a reading
from a binary that no longer exists. Treat every t/s figure here as carrying an implicit
build number, and re-measure the baseline before comparing against anything external.

And **this is the one variable no experiment in this notebook holds fixed on purpose.**
Section 0 pins the host, the model and the flags; the image is pinned by a floating tag.
Capture `docker run --rm <image> --version` alongside the numbers from here on.

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

### 11.8 A third-party claim that the shipped MTP head is untrained

Reported on the model's HuggingFace discussion page (community thread #10, "mtp.* tensors
look like random init, not trained weights"), **not verified here**. Recorded because it
bears directly on 11.2 and because it contradicts it.

**The claim.** Every projection in the shipped `mtp.*` tensors has a standard deviation of
exactly 0.0200 with clean Gaussian statistics — which is `initializer_range=0.02`, i.e. the
head was never trained. Norm weights sit near 0.02 where a trained head's would be ~1. The
same poster reports the head accepting **~13%** of drafted tokens in llama.cpp; a second
contributor's evaluation suite puts the native head at **37.2%** acceptance, mean accepted
length 2.116.

**Why that is a problem for 11.2.** This notebook measured the shipped head at **87–88%
acceptance** (173/198 and 142/162). Those are not the same regime, and one of the two is
measuring something other than what it says:

- 11.2 ran a fixed ~3.5k prompt with `--spec-draft-p-min 0.7`, so the drafter emits only
  tokens it is confident about. Acceptance conditional on *having drafted* can be high
  while the head is still poor — but 198 drafts inside a 256-token generation is not a
  drafter holding back.
- The community figures come from different harnesses (one from a separate Q8_0 draft
  model rather than `--spec-type draft-mtp`, one from a private evaluation suite), and
  llama.cpp's `draft acceptance` counter may not mean what either assumes.

**It does not change 11.2's recommendation, and it does weaken 11.2's explanation.** The
measurement stands — `draft-mtp` was 24% slower than `none` on this hardware — and
`SPEC_TYPE=none` remains right for the 3090 profile. What is now uncertain is the *reason*
given for it. 11.2 argued the loss is structural: a MoE token is too cheap for
verification to amortize, even at 87% acceptance. If the acceptance figure is not what it
appears, that argument rests on less than it seemed to. **A good drafter has not actually
been tested on this model here.**

**What is available if someone wants to settle it.** Two replacement heads and one
alternative drafter, all reported by third parties and none tried here:

| | source | reported acceptance |
|---|---|---|
| Qwen3.6-35B-A3B MTP head, grafted | direct graft, no training | 50.2%, mean run 2.51 |
| `shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY` | 12K KL distill | 69.3%, mean run 3.08 |
| Qwen3.6 DFlash head | `--spec-type draft-dflash`, ~236 MB at Q4 | 90% at 8k, decaying to ~49% at 252k |

The image this repo runs already has the flags for all three: `--spec-draft-model` (`-md`)
loads an external head, and `--spec-type` accepts `draft-dflash`. Neither is exposed in
`docker-compose.yml`, so trying one means adding a flag.

Two cautions carried over from the same thread, also unverified: the DFlash head is
reported to break multimodal use, and its acceptance is reported to decay with context —
crossing below the MTP head somewhere around 128k, which is inside the 262,144 this
profile actually runs at.

**The experiment worth running**, and the reason it is worth running despite 11.2: on a
card where the model fits, speculation lost because the token was cheap. In the
CPU-offload configuration of section 13 a token is **expensive** — 597 MiB over the memory
bus — which is exactly the regime where speculation is supposed to pay. Whether it does
depends on something not measured anywhere here: how much the union of experts selected by
several drafted tokens costs to read against the cost of one token's eight.

---

## 12. The smallest card: Ornith 1.0 9B on an RTX 3070 Laptop

The first three experiments ran on 24 GB cards in machines that do nothing else. This one
is an 8 GB laptop GPU, and both of its findings come from that difference rather than from
the model. Neither is about which setting is faster; both are about **measurements being
wrong in a way that does not announce itself** — one because the tool undershoots, one
because the hardware moves while you are measuring it.

### Fixed conditions

| | |
|---|---|
| Host | `personal` — Alienware m15, Ryzen 9 5900HX (8c/16t), 62 GiB RAM |
| GPU | RTX 3070 Laptop, **8,192 MiB** VRAM, compute capability 8.6 |
| Driver | 595.84 |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| Model | `Ornith-1.0-9B-UD-Q4_K_XL.gguf`, 5,980,251,232 B (5.98 GB), `unsloth/Ornith-1.0-9B-GGUF` |
| Architecture | `qwen35`, 32 blocks, hybrid — 8 with KV, `head_count_kv 4`, `key_length`/`value_length` 256 |
| Server | `-ngl all`, `-np 1`, `-fa on`, `-ctk q8_0 -ctv q8_0` |
| Sampling | `--temp 0.6 --top-p 0.95 --top-k 20` (the model card states these outright) |

This card differs from the other three in a way that matters beyond its size: **the RTX
also drives the external monitors** on this laptop, while the integrated Vega drives the
internal panel. VRAM headroom is not only a margin against OOM, it is what a monitor gets
plugged into.

### 12.1 `draft-mtp` is not available, and the upstream config says it is

The model's upstream `config.json` reports `mtp_num_hidden_layers: 1` — the original
model does ship MTP heads. `scripts/gguf-info.py` against the file that was actually
downloaded reports:

```
  MTP heads (nextn.*)      0  []
```

**Unsloth's conversion drops them.** That is why third-party repackagings with "MTP" in
the name exist for this model at all.

**The rule: the config of the original is not evidence about the file you downloaded.**
Experiment 11 later found the opposite case — a build where the head *did* survive — which
is what makes this worth stating as a check rather than as a fact about Ornith. Run
`gguf-info.py` on the file in your hands.

`SPEC_TYPE=ngram-simple` was set as a result. Whether it beats `none` here is still
unmeasured (12.4).

### 12.2 The `CTX` sweep, and `gguf-info.py` undershooting by 29%

VRAM in use at idle, container recreated at every value:

| `CTX` | VRAM | free of 8,192 MiB |
|---|---|---|
| 16,384 | 5,783 MiB | 2,409 |
| 32,768 | 6,135 MiB | 2,057 |
| 65,536 | 6,839 MiB | **1,353** |
| 98,304 | 7,543 MiB | 635 |
| 131,072 | **OOM** — `cudaMalloc failed: out of memory` on the compute buffers | — |

Exactly linear: **704 MiB per 32,768 tokens = 22.0 KiB/token**.

`gguf-info.py` computes **17.0 KiB/token**, so it undershoots by **29%**. The tool is not
wrong about the cache — the server confirms 2,176 MiB at 131,072 cells over 8 layers,
which is precisely what the tool predicts. The missing ~5 KiB/token is **context-dependent
compute buffers, which the formula does not model at all**.

The same comparison on the 35B in experiment 11.5 undershot by 17%. **The sign of the
error has been the same every time.** On a 24 GB card that is slack; on 8 GB it is the
difference between loading and not.

`CTX=65536` was chosen over the 98,304 that also loads. 635 MiB of headroom is not a
margin on a card that has to absorb an external monitor being plugged in.

**The OOM is not graceful.** It fails allocating compute buffers, after the weights are
already resident, minutes into loading — not with an up-front check that the request is
impossible.

### 12.3 The measurement that was thermal drift, not context

A first pass at "does raising `CTX` cost decode speed?" produced this, and it looks like a
clean answer:

| `CTX` | decode |
|---|---|
| 16,384 | 58.4 t/s |
| 98,304 | 56.4 t/s |

A 3.4% decline, monotonic, in the direction theory predicts — attention over a longer
cache should cost more. It is entirely an artifact.

The GPU went from **59 °C to 86 °C** over the course of the sweep. Re-running the **first**
configuration once the card was hot gave **56.5 t/s**: the whole effect reproduced with the
context unchanged. It was never about context.

**Result: raising `CTX` costs VRAM and nothing else, from 16k to 98k.** Consistent with
11.5, which found the same on the 3090 across 65k–262k.

**The transferable part is the method, not the number.** A laptop GPU throttles over a
benchmark block, so a sweep run in ascending order will manufacture a monotonic decline in
whatever variable happens to be on the x-axis. Two defences, both used from here on:

- **End every batch by repeating its first row.** If the control does not come back, the
  batch measured temperature.
- **Log `nvidia-smi --query-gpu=temperature.gpu` alongside every timing.**

Experiment 11's remark that "the 3090 does not thermally drift over a benchmark block the
way the laptop 3070 does" is a reference to this. On the 3090 the controls came back
identical — 161.00 vs 161.00 — which is what licensed reading its differences as effects.

### 12.4 Not measured

- **`SPEC_TYPE=none` against `ngram-simple`.** The open experiment on this model. `ngram`
  earned 1.07 tokens per forward pass on the 35B (experiment 3) and 15% acceptance in
  11.2, so the expected gain is small in either direction — but it is unmeasured here.
- **Decode throughput as a headline number.** The 56–58 t/s above were taken during the
  thermal investigation, not under a controlled workload, and are quoted only as evidence
  about the confound.
- **Prefill.** Never measured on this card.
- **The multimodal projector.** `unsloth/Ornith-1.0-9B-GGUF` ships an `mmproj-*.gguf` and
  the model is multimodal; it is served text-only here because the projector costs
  ~877 MiB and at `CTX=65536` this card does not have it. Untested, not unsupported.
- **Ornith 1.5 9B.** `Ornith-1.5-9B-Q4_K_M.gguf` (5.63 GB) is on this machine and has
  never been served. `gguf-info.py` says it is **dense** — `qwen35`, 32 blocks, 8 with KV,
  no expert tensors and no MTP heads — at 17.0 KiB/token computed, so the same 29%
  correction predicts ~22 KiB/token and the same `CTX` ceiling as the 1.0. It is the
  natural control for the offload experiment in section 13: a model that fits entirely in
  VRAM, against one four times larger that does not.

The full configuration is `docs/profiles/ornith-9b.env`.

---

## 13. A 20 GiB model on an 8 GiB card: MoE expert offload

Every experiment so far served a model that fits on its card. This one does not: Ornith 1.5
35B-A3B is 20.2 GiB and the RTX 3070 Laptop of section 12 has 8. It runs anyway, at 48 t/s,
because of a property of the architecture rather than a trick — and the interesting results
are not that it works but **which of this notebook's conclusions the offload regime
rewrites**.

### Fixed conditions

| | |
|---|---|
| Host | `personal` — Ryzen 9 5900HX (8c/16t), 62 GiB RAM, DDR4 dual channel |
| GPU | RTX 3070 Laptop, **8,192 MiB** |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| Model | `Ornith-1.5-35B-Q4_K_M.gguf`, 21,713,462,848 B — byte-identical to the file in section 11 |
| Server | `-ngl all`, `-np 1`, `-fa on`, `-b 2048 -ub 512`, `-c 65536` |
| Sampling | `--temp 1.0 --top-p 0.95 --top-k 20` (11.1) |
| Workload | fixed 13,798-byte prompt (3,879 tokens), `n_predict 256`, `cache_prompt false`, best of 3 |

Every batch ends by repeating its first row, per 12.3. **The controls came back**: 40.23 vs
40.20, 47.88 vs 48.14, 45.31 vs 44.94, 48.32 vs 47.96 — under 0.8% in all four, while the
GPU went from 61 °C to 79 °C over the session. Differences below ~1% are noise here;
everything reported as a result is larger than that.

Section 11 measured the same model on a dedicated RTX 3090 at **162.6 t/s**. That is the
number this section is against.

### 13.1 Why it fits at all: read the tensor index, not the file size

Summing the GGUF tensor index by name, splitting `*_exps` from everything else:

| | |
|---|---|
| expert tensors (`*_exps`) | **19,098 MiB — 92.3%** of the file |
| everything else | 1,599 MiB — 7.7% |
| per block | 466 MiB, over 41 blocks |
| active per token | **8 of 256 experts → 597 MiB** |

`-ncmoe N` keeps the experts of the first N blocks in host RAM. So the card holds 1,599 MiB
of dense weights plus the KV cache, and the memory bus carries 597 MiB per token — not the
20 GiB the file weighs, and not the 19 GiB that was offloaded.

Host RAM read bandwidth, measured with a STREAM-like probe over 1 GiB, 16 threads:
**42.5 GB/s**. That puts a hard ceiling of **68 t/s** on decode with every expert on the
CPU, before any inefficiency.

**This is why the decision is not the same as `-ngl`.** Offloading a dense layer means
every token pays to move it. Offloading a MoE block's experts means every token pays for
the experts it selects, which is 1/32 of them here.

### 13.2 The `N_CPU_MOE` sweep

Container recreated at every row. `CTX=65536`, `KV_TYPE=q8_0`, `SPEC_TYPE=none`.

| `N_CPU_MOE` | blocks in VRAM | VRAM used | prefill | decode |
|---|---|---|---|---|
| 41 | 0 | 2,826 MiB | 169.8 t/s | 40.23 t/s |
| 37 | 4 | 4,320 | 182.7 | 42.35 |
| 34 | 7 | 5,816 | 197.1 | 45.16 |
| **31** | **10** | **7,178** | **212.9** | **47.88** |
| 30 | 11 | 7,610 | 219.2 | **48.40** |
| 29 | 12 | — | — | **OOM** |
| 41 (control) | 0 | 2,826 | 169.6 | 40.20 |

Linear and unsurprising: **435 MiB and +0.77 t/s per block moved to the GPU**. Full offload
reaches **59% of the 68 t/s bandwidth ceiling**, which is about what a quantized gemv gets.

**Do not compute the headroom as `total - used`.** `nvidia-smi`'s own `memory.free` reads
**~350 MiB lower** than that subtraction, because `used` and `free` do not sum to `total` —
the driver's reservation appears in neither. At the chosen row the arithmetic says 1,014
MiB and the measured `memory.free` is **649 MiB**; at `N_CPU_MOE=30` it says 582 and the
real figure is ~233. Read the third column of `nvidia-smi --query-gpu=memory.free`, which
is what the OOM at 29 actually ran out of.

**`N_CPU_MOE=31` is the recommendation, not the 30 that measured fastest.** The last block
buys **0.5%** — inside the noise band established by the controls — and takes the real
headroom from 649 MiB to ~233. On this laptop the RTX also drives the external monitors
(12), so headroom is not only a margin against OOM: it is what a monitor gets plugged
into.

### 13.3 Speculation still loses — and the reason has changed completely

11.2 measured `draft-mtp` 24% slower than `none` on the 3090 and explained it by the token
being cheap: a MoE forward pass is ~3B active parameters, so verification never amortizes.
**That explanation predicts speculation should win here**, where a token costs 597 MiB over
a 42.5 GB/s bus. It does not.

At `N_CPU_MOE=34`:

| `SPEC_TYPE` | `N_MAX` | VRAM | prefill | decode | acceptance |
|---|---|---|---|---|---|
| **`none`** | — | 5,816 MiB | 197.8 t/s | **45.31 t/s** | — |
| `ngram-simple` | 8 | 5,816 | 197.5 | 44.82 | 8.3% (4/48) |
| `draft-mtp` | 8 | 7,082 | 195.9 | 41.86 | **79.8%** (99/124) |
| `draft-mtp` | 3 | 6,766 | 195.8 | 43.92 | 79.3% (92/116) |
| `none` (control) | — | 5,816 | 197.5 | 44.94 | — |

`draft-mtp` costs **7.6%** — much less than the 24% on the 3090, so the token really did
get more expensive, but not enough.

**And the raw comparison understates it, because in this regime VRAM converts directly into
decode speed.** `draft-mtp` took 1,266 MiB more than `none` — which is **three blocks of
experts** that had to go back to the CPU. Comparing at equal VRAM instead:

| | VRAM | decode |
|---|---|---|
| `draft-mtp`, `N_CPU_MOE=34` | 7,082 MiB | 41.86 t/s |
| **`none`, `N_CPU_MOE=31`** | 7,178 MiB | **47.88 t/s** |

**At the same VRAM, not speculating is 14.4% faster.** This is the finding of the section:
where the card is the binding constraint, any feature that costs VRAM is charged twice —
once for what it does, and once for the expert blocks it evicts. `draft-mtp` at 80%
acceptance is not a bad drafter. It is a drafter that has to pay rent.

**On the acceptance discrepancy of 11.8.** A third-party analysis claims this model's
shipped MTP head was never trained and accepts ~13% in llama.cpp. Measured here on
different hardware, in a different regime, months of drift apart in nothing but the file:
**79.8% and 79.3%**, reproducing 11.2's 87–88% far better than it reproduces 13%. Whatever
llama.cpp's `draft acceptance` counter means, it means the same thing on both machines, and
this notebook's figure is not a fluke of one host. That does not refute the tensor-statistics
claim — a head can be untrained and still have its drafts accepted by a model that would
have produced those tokens anyway — but it does say the 13% figure and this counter are not
measuring the same quantity.

### 13.4 `f16` KV, and the compression of every GPU-side gain

11.3 found `f16` KV **4.7% faster** than `q8_0` on the 3090, because this model's cache is
too small for quantization to save more bandwidth than dequantization costs. Here:

| `KV_TYPE` | `N_CPU_MOE` | VRAM | prefill | decode |
|---|---|---|---|---|
| `q8_0` | 31 | 7,178 MiB | 213.5 t/s | 48.32 t/s |
| `f16` | 31 | 7,778 | 213.6 | 48.88 |
| `f16` | 33 | 6,848 | 202.4 | 46.67 |
| `q8_0` (control) | 31 | 7,178 | 213.3 | 47.96 |

**+1.2%, against a control band of 0.7%.** The advantage is real and has nearly vanished,
for the same reason speculation's penalty shrank: the bottleneck left the GPU. Attention is
no longer on the critical path — the memory bus is — so an attention-side optimization has
almost nothing left to optimize.

And it costs 600 MiB, which at 435 MiB per block is more than one block. `f16` at
`N_CPU_MOE=33` — roughly the VRAM of `q8_0` at 31 — is **3.4% slower**. `q8_0` stays.

### 13.5 Result

**`N_CPU_MOE=31`, `CTX=65536`, `KV_TYPE=q8_0`, `SPEC_TYPE=none`** — 7,178 of 8,192 MiB,
**48.3 t/s decode, 213 t/s prefill**, 649 MiB genuinely free.

Host-side, the same configuration holds **15.9 GiB resident** — 15.0 GiB of it file-backed
pages of the `.gguf` rather than anonymous memory, because llama.cpp `mmap`s the weights.
Two consequences worth knowing. `free` reports it as buff/cache and not as used, so the
machine looks far emptier than it is; and the figure **grows with use** — 19,098 MiB of
experts exist but only the ones a request has actually selected are paged in. Under memory
pressure the kernel can drop them without swapping, at the cost of re-reading from disk.

Against the same file on a dedicated RTX 3090 (11.6):

| | RTX 3090, 24 GB | RTX 3070 Laptop, 8 GB |
|---|---|---|
| decode | 162.6 t/s | **48.3 t/s** — 3.4x slower |
| prefill | 3,271 t/s | **213 t/s** — **15x slower** |
| `CTX` | 262,144 | 65,536 |

**Decode degrades gracefully and prefill does not**, and that asymmetry is the thing to
decide on. Decode is 3.4x off a card that costs several times more, which for interactive
use is a good trade. Prefill is 15x off, because it is the one phase where the CPU must
touch nearly every expert: a batch of hundreds of tokens selects most of the 256 experts in
every block, so the 1/32 saving that makes decode viable does not apply. A 20k-token prompt
costs ~95 seconds before the first token.

For chat, that is tolerable. **For agent workloads it is the deciding number** — §4 measured
peak context of 22,564 tokens under real agent load, which is ~105 s of prefill on every
cache miss. `CACHE_RAM` is worth more here than anywhere else in this repo, and is set to
16384 accordingly.

The full configuration is `docs/profiles/ornith15-35b-offload.env`.

### 13.6 Not measured

- **Output quality**, as everywhere else in this notebook. All of the above is speed and
  VRAM. Nothing here checks that offloaded experts produce identical logits, though there
  is no mechanism by which they should not.
- **A trained MTP head.** 13.3 kills speculation with the *shipped* head. The distilled
  replacements of 11.8 report 69% acceptance against this head's ~1.6 mean run and were
  not tried, because obtaining one means converting and quantizing it by hand. 13.7 tested
  the other candidate — a published DFlash head — and found the reasoning that motivated
  both to be wrong.
- **`BATCH`/`UBATCH`.** Left at 2048/512. Prefill is CPU-compute-bound here, a regime
  neither experiment 2 (3090) nor experiment 10 (4090) covers, and `-ub` controls how many
  tokens' worth of experts must be resident per pass — plausibly the largest untouched
  lever on the 15x prefill gap.
- **`-t` / thread count.** llama.cpp's default was used throughout. With the FFN on an
  8-core CPU this is not obviously right.
- **A smaller quantization.** `Q4_K_M` was kept for comparability with section 11. Since
  bytes-per-token is now the binding constraint, a smaller quant should convert almost
  linearly into decode speed: `Q3_K_L` (17.4 GB) and `IQ3_XXS` (15.3 GB) predict **+25%**
  and **+42%** from the arithmetic alone. Untested, and a 3B-active MoE has little room to
  absorb quantization error — each expert is small.
- **Concurrency.** `-np 1`, as everywhere.

### 13.7 The DFlash head: a 236 MB file that costs 2,048 MiB

13.3 concluded that a drafter's VRAM is charged twice, and closed by predicting that the
small published heads — **236 MB at Q4** against `draft-mtp`'s 1,266 MiB — would therefore
be the version worth running. Three independent sources pointed the same way: the
HuggingFace thread of 11.8, a vLLM project serving Qwen3.8-27B on a 3090 that credits
DFlash for most of its speedup, and the flags already present in this image.

**The prediction was wrong, and instructively.**
`Anbeeld/Qwen3.6-35B-A3B-DFlash-GGUF`, `qwen36-35b-a3b-dflash-Q4_K_M.gguf`, 235,691,744 B,
loaded with `--spec-draft-model` against the same Ornith 1.5 35B-A3B:

| `SPEC_TYPE` | `N_CPU_MOE` | drafter on | VRAM | decode | acceptance |
|---|---|---|---|---|---|
| `none` | 41 | — | 2,826 MiB | 40.23 t/s | — |
| `none` | 37 | — | 4,320 | **42.35** | — |
| `draft-dflash` | 41 | GPU | **4,874** | 41.00 | 73–77% |
| `draft-dflash` | 41 | CPU (`DRAFT_NGL=0`) | 4,324 | **16.79** | — |
| `draft-dflash` | 33 | GPU | — | **OOM** — `failed to allocate compute pp buffers` | |
| `draft-dflash` | 31 | GPU | — | **OOM** — `unable to allocate CUDA0 buffer` | |

**A 236 MB file costs 2,048 MiB resident — 8.7x its own size, and more than the 1,266 MiB
of the in-file MTP path it was supposed to undercut.** The weights were never the expense;
the drafter's own context and compute buffers are, and those scale with `CTX` and the batch
rather than with the head. Judging a drafter by its download size is the same category of
error as judging `SPEC_TYPE` by a model's file size (11.2), one level down.

**It loses on its own terms too.** 41.00 t/s against 40.23 for no speculation at the same
`N_CPU_MOE` is +1.9%, barely outside the ±0.8% the controls establish. And the honest
comparison is at equal VRAM, where it does not merely fail to win: `none` at
`N_CPU_MOE=37` uses **554 MiB less** and returns **42.35 t/s**. The drafter is behind
before its rent is counted.

**Pinning it to the CPU does not escape the rent.** `DRAFT_NGL=0` still consumed 1,498 MiB
over baseline — the draft context stays on the GPU regardless — and collapsed decode to
**16.79 t/s, 2.4x slower than not speculating**, because the drafter's forward pass
serializes against expert reads that already have the CPU saturated. On a machine whose
decode is CPU-bound, moving anything else onto the CPU is not a free lunch, it is the
opposite of one.

**Where that leaves speculation on this configuration.** Three drafters measured — the
in-file MTP head (13.3), the n-gram drafter (13.3), and a published DFlash head — and all
three lose to `SPEC_TYPE=none`. The mechanism is the same each time and it is not about
draft quality: acceptance was 79.8%, 8.3% and 73–77% respectively, and the two good ones
both lost. **On a card where the model does not fit, VRAM is the currency and speculation
cannot pay for itself in it.**

That conclusion is scoped to this regime. On the 3090 where the model fits, speculation
also lost (11.2) but for the opposite reason — VRAM was free and the token was too cheap.
On the dense Qwen3.8-27B, `draft-mtp` wins by 31% (§1). Three regimes, three different
answers, one flag.

**Not measured:** a distilled MTP head (11.8), which needs hand conversion; DFlash at a
lower `CTX`, since its buffers scale with context and 65,536 may be most of the 2,048 MiB;
and any of this on a card with headroom, where the rent argument does not apply at all.
