# EXPERIMENTS

A lab notebook for the tuning done on this server: what was changed, what was measured,
and what the number turned out to mean. `README.md` documents the configuration that came
out of it; this file documents how it was arrived at, including the measurements that
argued *against* a change.

Every number here was measured on the machine described below. None of it is quoted from a
model card or a benchmark elsewhere.

---

## 0. Fixed conditions

Unless an experiment says otherwise, all of it ran under:

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
