# Serving a different model

Nothing in this repo is specific to Qwen3.8-27B. Switching models is an `.env` edit and a
restart — `docker-compose.yml` and the scripts stay untouched.

What follows is the checklist, then a second model worked all the way through as a real
example: **Ornith 1.0 35B**, which has been running on this setup on an RTX 4090 and
differs from the reference model in almost every way that matters. Comparing the two is
more instructive than either one alone.

---

## 1. The checklist

```bash
# 1. Fetch it, verified
./scripts/download-model.sh <hf-repo> <file.gguf>

# 2. Read what it actually is, before guessing at settings
python3 scripts/gguf-info.py "$MODELS_DIR/<file.gguf>"

# 3. Edit .env:  MODEL_FILE, MODEL_ALIAS, and everything section 2 below flags
$EDITOR .env

# 4. Restart and confirm with a real query
./scripts/serve.sh up && ./scripts/serve.sh test
```

Step 2 is the one people skip. It takes a second and answers the three questions that
decide whether the config will work at all:

| Question | Where the answer is |
|---|---|
| How much VRAM will the KV cache take? | the `KV cache per token` table |
| Does it have MTP heads, i.e. can `draft-mtp` be used? | the `MTP heads (nextn.*)` line |
| Is it hybrid, MoE, or a plain dense transformer? | `full_attention_interval`, `expert_count`, the block breakdown |

---

## 2. The four things that are model-specific

**Sampling.** Every model card carries its own values, and copying them across models
breaks nothing visibly — it just generates worse output, quietly.

> Note: the `general.sampling.*` keys embedded in these `.gguf` files are **not reliable**.
> Both models in the table below ship `temp 1.0, top_k 20` in their metadata, which
> contradicts Qwen's own card (`temp 0.6`) and Ornith's deployed configuration
> (`top_k 0`). Trust the model card, not the file.

**Speculative decoding.** `SPEC_TYPE=draft-mtp` requires the model to ship MTP heads.
Most do not. Check `gguf-info.py` for `MTP heads (nextn.*)`; if it reports 0, use
`ngram-simple` or `none`.

**KV cache size.** Driven by `head_count_kv`, `key_length`/`value_length` and — the term
that gets assumed rather than read — how many blocks actually carry a KV cache. Between
the two models below this varies by 3.4x per token.

**`CTX`.** Follows from the KV arithmetic and the free VRAM. See README §5.

---

## 3. Worked example: Ornith 1.0 35B

`unsloth/Ornith-1.0-35B-GGUF`, file `Ornith-1.0-35B-UD-Q4_K_XL.gguf`, 22.3 GB, MIT
licensed. Base model from Deepreinforce AI.

```ini
# --- Ornith 1.0 35B profile ---
MODEL_FILE=Ornith-1.0-35B-UD-Q4_K_XL.gguf
MODEL_ALIAS=ornith-35b
CONTAINER_NAME=ornith

CTX=131072              # its KV is cheap enough to afford this; see below
KV_TYPE=q8_0
NGL=all

TEMP=1.0                # Ornith's card, NOT Qwen's 0.6
TOP_P=0.95
TOP_K=0                 # zero, i.e. disabled — Qwen wants 20
MIN_P=0.0

REASONING_FORMAT=deepseek

SPEC_TYPE=ngram-simple  # it has NO MTP heads; draft-mtp is not available
BATCH=2048
UBATCH=512
CACHE_RAM=10240
```

### What `gguf-info.py` says about it

```
general.architecture             qwen35moe
qwen35moe.block_count            40
qwen35moe.full_attention_interval 4
qwen35moe.attention.head_count   16
qwen35moe.attention.head_count_kv 2
qwen35moe.expert_count           256
qwen35moe.expert_used_count      8
qwen35moe.context_length         262144

blocks: 40 total | 10 with KV (3, 7, 11 … 39) | 30 linear/SSM | 0 MTP
KV at q8_0: 10.6 KiB/token -> 1,360 MiB at CTX=131072
```

Three things follow, and none of them are guessable from "35B":

**It is a Mixture of Experts.** 256 experts, 8 active per token. The `.gguf` is 22.3 GB
but a forward pass only reads the active experts, so decode is far faster than the file
size suggests. Measured on the live deployment: **147.6 t/s** aggregate generation, against
63.2 t/s for the 27B dense-FFN Qwen. That is a 4090 rather than a 3090, but the cards are
within ~8% of each other on bandwidth — the 2.3x is the MoE, not the GPU.

**It is also hybrid**, same as Qwen3.8: `full_attention_interval 4`, so 10 of 40 blocks
carry KV and 30 are SSM. Combined with `head_count_kv 2` (Qwen has 4), KV costs
**10.6 KiB/token** versus Qwen's 36.1. That is why a 22.3 GB model fits at `CTX=131072`
on the same 24 GB card where a 17.9 GB model was left at 65,536: the bigger model has the
cheaper cache.

**It has no MTP heads.** `nextn.* = 0`. `--spec-type draft-mtp` is simply not an option,
which is why the deployed container uses `ngram-simple`. And that drafter earns very
little here: measured 35% acceptance over only 37 drafts in 6,201 decode steps —
**1.07 tokens per forward pass**, meaning speculation is contributing almost nothing. The
147 t/s above is essentially the unaided rate.

---

## 4. Side by side

| | Qwen3.8-27B | Ornith 1.0 35B |
|---|---|---|
| Architecture | `qwen35` — hybrid, dense FFN | `qwen35moe` — hybrid + MoE (8 of 256 experts) |
| File size | 17.9 GB | 22.3 GB |
| Blocks | 65 | 40 |
| Blocks carrying KV | 17 | 10 |
| `head_count_kv` | 4 | 2 |
| KV per token, `q8_0` | 36.1 KiB | **10.6 KiB** |
| KV at its deployed `CTX` | 2,312 MiB at 65,536 | 1,360 MiB at 131,072 |
| MTP heads | yes — `draft-mtp` | **no** — `ngram-simple` |
| `--temp` / `--top-k` | 0.6 / 20 | 1.0 / 0 |
| License | Apache-2.0 | MIT |
| Measured prefill | 988 t/s | 3,962 t/s |
| Measured decode | 63.2 t/s (with `draft-mtp`) | 147.6 t/s (speculation barely firing) |
| Tokens per forward pass | 3.17 | 1.07 |
| Deployed on | RTX 3090, 21,074 of 24,576 MiB | RTX 4090, 23,117 of 24,564 MiB |

Measurement conditions differ — different cards, different workloads, different context
sizes, and Ornith's deployment has no prompt cache configured. Treat the two decode
figures as "what each delivers in its own deployment", not as a controlled benchmark.

---

## 5. What the comparison is actually worth

Three defaults in this repo look universal and are not:

1. **`SPEC_TYPE=draft-mtp` is model-dependent.** It is the right default for the reference
   model and unavailable on Ornith. Pointing `.env` at a new model without checking will
   at best waste the flag.
2. **KV cost does not track model size.** The larger model here has a cache 3.4x cheaper
   per token. Sizing `CTX` from parameter count instead of from `gguf-info.py` gets this
   backwards.
3. **Throughput does not track model size either.** A 22.3 GB MoE decodes more than twice
   as fast as a 17.9 GB dense-FFN model, because a forward pass reads active weights, not
   stored ones.

The general lesson, and the reason `gguf-info.py` exists: **read the model, do not infer
it from its name.**

---

## 6. Note on the live Ornith deployment

The container currently running on the 4090 predates this repo and was configured by hand:

```
-c 131072 -ngl all -np 1 -b 512 -ub 256 -ctk q8_0 -ctv q8_0 -fa on
--temp 1.0 --top-p 0.95 --top-k 0 --min-p 0.0 --jinja
--reasoning-format deepseek --metrics --spec-type ngram-simple --alias ornith-35b
```

Two differences from what this repo would give it, both worth picking up if it is ever
recreated:

- **`-b 512 -ub 256` throttles prefill.** These are the values this repo moved away from;
  see `EXPERIMENTS.md` §2. Its measured 3,962 t/s is with the handbrake on.
- **No `--cache-ram`**, so there is no prompt cache at all. In an agent loop that is the
  difference between a warm turn and reprocessing the whole conversation —
  `EXPERIMENTS.md` §3 measured that at 35 seconds to first token on a 48k context.

Neither was changed while writing this: it is a shared machine and the container has been
serving for two days.
