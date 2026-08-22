# Profiles

Complete, working `.env` files, one per model. Copy one over `.env` and start.

```bash
cp docs/profiles/qwen38-27b.env .env
$EDITOR .env                      # MODELS_DIR is the one machine-specific line
./scripts/serve.sh up             # or ./scripts/serve-metal.sh up on macOS
```

| File | Model | Status |
|---|---|---|
| `qwen38-27b.env` | Qwen3.8-27B `UD-Q4_K_XL`, 17.9 GB | **in production** on an RTX 3090; every value measured |
| `ornith-35b.env` | Ornith 1.0 35B `UD-Q4_K_XL`, 22.3 GB | **in production** on a 4090; the exact serving config, A/B-verified optimal — `EXPERIMENTS.md` §10 |
| `ornith15-35b.env` | Ornith 1.5 35B-A3B `Q4_K_M`, 20.2 GB | **in production** on an RTX 3090; every value measured across five benchmark batches — `EXPERIMENTS.md` §11 |
| `ornith-9b.env` | Ornith 1.0 9B `UD-Q4_K_XL`, 5.98 GB | an **8 GB laptop RTX 3070**; `CTX` swept and measured, `SPEC_TYPE` still open — `EXPERIMENTS.md` §12 |
| `ornith15-35b-offload.env` | Ornith 1.5 35B-A3B `Q4_K_M`, 20.2 GB | the **same 8 GB card**, via MoE expert offload to host RAM; 48.3 t/s at 7,178 of 8,192 MiB — `EXPERIMENTS.md` §13 |

## Why these exist

`.env` is gitignored — it holds one machine's live configuration and must not be
committed. That is the right call, but it means a tuned configuration lives in exactly one
place and disappears with the machine.

These files are the durable copy. They are also the honest one: every non-obvious value
carries the measurement that justifies it, so a future reader can tell a tuned setting from
a default nobody thought about. `.env.example` remains the annotated template for someone
setting up from scratch; these are known-good end states.

## Before reusing a profile on a different model

Three of the values are model-dependent in ways that are not guessable from the model's
name, and no two profiles here agree on all three:

| | `qwen38-27b` | `ornith-35b` | `ornith15-35b` | `ornith-9b` |
|---|---|---|---|---|
| `SPEC_TYPE` | `draft-mtp` — it ships MTP heads | `ngram-simple` — it does not | `none` — it ships them and they still lose | `ngram-simple` — the original has heads, this build dropped them |
| KV per token at `q8_0` | 36.1 KiB | 10.6 KiB | 11.7 KiB computed, **13.6 measured** | 17.0 computed, **22.0 measured** |
| `TEMP` / `TOP_K` | 0.6 / 20 | 1.0 / 0 | 1.0 / 20 | 0.6 / 20 |

The third column is the cautionary one. `ornith15-35b` is a mixture of experts with ~3B
active parameters, and that single fact inverts two defaults this repo otherwise treats as
safe: `draft-mtp` costs 24% of decode instead of gaining 31%, and `q8_0` KV is *slower*
than `f16` rather than a free saving. Neither failure is visible without benchmarking —
both configurations load, report healthy, and answer correctly.

**Judge `SPEC_TYPE` by active parameters, not by file size**, and re-measure `KV_TYPE`
whenever a model's cache is small. `EXPERIMENTS.md` §11 has the numbers.

The fourth column adds a different warning, and it is about the tooling rather than the
model. `gguf-info.py` models the KV cache and not the context-dependent compute buffers
around it, so it **undershoots the real VRAM cost per token — by 17% on `ornith15-35b`
and by 29% on `ornith-9b`, in the same direction both times**. On a 24 GB card that is
slack you never notice. On the 8 GB card of §12 it decides whether the model loads. Size
from a measurement, not from the tool, whenever the headroom is thin.

Its `SPEC_TYPE` cell is the other half of the `ornith15-35b` lesson: having MTP heads
upstream is not the same as having them in the file you downloaded. The Unsloth build of
Ornith 1.0 drops them even though the model's own `config.json` declares
`mtp_num_hidden_layers: 1`.

## The two profiles for the same file

`ornith15-35b.env` and `ornith15-35b-offload.env` serve a byte-identical `.gguf` and
disagree about nearly everything, because one has 24 GB of VRAM and the other has 8:

| | 3090, fits in VRAM | 3070 Laptop, experts in RAM |
|---|---|---|
| `N_CPU_MOE` | 0 | **31** of 41 blocks |
| `CTX` | 262,144 | 65,536 |
| decode | 162.6 t/s | 48.3 t/s — 3.4x slower |
| prefill | 3,271 t/s | 213 t/s — **15x slower** |

The asymmetry is the point. Decode survives offload because a token reads 8 of 256
experts; prefill does not, because a batch of hundreds of tokens selects most of them. If
your workload sends long prompts — agent context, a repo tree — that 15x is the number to
judge on, not the 3.4x.

§13 also found that in the offload regime **anything costing VRAM is charged twice**, once
for itself and once for the expert blocks it evicts. That reverses the reasoning behind two
settings, not their values: `SPEC_TYPE=none` and `KV_TYPE=q8_0` are right on both cards for
opposite reasons.

## Before running any of these

Run `python3 scripts/gguf-info.py <model.gguf>` before adapting any of them. `docs/models.md`
is the full checklist. The +31% `draft-mtp` figure lives in EXPERIMENTS.md §1: it is real
for the dense model it was measured on, and does not generalize.
