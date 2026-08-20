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

| | `qwen38-27b` | `ornith-35b` | `ornith15-35b` |
|---|---|---|---|
| `SPEC_TYPE` | `draft-mtp` — it ships MTP heads | `ngram-simple` — it does not | `none` — it ships them and they still lose |
| KV per token at `q8_0` | 36.1 KiB | 10.6 KiB | 11.7 KiB computed, **13.6 measured** |
| `TEMP` / `TOP_K` | 0.6 / 20 | 1.0 / 0 | 1.0 / 20 |

The third column is the cautionary one. `ornith15-35b` is a mixture of experts with ~3B
active parameters, and that single fact inverts two defaults this repo otherwise treats as
safe: `draft-mtp` costs 24% of decode instead of gaining 31%, and `q8_0` KV is *slower*
than `f16` rather than a free saving. Neither failure is visible without benchmarking —
both configurations load, report healthy, and answer correctly.

**Judge `SPEC_TYPE` by active parameters, not by file size**, and re-measure `KV_TYPE`
whenever a model's cache is small. `EXPERIMENTS.md` §11 has the numbers.

Run `python3 scripts/gguf-info.py <model.gguf>` before adapting any of them. `docs/models.md`
is the full checklist. The +31% `draft-mtp` figure lives in EXPERIMENTS.md §1: it is real
for the dense model it was measured on, and does not generalize.
