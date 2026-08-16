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
| `ornith-35b.env` | Ornith 1.0 35B `UD-Q4_K_XL`, 22.3 GB | reconstructed from a working RTX 4090 deployment, with two corrections applied; not run in this exact form |

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
name, and two profiles here happen to differ on all three:

| | `qwen38-27b` | `ornith-35b` |
|---|---|---|
| `SPEC_TYPE` | `draft-mtp` — it ships MTP heads | `ngram-simple` — it does not |
| KV per token at `q8_0` | 36.1 KiB | 10.6 KiB |
| `TEMP` / `TOP_K` | 0.6 / 20 | 1.0 / 0 |

Run `python3 scripts/gguf-info.py <model.gguf>` before adapting either one. `docs/models.md`
is the full checklist.
