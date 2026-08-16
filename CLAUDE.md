# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A `docker compose` wrapper around `ghcr.io/ggml-org/llama.cpp:server-cuda` that serves a
GGUF model over an OpenAI-compatible HTTP API. It is model- and GPU-agnostic; the setup
the documentation measures against is an RTX 3090 (24 GB) serving **Qwen3.8-27B**
(`UD-Q4_K_XL`, 17.9 GB).

There is no application code here. The repo is four things: a compose file, an `.env` that
parameterizes it, two operational shell scripts, and the documentation. Read `README.md`
before changing anything — it explains every flag and the reasoning behind the defaults.

## Layout

```
docker-compose.yml         the service; all values come from .env
.env                       this machine's config — GITIGNORED, do not commit
.env.example               the versioned reference, with the comments
scripts/serve.sh           up | down | status | logs | test | vram
scripts/download-model.sh  fetch a .gguf from HuggingFace and verify it
scripts/gguf-info.py       read a .gguf header: architecture and KV cache arithmetic
README.md                  full documentation
EXPERIMENTS.md             the tuning lab notebook: what was measured and why
```

Models live in `~/models/` on the host, **outside the repo**, and are mounted read-only at
`/models` in the container.

## Ground rules

**Everything in this repo — code, comments, docs, commit messages — is written in
English.** The repo owner writes in Rioplatense Spanish; reply to them in Spanish, but do
not let Spanish into the files.

**Never delete a model file.** The `.gguf` files take hours to download over a shared link.
The mount is `:ro` specifically so that nothing inside the container can touch them. Do not
`rm` anything under `MODELS_DIR`.

**`.env` is not versioned and holds the live configuration.** Change it in place when the
task calls for it, but mirror any *new* variable into `.env.example` with a comment, or the
next person cannot reproduce the setup.

**This is a shared machine.** Do not install system packages to solve a problem that a
container or a script can solve. That is why `download-model.sh` uses `curl` rather than
`huggingface_hub`.

**Do not enable `restart: always`.** The model holds ~19 GB of the 24 GB card. Auto-starting
would take the GPU on days it is needed for training. Bringing it up is a manual act.

## Common tasks

```bash
cd ~/llm-server

./scripts/serve.sh up        # start; waits for /health, then prints VRAM in use
./scripts/serve.sh status    # container + health + which model is served
./scripts/serve.sh test      # real end-to-end query, prints reasoning/content/usage
./scripts/serve.sh vram      # nvidia-smi, one line
./scripts/serve.sh logs 100
./scripts/serve.sh down
```

**Changing the model** means editing `MODEL_FILE` and `MODEL_ALIAS` in `.env`, checking the
sampling values against the new model's card (they are model-specific — see README §4), and
running `serve.sh up`. It should never mean editing `docker-compose.yml`.

**Adding a flag** to `llama-server` goes in the `command:` block of `docker-compose.yml`,
and gets a line in README §4 explaining it. If the value should be tunable per machine, add
it as a `${VAR:-default}` and document it in `.env.example`.

## Verifying a change actually worked

`docker compose up -d` returning success only means the container started; llama.cpp can
still fail during model loading, minutes later. Confirm with a real query:

```bash
./scripts/serve.sh status    # health must answer, not "(not responding)"
./scripts/serve.sh test      # must print a non-empty content or reasoning
```

An empty `content` on its own is not a failure — with `--reasoning-format deepseek` the
output goes to `reasoning_content` first. README §4 covers this.

## Traps worth knowing

- **`/v1/chat/completions` is POST-only.** A GET returns
  `{"error":{"message":"File Not Found",...,"code":404}}`, which reads like a broken
  deployment but is just the wrong method. The browser-friendly route is `/`.
- **A trailing slash in `base_url`** produces `//v1/chat/completions` and the identical 404.
  Use `.../v1`, not `.../v1/`.
- **`BIND` must match how the server is exposed.** `tailscale serve` proxies to
  `127.0.0.1:8080`, so `BIND` has to stay `127.0.0.1` for it to work; `docker port qwen`
  tells you what is actually in effect. Setting `BIND` to the tailscale IP is the
  no-sudo alternative and makes `tailscale serve` redundant. README §7.
- **Container startup is slow by design.** `start_period: 300s` in the healthcheck exists
  because loading 18 GB into VRAM takes a while. Do not shorten it to make things "feel"
  faster.
- **VRAM is the binding constraint, not disk or RAM.** Before raising `CTX`, run
  `serve.sh vram` and check the headroom against README §5 and `EXPERIMENTS.md` §7. Note
  that the reference model is hybrid — only 17 of 65 blocks carry KV — so the usual
  "layers x heads x context" estimate overshoots several-fold. Never assume this either
  way for a new model: run `scripts/gguf-info.py` against the actual `.gguf`.

## Reaching the server

```
local     http://127.0.0.1:8080/v1
tailnet   https://<node>.<tailnet>.ts.net/v1        (via tailscale serve, TLS)
```

There is **no authentication** on the server. Anything that widens its exposure — changing
`BIND`, adding a `tailscale serve`/`funnel` config, publishing a port — is a security
decision. Ask before doing it.
