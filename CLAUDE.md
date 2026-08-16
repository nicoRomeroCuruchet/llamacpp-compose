# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A `docker compose` wrapper around `ghcr.io/ggml-org/llama.cpp:server-cuda` that serves a
GGUF model over an OpenAI-compatible HTTP API. It is model- and GPU-agnostic; the setup
the documentation measures against is an RTX 3090 (24 GB) serving **Qwen3.8-27B**
(`UD-Q4_K_XL`, 17.9 GB).

There is no application code here. The repo is a compose file, an `.env` that
parameterizes it, a few operational shell scripts, and the documentation.

**There are two runtimes, and the wrong one gives advice that silently does nothing.**
Establish which you are on before suggesting any command:

| | Linux + NVIDIA | macOS + Apple Silicon |
|---|---|---|
| how it runs | `docker compose`, CUDA image | native `llama-server`, Metal, **no Docker** |
| driven by | `scripts/serve.sh` | `scripts/serve-metal.sh` |
| memory | dedicated VRAM | unified, shared with the OS |

Docker on macOS is not a slower option, it is a broken one: the Linux VM cannot reach the
Apple GPU, so a containerised llama.cpp falls back to the CPU and reports nothing.

Read `README.md` before changing anything — it explains every flag and the reasoning
behind the defaults. `docs/architecture.md` explains how the pieces fit together and is
the right thing to read before any structural change.

## Layout

```
docker-compose.yml         the service; all values come from .env
.env                       this machine's config — GITIGNORED, do not commit
.env.example               the versioned reference, with the comments
scripts/serve.sh           up | down | status | logs | test | vram
scripts/download-model.sh  fetch a .gguf from HuggingFace and verify it
scripts/gguf-info.py       read a .gguf header: architecture and KV cache arithmetic
scripts/serve-metal.sh     the macOS/Metal equivalent of serve.sh — no Docker involved
README.md                  full documentation
EXPERIMENTS.md             the tuning lab notebook: what was measured and why
docs/architecture.md       how it works end to end; read before structural changes
docs/models.md             serving another model; the checklist and a worked second example
docs/profiles/*.env        complete known-good configs; the durable copy of a tuned .env
docs/macos.md              porting guide for Apple Silicon, with its open questions
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
next person cannot reproduce the setup. When a configuration has been tuned and measured,
also write it to `docs/profiles/<alias>.env` — otherwise it exists on one machine only and
dies with it.

**Do not install system packages to solve a problem a container or a script can solve.**
This rule exists because the reference host is shared with other users; it is why
`download-model.sh` uses `curl` rather than `huggingface_hub`. It is **not** a reason to
refuse the installs a platform genuinely requires: on macOS, `brew install llama.cpp` is
the supported path and there is no container alternative. Ask the user before installing
anything on a machine you have reason to think is shared.

**Do not enable `restart: always`.** The model holds ~19 GB of the 24 GB card. Auto-starting
would take the GPU on days it is needed for training. Bringing it up is a manual act.

**A flag change goes in both runtimes.** `docker-compose.yml` and `scripts/serve-metal.sh`
carry the same `llama-server` invocation. Changing one and not the other makes the two
platforms diverge with nothing to signal it.

## Setting it up for the first time

Establish the platform first — the two runtimes need different instructions, and the wrong
one fails quietly rather than loudly.

| | |
|---|---|
| **Linux + NVIDIA GPU** | README §3. Needs Docker and `nvidia-container-toolkit`. |
| **macOS + Apple Silicon** | README §3 → "On macOS", and `docs/macos.md`. Native `llama-server` via Homebrew, **no Docker**. Run `./scripts/serve-metal.sh preflight` before anything else: it checks that a Metal device is really present, that the build has the flags this repo uses, and that the model downloaded whole. All three fail silently otherwise. |
| **macOS + Intel, or Linux without an NVIDIA GPU** | there is no GPU path. Say so rather than proceeding; CPU inference on a 27B model is not usable. |

`docs/macos.md` §5 lists what is known to be unverified on Apple Silicon — the
wired-memory fraction, whether quantized KV works under Metal, and the throughput
estimates. Treat those as open questions to measure, not as documented facts, and tell the
user what you measured.

## Common tasks

On Linux. The macOS equivalents are the same subcommands on `scripts/serve-metal.sh`,
except `vram`, which is `mem` there.

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

Run `scripts/gguf-info.py` against the new `.gguf` before touching anything else.
`SPEC_TYPE=draft-mtp` only works on models that ship MTP heads and most do not, and KV cost
per token does not track model size — between the two models documented in
`docs/models.md` it varies by 3.4x, in the opposite direction to their file sizes.

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
- **On macOS, "it works but it is slow" usually means it never got the GPU.** Either it
  was run under Docker, or the build is CPU-only. Neither says so. `llama-server
  --list-devices` must list Metal; `serve-metal.sh preflight` checks it.
- **`--cache-ram` is host RAM on Linux and unified memory on macOS.** The same value is
  free of VRAM cost on one and competes with the weights on the other. Do not copy it
  across platforms.
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
