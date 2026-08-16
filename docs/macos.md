# Running this on macOS / Apple Silicon

This repo serves a GGUF model from a CUDA container. On a Mac neither half of
that applies, but the part that matters — the model, the flags, the tuning and
the way it is measured — carries over intact.

This document is the port, done as far as it can be done without a Mac to run
it on. `scripts/serve-metal.sh` is written and complete; what is left is
calibration that requires the hardware. The gaps are marked `TODO(you)` in the
script and listed in §5 here.

---

## 1. Why Docker is out

Not "slower". It does not work at all, and it fails quietly.

```
Linux + NVIDIA   Docker -> container -> llama-server (CUDA) -> GPU     works
macOS + Docker   Docker -> Linux VM -> container -> llama-server       no GPU
macOS native     llama-server (Metal) -> GPU                           works
```

On Linux the container shares the host kernel, and `nvidia-container-toolkit`
hands the GPU devices through. Docker Desktop on macOS runs a full Linux VM
instead, and that VM has no access to the Apple GPU — there is no Metal
passthrough. A containerised llama.cpp inside it runs on the CPU, roughly an
order of magnitude slower, and **prints nothing to say so**. That last part is
what makes it a trap rather than an inconvenience.

So: no Docker, no `docker-compose.yml`, no `serve.sh`. Native binary, Metal
backend, and `scripts/serve-metal.sh` to drive it.

What does transfer is everything above the backend. `-ngl`, `-c`, `-fa`,
`-ctk/-ctv`, `--jinja`, `--metrics`, `--spec-type` are `llama-server` flags, not
CUDA ones; `-ngl` offloads to whichever backend was compiled in, which on Apple
Silicon is Metal.

---

## 2. Install

```bash
brew install llama.cpp
llama-server --version
llama-server --list-devices        # a Metal device must appear here
```

That second check matters: a CPU-only build serves the model perfectly well at
a fraction of the speed and never mentions it. `serve-metal.sh preflight` checks
it for you along with everything else.

Building from source works too and is worth it if the bottled version is old
enough to lack `--spec-type` (see §4):

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON && cmake --build build -j --config Release
# then set LLAMA_BIN=/path/to/llama.cpp/build/bin/llama-server in .env
```

---

## 3. Setup

```bash
git clone https://github.com/nicoRomeroCuruchet/llamacpp-compose.git
cd llamacpp-compose
cp .env.example .env
$EDITOR .env                      # MODELS_DIR at minimum; see below

./scripts/download-model.sh unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf
./scripts/serve-metal.sh preflight
./scripts/serve-metal.sh up
./scripts/serve-metal.sh test
```

`serve-metal.sh` reads the **same `.env`** as the Linux path, so every variable
documented in `.env.example` means the same thing. Two additions:

| Variable | Purpose |
|---|---|
| `LLAMA_BIN` | full path to `llama-server`, if it is not on `PATH` |
| `MODELS_DIR` | wherever you keep `.gguf` files on the Mac |

`BIND`, `CONTAINER_NAME` and anything else Docker-shaped are ignored here.

Subcommands mirror `serve.sh`, plus two of their own:

```
preflight   check the assumptions before a long model load
up down status logs test
mem         the macOS answer to `serve.sh vram` — unified memory + KV arithmetic
```

---

## 4. What is different, and what to expect

### Memory is unified, and shared

There is no separate VRAM budget. Weights and KV cache come out of the same
pool as the OS and everything else running. macOS caps what the GPU may hold —
roughly 70% of installed RAM, though the exact fraction has moved between
releases:

```bash
sysctl hw.memsize                    # installed
sysctl iogpu.wired_limit_mb          # 0 means "system default"
sudo sysctl iogpu.wired_limit_mb=NNNNN    # raise it; not persistent across reboots
```

`./scripts/serve-metal.sh mem` prints all of this next to the model's actual KV
cost per token, which it gets from `scripts/gguf-info.py`.

**Worked example, 36 GB machine, Qwen3.8-27B `UD-Q4_K_XL`:**

```
weights                 17.9 GB
KV cache at CTX=65536    2.3 GB      (36.1 KiB/token — see README section 5)
                        --------
                        20.2 GB     against ~25 GB GPU-usable of 36 GB installed
```

It fits, with about 15 GB left for macOS and applications. The context ceiling
is set by that headroom, not by the model's declared 262,144.

### Speed will be lower, and the reason is bandwidth

Single-stream decode reads the whole model per token, so it is bound by memory
bandwidth:

| | Bandwidth | Ceiling for a 17.9 GB model |
|---|---|---|
| RTX 3090 (the reference setup) | 936 GB/s | 52 t/s — 40.8 measured without speculation, 78% of ceiling |
| M3 Pro | ~150 GB/s | ~8 t/s → **~6-7 t/s** at the same 78% |
| M4 Pro | ~273 GB/s | ~15 t/s → **~12 t/s** |

A 36 GB machine is the M3 Pro configuration, so that is likely the row that
applies — check with *About This Mac*. Speculative decoding should push this up
and arguably helps more here than on the 3090, since verifying several tokens
costs one weight read and Apple Silicon is long on compute relative to
bandwidth. **But that is reasoning, not a measurement**: on the 3090 speculation
delivered +31%, not the 3x that "3.17 tokens per forward pass" might suggest, so
there are overheads this argument does not capture. Measure it; do not assume it.

If it is too slow, `UD-Q3_K_XL` is 13.4 GB instead of 17.9 — about a third
faster for the same reason, with memory to spare.

### `CACHE_RAM` is more expensive here

On the Linux box the host-RAM prompt cache is a separate budget from VRAM. Here
it competes with the model for the same pool. `CACHE_RAM=10240` on a 36 GB
machine is a real 10 GB out of what is left after the weights. Start lower.

---

## 5. What is left for you

Everything below needs the machine in front of it.

1. **Confirm `LLAMA_BIN` resolves.** The script falls back to
   `./build/bin/llama-server`; set it explicitly in `.env` if neither path is
   right.
2. **Run `preflight` and read it.** It checks the Metal device, the model
   header, and which of `--spec-type`, `--cache-ram`, `--reasoning-format`,
   `--jinja`, `--metrics`, `-fa` your build actually has. If `--spec-type` is
   missing, either update llama.cpp or let `build_args()` skip it — the script
   already does the latter automatically, and you lose ~30% of decode speed.
3. **Check the wired-memory fraction.** The script assumes ~70% when
   `iogpu.wired_limit_mb` is 0. Verify against your macOS version; that number
   is what decides whether a given `CTX` fits.
4. **Pick `CTX` from `mem`, not from this repo's `.env`.** 65536 is right for a
   24 GB dedicated card. Your budget and your headroom are different.
5. **Verify quantized KV works on Metal.** `-ctk q8_0 -ctv q8_0` needs flash
   attention, which `-fa on` enables. It should be fine; if it errors, drop to
   `KV_TYPE=f16` and recompute the arithmetic — it roughly doubles the cache.
6. **Measure, then compare.** `EXPERIMENTS.md` documents how every number in it
   was obtained, from the same `/metrics` counters and `print_timing` lines this
   build emits. Running §1 (the speculative-decoding sweep) and §4 (throughput
   under real load) on Apple Silicon is the interesting result nobody here has.

If you get through this, a PR correcting the guesses in this file — the
bandwidth estimates, the wired-memory fraction, the flags your build turned out
to have — would be worth more than the guide itself.
