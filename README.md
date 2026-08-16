# llamacpp-compose

A local LLM server running **llama.cpp on the GPU**, packaged in Docker and driven with
`docker compose`. It exposes an **OpenAI-compatible** API, so any client that accepts a
`base_url` can talk to it unmodified.

Nothing here is tied to a particular model or GPU: point `.env` at any `.gguf` that fits
on the card and it serves that instead. Every flag has a documented default and a stated
reason for it.

### The reference setup

Throughput, VRAM and context figures appear throughout this file. They were all measured
on one configuration, named here so the numbers can be judged and reproduced rather than
taken on faith:

| | |
|---|---|
| GPU | NVIDIA RTX 3090, 24 GB |
| Model | Qwen3.8-27B, `UD-Q4_K_XL` (17.9 GB) |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda`, build 10335 |

**A different card or model will give different numbers**, and some of the conclusions
below depend on the model's architecture rather than on llama.cpp. `EXPERIMENTS.md`
records how each measurement was taken so you can rerun it on your own hardware;
`scripts/gguf-info.py` recomputes the memory arithmetic for any `.gguf`.

---

## 1. What the image is

`ghcr.io/ggml-org/llama.cpp:server-cuda` — the **official** llama.cpp image (`ggml-org` is
Georgi Gerganov's organization, the author of the project). 4.3 GB.

```
ENTRYPOINT: /app/llama-server
Cmd       : null
```

Two consequences worth being clear about:

- **The entrypoint is already the server binary.** Everything that follows the image name
  in `docker run` — or in the compose `command:` — is a **`llama-server` flag**, not a
  Docker one. That is why `-m`, `-c`, `-ngl` and friends appear there.
- **The image ships no model.** The `.gguf` files live on the host and come in through the
  volume. The same image serves Qwen, Ornith, or whatever you point it at.

What is inside:

| | |
|---|---|
| `/app/llama-server` | the HTTP server (build 10335) |
| `libggml-cuda.so` | CUDA backend — the reason for the 4.3 GB |
| `libggml-cpu-*.so` | one variant per microarchitecture (haswell, icelake, alderlake…), selected at runtime; the fallback for anything that does not fit in VRAM |

The **`-cuda`** suffix is what makes it useful here. There are much smaller CPU-only
variants that are of no use for this.

### Endpoints

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | inference, OpenAI format |
| `GET /v1/models` | returns the configured `--alias` |
| `GET /health` | `{"status":"ok"}` once loading has finished |
| `GET /metrics` | Prometheus metrics (enabled by `--metrics`) |
| `GET /` | llama.cpp's built-in web UI |

Note that `/v1/chat/completions` is **POST only**. Opening it in a browser issues a GET and
gets back `{"error":{"message":"File Not Found",...,"code":404}}`, which looks like the
server is misconfigured but is only the wrong method. Use `/` for a browser.

---

## 2. Host requirements

- An NVIDIA driver new enough for the image's CUDA build, and a GPU with enough VRAM —
  see the arithmetic in §5.
- **`nvidia-container-toolkit`**. Without it the container **cannot see
  the GPU** and llama.cpp silently falls back to the CPU. Check with:

  ```bash
  docker info | grep -i runtime          # must list "nvidia"
  docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
  ```

- `docker compose` v2 or newer.
- **No CUDA toolkit or `nvcc` needed on the host.** It all ships inside the image.

**On macOS this whole section does not apply.** Docker Desktop runs its containers in a
Linux VM with no access to the Apple GPU, so a containerised llama.cpp falls back to the
CPU without saying so. Apple Silicon runs `llama-server` natively against Metal instead:
see **`docs/macos.md`** and `scripts/serve-metal.sh`, which read the same `.env` and take
the same subcommands as everything below.

---

## 3. Getting started

```bash
git clone https://github.com/nicoRomeroCuruchet/llamacpp-compose.git
cd llamacpp-compose
cp .env.example .env
$EDITOR .env                                     # check MODELS_DIR and MODEL_FILE

./scripts/download-model.sh unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf
./scripts/serve.sh up
./scripts/serve.sh test
```

`serve.sh up` starts the container, **waits for `/health` to answer** (moving 18 GB from
disk into VRAM takes a while) and then prints how much VRAM ended up in use. If the
container dies during loading it stops waiting and shows you the logs instead of hanging.

### On macOS: Apple Silicon and Metal

The block above is for Linux with an NVIDIA GPU. **On a Mac none of it applies**, and the
reason is worth knowing before trying it anyway: Docker Desktop on macOS runs containers
inside a Linux VM, and that VM has no access to the Apple GPU — there is no Metal
passthrough. A containerised llama.cpp inside it falls back to the CPU, runs roughly an
order of magnitude slower, and **prints nothing to say so**. It looks like it worked.

So on Apple Silicon there is no Docker at all. `llama-server` runs natively against Metal,
driven by `scripts/serve-metal.sh`:

```bash
brew install llama.cpp
llama-server --list-devices          # a Metal device must appear in this list

git clone https://github.com/nicoRomeroCuruchet/llamacpp-compose.git
cd llamacpp-compose
cp .env.example .env
$EDITOR .env                         # MODELS_DIR at minimum

./scripts/download-model.sh unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf
./scripts/serve-metal.sh preflight   # check the assumptions before an 18 GB load
./scripts/serve-metal.sh up
./scripts/serve-metal.sh test
```

`serve-metal.sh` reads the **same `.env`** and takes the same subcommands as `serve.sh`,
so §4 to §8 below apply as written — substitute `serve-metal.sh` for `serve.sh`. Only the
backend and the process management differ. Two subcommands are its own:

| | |
|---|---|
| `preflight` | Verifies before committing to a long model load: that a Metal device is really present, that the binary resolves, that the `.gguf` header is genuine, and which of `--spec-type`, `--cache-ram`, `--reasoning-format`, `--jinja`, `--metrics` and `-fa` your build has. A missing `--spec-type` is skipped rather than fatal. |
| `mem` | The macOS answer to `serve.sh vram`. Reports installed RAM, the `iogpu.wired_limit_mb` cap and this model's KV cost per token, via `scripts/gguf-info.py`. |

Three things differ in kind rather than in degree:

**Memory is unified and shared.** There is no separate VRAM budget: weights and KV cache
come out of the same pool as the OS and everything else running. macOS caps what the GPU
may hold at roughly 70% of installed RAM — raise it with
`sudo sysctl iogpu.wired_limit_mb=NNNNN`, which does not survive a reboot. On a 36 GB
machine, 17.9 GB of weights plus 2.3 GB of KV at `CTX=65536` fits in the ~25 GB available
and leaves ~15 GB for the system.

**Decode is bandwidth-bound, and the bandwidth is much lower.** Each token reads the whole
model, so throughput tracks memory bandwidth almost directly: 936 GB/s on the reference
RTX 3090 against ~150 GB/s on an M3 Pro. Expect single digits to low teens of tokens per
second before speculative decoding, rather than the 40–60 in §4.

**`CACHE_RAM` competes with the weights.** On Linux the host-RAM prompt cache is a budget
separate from VRAM. Here it is the same pool, so the `10240` in `.env.example` is 10 GB
taken from what is left after the model loads. Start lower.

The full guide is **`docs/macos.md`**. Be aware of what it is: a port written without a
Mac to test on. The plumbing is complete; the calibration is not, and §5 of that file
lists precisely what is unverified — the wired-memory fraction, whether quantized KV works
under Metal, and the throughput figures above, which are extrapolated from memory
bandwidth and have not been measured.

### Operation

```bash
./scripts/serve.sh up       # start and wait until ready
./scripts/serve.sh status   # container + health + model being served
./scripts/serve.sh test     # a real end-to-end query
./scripts/serve.sh vram     # how much VRAM is in use
./scripts/serve.sh logs 100
./scripts/serve.sh down     # stop (leaves the .gguf files alone)
```

### Switching models

Edit `MODEL_FILE` and `MODEL_ALIAS` in `.env`, adjust the sampling settings (§4), and run
`serve.sh up`. `docker-compose.yml` stays untouched.

---

## 4. The flags, one by one

```
-m /models/<file>.gguf        the model; /models is the read-only volume
--alias <name>                the name clients see in /v1/models
--host 0.0.0.0 --port 8080    inside the container; outside it is published on loopback
-c 65536                      context window, in tokens
-ngl all                      ALL layers on the GPU (what makes this worth doing)
-np 1                         a single slot: the whole KV cache for one conversation
-b 2048 -ub 512               logical and physical prompt batch sizes (llama.cpp's defaults)
-ctk q8_0 -ctv q8_0           KV cache quantized to 8 bits (half of what f16 costs)
-fa on                        flash-attention: less memory, more speed
--jinja                       use the chat template embedded in the .gguf
--reasoning-format deepseek   split reasoning into a field of its own
--metrics                     enable /metrics
--spec-type draft-mtp         speculative decoding with the model's own MTP heads
--spec-draft-n-max 8          tokens drafted per step
--spec-draft-p-min 0.7        confidence floor for keeping a drafted token
```

### Speculative decoding: use the heads the model already ships

This `.gguf` carries **MTP (multi-token prediction) heads** — the `blk.64.nextn.*` tensors.
They are weights trained to predict the next few tokens ahead, and `--spec-type draft-mtp`
drafts with them. There is no second model to download and no extra VRAM for one: the
tensors are in the file either way.

With any other `--spec-type`, they are loaded and ignored, which the server says out loud
on startup:

```
W model has unused tensor blk.64.nextn.eh_proj.weight -- ignoring
```

**Speculation does not change the output.** Drafted tokens the model rejects are thrown
away, so the text is identical to what it would have generated unaided — the only thing at
stake is time. That makes it the one knob here with no quality trade-off, unlike the KV
cache type or the sampling values.

Whether it is paying off is in the logs:

```bash
./scripts/serve.sh logs 200 | grep 'draft acceptance'
```

`ngram-simple` drafts by replaying n-grams it has already seen, so it does well on
repetitive edits and poorly on prose — measured here between 4% and 37%, usually under 20%.
If acceptance stays low with `draft-mtp` too, lower `SPEC_P_MIN` to draft more freely, or
set `SPEC_TYPE=none`: rejected drafts are wasted compute, and a bad drafter is slower than
no drafter at all.

Measured on the reference setup, 700-token generation from a short coding prompt (the
full write-up, including the negative results, is in `EXPERIMENTS.md` §1):

| `SPEC_TYPE` | `N_MAX` | `P_MIN` | decode | acceptance | VRAM |
|---|---|---|---|---|---|
| `ngram-simple` | — | — | 40.8 t/s | 4–37% | 19.1 GB |
| `draft-mtp` | 4 | 0.7 | 50.9 t/s | 78–100% | 20.5 GB |
| `draft-mtp` | 8 | 0.5 | 52.0 t/s | 50% | 21.1 GB |
| **`draft-mtp`** | **8** | **0.7** | **53.5 t/s** | **62%** | **21.1 GB** |

Loosening `P_MIN` to 0.5 drafts more and lands *worse*: acceptance halves and the wasted
compute eats the gain. Raising `N_MAX` past 8 is not worth testing — mean accepted length
is 3.5, so the budget of 8 already goes unused.

### Sampling belongs to the model, not to the server

This is the easiest mistake to make when copying a config from one model to another:

| | Qwen3.8-27B | Ornith 1.0 35B |
|---|---|---|
| `--temp` | 0.6 | 1.0 |
| `--top-p` | 0.95 | 0.95 |
| `--top-k` | 20 | 0 |
| `--min-p` | 0.0 | 0.0 |

Every model card carries its own. Copying them across models breaks nothing visibly — it
just generates worse output, quietly.

### `--reasoning-format deepseek` and the empty `content`

The reasoning block does **not** go to `content`, it goes to **`reasoning_content`**. Ask
for a small `max_tokens` and the model spends it all thinking, so `content` arrives as
`''`.

**This is not a bug.** Give it several hundred tokens of headroom, or read
`reasoning_content`. It has cost more than one person an afternoon.

---

## 5. The VRAM arithmetic

The one thing to work out before picking a quantization and a context size:

```
total VRAM  ≥  .gguf size  +  KV cache  +  compute buffers
```

The `.gguf` size is on disk. The compute buffers are a few hundred MiB. **The KV cache is
the term people get wrong**, and the usual back-of-envelope can be out by a large factor,
so read it off the model instead of guessing:

```bash
python3 scripts/gguf-info.py /path/to/model.gguf
```

That prints the cost per token at `f16`, `q8_0` and `q4_0`, the total at several context
sizes, and — the part that catches people out — **how many blocks actually carry a KV
cache**. On a dense transformer that is all of them. On a **hybrid** model it is not:
architectures that interleave full attention with linear-attention or SSM blocks only
store KV in the full-attention ones, and the rest hold a recurrent state whose size does
not grow with the context at all. Assuming otherwise overestimates the cache several-fold
and makes a context size look impossible when it fits comfortably.

Note also that llama.cpp **allocates the entire KV cache when the model loads**, not as
the conversation grows. Raising `CTX` bills the VRAM immediately, whether or not the
window is ever used.

### Worked example: Qwen3.8-27B on a 24 GB card

The reference setup, as a template for doing the same on yours. The 17.9 GB of weights
leave ~7 GB for cache and buffers; measured in practice, **21,074 MiB of 24,576 MiB in
use**, so 3,502 MiB free. Check yours with `serve.sh vram`.

This model turns out to be hybrid — only **17 of its 65 blocks** carry a KV cache — which
works out to **36.1 KiB per token** at `q8_0`. So `CTX=65536` costs 2,312 MiB, and the
ceiling on this card is **131,072 tokens**, not the 262,144 the model declares. A dense
estimate would have said ~40k and been wrong by more than 3x. Full derivation, and the
measurement it was checked against, in **`EXPERIMENTS.md` §6–§7**.

Available quantizations of this model, to choose from with some basis:

| File | Size | Fits in 24 GB |
|---|---|---|
| `UD-Q3_K_XL` | 13.4 GB | yes, with plenty of context |
| **`UD-Q4_K_XL`** | **17.9 GB** | **yes — the one measured here** |
| `UD-Q5_K_XL` | 20.2 GB | barely, with a small context |
| `UD-Q6_K_XL` | 25.9 GB | **no** |

The `UD-*` ones are *Unsloth Dynamic*: they quantize differently per layer and outperform a
uniform quantization of the same size.

`CTX` is left at 65536 rather than raised to that ceiling, because peak context under real
agent load was 22,564 tokens — a third of the window — and the VRAM would be reserved
either way. The trigger to raise it is `truncated = 1` appearing in the logs, not free
VRAM existing. `EXPERIMENTS.md` §7.

**If it does not fit**, in order of preference: lower `CTX`, switch `KV_TYPE` to `q4_0`,
pick a smaller quantization. Touch `NGL` only as a last resort — offloading layers to the
CPU works but costs an order of magnitude in speed.

---

## 6. Usage

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 800
  }'
```

With the OpenAI SDK:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
r = c.chat.completions.create(model="qwen38-27b",
                              messages=[{"role": "user", "content": "Hello"}],
                              max_tokens=800)
print(r.choices[0].message.content)
```

`api_key` is required by the SDK but ignored by the server: there is no authentication.
That is precisely why the port **is published on loopback only**.

---

## 7. Reaching it from the tailnet

The container publishes its port **on loopback only** (`BIND=127.0.0.1` in `.env`). To
reach it from another machine on the tailnet, use `tailscale serve`, which terminates TLS
with a real tailnet certificate and proxies to `127.0.0.1:8080`:

```bash
sudo tailscale serve --bg 8080        # once; survives reboots
tailscale serve status                # show the active config
sudo tailscale serve --https=443 off  # tear it down
```

`tailscale status` prints your own tailnet name; substitute it for
`tailnet-name` below. It then answers at:

```
https://gpu-box.tailnet-name.ts.net/v1/chat/completions      model: qwen38-27b
https://gpu-box.tailnet-name.ts.net/                         web UI, in a browser
```

```bash
curl -s https://gpu-box.tailnet-name.ts.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"hello"}]}'
```

From the OpenAI SDK, on any tailnet node:

```python
from openai import OpenAI
c = OpenAI(base_url="https://gpu-box.tailnet-name.ts.net/v1", api_key="not-needed")
```

Note the `base_url` has **no trailing slash**. With one, the SDK concatenates into
`//v1/chat/completions`, which llama-server treats as an unknown route and answers with the
same `File Not Found` 404 as a GET does.

No router port needs to be opened: tailscale does not route internet traffic to these
nodes, only traffic between the machines on the tailnet. It does become visible to **every**
device on the tailnet, though, and the server has no authentication — so this is a
deliberate decision, not a default.

### Without sudo

If `sudo` is unavailable, the container can bind straight to the tailscale IP instead of
loopback — `tailscale ip -4` prints it:

```bash
BIND=100.x.y.z   # in .env, then ./scripts/serve.sh down && up
```

That serves it at `http://gpu-box:8080` (no TLS) and is still tailnet-only, because that IP
exists solely on the tailscale interface. In that case `tailscale serve` adds nothing and
should be turned off — otherwise there are two paths to the same port, and `serve` returns
502 while proxying to a loopback address nothing listens on anymore.

---

## 8. When something breaks

| Symptom | Usual cause |
|---|---|
| `/health` silent while the container is still alive | still loading; 18 GB takes time |
| Dies on startup, logs show `CUDA error: out of memory` | does not fit: lower `CTX`, see §5 |
| Runs but is extremely slow | it never got the GPU; check the toolkit (§2) and that `-ngl` is `all` |
| Empty `content` in the response | it is in `reasoning_content`, not a bug (§4) |
| `File Not Found` 404 on a valid route | GET instead of POST, or a double slash from a trailing `/` in `base_url` (§7) |
| `unknown flag` on startup | flag absent from this build; `docker run --rm <image> --help` |
| Docker cannot find the GPU | `nvidia-container-toolkit` missing from the host |

```bash
./scripts/serve.sh logs 100
docker run --rm ghcr.io/ggml-org/llama.cpp:server-cuda --help    # this build's flags
```

### On macOS

| Symptom | Usual cause |
|---|---|
| Runs, answers correctly, but is ~10x slower than expected | the commonest one, and it never announces itself: either you ran it in Docker (the Linux VM has no GPU) or the build is CPU-only. `llama-server --list-devices` must show Metal. `serve-metal.sh preflight` checks both |
| `ggml_metal_init` fails, or an allocation error on load | the model does not fit under the wired-memory cap. `serve-metal.sh mem`, then lower `CTX` or raise `iogpu.wired_limit_mb` |
| The whole machine starts swapping | unified memory: weights plus KV plus `CACHE_RAM` left nothing for the OS. Lower `CACHE_RAM` first — on Linux it is a separate budget, here it is not |
| `unknown flag` on startup | an older llama.cpp than these flags need. `serve-metal.sh preflight` lists which ones your build has; `brew upgrade llama.cpp` or build from source |
| `llama-server: command not found` | not on `PATH`; set `LLAMA_BIN` in `.env` to the full path |

```bash
./scripts/serve-metal.sh preflight     # run this first, it checks all of the above
./scripts/serve-metal.sh logs 100
```

---

## 9. Decisions taken

- **`restart: "no"`.** The model holds nearly all of the VRAM; starting on every reboot
  would keep the card busy even on days it is needed for training. It is brought up by
  hand.
- **Port on `127.0.0.1`.** The server has no authentication. Exposing it is an explicit
  step (`tailscale serve`), never the default.
- **Models mounted `:ro`.** They cost hours to download; the server only needs to read
  them.
- **`.gguf` files kept out of the repo.** They are tens of GB. `scripts/download-model.sh`
  fetches them and **verifies the exact size and the `GGUF` header** — a truncated file or
  a saved HTTP error response both go unnoticed if all you check is that "the file exists".
- **`curl` instead of `huggingface_hub`.** This is a shared machine; downloading one file
  is not worth installing Python packages for. `curl -C -` resumes, too.

---

## 10. License

MIT — see `LICENSE`.

Note that this covers the compose file, the scripts and the documentation only. The
llama.cpp image and any model you download carry their own licenses; Qwen3.8-27B is
Apache-2.0, but check the card for whatever you actually serve.
