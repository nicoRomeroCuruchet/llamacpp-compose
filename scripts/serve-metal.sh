#!/bin/bash
# Server operation on macOS / Apple Silicon, where llama-server runs natively
# against Metal instead of in a CUDA container.
#
#   ./scripts/serve-metal.sh up|down|status|logs|test|mem|preflight
#
# Same subcommands and the same .env as scripts/serve.sh, so the documentation
# in README.md applies unchanged; only the backend and the process management
# differ. Docker is not used at all here — Docker Desktop on macOS runs its
# containers inside a Linux VM that has no access to the Apple GPU, so a
# containerised llama.cpp would silently fall back to the CPU.
#
# ---------------------------------------------------------------------------
# STATUS: written against the flags and behaviour of the Linux/CUDA setup and
# NOT yet run on a Mac. Everything llama.cpp-side is backend-agnostic and should
# carry over; the macOS-side specifics are marked TODO(you) below. Run
# `preflight` first — it checks the assumptions rather than trusting them.
# ---------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ERROR: .env is missing. Copy it from the example:  cp .env.example .env" >&2
    exit 1
fi
# shellcheck disable=SC1091
set -a; . ./.env; set +a

PORT="${PORT:-8080}"
BASE="http://127.0.0.1:${PORT}"
MODEL="${MODELS_DIR:?set MODELS_DIR in .env}/${MODEL_FILE:?set MODEL_FILE in .env}"

# TODO(you): confirm this resolves. `brew install llama.cpp` puts llama-server on
# PATH; a source build leaves it in build/bin/llama-server. Override in .env with
# LLAMA_BIN=/full/path/to/llama-server if neither applies.
LLAMA_BIN="${LLAMA_BIN:-$(command -v llama-server || echo ./build/bin/llama-server)}"

PIDFILE=".llama-server.pid"
LOGFILE=".llama-server.log"

# --- the invocation --------------------------------------------------------
# Deliberately identical to the `command:` block in docker-compose.yml. None of
# these flags is CUDA-specific: -ngl offloads to whichever backend was compiled
# in, which on Apple Silicon is Metal.
build_args() {
    ARGS=(
        -m "$MODEL"
        --alias "${MODEL_ALIAS:-local-model}"
        --host 127.0.0.1 --port "$PORT"
        -c "${CTX:-65536}"
        -ngl "${NGL:-999}"
        -np 1
        -b "${BATCH:-2048}" -ub "${UBATCH:-512}"
        -ctk "${KV_TYPE:-q8_0}" -ctv "${KV_TYPE:-q8_0}"
        -fa on
        --temp "${TEMP:-0.6}" --top-p "${TOP_P:-0.95}"
        --top-k "${TOP_K:-20}" --min-p "${MIN_P:-0.0}"
        --jinja
        --reasoning-format "${REASONING_FORMAT:-deepseek}"
        --metrics
        --cache-ram "${CACHE_RAM:-8192}"
    )
    # Optional API key. Mirrors ${API_KEY:+--api-key} in docker-compose.yml:
    # absent when empty, which is the right default for a loopback port. Set it
    # before exposing the server past a trusted network -- there is no other
    # authentication. README section 7.
    [ -n "${API_KEY:-}" ] && ARGS+=( --api-key "${API_KEY}" )
    # Overrides the Jinja template baked into the .gguf. Mirrors
    # ${CHAT_TEMPLATE_FILE:+...} in docker-compose.yml. The file lives in
    # MODELS_DIR beside the model. README section 4.
    [ -n "${CHAT_TEMPLATE_FILE:-}" ] && ARGS+=( --chat-template-file "${MODELS_DIR}/${CHAT_TEMPLATE_FILE}" )

    # MoE expert offload to host RAM. Newer flag than the rest, so it is added
    # only if the build has it -- and only when asked for, since the default 0
    # is a no-op that an older build would still reject as unknown.
    #
    # CAREFUL on macOS: on Linux this moves weights from VRAM to a SEPARATE pool
    # of host RAM, which is the whole point. Here memory is unified, so it moves
    # them from one part of the same pool to another and buys nothing but the
    # loss of the GPU. Leave N_CPU_MOE unset unless you have measured otherwise.
    if [ "${N_CPU_MOE:-0}" != "0" ] && "$LLAMA_BIN" --help 2>&1 | grep -q -- '--n-cpu-moe'; then
        ARGS+=( -ncmoe "${N_CPU_MOE}" )
    fi

    # Speculative decoding is the one part that a slightly older build may not
    # have. preflight checks for it; skip rather than fail if it is missing.
    if "$LLAMA_BIN" --help 2>&1 | grep -q -- '--spec-type'; then
        ARGS+=( --spec-type "${SPEC_TYPE:-draft-mtp}"
                --spec-draft-n-max "${SPEC_N_MAX:-8}"
                --spec-draft-p-min "${SPEC_P_MIN:-0.7}" )
        # An external drafter in its own file -- a DFlash head, a grafted MTP
        # head. Opt-in: absent unless DRAFT_MODEL names a file. Mirrors the
        # two DRAFT_* lines in docker-compose.yml.
        [ -n "${DRAFT_MODEL:-}" ] && ARGS+=( --spec-draft-model "${MODELS_DIR}/${DRAFT_MODEL}" )
        [ -n "${DRAFT_NGL:-}" ]   && ARGS+=( --spec-draft-ngl "${DRAFT_NGL}" )
    fi
}

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

wait_until_ready() {
    echo -n "waiting for the model to load"
    for _ in $(seq 1 100); do
        if curl -sf -m 3 "$BASE/health" >/dev/null 2>&1; then
            echo " — ready"; return 0
        fi
        if ! running; then
            echo " — the process went down"
            echo "--- last lines of $LOGFILE ---"; tail -30 "$LOGFILE"
            return 1
        fi
        echo -n "."; sleep 5
    done
    echo " — timed out"; return 1
}

case "${1:-status}" in

    preflight)
        # Checks the things this script assumes, instead of discovering them the
        # hard way five minutes into a model load.
        rc=0
        echo "--- platform ---"
        echo "  $(uname -sm)  macOS $(sw_vers -productVersion 2>/dev/null || echo '?')"
        [ "$(uname -s)" = "Darwin" ] || { echo "  NOTE: not macOS — use scripts/serve.sh"; rc=1; }
        [ "$(uname -m)" = "arm64" ] || echo "  WARNING: not Apple Silicon; there is no Metal backend on Intel Macs"

        echo "--- llama-server ---"
        if [ -x "$LLAMA_BIN" ] || command -v "$LLAMA_BIN" >/dev/null 2>&1; then
            echo "  binary : $LLAMA_BIN"
            echo "  version: $("$LLAMA_BIN" --version 2>&1 | head -1)"
        else
            echo "  MISSING: $LLAMA_BIN"
            echo "           brew install llama.cpp   (or set LLAMA_BIN in .env)"
            rc=1
        fi

        echo "--- Metal backend ---"
        # A CPU-only build will happily serve the model at a tenth of the speed
        # and say nothing about it, so this is worth confirming up front.
        if "$LLAMA_BIN" --list-devices 2>&1 | grep -qi metal; then
            echo "  present: $("$LLAMA_BIN" --list-devices 2>&1 | grep -i metal | head -1)"
        else
            echo "  NOT DETECTED — check with: $LLAMA_BIN --list-devices"
            echo "  A CPU-only build runs but is roughly an order of magnitude slower."
            rc=1
        fi

        echo "--- flags this build supports ---"
        for f in --spec-type --cache-ram --reasoning-format --jinja --metrics -fa; do
            if "$LLAMA_BIN" --help 2>&1 | grep -q -- "$f"; then
                echo "  ok      $f"
            else
                echo "  MISSING $f   <- update llama.cpp, or drop it from build_args()"
                rc=1
            fi
        done

        echo "--- model ---"
        if [ -f "$MODEL" ]; then
            echo "  $MODEL"
            echo "  $(stat -f %z "$MODEL" 2>/dev/null || stat -c %s "$MODEL") bytes"
            [ "$(head -c 4 "$MODEL")" = "GGUF" ] && echo "  header : GGUF ok" \
                || { echo "  header : NOT GGUF — the file is not a model"; rc=1; }
        else
            echo "  MISSING: $MODEL"
            echo "           ./scripts/download-model.sh <repo> <file.gguf>"
            rc=1
        fi

        echo
        [ "$rc" = 0 ] && echo "preflight passed" || echo "preflight found problems (above)"
        exit "$rc"
        ;;

    mem)
        # The macOS equivalent of `serve.sh vram`. Unified memory, so the budget
        # is shared with the OS and everything else running.
        TOTAL=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
        echo "--- unified memory ---"
        printf "  installed        : %.1f GB\n" "$(awk "BEGIN{print $TOTAL/1e9}")"
        if [ "$WIRED" = "0" ]; then
            # TODO(you): confirm the actual fraction on your macOS version. It is
            # roughly 70% and has changed between releases; the exact number is
            # what decides whether a given CTX fits.
            printf "  GPU-usable       : system default, about 70%% -> ~%.1f GB\n" \
                   "$(awk "BEGIN{print $TOTAL*0.70/1e9}")"
            echo   "                     raise with: sudo sysctl iogpu.wired_limit_mb=NNNNN"
        else
            echo   "  GPU-usable       : ${WIRED} MB (iogpu.wired_limit_mb, set explicitly)"
        fi
        if [ -f "$MODEL" ]; then
            SZ=$(stat -f %z "$MODEL" 2>/dev/null || stat -c %s "$MODEL")
            printf "  model weights    : %.1f GB\n" "$(awk "BEGIN{print $SZ/1e9}")"
        fi
        echo
        echo "--- KV cache cost for this model (from its own metadata) ---"
        python3 scripts/gguf-info.py "$MODEL" 2>/dev/null \
            | sed -n '/KV cache per token/,$p' \
            || echo "  (run: python3 scripts/gguf-info.py \"$MODEL\")"
        echo
        echo "weights + KV at your CTX must fit in GPU-usable memory, with room"
        echo "left for the OS. README section 5 has the method."
        ;;

    up)
        if running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
        build_args
        echo "starting: $LLAMA_BIN"
        printf '  %s\n' "${ARGS[*]}"
        # Detached, output to a file, so `logs` behaves like the compose version.
        nohup "$LLAMA_BIN" "${ARGS[@]}" >"$LOGFILE" 2>&1 &
        echo $! > "$PIDFILE"
        wait_until_ready || exit 1
        "$0" mem
        ;;

    down)
        if running; then
            kill "$(cat "$PIDFILE")" && echo "stopped (pid $(cat "$PIDFILE"))"
            rm -f "$PIDFILE"
        else
            echo "not running"
            rm -f "$PIDFILE"
        fi
        echo "the .gguf files are untouched, they live in $MODELS_DIR"
        ;;

    status)
        running && echo "process : running (pid $(cat "$PIDFILE"))" || echo "process : not running"
        echo "--- health ---"
        curl -s -m 5 "$BASE/health" || echo "(not responding)"
        echo
        echo "--- model being served ---"
        curl -s -m 5 "$BASE/v1/models" 2>/dev/null \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null \
            || echo "(unavailable)"
        ;;

    logs)
        tail -n "${2:-60}" -f "$LOGFILE"
        ;;

    test)
        echo "--- test query ---"
        curl -s -m 180 "$BASE/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL_ALIAS}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 25*4? Reply with the number only.\"}],\"max_tokens\":600}" \
        | python3 -c '
import json, sys
d = json.load(sys.stdin)
m = d["choices"][0]["message"]
# With --reasoning-format deepseek the thinking block is returned separately; on
# a small max_tokens budget content can come back empty, which does not mean the
# server is broken.
print("reasoning:", (m.get("reasoning_content") or "")[:160].replace("\n", " "))
print("content  :", repr(m.get("content")))
print("usage    :", d.get("usage"))
'
        ;;

    *)
        echo "usage: $0 up|down|status|logs [N]|test|mem|preflight" >&2
        exit 1
        ;;
esac
