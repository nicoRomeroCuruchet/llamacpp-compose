#!/bin/bash
# Server operation. Everything goes through here so nobody has to type raw
# docker compose invocations.
#
#   ./scripts/serve.sh up|down|status|logs|test|vram
#   ./scripts/serve.sh fim up|down|status|test|logs
#
# The `fim` subcommands drive the second, independent server: the fill-in-the-
# middle model an editor plugin talks to, on FIM_PORT. It lives behind a compose
# profile, so the plain subcommands above never touch it and it never starts by
# accident.
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
FIM_PORT="${FIM_PORT:-8012}"
FIM_BASE="http://127.0.0.1:${FIM_PORT}"

# Both arguments are required. The liveness check has to name its service: with
# two of them in the project, a bare `docker compose ps --status running` is
# satisfied by whichever container is up, so a dead chat server would be waited
# on until the timeout while the FIM one kept the check happy.
wait_until_ready() {
    base="$1"; svc="$2"
    echo -n "waiting for the model to load"
    for _ in $(seq 1 100); do
        if curl -sf -m 3 "$base/health" >/dev/null 2>&1; then
            echo " — ready"
            return 0
        fi
        # If the container died there is no point in waiting any longer.
        if ! docker compose ps --status running --quiet "$svc" 2>/dev/null | grep -q .; then
            echo " — the container went down"
            echo "--- last lines ---"
            docker compose logs --tail 30 "$svc"
            return 1
        fi
        echo -n "."
        sleep 5
    done
    echo " — timed out"
    return 1
}

case "${1:-status}" in
    up)
        docker compose up -d llama
        wait_until_ready "$BASE" llama || exit 1
        "$0" vram
        ;;
    down)
        # Scoped to the chat service on purpose. `docker compose down` acts on
        # the whole project, and compose lists profiled services as part of it,
        # so the unscoped form would take the FIM server with it without saying
        # so. Stop that one with `serve.sh fim down`.
        docker compose rm -sf llama
        echo "stopped (the .gguf files are untouched, they live in $MODELS_DIR)"
        if docker compose ps --status running --quiet fim 2>/dev/null | grep -q .; then
            echo "note: the FIM server on port ${FIM_PORT} is still up (serve.sh fim down)"
        fi
        ;;
    status)
        # Scoped, like `down`: an unscoped `ps` lists the FIM container too and
        # reads as though the chat server were up when it is not.
        docker compose ps llama
        echo "--- health ---"
        curl -s -m 5 "$BASE/health" || echo "(not responding)"
        echo
        echo "--- model being served ---"
        curl -s -m 5 "$BASE/v1/models" 2>/dev/null \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null \
            || echo "(unavailable)"
        ;;
    logs)
        docker compose logs --tail "${2:-60}" -f llama
        ;;
    vram)
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
                   --format=csv,noheader
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
    fim)
        case "${2:-status}" in
            up)
                docker compose --profile fim up -d fim
                wait_until_ready "$FIM_BASE" fim || exit 1
                "$0" vram
                ;;
            down)
                docker compose rm -sf fim
                echo "FIM server stopped"
                ;;
            status)
                docker compose ps fim
                echo "--- health ---"
                curl -s -m 5 "$FIM_BASE/health" || echo "(not responding)"
                echo
                ;;
            logs)
                docker compose logs --tail "${3:-60}" -f fim
                ;;
            test)
                # /infill, not /v1/chat/completions. This is the endpoint the
                # editor actually uses, and the one that fails if the model
                # turns out to be an instruct checkpoint with no FIM tokens.
                echo "--- infill query ---"
                curl -s -m 60 "$FIM_BASE/infill" \
                    -H "Content-Type: application/json" \
                    -d '{"input_prefix":"def fibonacci(n: int) -> int:\n    if n < 2:\n        return n\n    ","input_suffix":"\n\nprint(fibonacci(10))\n","n_predict":48,"temperature":0.1}' \
                | python3 -c '
import json, sys
d = json.load(sys.stdin)
if "error" in d:
    print("ERROR:", d["error"]); sys.exit(1)
print("completion:", repr(d.get("content")))
t = d.get("timings", {})
print("prefill   : %s tok @ %.0f t/s" % (t.get("prompt_n"), t.get("prompt_per_second", 0)))
print("decode    : %s tok @ %.0f t/s" % (t.get("predicted_n"), t.get("predicted_per_second", 0)))
print("latency   : %.0f ms" % (t.get("prompt_ms", 0) + t.get("predicted_ms", 0)))
'
                ;;
            *)
                echo "usage: $0 fim up|down|status|logs [N]|test" >&2
                exit 1
                ;;
        esac
        ;;
    *)
        echo "usage: $0 up|down|status|logs [N]|test|vram" >&2
        echo "       $0 fim up|down|status|logs [N]|test" >&2
        exit 1
        ;;
esac
