#!/bin/bash
# Server operation. Everything goes through here so nobody has to type raw
# docker compose invocations.
#
#   ./scripts/serve.sh up|down|status|logs|test|vram
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

wait_until_ready() {
    echo -n "waiting for the model to load"
    for _ in $(seq 1 100); do
        if curl -sf -m 3 "$BASE/health" >/dev/null 2>&1; then
            echo " — ready"
            return 0
        fi
        # If the container died there is no point in waiting any longer.
        if ! docker compose ps --status running --quiet 2>/dev/null | grep -q .; then
            echo " — the container went down"
            echo "--- last lines ---"
            docker compose logs --tail 30
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
        docker compose up -d
        wait_until_ready || exit 1
        "$0" vram
        ;;
    down)
        docker compose down
        echo "stopped (the .gguf files are untouched, they live in $MODELS_DIR)"
        ;;
    status)
        docker compose ps
        echo "--- health ---"
        curl -s -m 5 "$BASE/health" || echo "(not responding)"
        echo
        echo "--- model being served ---"
        curl -s -m 5 "$BASE/v1/models" 2>/dev/null \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null \
            || echo "(unavailable)"
        ;;
    logs)
        docker compose logs --tail "${2:-60}" -f
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
    *)
        echo "usage: $0 up|down|status|logs [N]|test|vram" >&2
        exit 1
        ;;
esac
