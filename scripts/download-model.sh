#!/bin/bash
# Download a .gguf from HuggingFace into MODELS_DIR and verify it arrived whole.
#
#   ./scripts/download-model.sh <repo> <file.gguf>
#   ./scripts/download-model.sh unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf
#
# curl is used on purpose: it needs neither huggingface_hub nor pip, and `-C -`
# resumes an interrupted transfer. On a shared machine, installing nothing is
# the better trade.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] || { echo "ERROR: .env is missing (cp .env.example .env)" >&2; exit 1; }
# shellcheck disable=SC1091
set -a; . ./.env; set +a

REPO="${1:?usage: $0 <repo> <file.gguf>}"
FILE="${2:?usage: $0 <repo> <file.gguf>}"
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"
DEST="${MODELS_DIR}/${FILE}"

mkdir -p "$MODELS_DIR"

# The size HuggingFace declares, used to verify the download at the end. Under
# Xet Storage that value comes back in x-linked-size rather than in the
# content-length of the first 302.
echo "querying remote size..."
EXPECTED=$(curl -sIL "$URL" | grep -i '^x-linked-size:' | tail -1 | tr -dc '0-9')
[ -z "$EXPECTED" ] && EXPECTED=$(curl -sIL "$URL" | grep -i '^content-length:' | tail -1 | tr -dc '0-9')
[ -z "$EXPECTED" ] && { echo "ERROR: could not determine the remote size" >&2; exit 1; }
printf "remote: %s bytes (%.1f GB)\n" "$EXPECTED" "$(awk "BEGIN{print $EXPECTED/1e9}")"

if [ -f "$DEST" ] && [ "$(stat -c %s "$DEST")" = "$EXPECTED" ]; then
    echo "already complete, nothing to do: $DEST"
    exit 0
fi

echo "downloading to $DEST (resumable)..."
curl -L -C - --progress-bar -o "$DEST" "$URL"

# Verification. The file existing is not enough: an interrupted download leaves
# a perfectly valid half-file behind, and an HTTP error response can end up
# saved under the model's name.
ACTUAL=$(stat -c %s "$DEST")
if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "ERROR: size $ACTUAL != expected $EXPECTED. Re-run to resume." >&2
    exit 1
fi
if [ "$(head -c 4 "$DEST")" != "GGUF" ]; then
    echo "ERROR: the header is not GGUF. What came down is not a model." >&2
    exit 1
fi

echo "OK: $DEST — exact size and valid GGUF header"
echo "Remember to point MODEL_FILE=$FILE at it in .env"
