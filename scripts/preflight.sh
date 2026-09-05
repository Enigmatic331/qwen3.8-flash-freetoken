#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${QWEN38_CONFIG:-$REPO_DIR/.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config: $CONFIG_FILE (copy .env.example and edit it)" >&2
  exit 1
fi
set -a
source "$REPO_DIR/freetoken.lock"
source "$CONFIG_FILE"
set +a

fail=0
check_file() {
  if [[ ! -e "$1" ]]; then
    echo "MISSING: $1" >&2
    fail=1
  fi
}

check_file "$FREETOKEN_PYTHON"
check_file "$FREETOKEN_CHECKOUT/python/freetoken"
check_file "$FREETOKEN_NATIVE_EXTENSION_DIR"
check_file "$MODEL_PATH/model.safetensors.index.json"

actual_revision="$(git -C "$FREETOKEN_CHECKOUT" rev-parse HEAD 2>/dev/null || true)"
if [[ "$actual_revision" != "$FREETOKEN_REVISION" ]]; then
  echo "REVISION: expected $FREETOKEN_REVISION, got ${actual_revision:-unknown}" >&2
  fail=1
fi

if ! nvidia-smi -i "${EP2_VISION_DEVICE:-2}" --query-gpu=name --format=csv,noheader >/dev/null; then
  echo "VISION: GPU ${EP2_VISION_DEVICE:-2} is unavailable" >&2
  fail=1
fi

shopt -s nullglob
shards=("$MODEL_PATH"/model-*.safetensors)
if (( ${#shards[@]} != MODEL_SHARDS )); then
  echo "SHARDS: expected $MODEL_SHARDS, found ${#shards[@]}" >&2
  fail=1
fi

echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)"
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv
nvidia-smi topo -m
free -h

if (( fail != 0 )); then
  exit 1
fi
echo "Preflight passed. Validate direct P2P separately after any driver or BIOS change."
