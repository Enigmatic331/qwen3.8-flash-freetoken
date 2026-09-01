#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${QWEN38_CONFIG:-$REPO_DIR/.env}"
LOCK_FILE="$REPO_DIR/freetoken.lock"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config: $CONFIG_FILE (copy .env.example and edit it)" >&2
  exit 1
fi

set -a
source "$LOCK_FILE"
source "$CONFIG_FILE"
set +a

: "${FREETOKEN_CHECKOUT:?}"
: "${FREETOKEN_PYTHON:?}"
: "${FREETOKEN_NATIVE_EXTENSION_DIR:?}"
: "${MODEL_PATH:?}"

actual_revision="$(git -C "$FREETOKEN_CHECKOUT" rev-parse HEAD)"
if [[ "$actual_revision" != "$FREETOKEN_REVISION" ]]; then
  echo "FreeToken revision mismatch: expected $FREETOKEN_REVISION, got $actual_revision" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "Checkpoint index is missing: $MODEL_PATH" >&2
  exit 1
fi

shopt -s nullglob
model_shards=("$MODEL_PATH"/model-*.safetensors)
if (( ${#model_shards[@]} != MODEL_SHARDS )); then
  echo "Checkpoint needs $MODEL_SHARDS shards; found ${#model_shards[@]}" >&2
  exit 1
fi

export NCCL_P2P_DISABLE="${EP2_NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${EP2_NCCL_P2P_LEVEL:-PHB}"
export NCCL_IB_DISABLE=1
export PYTHONPATH="$FREETOKEN_CHECKOUT/python"
export FREETOKEN_NATIVE_EXTENSION_DIR
export FREETOKEN_FP8_EXPERTS=fp8
export FREETOKEN_EP_PACKED_WIRE_DTYPE="${EP2_PACKED_WIRE_DTYPE:-bf16}"
export FREETOKEN_EP_PACKED_ASYNC_SEND="${EP2_PACKED_ASYNC_SEND:-1}"
export FREETOKEN_EP_SKIP_INACTIVE_PREFILL_ROUTES="${EP2_SKIP_INACTIVE_PREFILL_ROUTES:-1}"
export FREETOKEN_EP_COMPACT_INACTIVE_PREFILL_ROUTES="${EP2_COMPACT_INACTIVE_PREFILL_ROUTES:-1}"
export FREETOKEN_GDN_PREFILL_TILE_TOKENS="${EP2_GDN_PREFILL_TILE_TOKENS:-8192}"
export FREETOKEN_EP_PREFILL_ROUTE_TILE_TOKENS="${EP2_PREFILL_ROUTE_TILE_TOKENS:-8192}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "$FREETOKEN_PYTHON" -m freetoken.cli serve \
  --model "$MODEL_PATH" \
  --served-model-name "${EP2_SERVED_MODEL_NAME:-qwen38-flash-next-fp8-ep2}" \
  --host "${EP2_HOST:-172.17.0.1}" \
  --port "${EP2_PORT:-1919}" \
  --gpu 0,1 \
  --tp-size 2 \
  --qwen4-exp-backbone-rank 0 \
  --max-running-requests 1 \
  --max-seq-len-override "${EP2_CONTEXT:-262144}" \
  --num-tokens "${EP2_KV_TOKENS:-262144}" \
  --max-prefill-length "${EP2_MAX_PREFILL_LENGTH:-32768}" \
  --memory-ratio 0.90 \
  --cuda-graph-max-bs 1 \
  --cache-type naive \
  --moe-backend offload \
  --expert-load parallel \
  --moe-cache-sizes "${EP2_MOE_CACHE_SIZES:-2048,4096}" \
  --moe-collect-stats \
  --sampling-defaults none \
  --reasoning-parser qwen3 \
  --decode-log-interval 32
