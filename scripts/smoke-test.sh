#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${QWEN38_BASE_URL:-http://172.17.0.1:1919}"
MODEL="${QWEN38_MODEL:-qwen38-flash-next-fp8-ep2}"

curl --fail --silent --show-error "$BASE_URL/health"
echo
curl --fail --silent --show-error "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: FREETOKEN_OK\"}],\"temperature\":0,\"max_tokens\":32}"
echo
