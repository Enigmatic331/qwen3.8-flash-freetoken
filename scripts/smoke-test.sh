#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${QWEN38_BASE_URL:-http://172.17.0.1:8080}"
MODEL="${QWEN38_MODEL:-qwen3.8-flash}"

curl --fail --silent --show-error "$BASE_URL/health"
echo
response="$(curl --fail --silent --show-error "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: FREETOKEN_OK\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"temperature\":0,\"max_tokens\":32}")"
printf '%s\n' "$response"
printf '%s\n' "$response" | python3 -c 'import json,sys; assert json.load(sys.stdin)["choices"][0]["message"]["content"].strip() == "FREETOKEN_OK"'
