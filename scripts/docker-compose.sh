#!/usr/bin/env bash
# Chạy từ thư mục gốc repo: bash scripts/docker-compose.sh <lệnh>
# Ví dụ: bash scripts/docker-compose.sh woe
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
cmd="${1:-}"
shift || true
case "$cmd" in
  build) docker compose build "$@" ;;
  shell|bash) docker compose run --rm app bash "$@" ;;
  woe|woe-analysis) docker compose run --rm woe-analysis "$@" ;;
  simplified|create-simplified) docker compose run --rm create-simplified "$@" ;;
  sample|sample-data) docker compose run --rm sample-data "$@" ;;
  train-softmax) docker compose run --rm train-mlp-softmax "$@" ;;
  validate-softmax) docker compose run --rm validate-mlp-softmax "$@" ;;
  test-softmax) docker compose run --rm test-mlp-softmax "$@" ;;
  train-coral) docker compose run --rm train-mlp-coral "$@" ;;
  validate-coral) docker compose run --rm validate-mlp-coral "$@" ;;
  test-coral) docker compose run --rm test-mlp-coral "$@" ;;
  *)
    echo "Usage: $0 {build|shell|woe|simplified|sample|train-softmax|validate-softmax|test-softmax|train-coral|validate-coral|test-coral} [extra args]"
    exit 1
    ;;
esac
