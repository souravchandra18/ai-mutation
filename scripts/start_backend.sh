#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ ! -d .venv ]]; then
  echo "Python virtual environment not found. Run:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  python -m pip install -r requirements.txt"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

echo "Starting FastAPI backend"
echo "  url: http://${BACKEND_HOST}:${BACKEND_PORT}"

uvicorn src.api:app --host "$BACKEND_HOST" --port "$BACKEND_PORT"

