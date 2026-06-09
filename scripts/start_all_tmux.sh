#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-ai-mutation}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install it with:"
  echo "  sudo apt install -y tmux"
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists. Attaching."
  tmux attach -t "$SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -c "$ROOT_DIR" "bash scripts/start_vllm.sh"
tmux split-window -h -t "${SESSION_NAME}:0" -c "$ROOT_DIR" "bash scripts/start_backend.sh"
tmux split-window -v -t "${SESSION_NAME}:0.1" -c "$ROOT_DIR" "bash scripts/start_ui.sh"
tmux select-layout -t "${SESSION_NAME}:0" tiled

echo "Started tmux session '${SESSION_NAME}' with vLLM, FastAPI, and Streamlit."
tmux attach -t "$SESSION_NAME"

