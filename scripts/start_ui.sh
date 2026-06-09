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
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Pick the UI framework. Default is Gradio (works inside AMD AI Developer
# Cloud JupyterLab); set UI_FRAMEWORK=streamlit to use the legacy app.
UI_FRAMEWORK="${UI_FRAMEWORK:-gradio}"
UI_HOST="${GRADIO_HOST:-${STREAMLIT_HOST:-0.0.0.0}}"
UI_PORT="${GRADIO_PORT:-${STREAMLIT_PORT:-8501}}"

export GRADIO_HOST="$UI_HOST"
export GRADIO_PORT="$UI_PORT"
export STREAMLIT_HOST="$UI_HOST"
export STREAMLIT_PORT="$UI_PORT"

print_access_urls() {
  local port="$1"
  local external_ip="${PUBLIC_IP:-${DROPLET_PUBLIC_IP:-${EXTERNAL_IP:-}}}"

  if [[ -z "$external_ip" ]] && command -v curl >/dev/null 2>&1; then
    external_ip="$(curl -fsS --max-time 2 https://api.ipify.org 2>/dev/null || true)"
  fi

  echo "  local:    http://localhost:${port}"

  if command -v hostname >/dev/null 2>&1; then
    read -r -a internal_ips <<<"$(hostname -I 2>/dev/null || true)"
    for ip in "${internal_ips[@]}"; do
      if [[ -n "$ip" ]]; then
        echo "  internal: http://${ip}:${port}"
      fi
    done
  fi

  if [[ -n "$external_ip" ]]; then
    echo "  external: http://${external_ip}:${port}"
  else
    echo "  external: http://<droplet-public-ip>:${port}"
  fi
}

echo "Using UI framework: ${UI_FRAMEWORK}"

if [[ "$UI_FRAMEWORK" == "streamlit" ]]; then
  echo "Starting Streamlit UI (legacy)"
  print_access_urls "$UI_PORT"
  exec streamlit run src/app.py --server.address "$UI_HOST" --server.port "$UI_PORT"
else
  if [[ ! -f src/gradio_app.py ]]; then
    echo "Gradio UI file not found: ${ROOT_DIR}/src/gradio_app.py"
    echo "Make sure the AMD machine has the latest project files copied."
    exit 1
  fi

  echo "Starting Gradio UI"
  print_access_urls "$UI_PORT"
  exec python -m src.gradio_app
fi
