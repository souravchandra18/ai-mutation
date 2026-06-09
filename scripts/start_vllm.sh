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

MODEL_ID="${MODEL_ID:-${AI_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8090}"
VLLM_CONTAINER_PORT="${VLLM_CONTAINER_PORT:-8000}"
VLLM_BACKEND="${VLLM_BACKEND:-auto}"

HF_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ARGS+=(--env "HF_TOKEN=${HF_TOKEN}")
fi

echo "Starting vLLM on AMD ROCm"
echo "  model: ${MODEL_ID}"
echo "  url:   http://localhost:${VLLM_HOST_PORT}/v1"
echo "  mode:  ${VLLM_BACKEND}"

docker_available() {
  command -v docker >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1
}

start_with_docker() {
  sudo docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "${HF_ARGS[@]}" \
    -p "127.0.0.1:${VLLM_HOST_PORT}:${VLLM_CONTAINER_PORT}" \
    --ipc=host \
    --entrypoint python3 \
    "$VLLM_IMAGE" \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_CONTAINER_PORT" \
    ${VLLM_ARGS:-}
}

start_with_python() {
  # Deactivate any existing virtual environment
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate 2>/dev/null || true
  fi
  
  # Try to use system vLLM first (for AMD notebooks with pre-configured ROCm)
  if command -v vllm >/dev/null 2>&1; then
    # Use system vLLM directly
    vllm serve "$MODEL_ID" \
      --host 127.0.0.1 \
      --port "$VLLM_HOST_PORT" \
      ${VLLM_ARGS:-}
  elif [[ ! -d .venv ]]; then
    echo "Python virtual environment not found. Run the setup steps first."
    exit 1
  else
    # Fallback to .venv if available
    # shellcheck disable=SC1091
    source .venv/bin/activate

    if ! python -c "import vllm" >/dev/null 2>&1; then
      echo "vLLM is not installed in .venv."
      echo "Install it inside the AMD JupyterLab terminal with:"
      echo "  source .venv/bin/activate"
      echo "  python -m pip install vllm"
      echo
      echo "If that install fails, your notebook image may not include ROCm/vLLM support."
      echo "Use a GPU droplet image with Docker, or an image with vLLM preinstalled."
      exit 1
    fi

    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_ID" \
      --host 127.0.0.1 \
      --port "$VLLM_HOST_PORT" \
      ${VLLM_ARGS:-}
  fi
}

case "$VLLM_BACKEND" in
  docker)
    start_with_docker
    ;;
  python)
    start_with_python
    ;;
  auto)
    if docker_available; then
      start_with_docker
    else
      echo "Docker is not available or the Docker daemon is not running."
      echo "Falling back to Python vLLM mode."
      start_with_python
    fi
    ;;
  *)
    echo "Unknown VLLM_BACKEND='${VLLM_BACKEND}'. Use auto, docker, or python."
    exit 1
    ;;
esac
