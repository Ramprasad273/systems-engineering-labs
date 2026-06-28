#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Define service name from docker-compose.yml
SERVICE_NAME="surprisal-gpt2-train"

echo "=== [Phase 0] Verifying Docker Environment ==="

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: docker command not found. Please install Docker." >&2
    exit 1
fi

# Check if docker compose or docker-compose is available
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "Error: Neither 'docker compose' nor 'docker-compose' could be found." >&2
    exit 1
fi

echo "Using Docker Compose command: ${DOCKER_COMPOSE}"

# Check GPU availability on host
echo "Checking GPU accessibility on host..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: 'nvidia-smi' not found on the host system. GPU acceleration may not be available."
else
    echo "NVIDIA GPU detected on host:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
fi

# Show WSL2 driver library paths available to the container
echo ""
echo "=== WSL2 Driver Library Paths ==="
if [ -d "/usr/lib/wsl/lib" ]; then
    echo "[host] Standard WSL2 lib: /usr/lib/wsl/lib"
fi
if [ -d "/usr/lib/wsl/drivers" ]; then
    echo "[host] Driver-specific WSL2 dirs:"
    ls -d /usr/lib/wsl/drivers/*/ 2>/dev/null || echo "  (none found)"
fi

# Build container
echo ""
echo "=== Building Docker Container ==="
${DOCKER_COMPOSE} build

# Run verification checks inside container
echo ""
echo "=== Verifying CUDA & Dependencies inside Container ==="
${DOCKER_COMPOSE} run --rm ${SERVICE_NAME} python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Version:', torch.version.cuda)
    print('Device Name:', torch.cuda.get_device_name(0))
    print('Device Memory (MB):', round(torch.cuda.get_device_properties(0).total_memory / 1e6))
else:
    print('WARNING: CUDA is NOT available in the PyTorch container.')
"

${DOCKER_COMPOSE} run --rm ${SERVICE_NAME} python -c "
import transformers, datasets, pynvml, yaml
print('All python dependencies (transformers, datasets, nvidia-ml-py3, pyyaml) imported successfully!')
"

echo ""
echo "=== Phase 0 Setup Verified & Completed Successfully ==="
