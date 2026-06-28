#!/usr/bin/env bash
# =============================================================================
# Entrypoint: Fix LD_LIBRARY_PATH for WSL2 host driver symbol resolution.
#
# On WSL2, NVIDIA driver libraries are split across two host paths:
#   1. /usr/lib/wsl/lib        — standard CUDA stubs (always present)
#   2. /usr/lib/wsl/drivers/   — arch-specific symbols for newer GPUs/drivers
#
# The CUDA Error 500 ("named symbol not found") happens when PyTorch can only
# find the stubs in (1) but not the real symbols from (2). This script detects
# the full driver path and prepends it to LD_LIBRARY_PATH before exec'ing.
# =============================================================================

set -euo pipefail

# --- Resolve WSL2 host driver library path ---
WSL_LIB="/usr/lib/wsl/lib"
WSL_DRIVER_LIB=""

# Look for the driver-specific lib directory (contains libcuda.so, etc.)
if [ -d "/usr/lib/wsl/drivers" ]; then
    # Prefer the directory that contains libcuda.so.1
    for dir in /usr/lib/wsl/drivers/*/; do
        if [ -f "${dir}libcuda.so.1" ] || [ -f "${dir}libcuda.so" ]; then
            WSL_DRIVER_LIB="${dir}"
            break
        fi
    done
    # Fallback: use the first drivers subdirectory found
    if [ -z "${WSL_DRIVER_LIB}" ]; then
        WSL_DRIVER_LIB=$(ls -d /usr/lib/wsl/drivers/*/ 2>/dev/null | head -n 1 || true)
    fi
fi

# Build LD_LIBRARY_PATH: driver-specific path takes priority over stub path
if [ -n "${WSL_DRIVER_LIB}" ]; then
    echo "[entrypoint] WSL2 driver lib detected: ${WSL_DRIVER_LIB}"
    export LD_LIBRARY_PATH="${WSL_DRIVER_LIB}:${WSL_LIB}:${LD_LIBRARY_PATH:-}"
else
    echo "[entrypoint] No driver-specific WSL2 lib found; using: ${WSL_LIB}"
    export LD_LIBRARY_PATH="${WSL_LIB}:${LD_LIBRARY_PATH:-}"
fi

echo "[entrypoint] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Execute the command passed to this container (default: bash)
exec "$@"
