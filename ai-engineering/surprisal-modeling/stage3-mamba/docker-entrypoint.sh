#!/usr/bin/env bash
# ==============================================================================
# Docker Entrypoint for Stage 3 Mamba Speed Layer (`docker-entrypoint.sh`)
# ==============================================================================

set -e

if [ "$1" == "--fast-dev-run" ] || [ -z "$1" ]; then
    echo "=== Running Stage 3 Verification & Cross-Stage Parity Suite via Docker (Fast Dev Mode) ==="
    exec ./run_paper_experiments.sh --fast-dev-run
elif [ "$1" == "--full" ] || [ "$1" == "full" ] || [ "$1" == "paper" ]; then
    echo "=== Running Full-Scale Stage 3 Pre-Training & Verification Suite (Multi-Epoch) ==="
    exec ./run_paper_experiments.sh
elif [ "$1" == "test" ] || [ "$1" == "pytest" ]; then
    echo "=== Running Unit Verification Suite ==="
    exec python3 -m pytest tests/test_mamba_shapes.py -v
elif [ "$1" == "compare" ]; then
    echo "=== Running Universal 3-Stage Cross-Comparison Engine ==="
    exec python3 scripts/compare_all_3_stages.py
else
    # Execute custom command (e.g. bash, python train.py ...)
    exec "$@"
fi
