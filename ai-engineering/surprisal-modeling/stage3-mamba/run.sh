#!/usr/bin/env bash
# ==============================================================================
# Quick Verification Runner (`run.sh`)
# ==============================================================================
# Executes unit verification suite, generates synthetic evaluation tables, and
# compiles publication figures for quick inspection without waiting for 24-layer GPU training.
# ==============================================================================

set -e

echo "=== Stage 3 Mamba Quick Verification Suite ==="
python3 -m pytest tests/test_mamba_shapes.py -v

echo ""
echo "=== Compiling Research Comparison Tables ==="
python3 scripts/analyze_stage3_results.py

echo ""
echo "=== Generating Feynman Publication Figures ==="
python3 scripts/generate_blog_figures.py

echo ""
echo "=== Verification Complete! Check 'results/' for outputs. ==="
