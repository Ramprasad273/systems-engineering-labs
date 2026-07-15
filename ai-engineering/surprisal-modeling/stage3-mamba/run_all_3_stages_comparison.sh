#!/usr/bin/env bash
# ==============================================================================
# Universal 3-Stage Comparison Runner (NVIDIA RTX 3060 Ti 8GB VRAM Parity)
# ==============================================================================
# Generates the universal cross-stage comparison table and tri-layer tradeoff figure
# comparing Stage 1 (GPT-2), Stage 2 (Qwen-2.5-3B QLoRA), and Stage 3 (Mamba & MambaLog).
#
# Usage:
#   chmod +x run_all_3_stages_comparison.sh
#   ./run_all_3_stages_comparison.sh
# ==============================================================================

set -e

echo "=== [Master Evaluation] Generating Universal 3-Stage Comparison & RTX 3060 Ti Parity Report ==="
python3 scripts/compare_all_3_stages.py

echo ""
echo "=== [SUCCESS] Universal Cross-Stage Analysis Generated! ==="
echo "Check the following files in 'results/':"
echo "  - results/all_3_stages_universal_comparison.md (Complete side-by-side markdown report)"
echo "  - results/all_3_stages_universal_comparison.csv (Raw CSV data)"
echo "  - results/all_3_stages_tradeoff_chart.png (Tri-layer VRAM & F1 vs Latency trade-off chart)"
