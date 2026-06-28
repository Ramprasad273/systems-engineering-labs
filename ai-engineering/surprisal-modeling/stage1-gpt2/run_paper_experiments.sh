#!/usr/bin/env bash
set -euo pipefail

FORCE=0
if [ "${1:-}" == "--force" ]; then
    FORCE=1
fi

echo "=============================================================================="
echo "          SURPRISAL MODELING: RESEARCH MANUSCRIPT BENCHMARK SUITE             "
echo "=============================================================================="
echo "Hardware Environment : CUDA GPU Acceleration"
echo "Real-Time Telemetry  : Active (tqdm ETA Spinners)"
echo "Execution CWD        : $(pwd)"
echo "Force Retrain Mode   : ${FORCE}"
echo "=============================================================================="

echo ""
if [ -f "data/checkpoints/checkpoint_10000.pt" ] && [ "${FORCE}" -eq 0 ]; then
    echo "=== [1/5] Pre-trained checkpoint_10000.pt detected! Skipping redundant 2.5h pre-training ==="
else
    echo "=== [1/5] Main Unsupervised Surprisal Model Training ==="
    python train.py config/stage1_config.yaml
fi

echo ""
if [ -f "data/stage1_eval_results.json" ] && [ "${FORCE}" -eq 0 ]; then
    echo "=== [2/5] Test benchmark evaluation results detected! Skipping redundant evaluation ==="
else
    echo "=== [2/5] Formal Test Set Evaluation & Threshold Calibration ==="
    python evaluate.py
fi

echo ""
echo "=== [3/5] Empirical Ablation Sweeps (Paper Table II & III) ==="

if [ -f "data/ablations/threshold_sensitivity.json" ] && [ "${FORCE}" -eq 0 ]; then
    echo "  [3a] Threshold sensitivity JSON detected! Skipping."
else
    python scripts/threshold_sensitivity.py
fi

if [ -f "data/ablations/ablation_depth.json" ] && [ "${FORCE}" -eq 0 ]; then
    echo "  [3b] Model depth ablation JSON detected! Skipping redundant 2.5h sweep."
else
    python scripts/ablation_depth.py
fi

if [ -f "data/ablations/ablation_vocab.json" ] && [ "${FORCE}" -eq 0 ]; then
    echo "  [3c] Vocab size ablation JSON detected! Skipping."
else
    python scripts/ablation_vocab.py
fi

if [ -f "data/ablations/token_stability.json" ] && [ "${FORCE}" -eq 0 ]; then
    echo "  [3d] Token stability JSON detected! Skipping."
else
    python scripts/token_stability_check.py
fi

echo ""
echo "=== [4/5] Compiling LaTeX Tables & Markdown Catalog ==="
python scripts/analyze_results.py

echo ""
echo "=============================================================================="
echo "          [SUCCESS] ALL RESEARCH EXPERIMENTS COMPLETED                        "
echo "=============================================================================="
echo "Generated Artifacts:"
echo "  1. Model Weights     : data/checkpoints/checkpoint_10000.pt"
echo "  2. Ablation Logs     : data/ablations/*.json"
echo "  3. Publication Table : data/ablations/ablation_summary.md"
echo "=============================================================================="
