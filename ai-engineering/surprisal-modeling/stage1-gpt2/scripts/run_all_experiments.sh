#!/usr/bin/env bash
set -e

echo "=========================================================================="
echo "AI RESEARCH AUTONOMOUS BENCHMARK SUITE (PHASES 0 - 4)"
echo "Smart Checkpointing & Thermal Cooldowns Enabled"
echo "=========================================================================="

echo ""
echo "--- [Phase 0] Data Quality Fixes & Baseline Extraction ---"
echo "[0/1] Extracting validation perplexities (B0c)..."
python evaluate.py --save_val_ppls data/val_perplexities.json
sleep 2

echo "[0/2] Running depth ablation re-calibration (B0a)..."
python scripts/ablation_depth.py
sleep 3

echo "[0/3] Running vocab ablation re-calibration (B0b)..."
python scripts/ablation_vocab.py
sleep 3

echo ""
echo "--- [Phase 1] Fast Analytics & Threshold Comparisons ---"
echo "[1/1] Evaluating multi-seed statistical significance (B1)..."
python scripts/run_b1_multiseed.py
sleep 2

echo "[1/2] Comparing EVT Gumbel vs Percentile thresholds (B3)..."
python scripts/run_b3_thresholds.py
sleep 2

echo "[1/3] Generating token surprisal error heatmaps (B6)..."
python scripts/run_b6_heatmaps.py
sleep 2

echo ""
echo "--- [Phase 2] Architectural & Packing Ablations ---"
echo "[2/1] Evaluating sequence packing vs truncation (B2)..."
python scripts/run_b2_packing.py
sleep 2

echo "[2/2] Evaluating positional embeddings RoPE vs Absolute (B4a)..."
python scripts/ablation_pos.py
sleep 3

echo "[2/3] Evaluating activation functions SwiGLU vs GELU (B4b)..."
python scripts/ablation_act.py
sleep 3

echo ""
echo "--- [Phase 3] Out-of-Distribution Generalization ---"
echo "[3/1] Evaluating cross-dataset BGL transferability (B5)..."
python scripts/run_b5_bgl.py
sleep 2

echo ""
echo "--- [Phase 4] Blog Figures & Visualization ---"
echo "[4/1] Generating empirical publication figures from experiment data..."
python scripts/generate_blog_figures.py

echo ""
echo "=========================================================================="
echo "BENCHMARK SUITE COMPLETED SUCCESSFULLY!"
echo "Figures saved to: data/figures/"
echo "All JSON artifacts in: data/ and data/ablations/"
echo "=========================================================================="
