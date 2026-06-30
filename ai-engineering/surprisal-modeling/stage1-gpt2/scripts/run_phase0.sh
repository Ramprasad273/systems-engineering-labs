#!/usr/bin/env bash
set -e

echo "============================================================"
echo "Starting Phase 0: Data Quality Fixes & Baseline Extraction"
echo "============================================================"

echo ""
echo "[1/3] Extracting raw validation perplexities (B0c)..."
python evaluate.py --save_val_ppls data/val_perplexities.json

echo ""
echo "[2/3] Running independent depth ablation re-calibration (B0a)..."
python scripts/ablation_depth.py --force

echo ""
echo "[3/3] Running independent vocabulary size ablation re-calibration (B0b)..."
python scripts/ablation_vocab.py --force

echo ""
echo "============================================================"
echo "Phase 0 Completed Successfully!"
echo "============================================================"
