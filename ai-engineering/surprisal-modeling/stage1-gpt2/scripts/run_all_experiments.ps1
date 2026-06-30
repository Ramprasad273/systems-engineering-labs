$ErrorActionPreference = "Stop"

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "AI RESEARCH AUTONOMOUS BENCHMARK SUITE (PHASES 0 - 4)" -ForegroundColor Cyan
Write-Host "Smart Checkpointing & Thermal Cooldowns Enabled" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

Write-Host "`n--- [Phase 0] Data Quality Fixes & Baseline Extraction ---" -ForegroundColor Yellow
Write-Host "[0/1] Extracting validation perplexities (B0c)..."
python evaluate.py --save_val_ppls data/val_perplexities.json
Start-Sleep -Seconds 2

Write-Host "[0/2] Running depth ablation re-calibration (B0a)..."
python scripts/ablation_depth.py
Start-Sleep -Seconds 3

Write-Host "[0/3] Running vocab ablation re-calibration (B0b)..."
python scripts/ablation_vocab.py
Start-Sleep -Seconds 3

Write-Host "`n--- [Phase 1] Fast Analytics & Threshold Comparisons ---" -ForegroundColor Yellow
Write-Host "[1/1] Evaluating multi-seed statistical significance (B1)..."
python scripts/run_b1_multiseed.py
Start-Sleep -Seconds 2

Write-Host "[1/2] Comparing EVT Gumbel vs Percentile thresholds (B3)..."
python scripts/run_b3_thresholds.py
Start-Sleep -Seconds 2

Write-Host "[1/3] Generating token surprisal error heatmaps (B6)..."
python scripts/run_b6_heatmaps.py
Start-Sleep -Seconds 2

Write-Host "`n--- [Phase 2] Architectural & Packing Ablations ---" -ForegroundColor Yellow
Write-Host "[2/1] Evaluating sequence packing vs truncation (B2)..."
python scripts/run_b2_packing.py
Start-Sleep -Seconds 2

Write-Host "[2/2] Evaluating positional embeddings RoPE vs Absolute (B4a)..."
python scripts/ablation_pos.py
Start-Sleep -Seconds 3

Write-Host "[2/3] Evaluating activation functions SwiGLU vs GELU (B4b)..."
python scripts/ablation_act.py
Start-Sleep -Seconds 3

Write-Host "`n--- [Phase 3] Out-of-Distribution Generalization ---" -ForegroundColor Yellow
Write-Host "[3/1] Evaluating cross-dataset BGL transferability (B5)..."
python scripts/run_b5_bgl.py
Start-Sleep -Seconds 2

Write-Host "`n--- [Phase 4] Blog Figures & Visualization ---" -ForegroundColor Yellow
Write-Host "[4/1] Generating empirical publication figures from experiment data..."
python scripts/generate_blog_figures.py
Start-Sleep -Seconds 1

Write-Host "`n==========================================================================" -ForegroundColor Green
Write-Host "BENCHMARK SUITE COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "Figures saved to: data/figures/" -ForegroundColor Green
Write-Host "All JSON artifacts in: data/ and data/ablations/." -ForegroundColor Green
