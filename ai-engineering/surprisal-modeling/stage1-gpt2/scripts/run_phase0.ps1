$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting Phase 0: Data Quality Fixes & Baseline Extraction" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Extracting raw validation perplexities (B0c)..." -ForegroundColor Yellow
python evaluate.py --save_val_ppls data/val_perplexities.json

Write-Host "`n[2/3] Running independent depth ablation re-calibration (B0a)..." -ForegroundColor Yellow
python scripts/ablation_depth.py --force

Write-Host "`n[3/3] Running independent vocabulary size ablation re-calibration (B0b)..." -ForegroundColor Yellow
python scripts/ablation_vocab.py --force

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "Phase 0 Completed Successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
