#!/usr/bin/env bash
set -euo pipefail

FORCE=0
if [ "${1:-}" == "--force" ]; then
    FORCE=1
fi

echo "=============================================================================="
echo "          STAGE 2: QLORA ROOT-CAUSE DIAGNOSIS BENCHMARK SUITE                 "
echo "=============================================================================="
echo "Hardware Environment : CUDA GPU Acceleration"
echo "Real-Time Telemetry  : Active (tqdm ETA Spinners)"
echo "Execution CWD        : $(pwd)"
echo "Force Retrain Mode   : ${FORCE}"
echo "=============================================================================="

export PYTHONPATH="."

echo ""
if [ -f "data/sft_dataset/train.jsonl" ] && [ "${FORCE}" -eq 0 ]; then
    echo "=== [1/5] SFT dataset splits detected! Skipping redundant dataset generation ==="
else
    echo "=== [1/5] Preparing HDFS SFT dataset splits (including 50 spot-check samples) ==="
    python scripts/prepare_sft_dataset.py
fi

echo ""
if [ -f "data/checkpoints/adapter_step_500.pt" ] && [ "${FORCE}" -eq 0 ]; then
    echo "=== [2/5] QLoRA adapter checkpoint_step_500.pt detected! Skipping redundant fine-tuning ==="
else
    echo "=== [2/5] Running QLoRA fine-tuning (Qwen-2.5-3B-Instruct in 4-bit NF4) ==="
    python finetune.py --config config/stage2_config.yaml
fi

echo ""
if [ -f "data/stage2_results.json" ] && [ -f "data/stage1_vs_stage2_comparison.json" ] && [ "${FORCE}" -eq 0 ]; then
    echo "=== [3/5] Formal evaluation and Stage 1 vs Stage 2 comparison detected! Skipping ==="
else
    echo "=== [3/5] Executing multi-seed formal evaluation & Stage 1 vs Stage 2 benchmark ==="
    python evaluate.py --config config/stage2_config.yaml --seeds 42 123 999
fi

echo ""
echo "=== [4/5] Empirical Ablation Suite (Paper Tables & Figures) ==="

if [ -f "data/ablations/ablation_dataset_size.json" ] && [ "${FORCE}" -eq 0 ]; then
    echo "  [4a] Dataset size ablation (B1) detected! Skipping."
else
    python scripts/ablation_dataset_size.py
fi

if [ -f "data/ablations/ablation_nf4_vs_fp16.json" ] && [ "${FORCE}" -eq 0 ]; then
    echo "  [4b] Quantization ablation (B2) detected! Skipping."
else
    python scripts/ablation_nf4_vs_fp16.py
fi

if [ -f "data/ablations/ablation_lora_rank.json" ] && [ "${FORCE}" -eq 0 ]; then
    echo "  [4c] LoRA rank ablation (B3) detected! Skipping."
else
    python scripts/ablation_lora_rank.py
fi

if [ -f "data/ablations/benchmark_latency.json" ] && [ "${FORCE}" -eq 0 ]; then
    echo "  [4d] Latency SLO benchmark (B4) detected! Skipping."
else
    python scripts/benchmark_latency.py
fi

echo ""
echo "=== [5/5] Compiling LaTeX Tables & Publication Figures ==="
python scripts/analyze_results.py
python scripts/generate_blog_figures.py

echo ""
echo "=============================================================================="
echo "          [SUCCESS] ALL STAGE 2 RESEARCH EXPERIMENTS COMPLETED                "
echo "=============================================================================="
echo "Generated Artifacts:"
echo "  1. LoRA Adapter Checkpoint : data/checkpoints/adapter_step_500.pt"
echo "  2. Formal Evaluation Report: data/stage2_results.json"
echo "  3. Stage 1 vs 2 Comparison : data/stage1_vs_stage2_comparison.json"
echo "  4. Publication Figures     : data/figures/*.png"
echo "=============================================================================="
