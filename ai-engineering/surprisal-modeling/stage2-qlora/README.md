# Stage 2: QLoRA Root-Cause Diagnosis Engine & Stage 1 vs Stage 2 Comparative Suite

This repository implements **Stage 2** of the Distributed System Log Analysis & Surprisal Modeling architecture. Building directly upon Stage 1 (GPT-2 Unsupervised Sequence Surprisal), Stage 2 fine-tunes **Qwen-2.5-3B-Instruct** using **4-bit NF4 Quantized Low-Rank Adaptation (QLoRA)** to transform raw distributed system logs into structured JSON diagnostic reports.

---

## Stage 1 vs Stage 2 Comparative Analysis

While Stage 1 provides rapid, unsupervised sequence surprisal scoring for anomaly detection, it outputs only scalar anomaly flags without actionable explanation. Stage 2 bridges this gap by delivering full structured root-cause explanations while simultaneously boosting binary detection F1.

| Architectural Dimension | Stage 1: GPT-2 Surprisal | Stage 2: Qwen-2.5-3B QLoRA |
| :--- | :--- | :--- |
| **Primary Task** | Unsupervised Token Surprisal Scoring | Supervised Instruction Fine-Tuning (SFT) |
| **Output Format** | Scalar Log-Probability / Surprisal Flag | Structured JSON Diagnosis Schema |
| **Binary Anomaly F1 Score** | `89.23%` | **`90.85%` (+1.62%)** |
| **Binary Anomaly Accuracy** | `95.28%` | **`96.10%` (+0.82%)** |
| **Root-Cause Explanation** | None | Structured Diagnostic Summary |
| **Severity Classification** | None | Multi-Class Macro F1: `0.8890` |
| **Automated Mitigation** | None | Concrete CLI / SRE Action Commands |
| **Peak VRAM Footprint** | `1,616 MB` (124M FP16) | `5,120 MB` (3.09B NF4 + LoRA) |
| **Inference SLO (p95)** | `< 50 ms` (Speed Layer) | `1,210 ms` (Batch/Serving Layer SLO < 2s) |

---

## First-Principles Architecture

To implement parameter-efficient fine-tuning without hidden abstractions, this codebase implements LoRA mathematics directly from first principles (`src/models/lora.py`):

```text
Delta_W = (alpha / r) * (B * A)
```

- **Zero-Initialization Invariant**: Matrix A (`r x d_in`) is initialized via Kaiming uniform distribution, while matrix B (`d_out x r`) is strictly zero-initialized (`B = 0`). This guarantees `Delta_W = 0` at training step `0`, ensuring the fine-tuned adapter begins as an exact identity transformation of the pre-trained weights.
- **Prompt Loss Masking**: During training (`src/dataset/sft_dataset.py`), all system instructions and user log input tokens are masked with `target = -100`. Gradients flow exclusively from generating the structured JSON completion tokens.

---

## Quick Start & Verification

### 1. Verification Suite (Offline & Unit Tests)
Run unit tests across LoRA math, dataset masking, and evaluation telemetry:
```bash
pytest -v
```

### 2. Run Complete Paper Suite (Idempotent)
Execute SFT dataset generation, fine-tuning, multi-seed evaluation, ablations (B1-B4), and figure generation:
```bash
bash run_paper_experiments.sh
```

### 3. Interactive Single-Sample Diagnosis Demo
```bash
python inference.py --mock
```

---

## Repository Structure

```text
stage2-qlora/
├── config/stage2_config.yaml         # Centralized hyperparameters & paths
├── scripts/
│   ├── prepare_sft_dataset.py        # ChatML structured dataset generator
│   ├── ablation_dataset_size.py      # B1: Dataset scaling sensitivity
│   ├── ablation_nf4_vs_fp16.py       # B2: Memory vs Accuracy tradeoff
│   ├── ablation_lora_rank.py         # B3: LoRA rank sensitivity
│   ├── benchmark_latency.py          # B4: Diagnostic latency SLO profile
│   ├── analyze_results.py            # LaTeX table report compiler
│   └── generate_blog_figures.py      # Matplotlib publication figure generator
├── src/
│   ├── models/lora.py                # LoRA math implementation (`Delta_W = BA`)
│   ├── dataset/
│   │   ├── sft_dataset.py            # ChatML dataset with target=-100 masking
│   │   └── data_loader.py            # DataLoader factory
│   └── utils/
│       ├── metrics.py                # JSON validation & Stage 1 vs 2 benchmark
│       └── vram_profiler.py          # Peak GPU VRAM tracking
├── tests/                            # Unit tests suite
├── finetune.py                       # QLoRA training engine
├── evaluate.py                       # Formal evaluation harness
├── inference.py                      # Interactive diagnosis demo
├── run_paper_experiments.sh          # Complete end-to-end suite runner
└── Dockerfile & docker-compose.yml   # Containerized reproducibility
```
