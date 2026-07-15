# AI Engineering Labs

This directory contains production machine learning architectures, sequence modeling pipelines, and statistical evaluation frameworks.

---

## Projects Overview

### Surprisal Modeling (`surprisal-modeling/`)
An advanced unsupervised anomaly detection pipeline for hyperscale distributed log streams based on statistical surprisal and cross-entropy evaluation.

- **`stage1-gpt2/`**: Unsupervised baseline anomaly detection utilizing custom GPT-2 Small transformer backbones with extreme value calibration (`τ = μ + 3σ`).
- **`stage2-qlora/`**: 4-bit QLoRA fine-tuning and inference engine using Qwen-2.5-3B for structured anomaly diagnosis.
- **`stage3-mamba/`**: Sub-quadratic state space sequence models (`Mamba S6`) and hybrid `MambaLog` architectures for real-time log ingestion.
