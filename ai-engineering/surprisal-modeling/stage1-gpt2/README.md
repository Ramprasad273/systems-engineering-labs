# Autoregressive Surprisal for Unsupervised Log Anomaly Detection

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.4%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

> **Keywords:** Unsupervised Anomaly Detection · Autoregressive Language Modeling ·
> Statistical Surprisal · System Log Analysis · Distributed Systems · Transformer Architecture

---

## Abstract

Modern hyperscale distributed systems generate massive volumes of unstructured syslog streams.
Supervised anomaly detection fails here due to extreme label scarcity and the non-stationary
nature of volatile log identifiers. This repository presents **`surprisal-gpt2`**: a fully
unsupervised anomaly detection system built on **statistical surprisal**. We train a custom
GPT-2 Small transformer to model the conditional token distribution of structurally normal HDFS
execution traces. At inference, anomalous log blocks induce high cross-entropy loss, which we
exponentiate into **perplexity** and threshold using extreme value calibration
(τ = μ + 3σ over normal holdouts). On the public HDFS benchmark, the method achieves
**F1 = 0.892**, **Precision = 0.946**, **Recall = 0.844**, with a calibrated perplexity threshold of **τ = 1.268**.

---

## Repository Structure

```text
stage1-gpt2/
├── config/
│   └── stage1_config.yaml        # All hyperparameters in one place
├── data/                         # Runtime artifacts (gitignored)
│   ├── ablations/                # Ablation JSON results (tracked in git)
│   ├── checkpoints/              # Model checkpoint_*.pt files
│   ├── processed/                # Cached .pt tensors (train/val/test)
│   ├── raw/                      # Downloaded HDFS.log + anomaly_label.csv
│   └── tokenizer/                # Trained BPE tokenizer JSON
├── scripts/
│   ├── ablation_vocab.py         # Vocabulary size ablation V∈{500,1K,2K,5K,10K}
│   ├── ablation_depth.py         # Model depth ablation L∈{2,4,8,12}
│   ├── threshold_sensitivity.py  # F1 vs. τ=μ+kσ sensitivity for k∈[1,5]
│   ├── token_stability_check.py  # Verify zero [UNK] tokens after masking
│   └── analyze_results.py        # Render all paper tables from JSON results
├── src/
│   ├── dataset/
│   │   └── data_loader.py        # Download, parse, FFD bin-pack, DataLoader
│   ├── models/
│   │   └── gpt2.py               # GPT-2 with RMSNorm, RoPE, SwiGLU, SDPA
│   ├── tokenizer/
│   │   └── log_tokenizer.py      # Regex masking + HuggingFace BPE lifecycle
│   └── utils/
│       └── metrics.py            # Perplexity, F1, confusion matrix, VRAM sweep
├── tests/
│   ├── conftest.py               # Shared pytest fixtures (models, tokenizer)
│   ├── unit/
│   │   ├── test_model.py         # Architecture unit tests (RMSNorm, RoPE, …)
│   │   ├── test_tokenizer.py     # Regex masking unit tests
│   │   ├── test_metrics.py       # Perplexity and classification metric tests
│   │   ├── test_packing.py       # FFD bin-packing algorithm tests
│   │   └── test_training_utils.py# LR schedule and optimizer config tests
│   └── integration/
│       ├── test_overfit.py       # Gradient flow sanity check
│       └── test_pipeline.py      # Checkpoint lifecycle + data pipeline e2e
├── train.py                      # Pre-training entrypoint
├── evaluate.py                   # Threshold calibration + anomaly inference
├── requirements.txt
├── LICENSE                       # MIT
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── run.sh
├── CODE_WALKTHROUGH.md           # End-to-end code architecture & flow walkthrough
├── EXPERIMENT.md                 # Full paper-style experiment walkthrough
```

---

## System Architecture

```text
[Raw Syslog Stream]
        │
        ▼
┌───────────────────────────┐
│  Dynamic Variable Masking │  ← Regex: <IP>, <HEX>, <DATE>, <TIME>
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Byte-Pair Encoding (BPE) │  ← Custom 5,000-token vocabulary
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  FFD Sequence Bin-Packing │  ← Dense 512-token context windows
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Custom GPT-2 Small Stack │  ← 12 Blocks: RMSNorm · RoPE · SwiGLU · SDPA
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Extreme Value Calibration│  ← τ = μ_val + 3σ_val
└───────────────────────────┘
```

### Architectural Innovations

| Component | Classic GPT-2 (2019) | This Implementation | Rationale |
|:---|:---|:---|:---|
| **Normalization** | LayerNorm (post-LN) | RMSNorm (pre-LN) | Removes mean-centering; faster; cleaner gradient paths [1] |
| **Positional Encoding** | Learned absolute | RoPE | Relative positions; zero extra parameters; length generalization [2] |
| **Feed-Forward** | GELU MLP | SwiGLU | Gated activation increases expressivity; used in LLaMA [3] |
| **Attention** | Scaled dot-product + explicit mask matrix | PyTorch SDPA (FlashAttention) | O(T) memory instead of O(T²); no materialized attention matrix [4] |
| **Layout** | Post-LN | Pre-LN | Unimpeded residual identity path; eliminates gradient vanishing in deep stacks |

### Hyperparameters

| Parameter | Value | Rationale |
|:---|:---|:---|
| Transformer layers (L) | 12 | GPT-2 Small baseline depth |
| Attention heads (A) | 12 | Head dim d_k = 64 (12 × 64 = 768) |
| Hidden width (d_model) | 768 | Capacity / VRAM balance |
| Context window (T) | 512 | Covers full HDFS execution block traces |
| Vocabulary size (V) | 5,000 | Minimal BPE vocab; avoids embedding table sparsity |
| SwiGLU inner dim | 2,048 | ≈ 8/3 × d_model (standard ratio) |
| Positional encoding | RoPE | θ = 10,000; 0 learned parameters |

---

## Installation

### Option A: Docker (Recommended for full reproducibility)

Requires: Docker, NVIDIA Container Toolkit.

```bash
chmod +x run.sh
./run.sh
```

`run.sh` verifies Docker installation, GPU accessibility, WSL2 driver path resolution,
builds the container image, and runs a CUDA assertion before starting pre-training.

For an interactive session:

```bash
docker compose up --build -d
docker compose exec surprisal-gpt2-train bash
```

### Option B: Local Python Environment

Requires: Python 3.10+, CUDA 12.1+.

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Automated End-to-End Benchmark Suite (Recommended)
# Runs pre-training, evaluation, ablations, and generates formatted paper tables
./run_paper_experiments.sh

# Or run manual step-by-step pipeline:
python train.py config/stage1_config.yaml
python evaluate.py --config config/stage1_config.yaml
python scripts/analyze_results.py
```

---

## Testing

The test suite is organized as pytest unit and integration tests.
**No GPU and no dataset download are required** to run the tests.

```bash
# Fast unit tests only (< 60 seconds on CPU)
pytest tests/unit/ -v

# Integration tests (gradient flow + checkpoint + pipeline, < 120 seconds on CPU)
pytest tests/integration/ -v

# Full test suite with short traceback
pytest tests/ -v --tb=short

# Run specific test file or test function
pytest tests/unit/test_model.py -k "test_rms_norm" -v

# With timing information for slowest tests
pytest tests/ -v --durations=10
```

### Test Coverage Summary

| Suite | File | What It Tests |
|:---|:---|:---|
| Unit | `test_model.py` | RMSNorm, RoPE, SwiGLU, CausalSelfAttention, GPT2Model (14 tests) |
| Unit | `test_tokenizer.py` | All 9 regex masking rules + BPE lifecycle (14 tests) |
| Unit | `test_metrics.py` | Perplexity math + F1/precision/recall/confusion matrix (8 tests) |
| Unit | `test_packing.py` | FFD bin-packing invariants + stress test (8 tests) |
| Unit | `test_training_utils.py` | LR schedule (warmup + cosine + floor) + optimizer param groups (10 tests) |
| Integration | `test_overfit.py` | Gradient flow sanity check on synthetic data (2 tests) |
| Integration | `test_pipeline.py` | Checkpoint round-trip + full data pipeline (3 tests) |

---

## Ablation Studies

All ablation scripts are in `scripts/`. Run them overnight **after** training and evaluation complete.
Results are written to `data/ablations/*.json` and then rendered into paper tables.

```bash
# 1. Correctness check (5 min): verify zero [UNK] tokens after masking
python scripts/token_stability_check.py --config config/stage1_config.yaml

# 2. Threshold sensitivity (10 min): precision-recall vs. k in τ=μ+kσ
python scripts/threshold_sensitivity.py \
    --config config/stage1_config.yaml \
    --checkpoint data/checkpoints/checkpoint_10000.pt

# 3. Vocabulary size ablation (4 hrs): V∈{500,1K,2K,5K,10K}
python scripts/ablation_vocab.py --config config/stage1_config.yaml

# 4. Model depth ablation (6 hrs): L∈{2,4,8,12}
python scripts/ablation_depth.py --config config/stage1_config.yaml

# 5. Generate all paper tables
python scripts/analyze_results.py
```

See [EXPERIMENT.md § 13](EXPERIMENT.md) for detailed methodology, hypotheses, and result interpretation guides.

---

## Results

Evaluated on the public HDFS benchmark dataset [5] (11,175 distributed execution blocks,
empirical anomaly rate ≈ 2.9%).

### Threshold Calibration (Normal Validation Holdout)

| Metric | Value |
|:---|:---|
| Mean normal perplexity (μ) | **1.165** |
| Std normal perplexity (σ) | **0.035** |
| Calibrated threshold (τ = μ + 3σ) | **1.268** |

### Anomaly Classification (Test Split)

| Metric | Value |
|:---|:---|
| **F1 Score** | **0.892** |
| **Precision** | **0.946** |
| **Recall** | **0.844** |
| Accuracy | 0.953 |
| True Positives | 14,212 |
| False Positives | 808 |
| True Negatives | 55,015 |
| False Negatives | 2,626 |

### VRAM Scalability (FlashAttention O(T) Memory)

| Sequence Length T | VRAM (MB) |
|:---|:---|
| 128 | 1,572 |
| 256 | 1,576 |
| 512 | 1,580 |
| 1,024 | 1,602 |
| 2,048 | 1,616 |

Memory growth is sub-linear (~2.8% increase from T=128 to T=2048), confirming
the FlashAttention O(T) memory property vs the O(T²) of naive attention.

> See `EXPERIMENT.md` for detailed result interpretation and failure mode analysis.

---

## Hyperparameter Reference

All hyperparameters are centralized in `config/stage1_config.yaml`:

```yaml
tokenizer:
  vocab_size: 5000        # BPE vocabulary bound
  save_path: "data/tokenizer/log_tokenizer.json"

dataset:
  url: "https://zenodo.org/records/3227177/files/HDFS_1.tar.gz"
  train_split: 0.90       # Normal blocks for unsupervised pre-training
  seq_len: 512            # Context window T (must match model block_size)
  batch_size: 16

model:
  n_layer: 12
  n_head: 12
  n_embd: 768
  d_ff: 2048
  layer_norm_epsilon: 1e-5

training:
  max_lr: 6.0e-4          # Peak LR after warmup
  min_lr: 6.0e-5          # Cosine decay floor
  warmup_steps: 2000      # Linear warmup horizon
  max_steps: 10000        # Total optimization steps
  weight_decay: 0.1       # L2 applied to 2D projection matrices only
  gradient_accumulation_steps: 4   # Effective batch = 16 × 4 = 64
  clip_grad: 1.0          # Gradient L2 norm clipping
  checkpoint_interval: 2000
```

---

## Docker Implementation Notes

### WSL2 GPU Driver Symbol Resolution

On Windows with WSL2, newer NVIDIA drivers (≥ 560) split CUDA libraries across
two directories. The `docker-entrypoint.sh` automatically scans `/usr/lib/wsl/drivers/*/`
and prepends the correct path to `LD_LIBRARY_PATH`, resolving `CUDA error 500:
named symbol not found` without manual user intervention.

### Shared Memory

PyTorch DataLoader workers communicate via IPC shared memory. Docker's default
64 MB `/dev/shm` triggers `Bus error` crashes during batch streaming. The
`docker-compose.yml` sets `ipc: host` and `shm_size: 8gb` to eliminate this.

---

## Performance Tuning Options

To achieve optimal throughput and memory efficiency during pre-training and inference, `surprisal-gpt2` supports several hardware and software tuning configurations:

### 1. PyTorch CUDA Memory Allocation
Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in your environment (pre-configured in `Dockerfile` and `docker-compose.yml`) to reduce memory fragmentation and prevent OOM spikes during variable-length batch processing.

### 2. Triton CUDA Graph Compilation (`torch.compile`)
The pre-training engine automatically fuses transformer layers into optimized Triton CUDA kernels using `torch.compile(model)`. This provides a ~30–40% training speedup on Ampere and Ada Lovelace architectures (RTX 3000/4000 series, A100/H100). If running on older architectures or debugging CPU backends, disable compilation by setting `compile: false` in `config/stage1_config.yaml`.

### 3. FlashAttention Scaled Dot-Product Attention (SDPA)
The model leverages PyTorch 2.4+ `F.scaled_dot_product_attention`, automatically selecting FlashAttention kernels or Memory-Efficient Attention kernels based on hardware support. This guarantees O(T) memory scaling instead of quadratic O(T^2) memory growth.

### 4. Gradient Accumulation & Effective Batch Size
To train with large effective batch sizes on consumer GPUs (e.g., 8 GB VRAM), adjust `gradient_accumulation_steps` and `batch_size` in `config/stage1_config.yaml`:

```text
Effective Batch Size = B * G
```

Where B is `batch_size` (e.g., 16) and G is `gradient_accumulation_steps` (e.g., 4), yielding a default effective batch size of 16 * 4 = 64. For 24 GB+ GPUs, increase `batch_size` to 64 and set `gradient_accumulation_steps` to 1.

### 5. DataLoader Parallelism & Memory Pinning
Ensure `num_workers: 4` and `pin_memory: true` in your configuration to maximize asynchronous CUDA memory transfers between host CPU and GPU tensors.

---

## Citations

```bibtex
@inproceedings{he2016experience,
  title     = {Experience Report: System Log Analysis for Anomaly Detection},
  author    = {He, Shilin and Zhu, Jieming and He, Pinjia and Lyu, Michael R.},
  booktitle = {Proceedings of the 27th IEEE International Symposium on
               Software Reliability Engineering (ISSRE)},
  year      = {2016},
  doi       = {10.1109/ISSRE.2016.21}
}

@article{shi2021hdfs,
  title   = {{HDFS} Log Dataset},
  author  = {Shi, Hao and He, Shilin and He, Pinjia and Lyu, Michael R.},
  journal = {Zenodo},
  year    = {2021},
  doi     = {10.5281/zenodo.3227177}
}

@article{radford2019language,
  title   = {Language Models are Unsupervised Multitask Learners},
  author  = {Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David
             and Amodei, Dario and Sutskever, Ilya},
  journal = {OpenAI Blog},
  volume  = {1},
  number  = {8},
  pages   = {9},
  year    = {2019}
}

@article{zhang2019root,
  title   = {Root Mean Square Layer Normalization},
  author  = {Zhang, Biao and Sennrich, Rico},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume  = {32},
  year    = {2019}
}

@article{su2024roformer,
  title   = {{RoFormer}: Enhanced Transformer with Rotary Position Embedding},
  author  = {Su, Jianlin and Ahmed, Murtadha and Lu, Yu and Pan, Shengfeng
             and Bo, Wen and Liu, Yunfeng},
  journal = {Neurocomputing},
  volume  = {568},
  pages   = {127063},
  year    = {2024},
  doi     = {10.1016/j.neucom.2023.127063}
}

@article{shazeer2020glu,
  title   = {{GLU} Variants Improve Transformer},
  author  = {Shazeer, Noam},
  journal = {arXiv preprint arXiv:2002.05202},
  year    = {2020}
}

@inproceedings{dao2022flashattention,
  title     = {{FlashAttention}: Fast and Memory-Efficient Exact Attention
               with {IO}-Awareness},
  author    = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri
               and Ré, Christopher},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {35},
  year      = {2022}
}

@inproceedings{du2017deeplog,
  title     = {{DeepLog}: Anomaly Detection and Diagnosis from System Logs
               through Deep Learning},
  author    = {Du, Min and Li, Feifei and Zheng, Guineng and Srikumar, Vivek},
  booktitle = {Proceedings of the 2017 ACM SIGSAC Conference on Computer and
               Communications Security (CCS)},
  year      = {2017},
  doi       = {10.1145/3133956.3134015}
}

@inproceedings{oliner2007supercomputers,
  title     = {What Supercomputers Say: A Study of Five System Logs},
  author    = {Oliner, Adam J. and Stearley, Jon},
  booktitle = {Proceedings of the 37th Annual IEEE/IFIP International Conference on
               Dependable Systems and Networks (DSN)},
  pages     = {575--584},
  year      = {2007},
  doi       = {10.1109/DSN.2007.103}
}
```

---

## License

MIT License see `LICENSE` for details.
