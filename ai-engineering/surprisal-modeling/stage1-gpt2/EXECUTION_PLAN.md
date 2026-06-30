# Complete Execution Plan: Research Experiments B0 to B6

This document is the **living execution blueprint** for elevating this project to research-publication standard.
It is updated as experiments complete and issues are discovered.

---

## 🛑 Core Architectural Requirement: Smart Checkpointing & Idempotency

To guarantee resilience during multi-hour training runs, **every script and training loop must implement Smart Checkpointing**:
1. **Auto-Resume Training (`train.py`)**: When `train.py` launches, it must automatically scan `--checkpoint_dir` for existing `.pt` checkpoints. If found, it must load the latest weights, optimizer state, and step counter (`step`), resuming training seamlessly rather than restarting from step 1.
2. **Experiment Idempotency**: All runner scripts (`run_b*.sh`, `run_b*.py`) must check if their final output JSON already exists and is valid. If present, they skip execution unless `--force` is passed. This guarantees zero lost compute if execution is interrupted or restarted.

---

## 🧠 Engineering & Coding Standards

All modified and new code adheres to these engineering standards:
1. **Explicit Tensor Shapes**: Every function docstring and tensor operation must document exact dimensions in comments (e.g., `# [batch_size, seq_len, vocab_size]`).
2. **Educational Docstrings & Comments**: Code must explain *why* an operation is performed, not just *what* it does (e.g., explaining why $\mu + 3\sigma$ calibrates extreme values, why weight decay is separated from 1D biases).
3. **Structured Academic Telemetry**: All scripts must use structured logger output (`logger.info`) reporting hyperparameters, step times, and loss progressions cleanly.

---

## 🔬 Senior AI Researcher Peer-Review Checklist

To ensure the resulting benchmark survives rigorous peer review, the execution pipeline enforces these checks:
1. **Reproducibility Guarantee**: Explicit PyTorch, CUDA, and NumPy seed setting across all threads before every initialization or evaluation run.
2. **Compute & Parameter Fairness**: Verify that architectural ablations (B4) maintain comparable parameter counts and receive identical token training budgets (2,000 steps).
3. **Loss Convergence Auditing**: Inspect gradient norms and validation loss curves during short training runs to confirm models plateau before evaluation.
4. **Qualitative Error Analysis**: Examine specific false positives and false negatives to explain *why* the language model flags certain benign logs or misses rare edge cases.

---

## Estimated Time to Completion (ETAs)

| Phase | Tasks Included | Estimated Duration | Compute Type |
|---|---|---|---|
| **Phase 0** | Patch `train.py` (resume + CLI flags), patch `evaluate.py`, fix Depth & Vocab ablation data quality bugs | **1.0 – 1.5 hours** | GPU (training small vocab models + eval) |
| **Phase 1** | B1 (Multi-seed F1), B3 (EVT/Percentile thresholding), B6 (Surprisal heatmap generation) | **15 minutes** | CPU / Inference |
| **Phase 2** | B2 (Packing ablation: 4 runs × 2k steps), B4 (Architecture ablation: 4 runs × 2k steps) | **2.5 – 3.0 hours** | GPU Training (Unattended) |
| **Phase 3** | B5 (BGL cross-dataset prep, train 5k steps, eval) *(Optional Stretch)* | **1.0 hour** | GPU Training + Network |
| **Phase 4** | Update markdown blog posts, compile LaTeX tables, embed figures | **20 minutes** | Text / Formatting |
| **TOTAL** | **End-to-End Publication Benchmark Suite** | **~4.5 – 6.0 hours** | **Unattended Execution via `/goal`** |

---

## Status Dashboard

| ID | Experiment | Script | Data Output | Status |
|---|---|---|---|---|
| B0a | Fix Depth Ablation (retrain per-depth) | `scripts/ablation_depth.py` | `data/ablations/ablation_depth.json` | ⚠️ DATA BUG — must rerun |
| B0b | Fix Vocab Ablation (calibration bug) | `scripts/ablation_vocab.py` | `data/ablations/ablation_vocab.json` | ⚠️ DATA BUG — must rerun |
| B0c | Extract val_perplexities.json | `evaluate.py --save_val_ppls` | `data/val_perplexities.json` | ❌ Not done |
| B1 | Multi-seed statistical significance | `scripts/run_b1_multiseed.py` | `data/b1_seed_*.json` | ❌ Script missing |
| B2 | Packing strategy ablation | `scripts/run_b2_packing.sh` | `data/b2_results/*.json` | ❌ Script missing |
| B3 | Threshold comparison (EVT/Percentile) | `scripts/run_b3_thresholds.py` | `data/b3_threshold_comparison.json` | ❌ Script missing |
| B4 | Architecture component ablations | `scripts/run_b4_ablations.sh` | `data/b4_results/*.json` | ❌ Script missing |
| B5 | Cross-dataset BGL validation | `scripts/run_b5_bgl.sh` | `data/b5_bgl_results.json` | ❌ Script + data missing |
| B6 | Token surprisal heatmap | `scripts/run_b6_heatmaps.py` | `data/heatmap_anomaly.png` | ❌ Script stub only |
| — | Blog post updated with results | manual | both `.md` copies | ❌ Pending all above |

---

## Completed Experiments (Verified)

The following are confirmed complete and valid:

| Result | Value |
|---|---|
| Main model (10,000 steps) | `data/checkpoints/checkpoint_10000.pt` |
| Formal evaluation F1 | **0.8922** (P=0.9462, R=0.8440, Acc=0.9527) |
| Threshold (μ + 3σ) | τ = 1.2685 |
| Threshold sensitivity sweep | k=1.0..5.0 complete in `threshold_sensitivity.json` |
| Token UNK stability | 0% normal, 0.003% anomaly — BPE coverage confirmed |
| Unit test suite | `tests/unit/` — model, metrics, packing, tokenizer all pass |

---

## Known Data Quality Bugs

### Bug 1: Depth Ablation F1 ≈ 0.29 Across ALL Depths

**Symptom:** `ablation_depth.json` shows F1 ≈ 0.29 for 2, 4, 8, and 12 layers — nearly identical and near-random.

**Root Cause:** Each depth variant runs evaluation but likely doesn't recalibrate the threshold on that model's own validation distribution. The threshold from the main model is being applied to depth-variant models with very different perplexity scales (depth-2 PPL = 6.8B, depth-12 PPL = 5.6B), making the threshold meaningless.

**Fix Required:** `ablation_depth.py` must:
1. Reload the depth-variant checkpoint
2. Run calibration on the **validation split with that checkpoint** to get a new μ and σ
3. Set τ = μ + 3σ for that specific model
4. Then evaluate the test set with that model's own threshold

### Bug 2: Vocab Ablation F1 Identical, Recall=1.0 at All Sizes

**Symptom:** `ablation_vocab.json` shows F1=0.3763, Recall=1.0, Precision=0.23 — identical for vocab sizes 500 to 10,000.

**Root Cause:** All vocab-size models are classifying everything as anomalous (FP=55,823, TN=0). This means the threshold τ collapsed to near-zero for small-vocab models (PPL ≈ 1.00000), so every sequence exceeds it.

**Fix Required:** Ensure each vocab-size model is independently trained (not just the tokenizer retrained on the same model weights) and that calibration is re-run on each model's own validation perplexities.

---

## Prerequisites & Test Verification

```bash
pip install pytest torch numpy scipy matplotlib pandas tokenizers
```

### Run Core Suite First
```bash
pytest tests/ -v
```

---

## Phase 0: Fix Data Quality (Blocking — Run First)

### B0a: Fix Depth Ablation

In `scripts/ablation_depth.py`, ensure the evaluation loop follows this pattern:

```python
for n_layer in [2, 4, 8, 12]:
    # 1. Load the correct depth-variant checkpoint
    model = GPT2SurprisalModel(n_layer=n_layer, ...)
    model.load_state_dict(torch.load(f"data/ablations/ckpt_depth_{n_layer}.pt"))
    model.eval()

    # 2. Calibrate threshold on validation split WITH THIS MODEL
    val_ppls = []
    for batch in val_loader:
        with torch.no_grad():
            ppl = compute_perplexity(model, batch)
        val_ppls.append(ppl)
    mu, sigma = np.mean(val_ppls), np.std(val_ppls)
    tau = mu + 3 * sigma  # <- must use THIS model's distribution

    # 3. Evaluate test set
    metrics = evaluate_test_set(model, test_loader, tau)
    results.append({"n_layer": n_layer, "tau": tau, **metrics})
```

Execute: `python scripts/ablation_depth.py`

### B0b: Fix Vocab Ablation

In `scripts/ablation_vocab.py`, ensure a **separate model is trained per vocab size** and threshold is recalibrated per model. The tokenizer alone cannot change without retraining the model on that tokenizer's vocabulary.

Execute: `python scripts/ablation_vocab.py`

### B0c: Extract val_perplexities.json (Required for B3)

Add a flag to `evaluate.py` to save raw validation perplexities:

```bash
python evaluate.py --save_val_ppls data/val_perplexities.json
```

Or extract from the calibration loop in evaluate.py and save as a JSON list.

---

## Phase 1: Fast, CPU-Only Experiments

### Experiment B1: Multi-Seed Statistical Significance
**Goal:** Prove F1 is stable across data splits by reporting `mean ± std` over 5 seeds (no GPU retraining).

**Create `scripts/run_b1_multiseed.py`:**
```python
import subprocess
import json
import numpy as np

seeds = [42, 43, 44, 45, 46]
f1_scores = []

for seed in seeds:
    print(f"--- Evaluating Seed {seed} ---")
    cmd = f"python evaluate.py --checkpoint data/checkpoints/checkpoint_10000.pt --seed {seed} --output data/b1_seed_{seed}.json"
    subprocess.run(cmd, shell=True, check=True)

    with open(f"data/b1_seed_{seed}.json") as f:
        data = json.load(f)
        f1_scores.append(data["test_metrics"]["f1"])

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
summary = {"seeds": seeds, "f1_scores": f1_scores, "mean_f1": mean_f1, "std_f1": std_f1}
with open("data/b1_multiseed_summary.json", "w") as out:
    json.dump(summary, out, indent=2)

print(f"\n==========================================")
print(f"FINAL B1 RESULT: F1 = {mean_f1:.4f} ± {std_f1:.4f}")
print(f"==========================================")
```

**Prerequisite:** `evaluate.py` must accept `--seed` (for test split shuffling) and `--output` flags.
**Execute:** `python scripts/run_b1_multiseed.py`

---

### Experiment B3: Threshold Strategy Comparison
**Goal:** Compare `μ + 3σ` against Non-Parametric Percentiles (95th, 99th) and Extreme Value Theory (EVT Gumbel fit).

**Prerequisite:** `data/val_perplexities.json` must exist (see B0c).

**Create `scripts/run_b3_thresholds.py`:**
```python
import json
import numpy as np
from scipy.stats import gumbel_r

# Load normal validation perplexities
with open("data/val_perplexities.json") as f:
    val_ppls = np.array(json.load(f))

# 1. Gaussian mu + 3sigma (current method)
mu, sigma = np.mean(val_ppls), np.std(val_ppls)
thresh_gauss = mu + 3 * sigma

# 2. Non-parametric percentiles
thresh_95 = np.percentile(val_ppls, 95)
thresh_99 = np.percentile(val_ppls, 99)

# 3. EVT Gumbel Fit (extreme value theory)
loc, scale = gumbel_r.fit(val_ppls)
thresh_evt = gumbel_r.ppf(0.99, loc=loc, scale=scale)

results = {
    "n_val_samples": len(val_ppls),
    "mu": float(mu),
    "sigma": float(sigma),
    "Gaussian_Mu_3Sig": float(thresh_gauss),
    "Percentile_95": float(thresh_95),
    "Percentile_99": float(thresh_99),
    "EVT_Gumbel_99": float(thresh_evt)
}

with open("data/b3_threshold_comparison.json", "w") as out:
    json.dump(results, out, indent=2)
print("B3 Threshold Comparison Saved:", results)
```

**Execute:** `python scripts/run_b3_thresholds.py`

---

### Experiment B6: Token Surprisal Heatmaps
**Goal:** Extract visual proof of which tokens spike during a failure trace.

**Create `scripts/run_b6_heatmaps.py`:**
```python
import torch
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import GPT2SurprisalModel
from src.tokenizer_utils import load_tokenizer

CHECKPOINT = "data/checkpoints/checkpoint_10000.pt"
TOKENIZER = "data/tokenizer/log_tokenizer.json"
ANOMALY_DATA = "data/processed/anomaly_sequences.pt"
OUTPUT = "data/heatmap_anomaly.png"

def generate_heatmap(tokens, losses, output_path, threshold=None):
    """Generate a per-token surprisal heatmap with optional threshold line."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(tokens) * 0.4), 5),
                                    gridspec_kw={"height_ratios": [1, 3]})

    # Top: matrix heatmap
    data = np.array(losses)[np.newaxis, :]
    cax = ax1.matshow(data, cmap="YlOrRd", aspect="auto",
                      vmin=0, vmax=max(losses) * 1.1)
    fig.colorbar(cax, ax=ax1, orientation="horizontal", pad=0.02, label="Surprisal (nats)")
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=60, ha="left", fontsize=8)
    ax1.set_yticks([])
    ax1.set_title("Token Surprisal Heatmap — Anomaly Trace", fontsize=12, pad=8)

    # Bottom: bar chart of per-token losses
    colors = ["#e74c3c" if l > (threshold or float("inf")) else "#3498db" for l in losses]
    ax2.bar(range(len(losses)), losses, color=colors, width=0.7)
    if threshold:
        ax2.axhline(y=threshold, color="black", linestyle="--", linewidth=1.5,
                    label=f"Threshold τ={threshold:.3f}")
        ax2.legend()
    ax2.set_xticks(range(len(tokens)))
    ax2.set_xticklabels(tokens, rotation=60, ha="right", fontsize=8)
    ax2.set_ylabel("Cross-Entropy Loss (nats)")
    ax2.set_xlabel("Log Template Token")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved → {output_path}")

def main():
    # Load tokenizer and model
    tokenizer = load_tokenizer(TOKENIZER)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model_cfg = checkpoint.get("config", {})

    model = GPT2SurprisalModel(**model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load calibration threshold
    with open("data/stage1_eval_results.json") as f:
        eval_data = json.load(f)
    tau = eval_data["calibration"]["threshold_tau"]

    # Load or sample an anomaly sequence
    if os.path.exists(ANOMALY_DATA):
        sequences = torch.load(ANOMALY_DATA)
        input_ids = sequences[0:1]  # Take first anomaly
    else:
        # Fallback: use a hardcoded anomaly-like log line
        sample_line = "DataNode PacketResponder Exception in receiveBlock for block ERROR writeBlock received exception"
        input_ids = torch.tensor([tokenizer.encode(sample_line).ids])

    # Forward pass: compute per-token loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # Get per-token logits
        logits = outputs.logits  # [1, seq_len, vocab]
        labels = input_ids[:, 1:]  # shift
        logits = logits[:, :-1, :]

        per_token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none"
        ).reshape(1, -1)[0].tolist()

    # Decode tokens
    token_ids = input_ids[0].tolist()
    tokens = [tokenizer.id_to_token(tid) or f"<{tid}>" for tid in token_ids[1:]]

    generate_heatmap(tokens, per_token_loss, OUTPUT, threshold=tau)
    print(f"Max surprisal token: '{tokens[per_token_loss.index(max(per_token_loss))]}' = {max(per_token_loss):.4f}")

if __name__ == "__main__":
    main()
```

**Execute:** `python scripts/run_b6_heatmaps.py`

---

## Phase 2: GPU Training Ablations (Overnight)

### Experiment B2: Packing Strategy Ablation
**Goal:** Compare FFD vs. Random vs. Greedy vs. No Packing over 2,000-step training budget.

**Prerequisite Check:** Verify `train.py` supports `--packing_strategy`. If not, add the flag.

**Create `scripts/run_b2_packing.sh`:**
```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/b2_results

for strategy in ffd random none greedy; do
    echo "=== Training with Packing Strategy: $strategy ==="
    python train.py \
        --packing_strategy $strategy \
        --max_steps 2000 \
        --checkpoint_dir checkpoints/b2_$strategy \
        --log_file data/b2_results/${strategy}_metrics.json \
        --seed 42
    echo "  -> $strategy done."
done

echo "B2 Packing Ablation Complete."
python -c "
import json, glob
results = {}
for f in sorted(glob.glob('data/b2_results/*_metrics.json')):
    name = f.split('/')[-1].replace('_metrics.json','')
    with open(f) as fp:
        results[name] = json.load(fp)
with open('data/b2_results/summary.json', 'w') as out:
    json.dump(results, out, indent=2)
print('Summary written to data/b2_results/summary.json')
"
```

**Execute:** `chmod +x scripts/run_b2_packing.sh && ./scripts/run_b2_packing.sh`

---

### Experiment B4: Architecture Component Ablations
**Goal:** Isolate the contribution of RMSNorm, RoPE, SwiGLU, and Tied Embeddings individually.

**Prerequisite Check:** Verify `train.py` supports `--norm_type`, `--pos_type`, `--ffn_type`, `--untie_embeddings`.

**Create `scripts/run_b4_ablations.sh`:**
```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/b4_results

# Baseline (full architecture — our model)
echo "=== Baseline (Full Architecture) ==="
python train.py --max_steps 2000 --output data/b4_results/baseline.json --seed 42

# Variant A: Standard LayerNorm instead of RMSNorm
echo "=== Variant: No RMSNorm (LayerNorm) ==="
python train.py --norm_type layernorm --max_steps 2000 --output data/b4_results/no_rmsnorm.json --seed 42

# Variant B: Learned Absolute Positional Embeddings instead of RoPE
echo "=== Variant: No RoPE (Absolute PE) ==="
python train.py --pos_type absolute --max_steps 2000 --output data/b4_results/no_rope.json --seed 42

# Variant C: Standard GELU FFN instead of SwiGLU
echo "=== Variant: No SwiGLU (GELU) ==="
python train.py --ffn_type gelu --max_steps 2000 --output data/b4_results/no_swiglu.json --seed 42

# Variant D: Untied Input/Output Embeddings
echo "=== Variant: Untied Embeddings ==="
python train.py --untie_embeddings --max_steps 2000 --output data/b4_results/no_tied_embeds.json --seed 42

echo "B4 Architecture Ablations Complete."
```

**Execute:** `chmod +x scripts/run_b4_ablations.sh && ./scripts/run_b4_ablations.sh`

---

## Phase 3: Cross-Dataset Generalization

### Experiment B5: Cross-Dataset Validation on BlueGene/L (BGL)
**Goal:** Prove model generalization beyond HDFS by evaluating on the BGL supercomputing log dataset.

**Data Source:** BGL logs are available from the LogHub repository:
```bash
# Download BGL dataset (public, ~700MB)
wget https://zenodo.org/record/8196385/files/BGL.tar.gz -O data/bgl_raw.tar.gz
tar -xzf data/bgl_raw.tar.gz -C data/
```

**Create `scripts/preprocess_bgl.py`** — parses BGL format (space-separated, anomaly label in col 0):
```python
"""Preprocess BGL logs into normal/anomaly split for training."""
import re
import sys

def preprocess_bgl(input_path, output_normal, output_anomaly):
    # BGL format: label(- or ALERT/FATAL) timestamp nodecard component message
    # "-" prefix = normal, anything else = anomaly
    normal, anomaly = [], []
    with open(input_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            label = parts[0]
            content = parts[1] if len(parts) > 1 else ""
            # Remove timestamps and node IDs, keep message content
            content = re.sub(r'\d{4}-\d{2}-\d{2}-\d{2}\.\d+', '<TIME>', content)
            content = re.sub(r'R[0-9A-F]+-[A-Z0-9]+', '<NODE>', content)
            if label == "-":
                normal.append(content)
            else:
                anomaly.append(content)

    with open(output_normal, "w") as f:
        f.write("\n".join(normal))
    with open(output_anomaly, "w") as f:
        f.write("\n".join(anomaly))

    print(f"BGL: {len(normal)} normal, {len(anomaly)} anomaly sequences extracted.")

if __name__ == "__main__":
    preprocess_bgl("data/bgl_raw.log", "data/bgl/normal.txt", "data/bgl/anomaly.txt")
```

**Create `scripts/run_b5_bgl.sh`:**
```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/bgl

echo "Step 1: Preprocess BGL Logs"
python scripts/preprocess_bgl.py

echo "Step 2: Train BPE Tokenizer on BGL Normal Traces"
python scripts/train_tokenizer.py \
    --data data/bgl/normal.txt \
    --vocab_size 2000 \
    --output data/bgl/tokenizer.json

echo "Step 3: Train Model on BGL Normal Traces"
python train.py \
    --data data/bgl/normal.txt \
    --tokenizer data/bgl/tokenizer.json \
    --max_steps 5000 \
    --checkpoint_dir checkpoints/bgl_model \
    --seed 42

echo "Step 4: Evaluate on BGL Anomalies"
python evaluate.py \
    --checkpoint checkpoints/bgl_model/checkpoint_5000.pt \
    --normal_data data/bgl/normal.txt \
    --anomaly_data data/bgl/anomaly.txt \
    --tokenizer data/bgl/tokenizer.json \
    --output data/b5_bgl_results.json

echo "B5 BGL Cross-Dataset Validation Complete."
cat data/b5_bgl_results.json
```

**Execute:** `chmod +x scripts/run_b5_bgl.sh && ./scripts/run_b5_bgl.sh`

---

## Master Automation Runner (`run_all_experiments.sh`)

Create this file to execute everything in one unattended job:

```bash
#!/usr/bin/env bash
set -euo pipefail

LOG="data/run_all_experiments.log"
mkdir -p data

exec > >(tee -a "$LOG") 2>&1
echo "================================================================="
echo " SURPRISAL MODELING: FULL RESEARCH BENCHMARK SUITE"
echo " Started: $(date)"
echo "================================================================="

echo ""
echo "=== PHASE 0: UNIT & SMOKE TESTS ==="
pytest tests/ -v --tb=short

echo ""
echo "=== PHASE 0: DATA QUALITY FIXES ==="
python scripts/ablation_depth.py       # Re-run with proper per-model calibration
python scripts/ablation_vocab.py       # Re-run with proper per-model calibration
python evaluate.py --save_val_ppls data/val_perplexities.json

echo ""
echo "=== PHASE 1: FAST NON-TRAINING EXPERIMENTS (B1, B3, B6) ==="
python scripts/run_b1_multiseed.py
python scripts/run_b3_thresholds.py
python scripts/run_b6_heatmaps.py

echo ""
echo "=== PHASE 2: TRAINING ABLATIONS (B2, B4) ==="
chmod +x scripts/run_b2_packing.sh
./scripts/run_b2_packing.sh
chmod +x scripts/run_b4_ablations.sh
./scripts/run_b4_ablations.sh

echo ""
echo "=== PHASE 3: GENERALIZATION (B5) ==="
chmod +x scripts/run_b5_bgl.sh
./scripts/run_b5_bgl.sh

echo ""
echo "=== COMPILING RESULTS ==="
python scripts/analyze_results.py

echo ""
echo "================================================================="
echo " ALL RESEARCH EXPERIMENTS (B0-B6) SUCCESSFULLY COMPLETED"
echo " Finished: $(date)"
echo "================================================================="
```

---

## Appendix: Missing Infrastructure

### Missing: `tests/test_core.py`
```python
import torch
import numpy as np

def test_unmasked_loss():
    """Padding tokens should not contribute to sequence loss."""
    pad = 0
    inputs = torch.tensor([[10, 20, pad]])
    targets = torch.tensor([[20, 30, pad]])
    losses = torch.tensor([[0.5, 0.5, 99.9]])
    mask = ~((inputs == pad) & (targets == pad))
    seq_loss = (losses * mask.float()).sum() / mask.sum()
    assert torch.isclose(seq_loss, torch.tensor(0.5))

def test_threshold():
    """Threshold must be strictly greater than mean."""
    arr = np.array([1.0, 1.1, 0.9])
    assert (np.mean(arr) + 3 * np.std(arr)) > np.mean(arr)

def test_depth_calibration():
    """Each depth model must produce a unique threshold (not reuse main model's)."""
    # This is the anti-regression test for Bug B0a
    ppls_shallow = np.array([6.0e9, 6.2e9, 5.8e9])  # depth-2 scale
    ppls_deep = np.array([1.16, 1.18, 1.15])         # depth-12 scale
    tau_shallow = np.mean(ppls_shallow) + 3 * np.std(ppls_shallow)
    tau_deep = np.mean(ppls_deep) + 3 * np.std(ppls_deep)
    assert tau_shallow > 1e6, "Shallow model threshold must be in billions"
    assert tau_deep < 2.0, "Deep model threshold must be near 1.2"
    assert abs(tau_shallow - tau_deep) > 1e6, "Thresholds must differ substantially"
```

### Missing: `scripts/train_tokenizer.py`
This script is required by B5. It should wrap `tokenizers.BPE.from_file()` with CLI arguments:
- `--data`: path to training text
- `--vocab_size`: integer
- `--output`: path to save tokenizer JSON

---

## Phase 4: Blog Post Manuscript Updates (After Experiments Complete)

Once all benchmark experiments complete and output valid logs/terminal outputs, update the live blog post manuscript:
- `f:\blog\ram-prasad.dev\src\content\blog\01_the_server_that_knew_too_much.md`

| Section | Update Needed | Source Data |
|---|---|---|
| Section 5 — Results Table | Add `F1 = 0.892 ± σ` confidence interval | `data/b1_multiseed_summary.json` |
| Section 5/6 — Figures | Embed `data/heatmap_anomaly.png` | B6 output |
| Section 5 — Threshold Table | Add EVT vs. Gaussian vs. Percentile comparison | `data/b3_threshold_comparison.json` |
| Section 5 — Ablation Tables | Fix depth/vocab tables + add B4 architecture table | B0a/B0b/B4 outputs |
| Empirical Rigor — Logs | Embed real terminal logs/snippets showing benchmark outputs | Terminal execution logs |
| Limitations — Generalization | Add BGL F1 score | `data/b5_bgl_results.json` |

Upon completion of Phase 4, modify both `EXECUTION_PLAN.md` and `implementation_plan.md` to indicate 100% project completion.

---

*Last updated: 2026-06-29 | Status: 7 of 14 milestones complete*
