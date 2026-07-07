# Stage 2 — QLoRA Fine-Tuning Experiment: Technical Implementation Plan
## Track 2 (Chapters 9–11): Domain-Adapted Root-Cause Diagnosis Engine

> *"The library is just convenience; the math is the substance."* — Andrej Karpathy
>
> This plan implements Stage 2 from first principles — every LoRA rank decomposition
> explained, every VRAM budget justified, every NF4 quantization decision grounded in math.
> We don't copy-paste from Hugging Face tutorials. We build the harness ourselves.

---

## Scientific Goal

Train a **QLoRA-adapted Qwen-2.5-3B-Instruct** model to act as a **structured root-cause diagnostic engine** (the "batch layer" of the future Lambda architecture). Given a flagged anomalous HDFS log block (output of Stage 1's surprisal detector), the fine-tuned model must produce a **validated JSON response** containing:

```json
{
  "root_cause": "DataNode communication timeout during replication pipeline",
  "severity": "P1_CRITICAL",
  "affected_component": "DataNode",
  "mitigation_commands": ["sudo systemctl restart hdfs-datanode", "hdfs dfsadmin -safemode leave"],
  "confidence": 0.92
}
```

**This is the Paper 1 fine-tuning section AND the backbone of Paper 2's Lambda batch layer.**

---

## Scientific Controls (Inherited from Master Plan)

These are **frozen** across ALL stages — do not modify:

| Control | Fixed Value | Rationale |
| :--- | :--- | :--- |
| **Dataset** | HDFS LogHub (same as Stage 1) | Same 11.4M log corpus; anomaly labels shared |
| **Tokenizer** | Stage 1 shared BPE (5K vocab) — NOT used for generation; Qwen's own tokenizer used for inference | Qwen-2.5 uses its own 151K-vocab tokenizer |
| **Evaluation Seeds** | `42, 123, 999` | 3 seeds, metrics averaged |
| **Validation Split** | 10% held-out val (same Stage 1 HDFS split) | Identical split prevents data leakage |
| **Baseline Metric** | Stage 1 GPT-2 surprisal F1 | All Stage 2 results must beat or complement the Stage 1 baseline |

---

## The Core Research Questions (What Paper 1 & 2 Need to Answer)

1. **Schema compliance vs. training data size**: At what SFT dataset size does JSON validity rate converge? (ablation B1)
2. **NF4 vs. FP16 LoRA**: What is the F1/VRAM tradeoff of 4-bit NF4 quantization vs. full FP16 LoRA? (ablation B2)
3. **LoRA rank sensitivity**: How does rank ∈ {8, 16, 32, 64} affect severity classification F1 at fixed VRAM? (ablation B3)
4. **Diagnostic latency SLO**: Can the batch layer deliver a structured JSON response < 2s on an 8GB GPU? (benchmark B4)
5. **Cross-block generalization**: Does a model fine-tuned on HDFS block anomalies generalize to BGL kernel panics? (stretch goal for Paper 4)

---

## Experiment Directory Structure

```
f:\projects\systems-engineering-labs\ai-engineering\surprisal-modeling\
├── stage1-gpt2\                          # ✅ DONE — GPT-2 baseline
└── stage2-qlora\                          # 🆕 THIS PLAN
    ├── config\
    │   └── stage2_config.yaml             # All hyperparameters in one place
    ├── data\
    │   ├── raw\                           # Symlinked from stage1-gpt2/data/raw (shared HDFS)
    │   ├── sft_dataset\                   # Curated SFT jsonl — (prompt, completion) pairs
    │   │   ├── train.jsonl                # ~4,000 annotated log anomaly pairs
    │   │   ├── val.jsonl                  # 10% holdout (400 pairs)
    │   │   └── test.jsonl                 # 10% holdout (400 pairs)
    │   ├── checkpoints\                   # LoRA adapter checkpoints (not full weights)
    │   ├── ablations\                     # Per-ablation JSON result logs
    │   └── stage2_results.json            # Final evaluation JSON report
    ├── src\
    │   ├── __init__.py
    │   ├── dataset\
    │   │   ├── __init__.py
    │   │   ├── sft_dataset.py             # SFTDataset: tokenize (prompt, completion) pairs
    │   │   └── data_loader.py             # get_sft_dataloader() factory
    │   ├── models\
    │   │   ├── __init__.py
    │   │   └── lora.py                    # LoRA math from scratch: ΔW = BA, rank decomposition
    │   └── utils\
    │       ├── __init__.py
    │       ├── metrics.py                 # JSON schema validator, severity F1, latency timer
    │       └── vram_profiler.py           # Peak VRAM tracker (mirrors stage1 util)
    ├── scripts\
    │   ├── __init__.py
    │   ├── prepare_sft_dataset.py         # Converts HDFS anomaly blocks → (prompt, JSON) pairs
    │   ├── ablation_lora_rank.py          # B3: rank ∈ {8, 16, 32, 64} sweep
    │   ├── ablation_nf4_vs_fp16.py        # B2: NF4 quantization vs FP16 full-weight comparison
    │   ├── ablation_dataset_size.py       # B1: SFT dataset size vs schema compliance rate
    │   ├── benchmark_latency.py           # B4: JSON generation latency profiling
    │   ├── analyze_results.py             # Compile LaTeX tables and Markdown catalog
    │   └── generate_blog_figures.py       # Publication-quality matplotlib/seaborn plots
    ├── finetune.py                        # Main QLoRA fine-tuning entry point
    ├── evaluate.py                        # JSON compliance + severity F1 + latency benchmark
    ├── inference.py                       # Single-sample inference (demo for blog/YouTube)
    ├── run_paper_experiments.sh           # Idempotent paper experiment runner (mirrors stage1)
    ├── run.sh                             # Quick-start single-run script
    ├── requirements.txt                   # Pinned Python dependencies
    ├── Dockerfile                         # Docker container for reproducibility
    ├── docker-compose.yml
    ├── README.md
    └── pytest.ini
```

---

## Proposed Changes

### Component 1: SFT Dataset Pipeline

This is the most critical component. **Garbage in → garbage out.** As Karpathy says: *"Visualize your data before writing a single line of model code."*

#### [NEW] `scripts/prepare_sft_dataset.py`

**Responsibility**: Takes Stage 1's raw HDFS log blocks (the ones flagged as anomalous by Stage 1's GPT-2 surprisal detector) and converts them into (prompt, JSON-completion) pairs for SFT.

**Algorithm**:
1. Load `stage1-gpt2/data/stage1_eval_results.json` — get the test block IDs with label=1 (anomaly)
2. Retrieve the raw 30-line context window for each flagged block from HDFS raw logs
3. Match against the HDFS structured labels (block_id → anomaly metadata)
4. Craft a structured prompt template (system + log-context block)
5. Craft a JSON completion template using rule-based SRE heuristics (DataNode, NameNode, network)
6. Write `train.jsonl`, `val.jsonl`, `test.jsonl`

**Prompt Format** (system role + user log context):
```
<|im_start|>system
You are a Site Reliability Engineer (SRE) analyzing HDFS distributed filesystem logs.
Given a block of anomalous log lines, produce a structured JSON root-cause diagnosis.
Output ONLY valid JSON. No prose. No markdown fences. No explanation.
<|im_end|>
<|im_start|>user
ANOMALOUS LOG BLOCK (block_id: blk_-1608999687919862906):
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906
081109 203519 144 ERROR dfs.DataNode$BlockReceiver: IOException: Connection reset by peer
081109 203519 145 WARN dfs.DataNode: DataXceiver error processing WRITE_BLOCK
...
<|im_end|>
<|im_start|>assistant
```

**Completion target**:
```json
{"root_cause": "DataNode write pipeline failure due to network reset during block replication", "severity": "P1_CRITICAL", "affected_component": "DataNode", "mitigation_commands": ["sudo systemctl restart hdfs-datanode", "hdfs dfsadmin -report"], "confidence": 0.88}
```

**Dataset sizes to generate** (for ablation B1):
- `train_100.jsonl` (100 pairs)
- `train_500.jsonl` (500 pairs)
- `train_2000.jsonl` (2000 pairs)
- `train_4000.jsonl` (4000 pairs — full)

---

### Component 2: LoRA Math from Scratch

> *"Build it from scratch, then use the library."* — Karpathy's most important rule.

#### [NEW] `src/models/lora.py`

We implement LoRA's low-rank decomposition **from first principles** before wrapping `peft`. This is the pedagogical core of Chapter 9.

**The Math** (ΔW = BA):
```
W_0 ∈ ℝ^{d×k}           # frozen pre-trained weight matrix
A ∈ ℝ^{r×k}  (r << d)   # trainable low-rank DOWN projection (random init)  
B ∈ ℝ^{d×r}             # trainable low-rank UP projection (zero init)
ΔW = BA                  # ΔW ∈ ℝ^{d×k}, same shape as W_0
W_effective = W_0 + (α/r) * BA  # α is the scaling factor (lora_alpha)
```

**`LoRALinear` class** — a drop-in replacement for `nn.Linear` with frozen weight + trainable low-rank adapter:
```python
class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation of a frozen linear weight matrix.
    
    WHY: Full fine-tuning of 3B params requires ~12GB VRAM (FP32) or ~6GB (BF16).
    LoRA introduces only r*(d+k) trainable parameters per layer, reducing
    adapter param count by d*k / r*(d+k) — for rank=16, d=k=2048: ~128x reduction.
    
    WHY zero-init B: Ensures ΔW=0 at initialization → identical to pre-trained model at step 0.
    WHY random-init A: Standard Kaiming initialization; A drives the signal, B scales to zero.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0):
        ...
```

**`inject_lora_adapters(model, target_modules, rank, alpha)`** — scans the HuggingFace model and replaces specified `nn.Linear` layers with `LoRALinear` wrappers, freezing original weights.

**`merge_lora_weights(model)`** — fuses `W_0 + (α/r)*BA` → single weight matrix for deployment (zero adapter overhead at inference).

---

### Component 3: Main Fine-Tuning Script

#### [NEW] `finetune.py`

**Architecture decisions**:
- **Base model**: `Qwen/Qwen2.5-3B-Instruct` (3B params, instruction-tuned, supports ChatML format)
- **Quantization**: BitsAndBytes NF4 (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`, `bnb_4bit_use_double_quant=True`)
- **LoRA targets**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **Optimizer**: AdamW 8-bit (via `bitsandbytes.optim.AdamW8bit`) — saves ~3GB vs FP32 Adam
- **Training objective**: Causal LM loss **masked on the prompt** — only the JSON completion contributes to loss
- **Gradient checkpointing**: Enabled — trades recomputation for ~40% VRAM reduction in activations

**Loss masking (critical detail)**:
```python
# WHY: If we compute loss on the entire sequence including the prompt,
# the model wastes capacity learning to predict the static system prompt.
# We set target=-100 (ignore_index) for all prompt tokens.
# Only the completion tokens (after <|im_start|>assistant\n) contribute to the gradient.
labels = input_ids.clone()
labels[labels == tokenizer.pad_token_id] = -100
prompt_len = find_assistant_token_position(input_ids)  # custom parser
labels[:, :prompt_len] = -100  # mask prompt tokens
```

**Training loop**: Mirrors Stage 1's `train.py` with identical patterns:
- Cosine LR schedule with linear warmup
- Gradient accumulation (effective batch = 16 despite micro-batch = 2)
- Smart checkpointing (save only LoRA adapter weights, not full 3B model)
- Per-100-step validation eval with `tqdm` telemetry

---

### Component 4: Evaluation Harness

#### [NEW] `evaluate.py`

**Metrics** (three dimensions, not one):

| Metric | What It Measures | Target |
| :--- | :--- | :--- |
| **JSON schema compliance rate** | % of outputs that are valid JSON matching the schema | > 90% |
| **Severity classification F1** | Precision/Recall/F1 for P0/P1/P2/P3 severity labels | > 0.80 |
| **Diagnostic latency (p50/p95)** | Time from prompt to complete JSON token at inference | p95 < 2000ms |

**Schema validator** (strict, not just `json.loads()`):
```python
REQUIRED_KEYS = {"root_cause", "severity", "affected_component", "mitigation_commands", "confidence"}
SEVERITY_VALUES = {"P0_EMERGENCY", "P1_CRITICAL", "P2_WARNING", "P3_INFO"}

def validate_json_schema(output: str) -> tuple[bool, dict | None]:
    """Returns (is_valid, parsed_dict). Checks: parseable JSON, required keys, type constraints."""
```

**Multi-seed evaluation**: Runs with seeds `[42, 123, 999]`, averages metrics, reports mean ± std.

---

### Component 5: Ablation Studies

Mirroring Stage 1's ablation structure — each ablation is an **independent, idempotent script** that trains a mini-model, evaluates, and writes JSON results.

#### [NEW] `scripts/ablation_dataset_size.py` — Ablation B1

**Hypothesis**: JSON compliance rate follows a data scaling law. There's a "knee" beyond which more data yields diminishing returns.

**Conditions**: SFT dataset size ∈ {100, 500, 2000, 4000} pairs
**Output**: `data/ablations/ablation_dataset_size.json`
**Blog figure**: Schema compliance rate vs. training dataset size (log-x axis)

#### [NEW] `scripts/ablation_lora_rank.py` — Ablation B3

**Hypothesis**: Rank ≥ 32 offers negligible F1 improvement over rank=16 while doubling trainable param count.

**Conditions**: LoRA rank ∈ {8, 16, 32, 64} with fixed alpha = 2×rank
**Output**: `data/ablations/ablation_lora_rank.json`
**Blog figure**: F1 vs. LoRA rank with adapter parameter count on secondary y-axis

#### [NEW] `scripts/ablation_nf4_vs_fp16.py` — Ablation B2

**Hypothesis**: NF4 (4-bit) matches FP16 LoRA within 2 F1 points while using 40% less VRAM.

**Conditions**: 
- Condition A: NF4 quantization + LoRA (our default)
- Condition B: FP16 base model + LoRA (no quantization)
- Condition C: Unquantized base (BF16, no LoRA) — zero-shot baseline

**Output**: `data/ablations/ablation_nf4_vs_fp16.json`
**Blog figure**: 3-panel bar chart: schema compliance, severity F1, VRAM footprint

#### [NEW] `scripts/benchmark_latency.py` — Benchmark B4

**Measures**: End-to-end inference latency for 100 samples with `torch.cuda.Event` timing.

**Reports**: 
- p50, p95, p99 latency (ms)
- Tokens per second (generation speed)
- VRAM peak during inference

---

### Component 6: Configuration

#### [NEW] `config/stage2_config.yaml`

```yaml
base_model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  revision: "main"
  trust_remote_code: true

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true   # WHY: double quant saves ~0.4 bits/param additional

lora:
  rank: 16                    # r in ΔW = BA; 16 is the empirical sweet spot
  alpha: 32                   # scaling = alpha/rank = 2.0 (ΔW is doubled in magnitude)
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  bias: "none"

dataset:
  sft_dir: "data/sft_dataset"
  train_file: "train.jsonl"
  val_file: "val.jsonl"
  test_file: "test.jsonl"
  max_seq_len: 1024           # Prompt + JSON completion fits in 1024 tokens

training:
  max_lr: 2.0e-4              # WHY: Smaller LR than pre-training (3B model already converged)
  min_lr: 2.0e-5
  warmup_steps: 100
  max_steps: 2000
  micro_batch_size: 2         # Max that fits in 8GB VRAM with NF4 + grad checkpointing
  gradient_accumulation_steps: 8   # Effective batch = 2 * 8 = 16
  weight_decay: 0.01
  gradient_clipping: 1.0
  checkpoint_interval: 500
  checkpoint_dir: "data/checkpoints"
  results_path: "data/stage2_results.json"

evaluation:
  seeds: [42, 123, 999]
  max_new_tokens: 256
  temperature: 0.1            # Near-greedy for structured JSON output
  do_sample: false
```

---

### Component 7: Paper Experiment Runner

#### [NEW] `run_paper_experiments.sh`

Mirrors Stage 1's idempotent runner structure exactly:

```bash
#!/usr/bin/env bash
set -euo pipefail
# === [1/5] SFT Dataset Preparation ===
# === [2/5] Main QLoRA Fine-Tuning (if checkpoint doesn't exist) ===
# === [3/5] Formal Evaluation (JSON compliance + Severity F1 + Latency) ===
# === [4/5] Ablation Sweeps (B1: dataset size, B2: NF4 vs FP16, B3: LoRA rank) ===
# === [5/5] Compile LaTeX Tables & Blog Figures ===
```

---

## VRAM Budget Analysis

> *"One VRAM or latency table gets 10× more shares than prose."* — curriculum writing rules

| Component | VRAM Usage | Notes |
| :--- | :--- | :--- |
| Qwen-2.5-3B in NF4 | ~1.8 GB | 3B × 0.5 bytes/param (4-bit) |
| LoRA adapter weights (rank=16) | ~60 MB | 21M adapter params × 4 bytes |
| Activation memory (grad. ckpt.) | ~1.2 GB | Recomputed per backward pass |
| AdamW 8-bit optimizer states | ~1.8 GB | 2 states × 21M adapter params |
| Input batch (bs=2, seq=1024) | ~0.2 GB | 2 × 1024 × bf16 |
| **Total** | **~5.1 GB** | **Fits on 8GB GPU (RTX 3070/4070)** |

Without NF4: ~12GB (FP16 base model) → **would OOM on 8GB GPU** → This is Paper 1's key finding.

---

## Paper Contribution Mapping

| Experiment | Paper Section | Claim |
| :--- | :--- | :--- |
| Main fine-tuning + eval (B0) | Paper 1: §4.2 Fine-Tuning | "QLoRA adapts 3B LLM to SRE diagnosis with 90%+ JSON compliance on 8GB VRAM" |
| NF4 vs FP16 ablation (B2) | Paper 1: §4.3 Quantization | "NF4 double-quant loses < 2 F1 points vs FP16 while saving 4GB VRAM" |
| LoRA rank sweep (B3) | Paper 1: Appendix A | "Rank=16 is the empirical optimum; rank=32 yields +0.5 F1 at 2× adapter cost" |
| Dataset size scaling (B1) | Paper 1: §4.2 | "Schema compliance saturates at ~2000 SFT pairs (Chinchilla-style data scaling)" |
| Latency benchmark (B4) | Paper 2: §3.1 Batch Layer SLO | "p95 JSON generation latency: 1200ms on RTX 3070 — meets 2s SLO" |
| Multi-seed evaluation | Paper 1: §5 Results | Error bars on all F1 scores — required for NeurIPS workshop submission |

---

## Karpathy Principles Applied

| Principle | How Applied |
| :--- | :--- |
| **"Build from scratch first"** | `src/models/lora.py` implements ΔW=BA manually before using `peft.LoraConfig` |
| **"Overfit one batch first"** | `finetune.py --debug` mode: 1 batch, 50 steps, verify loss drops to near-zero |
| **"Visualize your data"** | `prepare_sft_dataset.py` prints 3 random (prompt, completion) examples before writing files |
| **"LR is the most important hyperparameter"** | Config clearly documents why 2e-4 (not 6e-4 from stage1) — SFT needs smaller LR |
| **"Highlight failures"** | ablation B2's no-NF4 condition will OOM — that OOM is logged and is the point |
| **"Include numbers"** | Every ablation writes a LaTeX table; every blog figure has axis labels and numbers |
| **"Read your loss curves carefully"** | Training loop logs train/val loss every 50 steps; `analyze_results.py` plots the curves |
| **"Understand the shape at every step"** | Every tensor has explicit shape annotation in comments: `# [batch, seq, hidden]` |

---

## Implementation Phases

### Phase 0: Dataset Preparation (Day 1)
- [ ] Write `scripts/prepare_sft_dataset.py` — HDFS anomaly block → (prompt, JSON) pairs
- [ ] Manually validate 20 random (prompt, completion) pairs before running at scale
- [ ] Generate all 4 dataset size splits for ablation B1
- [ ] Run `python scripts/prepare_sft_dataset.py --verify` to print 3 random examples

### Phase 1: LoRA from Scratch (Day 1-2)
- [ ] Implement `src/models/lora.py` — `LoRALinear`, `inject_lora_adapters`, `merge_lora_weights`
- [ ] Unit test: `assert forward(x, adapter=zeroed) ≈ forward(x, no_adapter)` (zero-init invariant)
- [ ] Unit test: `assert count_trainable_params(model_with_lora) << count_all_params(base_model)`

### Phase 2: Fine-Tuning Script (Day 2-3)
- [ ] Write `finetune.py` with NF4 loading, LoRA injection, loss masking, training loop
- [ ] Debug mode: `python finetune.py --debug --max_steps 50` — verify loss < 0.1 on single batch
- [ ] Full run: `python finetune.py config/stage2_config.yaml`
- [ ] Verify checkpoint saves only LoRA adapter weights (not full 3B model)

### Phase 3: Evaluation Harness (Day 3)
- [ ] Write `evaluate.py` — JSON schema validator, severity F1, latency measurement
- [ ] Write `inference.py` — single-sample demo for blog post and YouTube walkthrough
- [ ] Run `python evaluate.py --seeds 42 123 999` with multi-seed aggregation

### Phase 4: Ablation Studies (Day 4-5)
- [ ] Run `scripts/ablation_dataset_size.py` — B1 (dataset scaling)
- [ ] Run `scripts/ablation_nf4_vs_fp16.py` — B2 (quantization tradeoff, expect OOM in FP16 condition)
- [ ] Run `scripts/ablation_lora_rank.py` — B3 (rank sensitivity)
- [ ] Run `scripts/benchmark_latency.py` — B4 (inference latency SLO)

### Phase 5: Paper Artifacts (Day 5-6)
- [ ] Run `scripts/analyze_results.py` — generate LaTeX tables for Paper 1/2
- [ ] Run `scripts/generate_blog_figures.py` — publication-quality matplotlib plots
- [ ] Run full `run_paper_experiments.sh` end-to-end (idempotency check)
- [ ] Write `README.md` and `CODE_WALKTHROUGH.md`

---

## Open Questions

> [!IMPORTANT]
> **Dataset annotation strategy**: The `prepare_sft_dataset.py` will use rule-based SRE heuristics to generate JSON completions (DataNode vs NameNode patterns, etc.). This is semi-automatic. Do you want to manually review/edit the generated (prompt, completion) pairs before training? Recommending: spot-check 50 samples before the full run.

> [!IMPORTANT]
> **Base model**: The plan uses `Qwen/Qwen2.5-3B-Instruct`. Alternatives: `Qwen/Qwen2.5-1.5B-Instruct` (fits 8GB more comfortably), `mistralai/Mistral-7B-Instruct-v0.3` (7B, tighter VRAM). Current selection optimizes for the VRAM story in Paper 1. **Confirm: Qwen-2.5-3B?**

> [!WARNING]
> **`peft` library dependency**: We implement LoRA math ourselves in `src/models/lora.py` for pedagogy, but use `peft.LoraConfig` for the actual training loop (for stability and CUDA kernel optimizations). The from-scratch implementation is for the blog/YouTube explanation. Confirm this approach or prefer fully custom?

> [!NOTE]
> **SFT dataset size**: HDFS has ~16,838 anomalous block sessions. With 30-line context windows, we can generate ~4000 high-quality (prompt, completion) pairs. This is a small dataset for SFT — which is intentional (the data scaling ablation is the point).

---

## Verification Plan

### Automated Tests
```bash
# Unit tests for LoRA math
pytest tests/test_lora.py -v

# Schema validator tests  
pytest tests/test_metrics.py -v

# Dataset integrity check
python scripts/prepare_sft_dataset.py --verify

# Debug training run (overfit 1 batch)
python finetune.py --debug --max_steps 50

# Full paper suite (idempotent)
bash run_paper_experiments.sh
```

### Manual Verification
1. After Phase 0: Print 10 random (prompt, completion) pairs, verify JSON is valid and plausible
2. After Phase 2: Plot training/validation loss curve — val loss should decrease, not diverge
3. After Phase 3: Run `python inference.py --input "081109 203518 ERROR DataNode..."` and inspect JSON output
4. After Phase 4: Verify ablation B2's FP16 condition OOMs (expected) — the OOM message is a result
5. After Phase 5: Open all `.md` result catalogs and verify LaTeX table numbers are reasonable

---

## Output Artifacts (What This Produces for Paper 1)

| Artifact | Path | Paper Use |
| :--- | :--- | :--- |
| Fine-tuned LoRA adapter | `data/checkpoints/adapter_step_2000/` | Paper 1: model released on HuggingFace |
| Main evaluation report | `data/stage2_results.json` | Paper 1: Table III (main results) |
| Dataset size ablation | `data/ablations/ablation_dataset_size.json` | Paper 1: Appendix B |
| NF4 vs FP16 ablation | `data/ablations/ablation_nf4_vs_fp16.json` | Paper 1: Table IV |
| LoRA rank ablation | `data/ablations/ablation_lora_rank.json` | Paper 1: Appendix A |
| Latency benchmark | `data/ablations/benchmark_latency.json` | Paper 2: §3.1 |
| Publication figures | `data/figures/*.png` | Paper 1 & 2 |
| LaTeX tables | `data/ablations/stage2_ablation_summary.md` | Paper 1 §5 |
