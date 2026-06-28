"""Vocabulary Size Ablation Study.

Trains a compact GPT-2 model with multiple BPE vocabulary sizes and evaluates
the effect on validation perplexity and anomaly detection performance. This
ablation validates the V=5,000 hyperparameter choice made in the main
experiment.

Ablation conditions
-------------------
V ∈ {500, 1_000, 2_000, 5_000, 10_000}

For each vocabulary size:
  1. Trains a BPE tokenizer on the same normal training corpus.
  2. Re-packs sequences using the new tokenizer.
  3. Trains a compact 4-layer GPT-2 for a fixed budget of 2,000 steps.
  4. Evaluates validation perplexity and test F1 under τ = μ + 3σ calibration.
  5. Records results to data/ablations/ablation_vocab.json.

Usage
-----
    python scripts/ablation_vocab.py --config config/stage1_config.yaml

Notes
-----
- This script uses a 4-layer model (n_layer=4) to reduce ablation runtime.
  The relative perplexity ordering across vocabulary sizes is stable across
  model depths.
- Expected runtime: ~2–4 hours on an RTX 3060 Ti (8 GB VRAM).
- Results from the paper's ablation are checked into data/ablations/.
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import torch
import yaml

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.data_loader import (
    download_hdfs_dataset,
    load_anomaly_labels,
    parse_and_group_logs,
    pack_sequences_ffd,
    PackedLogDataset,
)
from src.models.gpt2 import GPT2Config, GPT2Model
from src.tokenizer.log_tokenizer import LogTokenizer
from src.utils.metrics import calculate_perplexity, calculate_classification_metrics
from train import get_lr, configure_optimizers

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation hyperparameters (fixed across all conditions)
# ---------------------------------------------------------------------------

VOCAB_SIZES    = [500, 1_000, 2_000, 5_000, 10_000]
ABLATION_STEPS = 2_000          # short training budget for ablation
N_LAYER_ABLATION = 4            # reduced depth for faster runtime
WARMUP_STEPS   = 200
MAX_LR         = 6e-4
MIN_LR         = 6e-5
WEIGHT_DECAY   = 0.1
BETAS          = (0.9, 0.95)
BATCH_SIZE     = 16
GRAD_ACCUM     = 4
CLIP_GRAD      = 1.0
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                  "<EOS>", "<IP>", "<HEX>", "<DATE>", "<TIME>"]


def _train_tokenizer(normal_blocks: list, vocab_size: int, save_path: str) -> LogTokenizer:
    """Train a BPE tokenizer for a given vocabulary size."""
    corpus_path = save_path.replace(".json", "_corpus.txt")
    tok = LogTokenizer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for block in normal_blocks:
            for line in block["lines"]:
                f.write(line + "\n")
    tok.train(corpus_path, save_path)
    return tok


def _pack_dataset(
    blocks: list,
    tok: LogTokenizer,
    seq_len: int,
    eos_id: int,
) -> PackedLogDataset:
    """Tokenize, concatenate and FFD-pack a list of log blocks."""
    tokenized = []
    for block in blocks:
        toks = []
        masked = [tok.mask_variables(l) for l in block["lines"]]
        for enc in tok.tokenizer.encode_batch(masked):
            toks.extend(enc.ids + [eos_id])
        tokenized.append(toks)
    packed = pack_sequences_ffd(tokenized, max_len=seq_len, eos_token_id=eos_id)
    return PackedLogDataset(packed)


def _run_condition(
    vocab_size: int,
    normal_train: list,
    normal_val: list,
    anomaly_test: list,
    seq_len: int,
    ablation_dir: str,
    device: str,
    autocast_dtype: torch.dtype,
) -> dict:
    """Execute one ablation condition. Returns a metrics dictionary."""
    logger.info("=" * 60)
    logger.info(f"Ablation: vocab_size = {vocab_size:,}")
    logger.info("=" * 60)

    tok_path = os.path.join(ablation_dir, f"tok_v{vocab_size}.json")
    tok = _train_tokenizer(normal_train, vocab_size, tok_path)
    eos_id = tok.tokenizer.token_to_id("<EOS>")
    pad_id = tok.tokenizer.token_to_id("[PAD]")

    train_dataset = _pack_dataset(normal_train, tok, seq_len, eos_id)
    val_dataset   = _pack_dataset(normal_val,   tok, seq_len, eos_id)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=768,
        n_layer=N_LAYER_ABLATION,
        n_head=12,
        block_size=seq_len,
        d_ff=2048,
    )
    model = GPT2Model(cfg).to(device)
    optimizer = configure_optimizers(model, WEIGHT_DECAY, MAX_LR, BETAS)

    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()

    for step in range(ABLATION_STEPS):
        lr = get_lr(step, WARMUP_STEPS, ABLATION_STEPS, MAX_LR, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            x = batch["input_ids"].to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, x)
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        if step % 200 == 0 or step == ABLATION_STEPS - 1:
            elapsed = time.time() - t0
            logger.info(f"  step {step:4d}/{ABLATION_STEPS} | loss {accum_loss/GRAD_ACCUM:.4f} | {elapsed:.1f}s")

    # Validation perplexity calibration
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["input_ids"].to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, x)
            if loss is not None:
                val_losses.append(loss.item())
    mu  = sum(val_losses) / len(val_losses)
    sig = (sum((l - mu) ** 2 for l in val_losses) / len(val_losses)) ** 0.5
    tau = mu + 3 * sig
    val_ppl = calculate_perplexity(mu)
    logger.info(f"  Calibration: μ={mu:.4f} σ={sig:.4f} τ={tau:.4f} ppl={val_ppl:.4f}")

    # Test-set scoring
    def _seq_perplexity(block_lines: list) -> float:
        masked  = [tok.mask_variables(l) for l in block_lines]
        ids     = []
        for enc in tok.tokenizer.encode_batch(masked):
            ids.extend(enc.ids + [eos_id])
        ids = ids[:seq_len]
        ids += [pad_id] * (seq_len - len(ids))
        t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type=device, dtype=autocast_dtype):
            _, loss = model(t, t)
        return calculate_perplexity(loss.item()) if loss else float("inf")

    preds, labels = [], []
    all_blocks = [(b, 0) for b in normal_val] + [(b, 1) for b in anomaly_test]
    for block, label in all_blocks:
        ppl = _seq_perplexity(block["lines"])
        preds.append(1 if ppl > tau else 0)
        labels.append(label)

    m = calculate_classification_metrics(preds, labels)
    logger.info(f"  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    return {
        "vocab_size": vocab_size,
        "val_perplexity": val_ppl,
        "threshold_tau": tau,
        "calibration_mean": mu,
        "calibration_std": sig,
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "accuracy": m["accuracy"],
        "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
    }


def main():
    parser = argparse.ArgumentParser(description="Vocabulary size ablation study")
    parser.add_argument("--config", default="config/stage1_config.yaml")
    parser.add_argument("--output", default="data/ablations/ablation_vocab.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    seq_len = cfg["dataset"]["seq_len"]
    ablation_dir = os.path.dirname(args.output)
    os.makedirs(ablation_dir, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Ablation conditions: vocab_size ∈ {VOCAB_SIZES}")

    # Download and parse once
    raw_dir   = cfg["dataset"]["raw_dir"]
    download_hdfs_dataset(cfg["dataset"]["url"], raw_dir)
    log_path   = os.path.join(raw_dir, "HDFS.log")
    label_path = os.path.join(raw_dir, "anomaly_label.csv")
    labels_map = load_anomaly_labels(label_path)
    normal_all, anomaly = parse_and_group_logs(log_path, labels_map)

    split_idx    = int(len(normal_all) * cfg["dataset"]["train_split"])
    normal_train = normal_all[:split_idx]
    normal_val   = normal_all[split_idx:]
    logger.info(f"Normal train: {len(normal_train):,}  val: {len(normal_val):,}  anomaly: {len(anomaly):,}")

    results = []
    for v in VOCAB_SIZES:
        result = _run_condition(
            vocab_size=v,
            normal_train=normal_train,
            normal_val=normal_val,
            anomaly_test=anomaly,
            seq_len=seq_len,
            ablation_dir=ablation_dir,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        results.append(result)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 72)
    print("ABLATION: Vocabulary Size vs. Performance")
    print("=" * 72)
    print(f"{'V':>8}  {'Val PPL':>9}  {'τ':>8}  {'F1':>7}  {'Prec':>7}  {'Recall':>7}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['vocab_size']:>8,}  {r['val_perplexity']:>9.4f}  "
            f"{r['threshold_tau']:>8.4f}  {r['f1']:>7.4f}  "
            f"{r['precision']:>7.4f}  {r['recall']:>7.4f}"
        )
    print("=" * 72)
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
