"""Threshold Sensitivity Analysis.

Evaluates anomaly detection performance (F1, Precision, Recall) as a function
of the calibration multiplier k in the threshold formula:

    τ(k) = μ_val + k · σ_val

for k ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}

This analysis answers the question: "Is τ = μ + 3σ the optimal threshold,
or is there a better k that trades precision for recall?"

The k=3 choice corresponds to the 99.87th percentile of a Gaussian distribution
and is used as the default. This script generates the precision-recall tradeoff
curve for inclusion in the paper.

Usage
-----
    # Requires a trained model checkpoint and pre-computed val perplexities
    python scripts/threshold_sensitivity.py \\
        --config config/stage1_config.yaml \\
        --checkpoint data/checkpoints/checkpoint_10000.pt \\
        --output data/ablations/threshold_sensitivity.json

Output
------
    data/ablations/threshold_sensitivity.json : per-k metrics
    ASCII precision-recall curve printed to stdout
"""

import argparse
import json
import logging
import math
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.data_loader import get_dataloader
from src.models.gpt2 import GPT2Config, GPT2Model
from src.utils.metrics import calculate_perplexity, calculate_classification_metrics

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def _compute_sequence_perplexities(model, loader, device, autocast_dtype) -> list:
    """Returns per-sequence perplexity values from a DataLoader."""
    model.eval()
    ppls = []
    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, x)
            if loss is not None:
                ppls.append(calculate_perplexity(loss.item()))
    return ppls


def _ascii_pr_curve(results: list) -> str:
    """Render a basic ASCII precision-recall chart."""
    lines = ["\n  Precision-Recall Tradeoff (k = τ multiplier)"]
    lines.append("  " + "─" * 58)
    lines.append(f"  {'k':>5}  {'τ':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    lines.append("  " + "─" * 58)
    for r in results:
        f1_bar = "█" * int(r["f1"] * 20)
        lines.append(
            f"  {r['k']:>5.1f}  {r['tau']:>8.4f}  "
            f"{r['precision']:>10.4f}  {r['recall']:>8.4f}  "
            f"{r['f1']:>8.4f}  {f1_bar}"
        )
    lines.append("  " + "─" * 58)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Threshold sensitivity analysis")
    parser.add_argument("--config",     default="config/stage1_config.yaml")
    parser.add_argument("--checkpoint", default="data/checkpoints/checkpoint_10000.pt")
    parser.add_argument("--output",     default="data/ablations/threshold_sensitivity.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load model
    model_cfg = GPT2Config(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        n_embd=cfg["model"]["n_embd"],
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        block_size=cfg["dataset"]["seq_len"],
        d_ff=cfg["model"]["d_ff"],
        layer_norm_epsilon=cfg["model"]["layer_norm_epsilon"],
    )
    model = GPT2Model(model_cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    val_loader, tokenizer = get_dataloader(args.config, split="val")
    test_loader, _ = get_dataloader(args.config, split="test", tokenizer=tokenizer)

    from evaluate import evaluate_split_perplexities
    logger.info("Computing validation perplexities...")
    val_ppls, _, _ = evaluate_split_perplexities(model, val_loader, device, autocast_dtype=autocast_dtype)
    logger.info("Computing test set perplexities...")
    test_ppls, test_labels, _ = evaluate_split_perplexities(model, test_loader, device, autocast_dtype=autocast_dtype)

    mu  = sum(val_ppls) / len(val_ppls)
    sig = (sum((p - mu) ** 2 for p in val_ppls) / max(1, len(val_ppls) - 1)) ** 0.5
    logger.info(f"Normal val PPL distribution: μ={mu:.4f}  σ={sig:.4f}")

    all_preds  = test_ppls
    all_labels = test_labels

    results = []
    for k in K_VALUES:
        tau   = mu + k * sig
        preds = [1 if p > tau else 0 for p in all_preds]
        m     = calculate_classification_metrics(preds, all_labels)
        results.append({
            "k": k,
            "tau": tau,
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "accuracy": m["accuracy"],
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
        })
        logger.info(
            f"  k={k:.1f}  τ={tau:.4f}  F1={m['f1']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}"
        )

    best = max(results, key=lambda r: r["f1"])
    logger.info(f"\nBest F1={best['f1']:.4f} at k={best['k']}")

    with open(args.output, "w") as f:
        json.dump({"mu": mu, "sigma": sig, "results": results}, f, indent=2)

    print(_ascii_pr_curve(results))
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
