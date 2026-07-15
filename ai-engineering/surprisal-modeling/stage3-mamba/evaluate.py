"""Surprisal Threshold Calibration & Anomaly Classification Evaluation (`evaluate.py`).
- Evaluates unreduced per-token cross-entropy loss across valid log syntax (masking out padding `pad_token_id=5`).
- Calibrates Gaussian anomaly threshold (`tau = mu_val + 3 * sigma_val`) on healthy HDFS validation split.
- Evaluates precision, recall, accuracy, and F1 across all 72,661 held-out test log sequences.
- Supports multi-seed aggregation (`seeds: [42, 123, 999]`) and threshold sensitivity analysis across K-factors [1.5 -> 4.5].
"""

import os
import sys
import math
import json
import random
import argparse
import logging
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.mamba_lm import MambaLMHeadModel
from src.models.hybrid_mambalog import MambaLogLMHeadModel
from src.dataset.log_dataset import get_dataloader
from src.utils.metrics import (
    calculate_perplexity,
    calculate_surprisal_threshold,
    calculate_classification_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("stage3.eval")


def set_seed(seed: int):
    """Enforces deterministic seed state for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_split_perplexities(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    autocast_dtype: torch.dtype = torch.bfloat16,
    pad_token_id: int = 5,
    max_batches: int | None = None,
) -> tuple[list[float], list[int], list[str]]:
    """Evaluates per-sequence perplexity (`exp(mean_token_loss)`) masking out trailing padding tokens.

    WHY: Trailing padding (`pad_token_id=5`) artificially depresses sequence loss. By masking padding out,
    we ensure perplexity reflects true surprisal over valid log syntax.
    """
    model.eval()
    perplexities = []
    labels = []
    block_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating split perplexity", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            batch_labels = batch.get("label", torch.zeros(len(input_ids), dtype=torch.long)).cpu().tolist()
            batch_block_ids = batch.get("block_id", [""] * len(input_ids))

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            if torch.cuda.is_available() and device != "cpu":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits, _ = model(inputs)
            else:
                logits, _ = model(inputs)

            # Unreduced loss per token: [batch_size, seq_len - 1]
            loss_per_token = F.cross_entropy(
                logits.transpose(1, 2), targets, reduction="none"
            )
            valid_mask = ~((inputs == pad_token_id) & (targets == pad_token_id))
            valid_counts = valid_mask.sum(dim=1).clamp(min=1)

            loss_per_seq = (loss_per_token * valid_mask.float()).sum(dim=1) / valid_counts

            for seq_loss in loss_per_seq.cpu().tolist():
                perplexities.append(calculate_perplexity(seq_loss))

            labels.extend(batch_labels)
            block_ids.extend(batch_block_ids)

    return perplexities, labels, block_ids


def evaluate_single_seed(
    config: dict,
    config_path: str,
    seed: int,
    weights_path: str | None = None,
    device: str = "cuda",
    model_type: str = "mamba",
    k_factors: list[float] | None = None,
    max_batches: int | None = None,
) -> dict:
    """Evaluates surprisal threshold calibration and anomaly detection accuracy for a single seed run."""
    set_seed(seed)
    if k_factors is None:
        k_factors = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

    if model_type.lower() == "mambalog":
        model = MambaLogLMHeadModel(config)
    else:
        model = MambaLMHeadModel(config)

    # Resolve weights path
    if weights_path and os.path.exists(weights_path):
        ckpt_to_load = weights_path
    else:
        ckpt_dir = config["training"].get("checkpoint_dir", f"data/checkpoints_{model_type}")
        ckpt_to_load = os.path.join(ckpt_dir, f"{model_type}_seed_{seed}_final.pt")

    if os.path.exists(ckpt_to_load):
        logger.info(f"[Seed {seed}] Loading pre-trained checkpoint weights from: {ckpt_to_load}")
        ckpt = torch.load(ckpt_to_load, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(
            f"[Seed {seed}] Checkpoint {ckpt_to_load} not found. Using randomly initialized weights for verification."
        )

    model.to(device)
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pad_id = config["model"].get("pad_token_id", 5)

    logger.info(f"[Seed {seed}] Step 1: Calibrating baseline surprisal distribution on Validation split...")
    val_loader, _ = get_dataloader(config_path, split="val")
    val_ppls, _, _ = evaluate_split_perplexities(model, val_loader, device, autocast_dtype, pad_token_id=pad_id, max_batches=max_batches)

    tau_3sigma, mu_val, sigma_val = calculate_surprisal_threshold(val_ppls, k_factor=3.0)
    logger.info(
        f"[Seed {seed}] Calibration complete: mu={mu_val:.4f}, sigma={sigma_val:.4f} -> tau (3-sigma) = {tau_3sigma:.4f}"
    )

    logger.info(f"[Seed {seed}] Step 2: Evaluating anomaly classification across Test split...")
    test_loader, _ = get_dataloader(config_path, split="test")
    test_ppls, test_labels, test_block_ids = evaluate_split_perplexities(
        model, test_loader, device, autocast_dtype, pad_token_id=pad_id, max_batches=max_batches
    )

    # Base evaluation at k = 3.0
    preds_3sigma = [1 if p > tau_3sigma else 0 for p in test_ppls]
    base_metrics = calculate_classification_metrics(preds_3sigma, test_labels)
    logger.info(
        f"[Seed {seed} | k=3.0] F1: {base_metrics['f1']:.4f} | Precision: {base_metrics['precision']:.4f} | Recall: {base_metrics['recall']:.4f}"
    )

    # Sensitivity sweep across k-factors
    sensitivity_curve = {}
    for k in k_factors:
        tau_k = mu_val + k * sigma_val
        preds_k = [1 if p > tau_k else 0 for p in test_ppls]
        m_k = calculate_classification_metrics(preds_k, test_labels)
        sensitivity_curve[f"k_{k}"] = {
            "threshold": round(tau_k, 4),
            "f1": round(m_k["f1"], 4),
            "precision": round(m_k["precision"], 4),
            "recall": round(m_k["recall"], 4),
        }

    return {
        "seed": seed,
        "calibration": {
            "mu_val": round(mu_val, 4),
            "sigma_val": round(sigma_val, 4),
            "tau_3sigma": round(tau_3sigma, 4),
        },
        "metrics_3sigma": base_metrics,
        "sensitivity_curve": sensitivity_curve,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 Mamba anomaly detection across test splits.")
    parser.add_argument("--config", type=str, default="config/stage3_config.yaml", help="Path to config YAML.")
    parser.add_argument("--model_type", type=str, choices=["mamba", "mambalog"], default="mamba", help="Model type.")
    parser.add_argument("--weights", type=str, default=None, help="Explicit path to single checkpoint.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Seeds to evaluate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="data/stage3_results.json", help="Path to save evaluation summary JSON.")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to evaluate (for fast dev runs).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results = {}
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for seed in args.seeds:
        seed_res = evaluate_single_seed(
            config=config,
            config_path=args.config,
            seed=seed,
            weights_path=args.weights,
            device=args.device,
            model_type=args.model_type,
            max_batches=args.max_batches,
        )
        results[f"seed_{seed}"] = seed_res
        f1_scores.append(seed_res["metrics_3sigma"]["f1"])
        precision_scores.append(seed_res["metrics_3sigma"]["precision"])
        recall_scores.append(seed_res["metrics_3sigma"]["recall"])

    summary_stats = {
        "mean_f1": round(float(np.mean(f1_scores)), 4),
        "std_f1": round(float(np.std(f1_scores)), 4),
        "mean_precision": round(float(np.mean(precision_scores)), 4),
        "std_precision": round(float(np.std(precision_scores)), 4),
        "mean_recall": round(float(np.mean(recall_scores)), 4),
        "std_recall": round(float(np.std(recall_scores)), 4),
    }
    results["summary_statistics"] = summary_stats

    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(
        f"\n[Final Stage 3 Multi-Seed Evaluation Summary]\n"
        f"Mean F1: {summary_stats['mean_f1']:.4f} ± {summary_stats['std_f1']:.4f}\n"
        f"Mean Precision: {summary_stats['mean_precision']:.4f} ± {summary_stats['std_precision']:.4f}\n"
        f"Mean Recall: {summary_stats['mean_recall']:.4f} ± {summary_stats['std_recall']:.4f}\n"
        f"Results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
