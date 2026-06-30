#!/usr/bin/env python3
"""Activation Function Ablation Study (B4).

Pedagogical explanations of why SwiGLU gating improves feature representation over standard
GELU activations, explicit tensor shapes, structured telemetry, and idempotency checks.
"""

import argparse
import json
import logging
import os
import sys
import time
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.data_loader import get_dataloader
from src.models.gpt2 import GPT2Config, GPT2Model
from src.utils.metrics import calculate_perplexity, calculate_classification_metrics
from train import get_lr, configure_optimizers
from evaluate import evaluate_split_perplexities, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("surprisal.b4")

ABLATION_STEPS = 800
N_LAYER_ABLATION = 4
WARMUP_STEPS   = 100
MAX_LR         = 6e-4
MIN_LR         = 6e-5
WEIGHT_DECAY   = 0.1
BETAS          = (0.9, 0.95)
GRAD_ACCUM     = 4
CLIP_GRAD      = 1.0


def _run_act_condition(
    use_swiglu: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: dict,
    device: str,
    autocast_dtype: torch.dtype,
    ablation_dir: str,
) -> dict:
    cond_name = "SwiGLU (Gated)" if use_swiglu else "GELU (Standard)"
    logger.info("=" * 60)
    logger.info(f"Ablation Condition: Activation Function = {cond_name}")
    logger.info("=" * 60)

    model_cfg = GPT2Config(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        n_embd=cfg["model"]["n_embd"],
        n_layer=N_LAYER_ABLATION,
        n_head=cfg["model"]["n_head"],
        block_size=cfg["dataset"]["seq_len"],
        d_ff=cfg["model"]["d_ff"],
        layer_norm_epsilon=cfg["model"]["layer_norm_epsilon"],
        use_swiglu=use_swiglu
    )
    model = GPT2Model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    optimizer = configure_optimizers(model, WEIGHT_DECAY, MAX_LR, BETAS)
    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()

    for step in range(ABLATION_STEPS):
        lr = get_lr(step, WARMUP_STEPS, ABLATION_STEPS, MAX_LR, MIN_LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            # x: [batch_size, seq_len]
            x = batch["input_ids"].to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, x)
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        if step > 0 and step % 50 == 0:
            time.sleep(0.01)  # Thermal pacing

        if step % 500 == 0 or step == ABLATION_STEPS - 1:
            logger.info(f"  step {step:4d}/{ABLATION_STEPS} | loss {accum_loss/GRAD_ACCUM:.4f} | {time.time()-t0:.1f}s")

    # WHY: SwiGLU adds a multiplicative gating branch that enhances gradient propagation across FFN layers.
    val_ppls, _, _ = evaluate_split_perplexities(model, val_loader, device, autocast_dtype=autocast_dtype)
    mu  = sum(val_ppls) / max(1, len(val_ppls))
    sig = (sum((p - mu) ** 2 for p in val_ppls) / max(1, len(val_ppls) - 1)) ** 0.5
    tau = mu + 3 * sig
    logger.info(f"  Calibration: μ={mu:.4f}  σ={sig:.4f}  τ={tau:.4f}")

    test_ppls, test_labels, _ = evaluate_split_perplexities(model, test_loader, device, autocast_dtype=autocast_dtype)
    preds = [1 if ppl > tau else 0 for ppl in test_ppls]
    m = calculate_classification_metrics(preds, test_labels)
    logger.info(f"  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    ckpt_name = "ckpt_act_swiglu.pt" if use_swiglu else "ckpt_act_gelu.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": model_cfg.__dict__}, os.path.join(ablation_dir, ckpt_name))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(3.0)  # Thermal cooldown

    return {
        "condition": cond_name,
        "use_swiglu": use_swiglu,
        "n_params": n_params,
        "val_perplexity": mu,
        "threshold_tau": tau,
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "accuracy": m["accuracy"],
        "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
    }


def main():
    parser = argparse.ArgumentParser(description="Activation function ablation study")
    parser.add_argument("--config", default="config/stage1_config.yaml")
    parser.add_argument("--output", default="data/ablations/ablation_act.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--force", action="store_true", help="Force retraining.")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        try:
            with open(args.output, "r") as f:
                existing_data = json.load(f)
            if len(existing_data) == 2:
                logger.info(f"[IDEMPOTENCY] B4 activation ablation results already exist at {args.output}. Pass --force to override.")
                return
        except Exception:
            pass

    set_seed(args.seed)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ablation_dir = os.path.dirname(args.output)
    os.makedirs(ablation_dir, exist_ok=True)

    train_loader, tokenizer = get_dataloader(args.config, split="train")
    val_loader, _   = get_dataloader(args.config, split="val",   tokenizer=tokenizer)
    test_loader, _  = get_dataloader(args.config, split="test",  tokenizer=tokenizer)

    results = [
        _run_act_condition(True, train_loader, val_loader, test_loader, cfg, device, autocast_dtype, ablation_dir),
        _run_act_condition(False, train_loader, val_loader, test_loader, cfg, device, autocast_dtype, ablation_dir)
    ]

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
