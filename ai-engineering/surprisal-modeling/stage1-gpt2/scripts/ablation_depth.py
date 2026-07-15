"""Model Depth Ablation Study.

Explains why depth ablations require independent
threshold calibration, explicit shapes, structured telemetry, and idempotency.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.data_loader import get_dataloader
from src.models.gpt2 import GPT2Config, GPT2Model
from src.utils.metrics import calculate_perplexity, calculate_classification_metrics
from train import get_lr, configure_optimizers
from evaluate import evaluate_split_perplexities, set_seed

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEPTHS         = [2, 4, 12]
ABLATION_STEPS = 800
WARMUP_STEPS   = 100
MAX_LR         = 6e-4
MIN_LR         = 6e-5
WEIGHT_DECAY   = 0.1
BETAS          = (0.9, 0.95)
GRAD_ACCUM     = 4
CLIP_GRAD      = 1.0


def _count_parameters(model: GPT2Model) -> int:
    return sum(p.numel() for p in model.parameters())


def _run_depth_condition(
    n_layer: int,
    train_loader,
    val_loader,
    test_loader,
    cfg: dict,
    device: str,
    autocast_dtype: torch.dtype,
    ablation_dir: str,
) -> dict:
    logger.info("=" * 60)
    logger.info(f"Ablation Condition: n_layer = {n_layer}")
    logger.info("=" * 60)

    model_cfg = GPT2Config(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        n_embd=cfg["model"]["n_embd"],
        n_layer=n_layer,
        n_head=cfg["model"]["n_head"],
        block_size=cfg["dataset"]["seq_len"],
        d_ff=cfg["model"]["d_ff"],
        layer_norm_epsilon=cfg["model"]["layer_norm_epsilon"],
    )
    model = GPT2Model(model_cfg).to(device)
    n_params = _count_parameters(model)
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
            time.sleep(0.01)  # Thermal pacing to prevent GPU overheating

        if step % 500 == 0 or step == ABLATION_STEPS - 1:
            logger.info(
                f"  step {step:5d}/{ABLATION_STEPS} | "
                f"loss {accum_loss/GRAD_ACCUM:.4f} | {time.time()-t0:.1f}s"
            )

    # WHY: Each depth model has a drastically different baseline perplexity scale.
    # Applying a shared threshold from a 12-layer model marks 100% of sequences as anomalous for 2-layer models.
    # Independent threshold calibration per architecture is mandatory.
    val_ppls, _, _ = evaluate_split_perplexities(model, val_loader, device, autocast_dtype=autocast_dtype)
    mu  = sum(val_ppls) / len(val_ppls)
    sig = (sum((p - mu) ** 2 for p in val_ppls) / max(1, len(val_ppls) - 1)) ** 0.5
    tau = mu + 3 * sig
    val_ppl = mu
    logger.info(f"  Per-Model Calibration: μ={mu:.4f}  σ={sig:.4f}  τ={tau:.4f}")

    test_ppls, test_labels, _ = evaluate_split_perplexities(model, test_loader, device, autocast_dtype=autocast_dtype)
    preds = [1 if ppl > tau else 0 for ppl in test_ppls]
    m = calculate_classification_metrics(preds, test_labels)
    logger.info(f"  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    ckpt_path = os.path.join(ablation_dir, f"ckpt_depth_{n_layer}.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model_cfg.__dict__}, ckpt_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(3.0)  # Cooldown between runs

    return {
        "n_layer": n_layer,
        "n_params": n_params,
        "val_perplexity": val_ppl,
        "threshold_tau": tau,
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "accuracy": m["accuracy"],
        "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
    }


def main():
    parser = argparse.ArgumentParser(description="Model depth ablation study")
    parser.add_argument("--config", default="config/stage1_config.yaml")
    parser.add_argument("--output", default="data/ablations/ablation_depth.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--force", action="store_true", help="Force retraining.")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        try:
            with open(args.output, "r") as f:
                existing_data = json.load(f)
            if len(existing_data) == len(DEPTHS):
                logger.info(f"[IDEMPOTENCY] Depth ablation results already exist at {args.output}. Pass --force to override.")
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

    logger.info(f"Device: {device}")
    logger.info(f"Ablation conditions: n_layer ∈ {DEPTHS}")

    train_loader, tokenizer = get_dataloader(args.config, split="train")
    val_loader, _   = get_dataloader(args.config, split="val",   tokenizer=tokenizer)
    test_loader, _  = get_dataloader(args.config, split="test",  tokenizer=tokenizer)

    results = []
    for depth in DEPTHS:
        result = _run_depth_condition(
            n_layer=depth,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            cfg=cfg,
            device=device,
            autocast_dtype=autocast_dtype,
            ablation_dir=ablation_dir,
        )
        results.append(result)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print("ABLATION: Model Depth vs. Performance")
    print("=" * 72)
    print(f"{'L':>4}  {'Params':>12}  {'Val PPL':>9}  {'τ':>8}  {'F1':>7}  {'Prec':>7}  {'Recall':>7}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['n_layer']:>4}  {r['n_params']:>12,}  {r['val_perplexity']:>9.4f}  "
            f"{r['threshold_tau']:>8.4f}  {r['f1']:>7.4f}  "
            f"{r['precision']:>7.4f}  {r['recall']:>7.4f}"
        )
    print("=" * 72)
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
