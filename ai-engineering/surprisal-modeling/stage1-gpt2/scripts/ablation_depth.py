"""Model Depth Ablation Study.

Trains models with varying numbers of transformer layers (L ∈ {2, 4, 8, 12})
while holding all other hyperparameters constant, to determine whether GPT-2
Small's full 12-layer depth is necessary for the HDFS log anomaly detection task
or whether a shallower model suffices.

Ablation conditions
-------------------
L ∈ {2, 4, 8, 12}  (n_layer)
d_model = 768, n_head = 12, d_ff = 2048  (constant)

For each depth:
  1. Trains on the same packed corpus with V=5,000 tokenizer.
  2. Trains for 5,000 steps at full batch size.
  3. Calibrates τ = μ + 3σ on the validation split.
  4. Evaluates test F1 and records parameter count.
  5. Records results to data/ablations/ablation_depth.json.

Usage
-----
    python scripts/ablation_depth.py --config config/stage1_config.yaml

Notes
-----
- Expected runtime: ~4–8 hours on an RTX 3060 Ti (8 GB VRAM).
- The trained tokenizer from the main experiment is reused (V=5,000).
  If data/tokenizer/log_tokenizer.json does not exist, run train.py first
  to complete the tokenization phase.
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.data_loader import get_dataloader
from src.models.gpt2 import GPT2Config, GPT2Model
from src.utils.metrics import calculate_perplexity, calculate_classification_metrics
from train import get_lr, configure_optimizers

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation hyperparameters
# ---------------------------------------------------------------------------

DEPTHS         = [2, 4, 8, 12]
ABLATION_STEPS = 5_000
WARMUP_STEPS   = 500
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
    logger.info(f"Ablation: n_layer = {n_layer}")
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
            x = batch["input_ids"].to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, x)
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        if step % 500 == 0 or step == ABLATION_STEPS - 1:
            logger.info(
                f"  step {step:5d}/{ABLATION_STEPS} | "
                f"loss {accum_loss/GRAD_ACCUM:.4f} | {time.time()-t0:.1f}s"
            )

    # Calibration & Evaluation via evaluate.py standard loop
    from evaluate import evaluate_split_perplexities
    val_ppls, _, _ = evaluate_split_perplexities(model, val_loader, device, autocast_dtype=autocast_dtype)
    mu  = sum(val_ppls) / len(val_ppls)
    sig = (sum((p - mu) ** 2 for p in val_ppls) / max(1, len(val_ppls) - 1)) ** 0.5
    tau = mu + 3 * sig
    val_ppl = mu
    logger.info(f"  Calibration: μ={mu:.4f}  σ={sig:.4f}  τ={tau:.4f}")

    test_ppls, test_labels, _ = evaluate_split_perplexities(model, test_loader, device, autocast_dtype=autocast_dtype)
    preds = [1 if ppl > tau else 0 for ppl in test_ppls]
    m = calculate_classification_metrics(preds, test_labels)
    logger.info(f"  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    # Save checkpoint for this depth
    ckpt_path = os.path.join(ablation_dir, f"ckpt_depth_{n_layer}.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model_cfg.__dict__}, ckpt_path)

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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ablation_dir = os.path.dirname(args.output)
    os.makedirs(ablation_dir, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Ablation conditions: n_layer ∈ {DEPTHS}")

    # Reuse pre-built packed tensors from the main training run
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

    # Summary table
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
