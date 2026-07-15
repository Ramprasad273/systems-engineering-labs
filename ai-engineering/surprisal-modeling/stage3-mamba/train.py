"""Stage 3 Pre-Training Loop for Mamba S6 & MambaLog Architectures.

Pedagogical engineering following Karpathy clean code guidelines:
- Strict AdamW parameter decay separation (regularizing 2D weight projections while keeping 1D biases and RMSNorm scaling vectors unpenalized).
- Cosine annealing learning rate schedule with linear warmup (`warmup_steps=1000`).
- Multi-seed pre-training runner (`seeds: [42, 123, 999]`) ensuring strict statistical reproducibility.
- Automated validation perplexity tracking and fault-tolerant checkpointing.
"""

import os
import sys
import math
import json
import time
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
from src.utils.metrics import calculate_perplexity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("stage3.train")


def set_seed(seed: int):
    """Enforces deterministic seed state across CPU and GPU threads for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Deterministic seed enforced: {seed}")


def configure_optimizers(
    model: torch.nn.Module, weight_decay: float, learning_rate: float, betas: tuple[float, float] = (0.9, 0.95)
) -> torch.optim.AdamW:
    """Configures AdamW optimizer with strict parameter decay separation.

    WHY: 2D weight matrices (projections, embeddings) benefit from L2 shrinkage to prevent overfitting.
    1D parameters (biases, normalization scaling vectors, log-scale A/D parameters) regulate layer shifts;
    penalizing them distorts representation dynamics without reducing model capacity.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    logger.debug(
        f"Optimizer configured: {len(decay_params)} decayed tensors | {len(nodecay_params)} unpenalized 1D tensors"
    )
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Computes cosine annealing learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate_validation(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    device: str,
    autocast_dtype: torch.dtype = torch.bfloat16,
    max_val_batches: int = 40,
) -> tuple[float, float]:
    """Executes intermediate validation evaluation loop over healthy HDFS log sequences."""
    model.eval()
    val_loss_accum = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_val_batches:
                break
            input_ids = batch["input_ids"].to(device)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            if torch.cuda.is_available() and device != "cpu":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    _, loss = model(inputs, targets=targets)
            else:
                _, loss = model(inputs, targets=targets)

            val_loss_accum += loss.item()
            num_batches += 1

    mean_loss = val_loss_accum / max(num_batches, 1)
    val_ppl = calculate_perplexity(mean_loss)
    model.train()
    return mean_loss, val_ppl


def train_single_seed(
    config: dict,
    config_path: str,
    seed: int,
    device: str,
    model_type: str = "mamba",
) -> dict[str, float | str]:
    """Pre-trains a single model instance from scratch under a specific seed.

    Args:
        config: Loaded configuration dictionary.
        config_path: Filesystem path to active configuration YAML.
        seed: Random seed integer (`42`, `123`, or `999`).
        device: Active device (`cuda` or `cpu`).
        model_type: Model architecture (`mamba` or `mambalog`).

    Returns:
        Summary metrics dictionary recording `seed`, `final_val_loss`, `final_val_ppl`, and `checkpoint_path`.
    """
    set_seed(seed)
    train_cfg = config["training"]
    model_cfg = config["model"]

    # Instantiate model architecture matching requested capacity
    if model_type.lower() == "mambalog":
        logger.info(f"[Seed {seed}] Initializing ~125M Hybrid MambaLog (18 Mamba + 6 Attention blocks)...")
        model = MambaLogLMHeadModel(config)
    else:
        logger.info(f"[Seed {seed}] Initializing ~125M Mamba S6 Architecture (24 blocks)...")
        model = MambaLMHeadModel(config)

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Seed {seed}] Total trainable capacity: {total_params:,} parameters ({total_params / 1e6:.2f}M)")

    # Load dataloaders
    train_loader, _ = get_dataloader(config_path, split="train")
    val_loader, _ = get_dataloader(config_path, split="val")

    optimizer = configure_optimizers(
        model=model,
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        learning_rate=float(train_cfg.get("max_lr", 6e-4)),
    )

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_steps = int(train_cfg.get("max_steps", 10000))
    warmup_steps = int(train_cfg.get("warmup_steps", 1000))
    max_lr = float(train_cfg.get("max_lr", 6e-4))
    min_lr = float(train_cfg.get("min_lr", 6e-5))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 4))
    clip_grad = float(train_cfg.get("clip_grad", 1.0))
    step_sleep_sec = float(train_cfg.get("step_sleep_sec", 0.005))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 2000))

    checkpoint_dir = train_cfg.get("checkpoint_dir", f"data/checkpoints_{model_type}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    step = 0
    train_iterator = iter(train_loader)
    model.train()
    start_time = time.time()

    logger.info(f"[Seed {seed}] Commencing pre-training loop for {max_steps} optimization steps...")

    pbar = tqdm(total=max_steps, desc=f"Training {model_type} (Seed {seed})")
    while step < max_steps:
        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(grad_accum):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            input_ids = batch["input_ids"].to(device)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            if torch.cuda.is_available() and device != "cpu":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    _, loss = model(inputs, targets=targets)
            else:
                _, loss = model(inputs, targets=targets)

            loss = loss / grad_accum
            loss_accum += loss.item()
            loss.backward()

        if clip_grad > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        if step_sleep_sec > 0.0:
            time.sleep(step_sleep_sec)
        step += 1
        pbar.update(1)

        if step % 25 == 0 or step == 1:
            pbar.set_postfix({"loss": f"{loss_accum * grad_accum:.4f}", "lr": f"{lr:.2e}"})

        if step % checkpoint_interval == 0 or step == max_steps:
            val_loss, val_ppl = evaluate_validation(model, val_loader, device, autocast_dtype)
            logger.info(
                f"[Seed {seed} | Step {step}/{max_steps}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.4f}"
            )
            ckpt_name = f"{model_type}_seed_{seed}_step_{step}.pt"
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            torch.save(
                {
                    "step": step,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "config": config,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

    pbar.close()
    final_val_loss, final_val_ppl = evaluate_validation(model, val_loader, device, autocast_dtype)
    final_ckpt_path = os.path.join(checkpoint_dir, f"{model_type}_seed_{seed}_final.pt")
    torch.save(
        {
            "step": step,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "val_loss": final_val_loss,
            "val_ppl": final_val_ppl,
            "config": config,
        },
        final_ckpt_path,
    )
    logger.info(f"[Seed {seed}] Pre-training complete. Final Val PPL: {final_val_ppl:.4f}")

    return {
        "seed": seed,
        "final_val_loss": round(final_val_loss, 4),
        "final_val_ppl": round(final_val_ppl, 4),
        "checkpoint_path": final_ckpt_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-train Stage 3 Mamba or MambaLog models across seeds.")
    parser.add_argument("--config", type=str, default="config/stage3_config.yaml", help="Path to config YAML.")
    parser.add_argument("--model_type", type=str, choices=["mamba", "mambalog"], default="mamba", help="Model type.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds to evaluate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Path to save multi-seed training summary JSON.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results = {}
    for seed in args.seeds:
        run_res = train_single_seed(
            config=config,
            config_path=args.config,
            seed=seed,
            device=args.device,
            model_type=args.model_type,
        )
        results[f"seed_{seed}"] = run_res

    output_path = args.output or config["training"].get("results_path", f"data/{args.model_type}_training_summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"All seed training runs complete. Summary written to: {output_path}")


if __name__ == "__main__":
    main()
