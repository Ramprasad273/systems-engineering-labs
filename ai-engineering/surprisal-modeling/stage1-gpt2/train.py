"""Autoregressive Pre-training Pipeline for Unsupervised Surprisal Log Anomaly Modeling.

Details architectural choices, explicit tensor dimension
annotations [batch_size, seq_len, dim], structured telemetry, and robust fault-tolerant smart checkpointing.
"""

import os
import sys
import yaml
import math
import json
import time
import glob
import random
import logging
import argparse
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from src.models.gpt2 import GPT2Config, GPT2Model
from src.dataset.data_loader import get_dataloader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("surprisal.train")


def set_seed(seed: int):
    """Enforces deterministic random seed setting across CPU and GPU threads for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global deterministic random seed enforced: {seed}")


def configure_optimizers(model: torch.nn.Module, weight_decay: float, learning_rate: float, betas: tuple[float, float]) -> torch.optim.AdamW:
    """Configures AdamW optimizer with strict parameter weight decay separation.

    WHY: 2D weight matrices (projections, embeddings) benefit from L2 shrinkage to prevent overfitting.
    1D parameters (biases, normalization scaling vectors) regulate layer shifts; penalizing them distorts
    representation dynamics without reducing capacity.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    logger.debug(f"Optimizer setup: {len(decay_params)} decayed tensors | {len(nodecay_params)} unregularized 1D tensors")
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Computes cosine annealing learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate_validation(
    model: torch.nn.Module, 
    val_dataloader: torch.utils.data.DataLoader, 
    device: str, 
    autocast_dtype: torch.dtype = torch.bfloat16, 
    max_val_batches: int = 20
) -> tuple[float, float]:
    """Executes intermediate validation evaluation loop over masked log sequences."""
    model.eval()
    val_loss_accum = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_val_batches:
                break
            # input_ids: [batch_size, seq_len]
            input_ids = batch["input_ids"].to(device)
            # Predict next token autoregressively: inputs=[B, T-1], targets=[B, T-1]
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(inputs, targets)
            
            val_loss_accum += loss.item()
            num_batches += 1
            
    model.train()
    if num_batches == 0:
        return 0.0, 1.0
    mean_val_loss = val_loss_accum / num_batches
    try:
        val_perplexity = math.exp(mean_val_loss)
    except OverflowError:
        val_perplexity = float('inf')
    return mean_val_loss, val_perplexity


def load_latest_checkpoint(checkpoint_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, dict]:
    """Scans checkpoint directory for existing weights and resumes training state (Smart Checkpointing)."""
    if not os.path.exists(checkpoint_dir):
        return 0, {"steps": [], "train_loss": [], "val_loss": [], "val_perplexity": [], "lr": [], "step_times": []}
        
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    if not checkpoints:
        return 0, {"steps": [], "train_loss": [], "val_loss": [], "val_perplexity": [], "lr": [], "step_times": []}
        
    # Sort by step integer extracted from filename
    def extract_step(path):
        try:
            base = os.path.basename(path)
            return int(base.split("_")[1].split(".")[0])
        except Exception:
            return -1
            
    latest_ckpt = max(checkpoints, key=extract_step)
    logger.info(f"[SMART CHECKPOINTING] Resuming training from latest checkpoint: {latest_ckpt}")
    
    checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint.get('step', 0)
    metrics = checkpoint.get('metrics', {"steps": [], "train_loss": [], "val_loss": [], "val_perplexity": [], "lr": [], "step_times": []})
    logger.info(f"[SMART CHECKPOINTING] Successfully restored model weights and optimizer state at step {step}")
    return step, metrics


def main():
    parser = argparse.ArgumentParser(description="Autoregressive pre-training for surprisal log models.")
    parser.add_argument("--config", default="config/stage1_config.yaml", help="Path to config YAML.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps.")
    parser.add_argument("--packing_strategy", default=None, help="Override sequence packing strategy.")
    parser.add_argument("--rotary", action="store_true", help="Enable RoPE positional embeddings.")
    parser.add_argument("--swiglu", action="store_true", help="Enable SwiGLU activations.")
    parser.add_argument("--checkpoint_dir", default=None, help="Override checkpoint save directory.")
    parser.add_argument("--results_path", default=None, help="Override results JSON path.")
    args = parser.parse_args()

    set_seed(args.seed)
    
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        
    # Apply CLI overrides for ablation studies
    if args.packing_strategy:
        config_dict.setdefault("dataset", {})["packing_strategy"] = args.packing_strategy
    if args.rotary:
        config_dict.setdefault("model", {})["use_rotary"] = True
    if args.swiglu:
        config_dict.setdefault("model", {})["use_swiglu"] = True
        
    gpt_config = GPT2Config.from_dict(config_dict)
    train_cfg = config_dict.get("training", {})
    
    max_lr = float(train_cfg.get("max_lr", 6.0e-4))
    min_lr = float(train_cfg.get("min_lr", 6.0e-5))
    warmup_steps = int(train_cfg.get("warmup_steps", 2000))
    max_steps = args.max_steps if args.max_steps is not None else int(train_cfg.get("max_steps", 50000))
    weight_decay = float(train_cfg.get("weight_decay", 0.1))
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 4))
    clip_grad = float(train_cfg.get("clip_grad", 1.0))
    step_sleep_sec = float(train_cfg.get("step_sleep_sec", 0.10))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 5000))
    checkpoint_dir = args.checkpoint_dir or train_cfg.get("checkpoint_dir", "data/checkpoints")
    results_path = args.results_path or train_cfg.get("results_path", "data/stage1_results.json")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Hardware compute backend: {device.upper()}")
    
    train_dataloader, _ = get_dataloader(args.config, split="train")
    val_dataloader, _ = get_dataloader(args.config, split="val")
    
    model = GPT2Model(gpt_config).to(device)
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    if device == "cuda" and sys.platform != "win32":
        try:
            model = torch.compile(model)
            logger.info("Neural network successfully compiled via Triton.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Running eager mode.")
            
    optimizer = configure_optimizers(model, weight_decay, max_lr, betas=(0.9, 0.95))
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Auto-resume via Smart Checkpointing
    step, metrics = load_latest_checkpoint(checkpoint_dir, model, optimizer)
    
    if step >= max_steps:
        logger.info(f"[IDEMPOTENCY] Training already completed at step {step}/{max_steps}. Results preserved at {results_path}.")
        return

    epoch = 0
    data_iter = iter(train_dataloader)
    accumulated_loss = 0.0
    model.train()
    
    with tqdm(initial=step, total=max_steps, desc="Pre-training GPT-2", unit="step", dynamic_ncols=True) as pbar:
        while step < max_steps:
            lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            start_time = time.time()
            optimizer.zero_grad()
            micro_loss_sum = 0.0
            
            for _ in range(grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    data_iter = iter(train_dataloader)
                    batch = next(data_iter)
                    
                input_ids = batch["input_ids"].to(device)
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                
                with torch.autocast(device_type=device, dtype=autocast_dtype):
                    _, loss = model(inputs, targets)
                    scaled_loss = loss / grad_accum_steps
                    
                scaled_loss.backward()
                micro_loss_sum += loss.item()
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()
            
            if step_sleep_sec > 0:
                time.sleep(step_sleep_sec)
                
            accumulated_loss += micro_loss_sum
            step_time = time.time() - start_time
            step += 1
            pbar.update(1)
            
            if step % 100 == 0 or step == 1 or step == max_steps:
                avg_train_loss = accumulated_loss / (100 if step > 1 else 1)
                accumulated_loss = 0.0
                val_loss, val_ppl = evaluate_validation(model, val_dataloader, device, autocast_dtype=autocast_dtype)
                
                metrics["steps"].append(step)
                metrics["train_loss"].append(avg_train_loss)
                metrics["val_loss"].append(val_loss)
                metrics["val_perplexity"].append(val_ppl)
                metrics["lr"].append(lr)
                metrics["step_times"].append(step_time)
                
                pbar.set_postfix(loss=f"{avg_train_loss:.4f}", val_ppl=f"{val_ppl:.2f}", lr=f"{lr:.1e}")
                
                with open(results_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                    
            if step % checkpoint_interval == 0 or step == max_steps:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    'step': step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'config': config_dict
                }, checkpoint_path)
                
    logger.info(f"=== Stage 1 Pre-Training Completed! Telemetry saved to {results_path} ===")


if __name__ == "__main__":
    main()
