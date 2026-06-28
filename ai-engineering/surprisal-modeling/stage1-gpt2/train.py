"""Autoregressive Pre-training Pipeline for Unsupervised Surprisal Log Anomaly Modeling.

This script orchestrates the end-to-end pre-training lifecycle of the custom GPT-2 Small baseline.
Integrates hardware-aware optimization patterns including: 2D/1D parameter weight decay separation,
cosine annealing learning rate schedules with linear warmup, bfloat16 mixed-precision autocasting,
Triton CUDA kernel fusion via torch.compile, micro-step gradient accumulation, gradient norm clipping,
and thermal dissipation sleep cycles to maintain GPU clock stability during prolonged training runs.
"""

import os
import sys
import yaml
import math
import json
import time
import logging
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from src.models.gpt2 import GPT2Config, GPT2Model
from src.dataset.data_loader import get_dataloader
from tqdm import tqdm

# Configure structured academic logging output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("surprisal.train")


def configure_optimizers(model: torch.nn.Module, weight_decay: float, learning_rate: float, betas: tuple[float, float]) -> torch.optim.AdamW:
    """Configures AdamW optimizer with strict parameter weight decay separation.

    Separates multi-dimensional weight matrices (linear projections, embedding tables) that require
    L2 weight decay regularisation from 1D vectors (biases, RMSNorm scale parameters) that should
    remain unpenalized. This prevents optimization landscape distortion.

    Args:
        model: Active PyTorch neural network module.
        weight_decay: L2 regularisation coefficient ($\lambda$).
        learning_rate: Peak maximum learning rate ($\eta_{\text{max}}$).
        betas: AdamW exponential decay coefficients ($\beta_1, \beta_2$).

    Returns:
        Configured AdamW optimizer instance with distinct parameter groups.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # 2D tensors (weights of linear projections and token embeddings) receive weight decay
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # 1D tensors (biases and normalization scale weights) bypass weight decay
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    logger.debug(f"Optimizer configuration: {len(decay_params)} decayed tensors | {len(nodecay_params)} unregularized 1D tensors")
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Computes cosine annealing learning rate schedule with linear warmup.

    Args:
        step: Current global optimization step index.
        warmup_steps: Initial linear warmup horizon to stabilize early gradients.
        max_steps: Total global pre-training step budget.
        max_lr: Peak learning rate post-warmup.
        min_lr: Asymptotic minimum learning rate threshold.

    Returns:
        Interpolated learning rate scalar for the current step.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate_validation(
    model: torch.nn.Module, 
    val_dataloader: torch.utils.data.DataLoader, 
    device: str, 
    autocast_dtype: torch.dtype = torch.bfloat16, 
    max_val_batches: int = 20
) -> tuple[float, float]:
    """Executes a rapid intermediate validation loop to monitor generalization loss and perplexity.

    Args:
        model: Active PyTorch neural network module.
        val_dataloader: Validation split data loader stream.
        device: Active hardware device string.
        autocast_dtype: Mixed-precision target data type.
        max_val_batches: Capped evaluation batch limit to minimize training interruptions.

    Returns:
        Tuple containing mean validation cross-entropy loss and exponentiated perplexity.
    """
    model.eval()
    val_loss_accum = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_val_batches:
                break
            input_ids = batch["input_ids"].to(device)
            # Causal shifting: predict target sequence shifted 1 position into the future
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


def main():
    """Main execution entrypoint for Stage 1 autoregressive pre-training."""
    config_path = "config/stage1_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    logger.info(f"Loading pre-training runtime hyperparameters from: {config_path}")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    # Instantiate architectural and training configs
    gpt_config = GPT2Config.from_dict(config_dict)
    train_cfg = config_dict.get("training", {})
    
    # Extract runtime hyperparameters
    max_lr = float(train_cfg.get("max_lr", 6.0e-4))
    min_lr = float(train_cfg.get("min_lr", 6.0e-5))
    warmup_steps = int(train_cfg.get("warmup_steps", 2000))
    max_steps = int(train_cfg.get("max_steps", 50000))
    weight_decay = float(train_cfg.get("weight_decay", 0.1))
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 4))
    clip_grad = float(train_cfg.get("clip_grad", 1.0))
    step_sleep_sec = float(train_cfg.get("step_sleep_sec", 0.10))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 5000))
    checkpoint_dir = train_cfg.get("checkpoint_dir", "data/checkpoints")
    results_path = train_cfg.get("results_path", "data/stage1_results.json")
    
    # Hardware accelerator discovery
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Hardware computing backend initialized: {device.upper()}")
    
    # Initialize data ingestion streams
    logger.info("Instantiating bin-packed HDFS datasets...")
    train_dataloader, _ = get_dataloader(config_path, split="train")
    val_dataloader, _ = get_dataloader(config_path, split="val")
    
    # Initialize transformer model
    logger.info("Constructing GPT-2 Small transformer backbone...")
    model = GPT2Model(gpt_config).to(device)
    
    # Select mixed-precision target based on hardware capabilities
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Hardware-fused Triton kernel compilation (Linux/PyTorch 2.0+ optimization)
    # Note: Triton inductor backend does not natively support Windows (win32).
    if device == "cuda" and sys.platform != "win32":
        logger.info("Executing CUDA graph compilation via torch.compile...")
        try:
            model = torch.compile(model)
            logger.info("Neural network successfully fused into Triton CUDA kernels.")
        except Exception as e:
            logger.warning(f"torch.compile compilation exception: {e}. Falling back to eager execution.")
    elif sys.platform == "win32":
        logger.info("Windows (win32) environment detected: Bypassing Triton torch.compile and running in native CUDA eager mode.")
        
    # Configure AdamW optimizer
    optimizer = configure_optimizers(model, weight_decay, max_lr, betas=(0.9, 0.95))
    
    # Initialize experimental tracking telemetry
    metrics = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "lr": [],
        "step_times": []
    }
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    logger.info(f"Initiating autoregressive pre-training loop for N={max_steps} steps...")
    step = 0
    epoch = 0
    data_iter = iter(train_dataloader)
    accumulated_loss = 0.0
    
    model.train()
    
    with tqdm(total=max_steps, desc="Pre-training GPT-2", unit="step", dynamic_ncols=True) as pbar:
        while step < max_steps:
            # Update learning rate per schedule
            lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            start_time = time.time()
            optimizer.zero_grad()
            
            # Accumulate gradients across micro-steps to simulate large effective batch size
            micro_loss_sum = 0.0
            for _ in range(grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    tqdm.write(f"Corpus stream exhausted. Initiating Epoch {epoch}...")
                    data_iter = iter(train_dataloader)
                    batch = next(data_iter)
                    
                input_ids = batch["input_ids"].to(device)
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                
                # Execute mixed-precision forward pass
                with torch.autocast(device_type=device, dtype=autocast_dtype):
                    _, loss = model(inputs, targets)
                    scaled_loss = loss / grad_accum_steps
                    
                scaled_loss.backward()
                micro_loss_sum += loss.item()
                
            # Clip gradient norms to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()
            
            # Thermal mitigation: pause between optimization steps to dissipate GPU core heat
            if step_sleep_sec > 0:
                time.sleep(step_sleep_sec)
            
            accumulated_loss += micro_loss_sum
            step_time = time.time() - start_time
            step += 1
            pbar.update(1)
            
            # Telemetry logging and periodic evaluation
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
                
                pbar.set_postfix(
                    loss=f"{avg_train_loss:.4f}",
                    val_ppl=f"{val_ppl:.2f}",
                    lr=f"{lr:.1e}"
                )
                
                with open(results_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                    
            # Periodic model checkpointing
            if step % checkpoint_interval == 0 or step == max_steps:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
                tqdm.write(f"Serializing model checkpoint to: {checkpoint_path}")
                
                # Extract uncompiled raw module weights
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                
                torch.save({
                    'step': step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'config': config_dict
                }, checkpoint_path)
                
    logger.info(f"=== Stage 1 Pre-Training Successfully Completed! Telemetry saved to {results_path} ===")


if __name__ == "__main__":
    main()
