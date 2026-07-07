"""QLoRA Supervised Fine-Tuning Engine for Qwen-2.5-3B.

Orchestrates 4-bit NF4 quantization loading, first-principles LoRA adapter injection,
prompt loss masking (`target = -100`), gradient accumulation, gradient checkpointing, and
memory-efficient checkpoint serialization (saving only ~21M adapter parameters).
"""

import os
import sys
import yaml
import math
import time
import random
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

# Polyfill set_submodule for PyTorch < 2.5.0 compatibility with latest transformers/bitsandbytes
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        if target == "":
            raise ValueError("Cannot set the root module")
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self.get_submodule(".".join(atoms)) if atoms else self
        setattr(mod, name, module)
    nn.Module.set_submodule = _set_submodule
from torch.optim import AdamW

try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        get_cosine_schedule_with_warmup
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from src.models.lora import inject_lora_adapters, count_trainable_parameters
from src.dataset.data_loader import get_sft_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("stage2.finetune")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed enforced: {seed}")


def prepare_model_for_qlora_training(model):
    """Prepares a 4-bit NF4 quantized model for QLoRA training from first principles.

    WHY:
    1. Freezes all base model parameters so only LoRA adapters receive gradients.
    2. Keeps all layers (norms, embeddings, lm_head) in native bfloat16. On Ampere GPUs
       (RTX 3060 Ti) under Windows WSL2 WDDM, casting the massive 152,064-token vocabulary
       head to float32 forces non-Tensor Core fp32 GEMM and heavy memory allocation that
       triggers WDDM driver timeouts (`CUDA driver error: device not ready`).
    3. Keeps gradient checkpointing disabled: without gradient checkpointing, a 3B NF4 model
       with bs=1, seq=512 fits comfortably in 8GB VRAM and avoids async BitsAndBytes backward
       kernel recomputation faults on WSL2.
    """
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Model prepared for QLoRA: base frozen, native bfloat16 preserved, grad checkpointing OFF.")
    return model


class MockModel(nn.Module):
    """Simulated model for fast pipeline verification and CPU/mock execution."""
    def __init__(self, vocab_size=10240, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        x = torch.relu(self.q_proj(x) + self.v_proj(x))
        logits = self.head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        return type("CausalLMOutput", (), {"loss": loss, "logits": logits})()


class MockTokenizer:
    """Mock tokenizer for offline pipeline validation."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def __call__(self, text, truncation=True, max_length=1024, padding=False, add_special_tokens=False):
        words = text.split()
        ids = [abs(hash(w)) % 10000 + 10 for w in words][:max_length]
        if padding == "max_length":
            ids = ids + [self.pad_token_id] * max(0, max_length - len(ids))
        mask = [1 if i != self.pad_token_id else 0 for i in ids]
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=True):
        return '{"root_cause": "Mock DataNode timeout diagnosis", "severity": "P1_CRITICAL", "affected_component": "DataNode", "mitigation_commands": ["sudo systemctl restart hdfs-datanode"], "confidence": 0.90, "is_anomaly": true}'


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Engine.")
    parser.add_argument("--config", default="config/stage2_config.yaml", help="Path to YAML config.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: overfit single micro-batch.")
    parser.add_argument("--mock", action="store_true", help="Mock mode for CPU verification.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps.")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint in checkpoint_dir if available.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to specific checkpoint .pt file to resume from.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    model_cfg = config.get("base_model", {})
    quant_cfg = config.get("quantization", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() and not args.mock else "cpu"
    logger.info(f"Execution accelerator initialized: {device.upper()}")

    if not HAS_TRANSFORMERS or args.mock or device == "cpu":
        logger.warning("Transformers/CUDA unavailable or --mock specified. Initializing Mock model & tokenizer for verification.")
        tokenizer = MockTokenizer()
        model = MockModel().to(device)
    else:
        logger.info(f"Loading tokenizer: {model_cfg['name']}...")
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True)
        )
        logger.info(f"Loading 4-bit NF4 quantized base model: {model_cfg['name']}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_qlora_training(model)

    # Inject first-principles LoRA adapters
    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
    rank = lora_cfg.get("rank", 16)
    alpha = lora_cfg.get("alpha", 32.0)
    
    inject_lora_adapters(model, target_modules=target_modules, rank=rank, alpha=alpha)
    trainable, total, pct = count_trainable_parameters(model)
    logger.info(f"Parameter budget: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")

    ckpt_dir = train_cfg.get("checkpoint_dir", "data/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_step = 0

    resume_path = args.resume_from
    if args.resume and not resume_path:
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("adapter_step_") and f.endswith(".pt")]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.replace("adapter_step_", "").replace(".pt", "")))
                resume_path = os.path.join(ckpt_dir, ckpts[-1])

    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming training from checkpoint: {resume_path}...")
        try:
            ckpt_data = torch.load(resume_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt_data = torch.load(resume_path, map_location="cpu")
        if "adapter_state_dict" in ckpt_data:
            model.load_state_dict(ckpt_data["adapter_state_dict"], strict=False)
            start_step = ckpt_data.get("step", 0)
            logger.info(f"Successfully loaded adapter weights from step {start_step}.")
        else:
            model.load_state_dict(ckpt_data, strict=False)
            logger.info("Successfully loaded raw adapter state dict.")

    train_dataloader = get_sft_dataloader(args.config, split="train", tokenizer=tokenizer)
    
    lr = float(train_cfg.get("max_lr", 2e-4))
    # Why foreach=False and fused=False: In PyTorch 2.x on Windows/WSL2 consumer GPUs (RTX 3060 Ti),
    # multi-tensor foreach AdamW (_foreach_sqrt) launches concurrent kernels across 500+ LoRA tensors
    # simultaneously, which overflows the WSL2 driver stream queue and triggers "CUDA driver error: device not ready".
    # Setting foreach=False forces sequential single-tensor updates, ensuring 100% driver stability and lower peak VRAM.
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
        foreach=False,
        fused=False
    )
    
    max_steps = args.max_steps or train_cfg.get("max_steps", 2000)
    warmup_steps = train_cfg.get("warmup_steps", 100)
    if args.debug:
        max_steps = 50
        warmup_steps = 5
        logger.info("[DEBUG MODE] Overfitting single batch for 50 steps.")

    # Cosine LR scheduler with linear warmup — critical for training stability
    # at high LR (2e-4) on large frozen base models
    if HAS_TRANSFORMERS and not args.mock:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps
        )
    else:
        scheduler = None

    accum_steps = train_cfg.get("gradient_accumulation_steps", 8)
    
    model.train()
    optimizer_step = start_step      # counts actual optimizer (gradient) steps
    micro_batch_count = start_step * accum_steps   # counts total forward passes (micro-batches)
    running_loss = 0.0
    start_time = time.time()

    debug_batch = next(iter(train_dataloader)) if args.debug else None

    with tqdm(initial=start_step, total=max_steps, desc="Training QLoRA adapter") as pbar:
        while optimizer_step < max_steps:
            for batch in train_dataloader:
                if batch is None:
                    # Entire batch was filtered out (all records had prompts longer than max_seq_len)
                    continue

                if args.debug:
                    batch = debug_batch

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accum_steps
                
                # NaN guard: with CUDA_LAUNCH_BLOCKING=1 all ops are synchronous so this check
                # is reliable. Abort training immediately on NaN — don't silently skip batches.
                if not torch.isfinite(loss):
                    logger.error(
                        f"Non-finite loss={loss.item():.6f} at optimizer_step={optimizer_step}, "
                        f"micro_batch={micro_batch_count}. Input shape: {input_ids.shape}. "
                        f"Labels unique: {labels.unique().tolist()}. Aborting."
                    )
                    raise RuntimeError(f"Training aborted: non-finite loss={loss.item()} at step {optimizer_step}.")
                
                loss.backward()

                running_loss += loss.item() * accum_steps
                micro_batch_count += 1

                # Thermal relief micro-sleep after every backward pass to prevent continuous 100% GPU duty cycle
                # and keep GPU temperatures safely below thermal throttling limits (84°C) on consumer GPUs.
                time.sleep(0.01)

                # Optimizer step fires every accum_steps micro-batches
                if micro_batch_count % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("gradient_clipping", 1.0), foreach=False)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    
                    # Thermal relief micro-sleep to prevent GPU overheating on consumer cards (RTX 3060 Ti)
                    step_sleep_sec = float(train_cfg.get("step_sleep_sec", 0.10))
                    if step_sleep_sec > 0:
                        time.sleep(step_sleep_sec)
                        
                    optimizer_step += 1
                    pbar.update(1)
                    current_lr = scheduler.get_last_lr()[0] if scheduler else lr
                    pbar.set_postfix({"loss": f"{running_loss / max(1, micro_batch_count):.4f}", "lr": f"{current_lr:.2e}"})

                    # Periodic checkpoint saving
                    ckpt_interval = train_cfg.get("checkpoint_interval", 100)
                    if ckpt_interval > 0 and optimizer_step % ckpt_interval == 0 and optimizer_step < max_steps:
                        interim_path = os.path.join(ckpt_dir, f"adapter_step_{optimizer_step}.pt")
                        adapter_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
                        torch.save({"adapter_state_dict": adapter_state, "step": optimizer_step, "rank": rank, "alpha": alpha}, interim_path)
                        logger.info(f"Saved periodic checkpoint at step {optimizer_step} -> {interim_path}")

                if optimizer_step >= max_steps:
                    break

    ckpt_dir = train_cfg.get("checkpoint_dir", "data/checkpoints")
    final_path = os.path.join(ckpt_dir, f"adapter_step_{max_steps}.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    adapter_state = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({"adapter_state_dict": adapter_state, "step": max_steps, "rank": rank, "alpha": alpha}, final_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Successfully serialized LoRA adapter checkpoint ({len(adapter_state)} tensors) -> {final_path}")
    logger.info(f"Total training duration: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
