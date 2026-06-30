"""Surprisal Threshold Calibration, Anomaly Classification, and Evaluation Telemetry.

Pedagogical explanations of why statistical boundary thresholds work, explicit tensor
dimension annotations [batch_size, seq_len], structured telemetry, and seed reproducibility.
"""

import os
import sys
import glob
import yaml
import math
import json
import random
import argparse
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from src.models.gpt2 import GPT2Config, GPT2Model
from src.dataset.data_loader import get_dataloader
from src.utils.metrics import (
    calculate_perplexity,
    calculate_classification_metrics,
    get_peak_vram,
    sweep_vram_footprint
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("surprisal.eval")


def set_seed(seed: int):
    """Enforces deterministic random seed setting across CPU and GPU threads for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global deterministic random seed enforced: {seed}")


def evaluate_split_perplexities(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: str, 
    autocast_dtype: torch.dtype = torch.bfloat16, 
    pad_token_id: int = 5
) -> tuple[list[float], list[int], list[str]]:
    """Evaluates per-sequence perplexity over valid log tokens across a dataset split.

    WHY: Trailing padding tokens distort mean loss. We compute unreduced token-level cross-entropy
    and mask out padding locations so that perplexity reflects true surprisal over valid log syntax.
    """
    model.eval()
    perplexities = []
    labels = []
    block_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating split", leave=False):
            # input_ids: [batch_size, seq_len]
            input_ids = batch["input_ids"].to(device)
            batch_labels = batch.get("label", torch.zeros(len(input_ids))).cpu().tolist()
            batch_block_ids = batch.get("block_id", [""] * len(input_ids))
            
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                logits, _ = model(inputs)
                # loss_per_token: [batch_size, seq_len - 1]
                loss_per_token = F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
                
                valid_mask = ~((inputs == pad_token_id) & (targets == pad_token_id))
                valid_counts = valid_mask.sum(dim=1).clamp(min=1)
                
                loss_per_seq = (loss_per_token * valid_mask.float()).sum(dim=1) / valid_counts
                
            for seq_loss in loss_per_seq.cpu().tolist():
                ppl = calculate_perplexity(seq_loss)
                perplexities.append(ppl)
                
            labels.extend(batch_labels)
            block_ids.extend(batch_block_ids)
            
    return perplexities, labels, block_ids


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Discovers the most recent pre-trained model checkpoint file on disk by step index."""
    if not os.path.exists(checkpoint_dir):
        return None
    pt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    if not pt_files:
        return None
        
    def extract_step(path: str) -> int:
        try:
            basename = os.path.basename(path)
            return int(basename.split("_")[1].split(".")[0])
        except ValueError:
            return -1
            
    pt_files.sort(key=extract_step, reverse=True)
    return pt_files[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate surprisal-gpt2 anomaly detection pipeline.")
    parser.add_argument("--config", default="config/stage1_config.yaml", help="Path to runtime configuration YAML.")
    parser.add_argument("--checkpoint", default=None, help="Explicit path to checkpoint file.")
    parser.add_argument("--results", default="data/stage1_eval_results.json", help="Destination path for evaluation JSON report.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--save_val_ppls", default=None, help="Optional path to save validation perplexities list as JSON.")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if results exist.")
    args = parser.parse_args()
    
    if os.path.exists(args.results) and (args.save_val_ppls is None or os.path.exists(args.save_val_ppls)) and not args.force:
        logger.info(f"[IDEMPOTENCY] Evaluation results already exist at {args.results}. Pass --force to override.")
        return

    set_seed(args.seed)
    
    logger.info(f"Loading evaluation runtime hyperparameters from: {args.config}")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        
    gpt_config = GPT2Config.from_dict(config_dict)
    train_cfg = config_dict.get("training", {})
    checkpoint_dir = train_cfg.get("checkpoint_dir", "data/checkpoints")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info(f"Hardware inference accelerator initialized: {device.upper()}")
    
    model = GPT2Model(gpt_config).to(device)
    checkpoint_path = args.checkpoint or find_latest_checkpoint(checkpoint_dir)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Restoring pre-trained model checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    else:
        logger.warning(f"No checkpoint discovered in {checkpoint_dir}. Evaluating uninitialized baseline weights.")
        
    logger.info("Instantiating validation holdout split for threshold calibration...")
    val_dataloader, tokenizer = get_dataloader(args.config, split="val")
    
    logger.info("Computing validation perplexity distribution...")
    val_ppls, _, _ = evaluate_split_perplexities(model, val_dataloader, device, autocast_dtype=autocast_dtype)
    
    if args.save_val_ppls and val_ppls:
        os.makedirs(os.path.dirname(args.save_val_ppls), exist_ok=True)
        with open(args.save_val_ppls, "w") as f:
            json.dump(val_ppls, f, indent=4)
        logger.info(f"Saved validation perplexities list to: {args.save_val_ppls}")
        
    if val_ppls:
        mean_ppl_val = sum(val_ppls) / len(val_ppls)
        variance = sum((x - mean_ppl_val) ** 2 for x in val_ppls) / len(val_ppls)
        std_ppl_val = math.sqrt(variance)
    else:
        mean_ppl_val, std_ppl_val = 0.0, 0.0
        
    # WHY: Setting threshold at mu + 3*sigma ensures bounds on false positives under Gaussian assumptions.
    tau = mean_ppl_val + (3 * std_ppl_val)
    logger.info("=== Threshold Calibration Summary ===")
    logger.info(f"Validation Perplexity Mean (mu)      : {mean_ppl_val:.4f}")
    logger.info(f"Validation Perplexity Std  (sigma)   : {std_ppl_val:.4f}")
    logger.info(f"Calibrated Anomaly Threshold (tau)   : {tau:.4f}")
    
    logger.info("Instantiating benchmark test split...")
    test_dataloader, _ = get_dataloader(args.config, split="test", tokenizer=tokenizer)
    
    logger.info("Evaluating test split sequence perplexities...")
    test_ppls, test_labels, _ = evaluate_split_perplexities(model, test_dataloader, device, autocast_dtype=autocast_dtype)
    
    predictions = [1 if ppl > tau else 0 for ppl in test_ppls]
    metrics = calculate_classification_metrics(predictions, test_labels)
    
    logger.info("=== Test Set Benchmark Evaluation Results ===")
    logger.info(f"Total Test Sequences Evaluated : {len(test_labels)}")
    logger.info(f"Accuracy                       : {metrics['accuracy']:.4f}")
    logger.info(f"Precision                      : {metrics['precision']:.4f}")
    logger.info(f"Recall                         : {metrics['recall']:.4f}")
    logger.info(f"F1 Score                       : {metrics['f1']:.4f}")
    
    logger.info("Executing VRAM resident memory scalability sweep...")
    vram_sweep = sweep_vram_footprint(
        model, 
        device=device, 
        seq_lengths=[128, 256, 512, 1024, 2048], 
        vocab_size=gpt_config.vocab_size
    )
    
    eval_report = {
        "calibration": {
            "mean_val_perplexity": mean_ppl_val,
            "std_val_perplexity": std_ppl_val,
            "threshold_tau": tau
        },
        "test_metrics": metrics,
        "vram_sweep_mb": vram_sweep,
        "checkpoint_evaluated": checkpoint_path or "initialized_weights"
    }
    
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(eval_report, f, indent=4)
    logger.info(f"Successfully serialized benchmark evaluation report to: {args.results}")


if __name__ == "__main__":
    main()
