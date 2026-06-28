"""Surprisal Threshold Calibration, Anomaly Classification, and Evaluation Telemetry.

This module executes inference evaluation over validation and test holdout splits.
Computes token-level cross-entropy loss over unmasked sequence activations (ignoring trailing
padding delimiters), calibrates extreme value anomaly thresholds ($\tau = \mu + 3\sigma$),
derives binary classification benchmark statistics (F1, Precision, Recall, Accuracy), and
executes GPU resident memory footprint sweeps across sequence length horizons.
"""

import os
import sys
import glob
import yaml
import math
import json
import argparse
import logging
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

# Configure structured academic logging output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("surprisal.eval")


def evaluate_split_perplexities(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: str, 
    autocast_dtype: torch.dtype = torch.bfloat16, 
    pad_token_id: int = 5
) -> tuple[list[float], list[int], list[str]]:
    """Evaluates per-sequence perplexity over valid log tokens across a dataset split.

    Computes unreduced cross-entropy loss per token, masks out trailing `<EOS>` padding
    tokens to avoid artificial loss dilution, and averages cross-entropy over active sequence lengths.

    Args:
        model: Active PyTorch neural network module.
        dataloader: Dataset split loader stream.
        device: Active hardware accelerator string.
        autocast_dtype: Target mixed-precision data type.
        pad_token_id: Integer token identifier for `<EOS>` delimiter.

    Returns:
        Tuple containing:
            - `perplexities`: List of computed per-sequence perplexity floating-point values.
            - `labels`: List of ground-truth binary block anomaly labels.
            - `block_ids`: List of HDFS block identifier provenance strings.
    """
    model.eval()
    perplexities = []
    labels = []
    block_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating split", leave=False):
            input_ids = batch["input_ids"].to(device)
            batch_labels = batch.get("label", torch.zeros(len(input_ids))).cpu().tolist()
            batch_block_ids = batch.get("block_id", [""] * len(input_ids))
            
            # Causal shifting
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                logits, _ = model(inputs)
                # Compute unreduced token-level cross entropy
                loss_per_token = F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
                
                # Mask out trailing padding blocks where both input and target match pad delimiter
                valid_mask = ~((inputs == pad_token_id) & (targets == pad_token_id))
                valid_counts = valid_mask.sum(dim=1).clamp(min=1)
                
                # Normalize sequence loss strictly over valid active log tokens
                loss_per_seq = (loss_per_token * valid_mask.float()).sum(dim=1) / valid_counts
                
            for seq_loss in loss_per_seq.cpu().tolist():
                ppl = calculate_perplexity(seq_loss)
                perplexities.append(ppl)
                
            labels.extend(batch_labels)
            block_ids.extend(batch_block_ids)
            
    return perplexities, labels, block_ids


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Discovers the most recent pre-trained model checkpoint file on disk by step index.

    Args:
        checkpoint_dir: Filesystem directory containing checkpoint_*.pt artifacts.

    Returns:
        Absolute filesystem path to the latest checkpoint, or None if directory is empty.
    """
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
    """Main execution entrypoint for surprisal threshold calibration and test split evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate surprisal-gpt2 anomaly detection pipeline.")
    parser.add_argument("--config", default="config/stage1_config.yaml", help="Path to runtime configuration YAML.")
    parser.add_argument("--checkpoint", default=None, help="Explicit path to checkpoint file. If None, auto-discovers latest.")
    parser.add_argument("--results", default="data/stage1_eval_results.json", help="Destination path for evaluation JSON report.")
    args = parser.parse_args()
    
    logger.info(f"Loading evaluation runtime hyperparameters from: {args.config}")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        
    gpt_config = GPT2Config.from_dict(config_dict)
    train_cfg = config_dict.get("training", {})
    checkpoint_dir = train_cfg.get("checkpoint_dir", "data/checkpoints")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info(f"Hardware inference accelerator initialized: {device.upper()}")
    
    # 1. Load trained model weights
    model = GPT2Model(gpt_config).to(device)
    checkpoint_path = args.checkpoint or find_latest_checkpoint(checkpoint_dir)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Restoring pre-trained model checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Strip torch.compile graph wrapper prefixes if present
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    else:
        logger.warning(f"No checkpoint discovered in {checkpoint_dir}. Evaluating uninitialized baseline weights.")
        
    # 2. Calibrate extreme value surprisal threshold over normal validation holdouts
    logger.info("Instantiating validation holdout split for threshold calibration...")
    val_dataloader, tokenizer = get_dataloader(args.config, split="val")
    
    logger.info("Computing validation perplexity distribution...")
    val_ppls, _, _ = evaluate_split_perplexities(model, val_dataloader, device, autocast_dtype=autocast_dtype)
    
    if val_ppls:
        mean_ppl_val = sum(val_ppls) / len(val_ppls)
        variance = sum((x - mean_ppl_val) ** 2 for x in val_ppls) / len(val_ppls)
        std_ppl_val = math.sqrt(variance)
    else:
        mean_ppl_val, std_ppl_val = 0.0, 0.0
        
    # Extreme value surprisal threshold calibration: \tau = \mu + 3\sigma
    tau = mean_ppl_val + (3 * std_ppl_val)
    logger.info("=== Threshold Calibration Summary ===")
    logger.info(f"Validation Perplexity Mean (mu)      : {mean_ppl_val:.4f}")
    logger.info(f"Validation Perplexity Std  (sigma)   : {std_ppl_val:.4f}")
    logger.info(f"Calibrated Anomaly Threshold (tau)   : {tau:.4f}")
    
    # 3. Evaluate test split anomaly classification performance
    logger.info("Instantiating benchmark test split (combining normal holdouts and anomaly traces)...")
    test_dataloader, _ = get_dataloader(args.config, split="test", tokenizer=tokenizer)
    
    logger.info("Evaluating test split sequence perplexities...")
    test_ppls, test_labels, _ = evaluate_split_perplexities(model, test_dataloader, device, autocast_dtype=autocast_dtype)
    
    # Classify sequence as anomalous if empirical perplexity exceeds calibrated threshold tau
    predictions = [1 if ppl > tau else 0 for ppl in test_ppls]
    metrics = calculate_classification_metrics(predictions, test_labels)
    
    logger.info("=== Test Set Benchmark Evaluation Results ===")
    logger.info(f"Total Test Sequences Evaluated : {len(test_labels)}")
    logger.info(f"Accuracy                       : {metrics['accuracy']:.4f}")
    logger.info(f"Precision                      : {metrics['precision']:.4f}")
    logger.info(f"Recall                         : {metrics['recall']:.4f}")
    logger.info(f"F1 Score                       : {metrics['f1']:.4f}")
    logger.info(
        f"Confusion Matrix               : TP={metrics['tp']} | FP={metrics['fp']} | "
        f"TN={metrics['tn']} | FN={metrics['fn']}"
    )
    
    # 4. Benchmark hardware resident memory scalability across sequence horizons
    logger.info("Executing VRAM resident memory scalability sweep (T=128 to 2048)...")
    vram_sweep = sweep_vram_footprint(
        model, 
        device=device, 
        seq_lengths=[128, 256, 512, 1024, 2048], 
        vocab_size=gpt_config.vocab_size
    )
    logger.info(f"Empirical VRAM Footprint Sweep Results (MB): {vram_sweep}")
    
    # Assemble comprehensive evaluation report
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
