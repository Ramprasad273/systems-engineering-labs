#!/usr/bin/env python3
"""Surprisal Heatmaps Generation (B6).

Pedagogical explanations of why token-level surprisal pinpoints error localization,
explicit tensor shapes [1, seq_len], and idempotency checks.
"""

import argparse
import json
import logging
import os
import sys
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.gpt2 import GPT2Config, GPT2Model
from src.tokenizer.log_tokenizer import LogTokenizer
from evaluate import find_latest_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("surprisal.b6")


def extract_token_surprisals(model: torch.nn.Module, tokenizer: LogTokenizer, line: str, device: str) -> list[dict]:
    """Computes token-by-token cross-entropy loss (surprisal) for a given log sequence.

    WHY: Aggregate sequence perplexity indicates that an anomaly occurred, but token surprisal (-log P(x_t | x_<t))
    highlights the exact token (e.g., unexpected error code or state transition) triggering the alert.
    """
    model.eval()
    masked = tokenizer.mask_variables(line)
    encoded = tokenizer.tokenizer.encode(masked)
    ids = encoded.ids
    tokens = encoded.tokens
    
    if len(ids) < 2:
        return [{"token": t, "surprisal": 0.0} for t in tokens]
        
    # t_tensor: [1, seq_len]
    t_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        inputs = t_tensor[:, :-1]
        targets = t_tensor[:, 1:]
        logits, _ = model(inputs)
        # loss_per_token: [1, seq_len - 1]
        loss_per_token = F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')[0].cpu().tolist()
        
    results = [{"token": tokens[0], "surprisal": 0.0}]
    for token_str, loss_val in zip(tokens[1:], loss_per_token):
        results.append({"token": token_str, "surprisal": float(loss_val)})
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate per-token surprisal heatmaps.")
    parser.add_argument("--config", default="config/stage1_config.yaml", help="Config YAML path.")
    parser.add_argument("--tokenizer", default="data/tokenizer/log_tokenizer.json", help="Tokenizer JSON path.")
    parser.add_argument("--output", default="data/ablations/b6_heatmaps.json", help="Output JSON path.")
    parser.add_argument("--force", action="store_true", help="Force recomputation.")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        logger.info(f"[IDEMPOTENCY] B6 heatmaps already exist at {args.output}. Pass --force to override.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    gpt_config = GPT2Config.from_dict(cfg)
    model = GPT2Model(gpt_config).to(device)
    
    ckpt_dir = cfg.get("training", {}).get("checkpoint_dir", "data/checkpoints")
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path:
        logger.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state)
    else:
        logger.warning("No checkpoint found. Using baseline weights.")

    tok = LogTokenizer(vocab_size=cfg["tokenizer"]["vocab_size"])
    if os.path.exists(args.tokenizer):
        tok.load(args.tokenizer)
    else:
        logger.warning("Tokenizer artifact not found. Initializing raw tokenizer.")

    # 5 representative normal and 5 anomalous HDFS log traces
    normal_traces = [
        "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "BLOCK* NameSystem.allocateBlock: /mnt/hadoop/dfs/data/current/rbw/blk_-1608999687919862906",
        "PacketResponder 1 for block blk_-1608999687919862906 terminating",
        "Received block blk_-1608999687919862906 of size 67108864 from /10.250.19.102",
        "Verification succeeded for blk_-1608999687919862906"
    ]
    anomalous_traces = [
        "Exception in receiveBlock for block blk_-3544583377289625738 java.io.IOException: Connection reset by peer",
        "ERROR DataNode: DataTransfer: java.io.EOFException at java.io.DataInputStream.readFully",
        "FATAL DataNode terminating due to unrecoverable disk IO error on /mnt/hadoop/dfs/data1",
        "WARN NameSystem: Block blk_8472910482910 missing replicas across all available storage nodes",
        "Unexpected packet sequence number 99482 received for blk_91827364510 expected 102"
    ]

    heatmap_data = {
        "normal": [extract_token_surprisals(model, tok, line, device) for line in normal_traces],
        "anomalous": [extract_token_surprisals(model, tok, line, device) for line in anomalous_traces]
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(heatmap_data, f, indent=4)
    logger.info(f"Successfully saved B6 surprisal heatmaps to {args.output}")


if __name__ == "__main__":
    main()
