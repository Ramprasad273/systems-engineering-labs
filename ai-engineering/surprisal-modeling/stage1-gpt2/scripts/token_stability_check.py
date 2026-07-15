"""Token Stability Analysis.

Verifies that the dynamic variable masking pipeline does NOT produce [UNK]
tokens for any structural keywords in the trained BPE vocabulary. An [UNK]
token indicates that the masking regex has aggressively collapsed a structural
token (e.g., 'DataNode', 'PacketResponder') into a placeholder, destroying
the model's ability to learn from that structural pattern.

This is a critical correctness check that must pass before paper submission:
if any structural token maps to [UNK], the model's perplexity measurements
on those token positions are meaningless.

Analysis
--------
For each log line in the test corpus (both normal and anomaly):
  1. Apply mask_variables() regex pipeline.
  2. Encode with the trained BPE tokenizer.
  3. Record any token IDs equal to the [UNK] token ID.
  4. Recover the original sub-string that produced the [UNK].

Outputs
-------
  data/ablations/token_stability.json : full UNK analysis
  Summary table printed to stdout

Usage
-----
    python scripts/token_stability_check.py --config config/stage1_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.data_loader import (
    download_hdfs_dataset,
    load_anomaly_labels,
    parse_and_group_logs,
)
from src.tokenizer.log_tokenizer import LogTokenizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _analyze_unk_tokens(
    blocks: list,
    tok: LogTokenizer,
    unk_id: int,
    label: str,
    max_blocks: int = 5_000,
) -> dict:
    """Scan up to max_blocks for [UNK] tokens and collect statistics."""
    total_tokens   = 0
    total_unk      = 0
    unk_contexts   = defaultdict(int)   # masked line → count

    for block in blocks[:max_blocks]:
        for line in block["lines"]:
            masked = tok.mask_variables(line)
            enc    = tok.encode(masked)
            ids    = enc.ids
            tokens = tok.tokenizer.id_to_token

            total_tokens += len(ids)
            for tok_id in ids:
                if tok_id == unk_id:
                    total_unk += 1
                    unk_contexts[masked[:80]] += 1

    unk_rate = total_unk / total_tokens if total_tokens > 0 else 0.0
    top_contexts = sorted(unk_contexts.items(), key=lambda x: -x[1])[:10]

    return {
        "label":          label,
        "blocks_analyzed": min(max_blocks, len(blocks)),
        "total_tokens":   total_tokens,
        "total_unk":      total_unk,
        "unk_rate":       unk_rate,
        "top_unk_contexts": [{"masked_line": c, "count": n} for c, n in top_contexts],
    }


def main():
    parser = argparse.ArgumentParser(description="Token stability analysis")
    parser.add_argument("--config",  default="config/stage1_config.yaml")
    parser.add_argument("--output",  default="data/ablations/token_stability.json")
    parser.add_argument("--max-blocks", type=int, default=5000,
                        help="Maximum blocks to analyze per split (default: 5000)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load tokenizer
    tok_path = cfg["tokenizer"]["save_path"]
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}. "
            "Run train.py first to train the BPE tokenizer."
        )
    tok = LogTokenizer(vocab_size=cfg["tokenizer"]["vocab_size"])
    tok.load(tok_path)
    unk_id = tok.tokenizer.token_to_id("[UNK]")
    logger.info(f"Loaded tokenizer from {tok_path}  |  [UNK] ID = {unk_id}")

    raw_dir   = cfg["dataset"]["raw_dir"]
    download_hdfs_dataset(cfg["dataset"]["url"], raw_dir)
    log_path   = os.path.join(raw_dir, "HDFS.log")
    label_path = os.path.join(raw_dir, "anomaly_label.csv")
    labels_map = load_anomaly_labels(label_path)
    normal_all, anomaly = parse_and_group_logs(log_path, labels_map)

    split_idx  = int(len(normal_all) * cfg["dataset"]["train_split"])
    normal_val = normal_all[split_idx:]

    logger.info("Analyzing normal validation blocks...")
    normal_analysis = _analyze_unk_tokens(normal_val, tok, unk_id, "normal", args.max_blocks)

    logger.info("Analyzing anomaly blocks...")
    anomaly_analysis = _analyze_unk_tokens(anomaly, tok, unk_id, "anomaly", args.max_blocks)

    results = {
        "tokenizer_path": tok_path,
        "vocab_size": cfg["tokenizer"]["vocab_size"],
        "unk_token_id": unk_id,
        "normal": normal_analysis,
        "anomaly": anomaly_analysis,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("TOKEN STABILITY ANALYSIS")
    print("=" * 60)
    for split in [normal_analysis, anomaly_analysis]:
        print(f"\nSplit: {split['label'].upper()}")
        print(f"  Blocks analyzed : {split['blocks_analyzed']:,}")
        print(f"  Total tokens    : {split['total_tokens']:,}")
        print(f"  [UNK] tokens    : {split['total_unk']:,}")
        print(f"  UNK rate        : {split['unk_rate']:.6f} ({split['unk_rate']*100:.4f}%)")
        if split["top_unk_contexts"]:
            print(f"  Top UNK contexts:")
            for ctx in split["top_unk_contexts"][:3]:
                print(f"    [{ctx['count']:4d}x] {ctx['masked_line'][:70]}")
        else:
            print("  [PASS] No [UNK] tokens detected - masking pipeline is stable.")

    if normal_analysis["total_unk"] == 0 and anomaly_analysis["total_unk"] == 0:
        print("\n[PASS] Zero [UNK] tokens in both normal and anomaly splits.")
        print("  The masking pipeline preserves all structural vocabulary tokens.")
    else:
        unk_rate = normal_analysis["unk_rate"] + anomaly_analysis["unk_rate"]
        print(f"\n[WARNING] [UNK] tokens detected. Combined UNK rate: {unk_rate:.6f}")
        print("  Review masking regexes for over-aggressive pattern matching.")

    print(f"\nFull analysis written to: {args.output}")


if __name__ == "__main__":
    main()
