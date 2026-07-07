"""SFT Dataset Generator for QLoRA Root-Cause Diagnosis Engine.

Parses raw HDFS distributed execution traces, matches ground-truth block anomaly labels,
and applies rule-based SRE diagnostic heuristics to generate structured (prompt, JSON-completion)
pairs for supervised fine-tuning.

Guarantees exact comparability with Stage 1 anomaly detection baseline by including explicit
`is_anomaly` classification flags alongside structured root-cause diagnosis.
"""

import os
import sys
import csv
import re
import json
import random
import argparse
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("stage2.dataset")


SYSTEM_PROMPT = (
    "You are a Site Reliability Engineer (SRE) analyzing HDFS distributed filesystem logs. "
    "Given a block of log lines, produce a structured JSON root-cause diagnosis. "
    "Output ONLY valid JSON. No prose. No markdown fences. No explanation."
)


def load_anomaly_labels(label_file_path: str) -> dict[str, int]:
    """Parses ground-truth anomaly classifications from HDFS anomaly_label.csv."""
    if not os.path.exists(label_file_path):
        raise FileNotFoundError(f"Missing label CSV file at: {label_file_path}")
    labels = {}
    with open(label_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                labels[row[0]] = 1 if row[1] == "Anomaly" else 0
    logger.info(f"Loaded {len(labels)} ground-truth block anomaly labels.")
    return labels


def parse_raw_hdfs_blocks(log_file_path: str, max_blocks: int = 15000) -> dict[str, list[str]]:
    """Streams HDFS log lines and groups execution traces by block ID up to max_blocks."""
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Missing HDFS log corpus at: {log_file_path}")
    
    blocks = defaultdict(list)
    block_pattern = re.compile(r'(blk_-?\d+)')
    
    logger.info(f"Streaming HDFS execution corpus from: {log_file_path}...")
    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = block_pattern.search(line)
            if match:
                blk_id = match.group(1)
                blocks[blk_id].append(line)
                if len(blocks) >= max_blocks and blk_id not in blocks:
                    break
                    
    logger.info(f"Successfully extracted {len(blocks)} unique log blocks.")
    return blocks


def apply_sre_heuristic(block_id: str, lines: list[str], label: int) -> dict:
    """Generates structured diagnosis JSON based on log text semantics and ground-truth label."""
    combined_text = " ".join(lines)
    
    if label == 0:
        return {
            "is_anomaly": False,
            "root_cause": "Normal block allocation and replication execution without error",
            "severity": "P3_INFO",
            "affected_component": "None",
            "mitigation_commands": [],
            "confidence": 0.99
        }
        
    # Anomaly heuristics matching SRE failure modes
    if "Connection reset by peer" in combined_text or "DataXceiver error" in combined_text:
        return {
            "is_anomaly": True,
            "root_cause": "DataNode write pipeline failure due to network socket reset during block replication",
            "severity": "P1_CRITICAL",
            "affected_component": "DataNode",
            "mitigation_commands": ["sudo systemctl restart hdfs-datanode", "hdfs dfsadmin -report"],
            "confidence": 0.92
        }
    elif "PacketResponder" in combined_text or "Exception in receiveBlock" in combined_text:
        return {
            "is_anomaly": True,
            "root_cause": "DataNode packet transmission failure across downstream replication nodes",
            "severity": "P1_CRITICAL",
            "affected_component": "DataNode",
            "mitigation_commands": ["hdfs dfs -checknv -files", "sudo systemctl restart hdfs-datanode"],
            "confidence": 0.88
        }
    elif "NameNode" in combined_text or "LeaseExpiredException" in combined_text:
        return {
            "is_anomaly": True,
            "root_cause": "NameNode metadata synchronization failure or client lease expiration",
            "severity": "P0_EMERGENCY",
            "affected_component": "NameNode",
            "mitigation_commands": ["hdfs dfsadmin -safemode enter", "hdfs dfsadmin -saveNamespace"],
            "confidence": 0.95
        }
    elif "Timeout" in combined_text or "timed out" in combined_text.lower():
        return {
            "is_anomaly": True,
            "root_cause": "Inter-node communication timeout during distributed block transfer",
            "severity": "P2_WARNING",
            "affected_component": "Network",
            "mitigation_commands": ["check network latency between rack nodes", "adjust dfs.socket.timeout"],
            "confidence": 0.85
        }
    else:
        return {
            "is_anomaly": True,
            "root_cause": "Unspecified distributed block I/O exception during HDFS execution trace",
            "severity": "P2_WARNING",
            "affected_component": "HDFS Core",
            "mitigation_commands": ["hdfs fsck / -files -blocks -locations", "inspect datanode syslog"],
            "confidence": 0.80
        }


def format_chatml_pair(block_id: str, lines: list[str], completion_json: dict, ground_truth_label: int, max_lines: int = 8) -> dict:
    """Formats system/user prompt and assistant completion into standard ChatML JSONL entry.

    CRITICAL: label field is sourced from ground-truth anomaly_label.csv (ground_truth_label),
    NOT from the SRE heuristic completion. This ensures evaluation binary anomaly metrics
    are directly comparable to Stage 1's calculate_classification_metrics() which uses the
    same ground-truth CSV labels. Without this, evaluation measures heuristic reproduction,
    not actual anomaly detection accuracy.
    """
    truncated_lines = lines[:max_lines]
    user_prompt = f"LOG BLOCK (block_id: {block_id}):\n" + "\n".join(truncated_lines)
    assistant_completion = json.dumps(completion_json, ensure_ascii=False)
    
    prompt_str = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full_text = prompt_str + assistant_completion + "<|im_end|>"
    
    return {
        "block_id": block_id,
        "prompt": prompt_str,
        "completion": assistant_completion + "<|im_end|>",
        "full_text": full_text,
        "label": ground_truth_label  # Ground-truth from anomaly_label.csv, not heuristic
    }


def write_jsonl(file_path: str, records: list[dict]):
    """Writes dataset records to disk in JSONL format."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Serialized {len(records)} pairs -> {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare HDFS SFT dataset splits.")
    parser.add_argument("--raw_log", default="../stage1-gpt2/data/raw/HDFS.log")
    parser.add_argument("--label_csv", default="../stage1-gpt2/data/raw/anomaly_label.csv")
    parser.add_argument("--output_dir", default="data/sft_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", help="Print 3 random verification samples.")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if not os.path.exists(args.raw_log):
        logger.warning(f"Raw HDFS log not found at {args.raw_log}. Generating synthetic sample pairs for pipeline verification.")
        # Create synthetic pairs if raw file path is relative from different working dir
        blocks = {
            "blk_1001": ["INFO dfs.DataNode$DataXceiver: Receiving block blk_1001", "ERROR dfs.DataNode$BlockReceiver: IOException: Connection reset by peer"],
            "blk_1002": ["INFO dfs.DataNode$DataXceiver: Receiving block blk_1002", "INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_1002 terminating"],
            "blk_1003": ["INFO dfs.DataNode$DataXceiver: Receiving block blk_1003", "WARN dfs.DataNode: DataXceiver error processing WRITE_BLOCK"],
            "blk_1004": ["INFO dfs.DataNode$DataXceiver: Receiving block blk_1004", "INFO block blk_1004 written successfully"]
        }
        labels = {"blk_1001": 1, "blk_1002": 0, "blk_1003": 1, "blk_1004": 0}
    else:
        labels = load_anomaly_labels(args.label_csv)
        blocks = parse_raw_hdfs_blocks(args.raw_log, max_blocks=12000)
        
    normal_pairs = []
    anomaly_pairs = []
    
    for blk_id, lines in blocks.items():
        lbl = labels.get(blk_id, 0)
        completion = apply_sre_heuristic(blk_id, lines, lbl)
        pair = format_chatml_pair(blk_id, lines, completion, ground_truth_label=lbl)
        if lbl == 1:
            anomaly_pairs.append(pair)
        else:
            normal_pairs.append(pair)
            
    logger.info(f"Prepared {len(normal_pairs)} normal pairs and {len(anomaly_pairs)} anomaly pairs.")
    
    random.shuffle(normal_pairs)
    random.shuffle(anomaly_pairs)
    
    # 1. Generate 50 spot-check samples (25 anomaly, 25 normal)
    spot_check = anomaly_pairs[:25] + normal_pairs[:25]
    random.shuffle(spot_check)
    write_jsonl(os.path.join(args.output_dir, "spot_check_50.jsonl"), spot_check)
    
    # 2. Holdout split: val (200 anomaly + 200 normal = 400), test (200 anomaly + 200 normal = 400)
    val_set = anomaly_pairs[25:225] + normal_pairs[25:225]
    test_set = anomaly_pairs[225:425] + normal_pairs[225:425]
    random.shuffle(val_set)
    random.shuffle(test_set)
    write_jsonl(os.path.join(args.output_dir, "val.jsonl"), val_set)
    write_jsonl(os.path.join(args.output_dir, "test.jsonl"), test_set)
    
    # 3. Training size ablation datasets (B1)
    train_anom_pool = anomaly_pairs[425:]
    train_norm_pool = normal_pairs[425:]
    
    for size in [100, 500, 2000, 4000]:
        half = min(size // 2, len(train_anom_pool), len(train_norm_pool))
        subset = train_anom_pool[:half] + train_norm_pool[:half]
        random.shuffle(subset)
        write_jsonl(os.path.join(args.output_dir, f"train_{size}.jsonl"), subset)
        if size == 4000:
            write_jsonl(os.path.join(args.output_dir, "train.jsonl"), subset)
            
    if args.verify:
        logger.info("\n=== VERIFICATION SAMPLE FROM SPOT-CHECK SET ===")
        sample = spot_check[0] if spot_check else pair
        print("--- PROMPT ---")
        print(sample["prompt"])
        print("--- COMPLETION ---")
        print(sample["completion"])
        print(f"--- GROUND TRUTH LABEL (from CSV) ---")
        print(f"label: {sample['label']} ({'Anomaly' if sample['label'] == 1 else 'Normal'})")
        print("===============================================")


if __name__ == "__main__":
    main()
