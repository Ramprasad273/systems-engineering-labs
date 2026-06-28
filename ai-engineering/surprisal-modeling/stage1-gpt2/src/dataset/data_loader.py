"""HDFS Dataset Ingestion, Anomaly Label Mapping, and Sequence Bin-Packing Pipeline.

This module orchestrates the automated downloading of benchmark HDFS distributed system logs,
parsing block-level anomaly labels, grouping syslog streams by block ID, and bin-packing
variable-length log sequences into fixed-size context blocks (512 tokens) using the First-Fit
Decreasing (FFD) heuristic. This ensures zero padding waste and maximizes GPU throughput.
"""

import os
import csv
import re
import urllib.request
import tarfile
import yaml
import bisect
import random
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

logger = logging.getLogger(__name__)


class PackedLogDataset(Dataset):
    """PyTorch Dataset abstraction for bin-packed autoregressive log sequences.

    Attributes:
        input_ids (torch.Tensor): LongTensor of shape (num_sequences, seq_len) containing token IDs.
        labels (torch.Tensor | None): LongTensor containing binary block anomaly labels (0=Normal, 1=Anomaly).
        block_ids (list[str] | None): List of HDFS block identifier strings corresponding to each sequence.
    """

    def __init__(self, sequences: list[list[int]], labels: list[int] = None, block_ids: list[str] = None):
        """Initializes the PackedLogDataset with tokenized sequences and optional metadata.

        Args:
            sequences: List of packed integer token ID sequences of fixed length T.
            labels: Optional binary anomaly labels corresponding to each sequence block.
            block_ids: Optional list of raw HDFS block IDs tracking provenance.
        """
        self.input_ids = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.block_ids = block_ids
        
    def __len__(self) -> int:
        """Returns the total number of packed sequences in the dataset."""
        return len(self.input_ids)
        
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Retrieves a single packed sequence dictionary by index.

        Args:
            idx: Sequence index.

        Returns:
            Dictionary containing `input_ids` tensor and optional `label` and `block_id`.
        """
        item = {"input_ids": self.input_ids[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        if self.block_ids is not None:
            item["block_id"] = self.block_ids[idx]
        return item


def download_hdfs_dataset(url: str, dest_dir: str, raw_file_name: str = "HDFS.log") -> str:
    """Downloads and unpacks the benchmark HDFS dataset tarball from Zenodo/LogHub.

    Args:
        url: Direct HTTP download URL for the HDFS tarball artifact.
        dest_dir: Target filesystem directory to store extracted logs and label CSVs.
        raw_file_name: Expected extracted filename for raw syslog text.

    Returns:
        Absolute filesystem path to the extracted HDFS log file.

    Raises:
        FileNotFoundError: If target files are missing post-extraction.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, raw_file_name)
    label_path = os.path.join(dest_dir, "anomaly_label.csv")
    
    if os.path.exists(dest_path) and os.path.exists(label_path):
        logger.info(f"HDFS benchmark log corpus and anomaly labels already present at: {dest_dir}")
        return dest_path
    
    tar_path = os.path.join(dest_dir, "HDFS_1.tar.gz")
    logger.info(f"Downloading HDFS dataset tarball from {url} to destination: {tar_path}")
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(tar_path, 'wb') as out_file:
        # Stream download in 1MB chunks to control memory pressure
        block_size = 1024 * 1024
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            out_file.write(buffer)
            
    logger.info("Download complete. Unpacking HDFS.log and anomaly_label.csv...")
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        extracted_files = []
        for member in members:
            basename = os.path.basename(member.name)
            if basename in ["HDFS.log", "anomaly_label.csv"]:
                member.name = basename
                tar.extract(member, path=dest_dir)
                extracted_files.append(basename)
        if "HDFS.log" not in extracted_files:
            raise FileNotFoundError("Critical failure: HDFS.log missing from downloaded tarball.")
        
    if os.path.exists(tar_path):
        os.remove(tar_path)
        
    logger.info(f"HDFS dataset and ground-truth labels successfully extracted to: {dest_dir}")
    return dest_path


def load_anomaly_labels(label_file_path: str) -> dict[str, int]:
    """Parses ground-truth block anomaly classifications from CSV.

    Args:
        label_file_path: Filesystem path to anomaly_label.csv.

    Returns:
        Dictionary mapping HDFS block identifier strings (e.g., 'blk_123') to binary flags (1=Anomaly, 0=Normal).
    """
    labels = {}
    logger.info(f"Loading ground-truth anomaly classifications from: {label_file_path}")
    with open(label_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)  # Skip header row (BlockId, Label)
        for row in reader:
            if len(row) >= 2:
                block_id, label_str = row[0], row[1]
                labels[block_id] = 1 if label_str == "Anomaly" else 0
    logger.info(f"Successfully loaded {len(labels)} block label entries.")
    return labels


def parse_and_group_logs(log_file_path: str, labels_map: dict[str, int]) -> tuple[list[dict], list[dict]]:
    """Streams raw syslog text, extracts block IDs via regex, and aggregates execution traces.

    Args:
        log_file_path: Filesystem path to raw HDFS syslog text file.
        labels_map: Mapping of block IDs to ground-truth anomaly labels.

    Returns:
        Tuple containing lists of grouped block dictionaries for normal and anomalous splits.
    """
    blocks = defaultdict(list)
    # Fast compiled regex to match HDFS block allocation tracking tokens
    block_pattern = re.compile(r'(blk_-?\d+)')
    
    logger.info(f"Streaming and parsing distributed execution logs from: {log_file_path}")
    unmapped_lines_count = 0
    total_lines = 0
    
    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            match = block_pattern.search(line)
            if match:
                block_id = match.group(1)
                blocks[block_id].append(line)
            else:
                unmapped_lines_count += 1
                
    logger.info(f"Corpus ingestion complete. Total syslog lines analyzed: {total_lines}")
    logger.info(f"Unmapped orphan log lines skipped: {unmapped_lines_count}")
    logger.info(f"Unique distributed execution blocks identified: {len(blocks)}")
    
    normal_blocks = []
    anomaly_blocks = []
    
    for block_id, lines in blocks.items():
        label = labels_map.get(block_id, 0)
        block_info = {
            "block_id": block_id,
            "lines": lines,
            "label": label
        }
        if label == 1:
            anomaly_blocks.append(block_info)
        else:
            normal_blocks.append(block_info)
            
    anomaly_rate = (len(anomaly_blocks) / len(blocks) * 100) if blocks else 0.0
    logger.info(
        f"Corpus split distribution: {len(normal_blocks)} Normal blocks | "
        f"{len(anomaly_blocks)} Anomaly blocks (Empirical Anomaly Rate: {anomaly_rate:.2f}%)"
    )
    return normal_blocks, anomaly_blocks


def pack_sequences_ffd(tokenized_logs: list[list[int]], max_len: int = 512, eos_token_id: int = 5) -> list[list[int]]:
    """Bin-packs variable-length tokenized log traces into fixed context windows via FFD.

    First-Fit Decreasing (FFD) sorts sequences in descending order of length and places each
    sequence into the bin with the smallest remaining capacity that can accommodate it. This
    minimizes trailing padding tokens and optimizes attention matrix density.

    Args:
        tokenized_logs: List of variable-length integer token ID sequences.
        max_len: Fixed sequence length context window cap (T).
        eos_token_id: Integer identifier for `<EOS>` delimiter token.

    Returns:
        List of densely packed token sequences of uniform length T.
    """
    chunk_size = 100000
    packed_blocks = []
    
    # Process bin-packing in chunks to maintain sub-quadratic time complexity
    for chunk_start in range(0, len(tokenized_logs), chunk_size):
        chunk = tokenized_logs[chunk_start : chunk_start + chunk_size]
        # Sort descending by length (core FFD heuristic requirement)
        chunk_sorted = sorted(chunk, key=len, reverse=True)
        
        bins = []
        bin_capacities = []
        # Index bins by their remaining free capacity for O(1) candidate lookup
        bins_by_cap = {c: [] for c in range(max_len + 1)}
        
        for seq in chunk_sorted:
            n = len(seq)
            if n > max_len:
                # Hard truncate over-length sequences to prevent context overflow
                seq = seq[:max_len]
                n = max_len
            
            # Find candidate bin with minimal sufficient remaining capacity (best-fit approximation)
            best_bin_idx = None
            best_cap = None
            for c in range(n, max_len + 1):
                if bins_by_cap[c]:
                    first_idx = bins_by_cap[c][0]
                    if best_bin_idx is None or first_idx < best_bin_idx:
                        best_bin_idx = first_idx
                        best_cap = c
                        
            if best_bin_idx is not None:
                # Place sequence into identified active bin
                bins_by_cap[best_cap].pop(0)
                bins[best_bin_idx].extend(seq)
                new_cap = best_cap - n
                bin_capacities[best_bin_idx] = new_cap
                bisect.insort(bins_by_cap[new_cap], best_bin_idx)
            else:
                # Allocate a new context bin
                new_bin_idx = len(bins)
                bins.append(list(seq))
                new_cap = max_len - n
                bin_capacities.append(new_cap)
                bins_by_cap[new_cap].append(new_bin_idx)
                
        # Fill residual bin slack with EOS delimiter tokens
        for b, cap in zip(bins, bin_capacities):
            if cap > 0:
                b.extend([eos_token_id] * cap)
            assert len(b) == max_len, f"Packing assertion error: Expected length {max_len}, got {len(b)}"
            packed_blocks.append(b)
            
    return packed_blocks


def get_dataloader(config_path: str, split: str = "train", tokenizer = None) -> tuple[DataLoader, object]:
    """Constructs a PyTorch DataLoader for the requested split, handling caching and tokenization.

    Args:
        config_path: Filesystem path to stage1_config.yaml.
        split: Dataset split string ('train', 'val', or 'test').
        tokenizer: Optional pre-loaded LogTokenizer instance. If None, loaded from config path.

    Returns:
        Tuple containing configured PyTorch DataLoader and active LogTokenizer instance.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    dataset_cfg = config["dataset"]
    tokenizer_cfg = config["tokenizer"]
    processed_dir = dataset_cfg["processed_dir"]
    processed_file_path = os.path.join(processed_dir, f"{split}_packed.pt")
    
    # Check disk cache to bypass redundant download/pre-processing cycles
    if os.path.exists(processed_file_path):
        logger.info(f"Loading pre-computed cached dataset artifact: {processed_file_path}")
        data = torch.load(processed_file_path, weights_only=False)
        
        tokenizer_save_path = tokenizer_cfg["save_path"]
        if tokenizer is None:
            from src.tokenizer.log_tokenizer import LogTokenizer
            tokenizer = LogTokenizer(
                vocab_size=tokenizer_cfg["vocab_size"],
                special_tokens=tokenizer_cfg.get("special_tokens")
            )
            tokenizer.load(tokenizer_save_path)
            
        if isinstance(data, dict):
            dataset = PackedLogDataset(
                data["input_ids"], 
                labels=data.get("labels"), 
                block_ids=data.get("block_ids")
            )
        else:
            dataset = PackedLogDataset(data)
            
        dataloader = DataLoader(
            dataset, 
            batch_size=dataset_cfg["batch_size"], 
            shuffle=(split == "train"), 
            pin_memory=True
        )
        return dataloader, tokenizer
        
    # Execute full raw download and ingestion pipeline if disk cache is absent
    raw_dir = dataset_cfg["raw_dir"]
    url = dataset_cfg["url"]
    raw_file_name = "HDFS.log"
    raw_file_path = os.path.join(raw_dir, raw_file_name)
    label_file_path = os.path.join(raw_dir, "anomaly_label.csv")
    
    if not os.path.exists(raw_file_path) or not os.path.exists(label_file_path):
        download_hdfs_dataset(url, raw_dir, raw_file_name)
        
    labels_map = load_anomaly_labels(label_file_path)
    normal_blocks, anomaly_blocks = parse_and_group_logs(raw_file_path, labels_map)
    
    # Sort blocks deterministically by ID before seeding random shuffle
    normal_blocks = sorted(normal_blocks, key=lambda x: x["block_id"])
    random.Random(42).shuffle(normal_blocks)
    
    split_idx = int(len(normal_blocks) * dataset_cfg["train_split"])
    if len(normal_blocks) > 0:
        split_idx = max(1, split_idx)
    if len(normal_blocks) > 1 and split_idx == len(normal_blocks):
        split_idx = len(normal_blocks) - 1
        
    train_normal = normal_blocks[:split_idx]
    val_normal = normal_blocks[split_idx:]
    
    tokenizer_save_path = tokenizer_cfg["save_path"]
    if tokenizer is None:
        from src.tokenizer.log_tokenizer import LogTokenizer
        tokenizer = LogTokenizer(
            vocab_size=tokenizer_cfg["vocab_size"],
            special_tokens=tokenizer_cfg.get("special_tokens")
        )
        if split == "train" or not os.path.exists(tokenizer_save_path):
            train_temp_path = "data/tokenizer/train_raw.log"
            os.makedirs(os.path.dirname(train_temp_path), exist_ok=True)
            with open(train_temp_path, "w", encoding="utf-8") as tf:
                for block in train_normal:
                    for line in block["lines"]:
                        tf.write(line + "\n")
            tokenizer.train(train_temp_path, tokenizer_save_path)
            if os.path.exists(train_temp_path):
                os.remove(train_temp_path)
        else:
            tokenizer.load(tokenizer_save_path)
            
    eos_token_id = tokenizer.tokenizer.token_to_id("<EOS>")
    
    if split == "train":
        logger.info("Executing stream tokenization over normal training blocks...")
        tokenized_logs = []
        for block in train_normal:
            block_tokens = []
            masked_lines = [tokenizer.mask_variables(line) for line in block["lines"]]
            encodings = tokenizer.tokenizer.encode_batch(masked_lines)
            for enc in encodings:
                block_tokens.extend(enc.ids + [eos_token_id])
            tokenized_logs.append(block_tokens)
            
        logger.info(f"Executing First-Fit Decreasing (FFD) packing over {len(tokenized_logs)} traces...")
        packed_sequences = pack_sequences_ffd(
            tokenized_logs, 
            max_len=dataset_cfg["seq_len"], 
            eos_token_id=eos_token_id
        )
        logger.info(f"Successfully packed into {len(packed_sequences)} dense context tensors (T={dataset_cfg['seq_len']}).")
        
        os.makedirs(processed_dir, exist_ok=True)
        torch.save(packed_sequences, processed_file_path)
        dataset = PackedLogDataset(packed_sequences)
        
    else:
        if split == "val":
            selected_blocks = val_normal
        else:  # test split combines normal validation holdouts with all anomaly traces
            selected_blocks = val_normal + anomaly_blocks
            
        logger.info(f"Tokenizing and assembling {len(selected_blocks)} discrete block sequences for split: '{split}'...")
        sequences = []
        labels = []
        block_ids = []
        
        # Tokenize block-by-block without cross-block FFD mixing to preserve discrete inference evaluation
        for block in selected_blocks:
            block_tokens = []
            masked_lines = [tokenizer.mask_variables(line) for line in block["lines"]]
            encodings = tokenizer.tokenizer.encode_batch(masked_lines)
            for enc in encodings:
                block_tokens.extend(enc.ids + [eos_token_id])
                
            # Pad or truncate individual traces to exact block_size window cap
            padded = block_tokens[:512] + [eos_token_id] * max(0, 512 - len(block_tokens))
            sequences.append(padded)
            labels.append(block["label"])
            block_ids.append(block["block_id"])
            
        data_to_save = {
            "input_ids": sequences,
            "labels": labels,
            "block_ids": block_ids
        }
        
        os.makedirs(processed_dir, exist_ok=True)
        torch.save(data_to_save, processed_file_path)
        logger.info(f"Serialized processed '{split}' evaluation tensor to cache: {processed_file_path}")
        
        dataset = PackedLogDataset(
            sequences, 
            labels=labels, 
            block_ids=block_ids
        )
        
    dataloader = DataLoader(
        dataset, 
        batch_size=dataset_cfg["batch_size"], 
        shuffle=(split == "train"), 
        pin_memory=True
    )
    return dataloader, tokenizer


# Backward compatibility alias for ablation scripts
group_logs_by_block = parse_and_group_logs
