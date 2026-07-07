"""Data loader factory for Stage 2 QLoRA fine-tuning and evaluation."""

import os
import yaml
import logging
import torch
from torch.utils.data import DataLoader
from src.dataset.sft_dataset import SFTDataset

logger = logging.getLogger("stage2.data_loader")


def _collate_skip_none(batch: list) -> dict | None:
    """Collate function that filters out None items returned by SFTDataset.

    WHY: SFTDataset.__getitem__ returns None for records where the prompt fills
    the entire context window (leaving no completion tokens). If we pass these
    to CrossEntropyLoss, we get 0/0 = NaN loss. This collate_fn filters them
    out so such samples are silently dropped from training without causing NaN.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "block_id": [item["block_id"] for item in batch],
        "label": [item["label"] for item in batch],
        "prompt_text": [item["prompt_text"] for item in batch],
        "completion_text": [item["completion_text"] for item in batch],
    }


def get_sft_dataloader(config_path: str, split: str, tokenizer, batch_size: int = None) -> DataLoader:
    """Creates a PyTorch DataLoader for the specified SFT dataset split.

    Args:
        config_path: Path to runtime YAML config.
        split: Dataset split ('train', 'val', 'test', 'spot_check').
        tokenizer: PreTrainedTokenizer instance.
        batch_size: Optional override for batch size.

    Returns:
        Configured PyTorch DataLoader.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ds_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    
    sft_dir = ds_cfg.get("sft_dir", "data/sft_dataset")
    max_seq_len = ds_cfg.get("max_seq_len", 1024)

    if split == "train":
        filename = ds_cfg.get("train_file", "train.jsonl")
        bs = batch_size or train_cfg.get("micro_batch_size", 2)
        shuffle = True
    elif split == "val":
        filename = ds_cfg.get("val_file", "val.jsonl")
        bs = batch_size or train_cfg.get("micro_batch_size", 2)
        shuffle = False
    elif split == "test":
        filename = ds_cfg.get("test_file", "test.jsonl")
        bs = batch_size or 1
        shuffle = False
    elif split == "spot_check":
        filename = ds_cfg.get("spot_check_file", "spot_check_50.jsonl")
        bs = batch_size or 1
        shuffle = False
    else:
        filename = f"{split}.jsonl"
        bs = batch_size or 1
        shuffle = False

    file_path = os.path.join(sft_dir, filename)
    dataset = SFTDataset(file_path, tokenizer, max_seq_len=max_seq_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=_collate_skip_none
    )
    logger.info(f"Initialized DataLoader for split '{split}' ({len(dataset)} items, bs={bs}).")
    return dataloader
