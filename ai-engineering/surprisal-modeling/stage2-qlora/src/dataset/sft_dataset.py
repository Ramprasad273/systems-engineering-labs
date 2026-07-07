"""Supervised Fine-Tuning (SFT) Dataset with Prompt Loss Masking.

Orchestrates tokenization of structured (prompt, completion) pairs using ChatML format.
Applies rigorous loss masking (`target = -100`) across all system and user prompt tokens
so that gradients flow exclusively from generating the structured JSON diagnostic output.
"""

import os
import json
import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("stage2.sft_dataset")


class SFTDataset(Dataset):
    """PyTorch Dataset for ChatML prompt-completion pairs with loss masking.

    Attributes:
        records (list[dict]): Raw parsed JSONL records containing prompt, completion, and metadata.
        tokenizer: HuggingFace PreTrainedTokenizer instance.
        max_seq_len (int): Maximum sequence length window cap.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 1024):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"SFT dataset file missing at: {jsonl_path}")

        raw_records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cached_samples = []

        logger.info(f"Pre-tokenizing {len(raw_records)} SFT records from {jsonl_path} (max_len={max_seq_len})...")
        MIN_COMPLETION_TOKENS = 8
        skipped_count = 0

        for idx, record in enumerate(raw_records):
            prompt_str = record["prompt"]
            full_str = record["full_text"]

            # Tokenize prompt to determine mask cutoff index
            prompt_enc = self.tokenizer(
                prompt_str,
                truncation=True,
                max_length=self.max_seq_len,
                add_special_tokens=False
            )
            prompt_len = len(prompt_enc["input_ids"])

            # Tokenize full ChatML sequence
            full_enc = self.tokenizer(
                full_str,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
                add_special_tokens=False
            )

            input_ids = torch.tensor(full_enc["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(full_enc["attention_mask"], dtype=torch.long)

            # Clone input_ids to create target labels
            labels = input_ids.clone()

            # WHY: Set labels to -100 for all prompt tokens and padding tokens.
            # -100 is the default ignore_index in PyTorch CrossEntropyLoss.
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100

            if (labels != -100).sum().item() < MIN_COMPLETION_TOKENS:
                skipped_count += 1
                continue

            self.cached_samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "block_id": record.get("block_id", f"sample_{idx}"),
                "label": record.get("label", 1),
                "prompt_text": prompt_str,
                "completion_text": record.get("completion", "")
            })

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} records where prompt filled context window (<{MIN_COMPLETION_TOKENS} completion tokens).")
        logger.info(f"Successfully cached {len(self.cached_samples)} valid tokenized SFT samples in memory.")

    def __len__(self) -> int:
        return len(self.cached_samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        return self.cached_samples[idx]
