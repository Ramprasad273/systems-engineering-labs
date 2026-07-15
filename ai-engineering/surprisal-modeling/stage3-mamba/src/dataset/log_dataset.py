"""Packed Sequence Dataset Loader for Stage 3 Mamba & MambaLog Experiments.

Pedagogical and clean implementation following Karpathy coding guidelines:
- Zero padding waste via pre-packed sequences (seq_len=512) inherited from Stage 1.
- Explicit shape annotations (`[num_sequences, seq_len]`) and docstrings.
- Clean fallback path resolution across shared repository structures.
"""

import os
import yaml
import logging
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class PackedLogDataset(Dataset):
    """PyTorch Dataset abstraction for bin-packed autoregressive log sequences.

    Attributes:
        input_ids (torch.Tensor): Tensor of shape [num_sequences, seq_len] containing token IDs.
        labels (torch.Tensor | None): Binary block anomaly labels (0=Normal, 1=Anomaly) of shape [num_sequences].
        block_ids (list[str] | None): List of HDFS block identifier strings tracking log provenance.
    """

    def __init__(
        self,
        sequences: torch.Tensor | list[list[int]],
        labels: torch.Tensor | list[int] | None = None,
        block_ids: list[str] | None = None,
    ):
        """Initializes PackedLogDataset with tokenized sequences and optional metadata.

        Args:
            sequences: Packed token ID sequences of fixed length T (`[num_sequences, seq_len]`).
            labels: Optional binary anomaly labels corresponding to each sequence block.
            block_ids: Optional list of raw HDFS block IDs tracking provenance.
        """
        if isinstance(sequences, torch.Tensor):
            self.input_ids = sequences.long()
        else:
            self.input_ids = torch.tensor(sequences, dtype=torch.long)

        if labels is not None:
            if isinstance(labels, torch.Tensor):
                self.labels = labels.long()
            else:
                self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None

        self.block_ids = block_ids

    def __len__(self) -> int:
        """Returns the total number of packed sequences in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Retrieves a single packed sequence dictionary by index.

        Args:
            idx: Sequence index.

        Returns:
            Dictionary containing `input_ids` [seq_len] tensor and optional `label` and `block_id`.
        """
        item = {"input_ids": self.input_ids[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        if self.block_ids is not None:
            item["block_id"] = self.block_ids[idx]
        return item


def _resolve_path(config_path: str, target_path: str) -> str:
    """Resolves relative file paths against config location or workspace root."""
    if os.path.isabs(target_path):
        return target_path
    config_dir = os.path.dirname(os.path.abspath(config_path))
    resolved_from_config = os.path.normpath(os.path.join(config_dir, target_path))
    if os.path.exists(resolved_from_config):
        return resolved_from_config
    # Fallback checking relative to project root (parent of config_dir)
    project_dir = os.path.dirname(config_dir)
    resolved_from_project = os.path.normpath(os.path.join(project_dir, target_path))
    return resolved_from_project


def get_dataloader(
    config_path: str, split: str = "train", tokenizer=None
) -> tuple[DataLoader, object]:
    """Constructs a PyTorch DataLoader for the requested split (`train`, `val`, or `test`).

    Directly loads pre-computed Stage 1 bin-packed tensor caches (`{split}_packed.pt`)
    to ensure exact token and data split parity across stages.

    Args:
        config_path: Filesystem path to stage3_config.yaml or mambalog_config.yaml.
        split: Dataset split ('train', 'val', or 'test').
        tokenizer: Optional LogTokenizer instance. If None, loaded from config path.

    Returns:
        Tuple containing configured PyTorch DataLoader and active LogTokenizer instance.

    Raises:
        FileNotFoundError: If the pre-computed packed dataset file cannot be resolved.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    tokenizer_cfg = config.get("tokenizer", {})

    processed_dir = _resolve_path(config_path, dataset_cfg["processed_dir"])
    processed_file_path = os.path.join(processed_dir, f"{split}_packed.pt")

    if not os.path.exists(processed_file_path):
        raise FileNotFoundError(
            f"Pre-computed packed sequence file not found: {processed_file_path}. "
            "Ensure stage1-gpt2 data preprocessing (`stage1-gpt2/data/processed/`) is accessible."
        )

    logger.info(f"Loading packed sequence split '{split}' from: {processed_file_path}")
    data = torch.load(processed_file_path, weights_only=False)

    if tokenizer is None and "save_path" in tokenizer_cfg:
        tokenizer_path = _resolve_path(config_path, tokenizer_cfg["save_path"])
        if os.path.exists(tokenizer_path):
            try:
                import sys
                # Add stage1-gpt2 to sys.path if needed to load LogTokenizer class cleanly
                stage1_dir = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.abspath(config_path)), "../../stage1-gpt2")
                )
                if stage1_dir not in sys.path and os.path.exists(stage1_dir):
                    sys.path.insert(0, stage1_dir)
                from src.tokenizer.log_tokenizer import LogTokenizer

                tokenizer = LogTokenizer(
                    vocab_size=tokenizer_cfg.get("vocab_size", 5000)
                )
                tokenizer.load(tokenizer_path)
            except Exception as e:
                logger.warning(f"Could not load custom LogTokenizer ({e}); continuing with raw IDs.")
                tokenizer = None

    if isinstance(data, dict):
        dataset = PackedLogDataset(
            data["input_ids"],
            labels=data.get("labels"),
            block_ids=data.get("block_ids"),
        )
    else:
        dataset = PackedLogDataset(data)

    use_cuda = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=dataset_cfg.get("batch_size", 16),
        shuffle=(split == "train"),
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0,
        prefetch_factor=2 if use_cuda else None,
    )
    return dataloader, tokenizer
