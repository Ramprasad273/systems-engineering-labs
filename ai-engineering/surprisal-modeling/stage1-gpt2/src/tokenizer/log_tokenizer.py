"""Dynamic Regex Pre-tokenization and Byte-Pair Encoding (BPE) Tokenizer Lifecycle.

This module implements dynamic variable masking to strip high-entropy, non-stationary
identifiers (e.g., IP addresses, block IDs, timestamps, hexadecimal strings) from raw
syslog data prior to constructing a custom Byte-Pair Encoding vocabulary. This prevents
vocabulary fragmentation and preserves autoregressive model capacity for structural log patterns.
"""

import os
import re
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

logger = logging.getLogger(__name__)


class LogTokenizer:
    """Orchestrates dynamic regex variable masking and BPE vocabulary construction.

    Attributes:
        vocab_size (int): Maximum vocabulary capacity for the BPE model.
        special_tokens (list[str]): Reserved control and placeholder tokens.
        tokenizer (Tokenizer | None): Underlying HuggingFace Tokenizer instance.
    """

    def __init__(self, vocab_size: int = 5000, special_tokens: list[str] = None):
        """Initializes the LogTokenizer with regex heuristics and vocabulary bounds.

        Args:
            vocab_size: Maximum number of tokens in the trained BPE vocabulary.
            special_tokens: Custom list of special control and placeholder tokens.
                If None, defaults to standard structural and masked entity tokens.
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", 
            "<EOS>", "<IP>", "<HEX>", "<DATE>", "<TIME>"
        ]
        self.tokenizer = None
        
        # Pre-compile regex patterns for high-throughput stream processing
        # Matches IPv4 addresses with optional ports or leading/trailing slashes
        self.ip_pattern = re.compile(r'/?\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b')
        # Matches HDFS block identifiers (e.g., blk_-3509323198988774369, blk_12345)
        self.block_pattern = re.compile(r'\bblk_-?\d+\b')
        # Matches standard hexadecimal memory addresses or error codes (e.g., 0x9f31a)
        self.hex_pattern = re.compile(r'\b0x[0-9a-fA-F]+\b')
        # Matches ISO dates (YYYY-MM-DD)
        self.date_pattern_1 = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
        # Matches slash-separated dates (YY/MM/DD)
        self.date_pattern_2 = re.compile(r'\b\d{2}/\d{2}/\d{2}\b')
        # Matches hyphen-separated short dates (YY-MM-DD)
        self.date_pattern_3 = re.compile(r'\b\d{2}-\d{2}-\d{2}\b')
        # Matches standard 24-hour timestamps (HH:MM:SS)
        self.time_pattern = re.compile(r'\b\d{2}:\d{2}:\d{2}\b')
        # Matches compact 6-digit HDFS syslog date prefixes (e.g., 081109)
        self.hdfs_date_prefix = re.compile(r'^\b\d{6}\b')
        # Matches compact HDFS date-time headers post-date normalization
        self.hdfs_datetime_prefix = re.compile(r'^<DATE>\s+\b\d{6}\b')
        
    def mask_variables(self, text: str) -> str:
        """Applies deterministic regular expressions to replace dynamic entities with placeholders.

        Args:
            text: Raw input log line string.

        Returns:
            Normalized log string containing invariant structural tokens and placeholders.
        """
        # 1. Mask IPv4 network addresses
        text = self.ip_pattern.sub('<IP>', text)

        # 2. Mask distributed filesystem Block IDs
        text = self.block_pattern.sub('<HEX>', text)

        # 3. Mask hexadecimal memory offsets and identifiers
        text = self.hex_pattern.sub('<HEX>', text)

        # 4. Mask calendar date strings across multiple common formats
        text = self.date_pattern_1.sub('<DATE>', text)
        text = self.date_pattern_2.sub('<DATE>', text)
        text = self.date_pattern_3.sub('<DATE>', text)
        
        # 5. Mask clock timestamps
        text = self.time_pattern.sub('<TIME>', text)

        # 6. Mask specialized HDFS syslog leading headers
        text = self.hdfs_date_prefix.sub('<DATE>', text)
        text = self.hdfs_datetime_prefix.sub('<DATE> <TIME>', text)

        return text

    def train(self, input_file_path: str, save_path: str):
        """Executes the BPE training loop over masked log lines and serializes the model.

        Args:
            input_file_path: Filesystem path to the raw input log corpus.
            save_path: Filesystem destination to store the JSON tokenizer artifact.

        Raises:
            FileNotFoundError: If the specified input corpus does not exist.
        """
        logger.info(f"Preparing data for BPE training from corpus: {input_file_path}")
        temp_masked_path = input_file_path + ".masked"
        
        # Stream corpus line-by-line to minimize resident memory footprint during pre-processing
        with open(input_file_path, "r", encoding="utf-8") as fin, open(temp_masked_path, "w", encoding="utf-8") as fout:
            for line in fin:
                masked_line = self.mask_variables(line.strip())
                fout.write(masked_line + "\n")
        
        logger.info("Initializing HuggingFace BPE Tokenizer engine...")
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens
        )
        
        logger.info(f"Training BPE vocabulary capped at V={self.vocab_size} tokens...")
        self.tokenizer.train(files=[temp_masked_path], trainer=trainer)
        
        # Clean up intermediate masked pre-processing file
        if os.path.exists(temp_masked_path):
            os.remove(temp_masked_path)
            
        # Ensure destination directory hierarchy exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.tokenizer.save(save_path)
        logger.info(f"Tokenizer trained and successfully serialized to: {save_path}")

    def load(self, load_path: str):
        """Loads a pre-trained BPE tokenizer configuration from disk.

        Args:
            load_path: Filesystem path to the serialized JSON tokenizer artifact.

        Raises:
            FileNotFoundError: If the artifact cannot be located at load_path.
        """
        if not os.path.exists(load_path):
            logger.error(f"Tokenizer artifact missing at path: {load_path}")
            raise FileNotFoundError(f"No tokenizer file found at {load_path}")
        self.tokenizer = Tokenizer.from_file(load_path)
        logger.info(f"Successfully loaded pre-trained tokenizer from: {load_path}")

    def encode(self, text: str):
        """Masks variables and converts a log line into a sequence of integer token IDs.

        Args:
            text: Raw input log string.

        Returns:
            Encoding object containing `ids` and `tokens` attributes.

        Raises:
            ValueError: If invoked prior to training or loading a valid tokenizer.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer instance is uninitialized. Call train() or load() first.")
        masked = self.mask_variables(text)
        return self.tokenizer.encode(masked)

    def decode(self, ids: list[int]) -> str:
        """Converts integer token IDs back into a human-readable log string.

        Args:
            ids: List of integer token identifiers.

        Returns:
            Decoded text string with special tokens preserved or stripped.

        Raises:
            ValueError: If invoked prior to training or loading a valid tokenizer.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer instance is uninitialized. Call train() or load() first.")
        return self.tokenizer.decode(ids)
