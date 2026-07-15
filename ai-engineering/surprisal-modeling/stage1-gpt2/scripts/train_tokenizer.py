#!/usr/bin/env python3
"""
CLI wrapper for training custom Byte-Pair Encoding (BPE) log tokenizers.

Details why dynamic variable masking precedes BPE training,
structured telemetry, and idempotency checks.
"""
import argparse
import logging
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.tokenizer.log_tokenizer import LogTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train custom BPE tokenizer on log corpus.")
    parser.add_argument("--data", required=True, help="Path to raw training text file.")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Vocabulary size integer.")
    parser.add_argument("--output", required=True, help="Path to save tokenizer JSON file.")
    parser.add_argument("--force", action="store_true", help="Force retraining even if output JSON exists.")
    args = parser.parse_args()

    # Idempotency Check
    if os.path.exists(args.output) and not args.force:
        logger.info(f"[IDEMPOTENCY] Tokenizer already exists at {args.output}. Skipping training. Pass --force to override.")
        return

    if not os.path.exists(args.data):
        logger.error(f"Input data file not found: {args.data}")
        sys.exit(1)

    logger.info(f"Initializing LogTokenizer with vocab_size={args.vocab_size}")
    tokenizer = LogTokenizer(vocab_size=args.vocab_size)
    
    # WHY: Training BPE directly on raw syslog files causes high-entropy strings (IPs, timestamps, block IDs)
    # to consume vocabulary slots, starving the model of structural log pattern tokens.
    # LogTokenizer applies regex masking prior to BPE statistics accumulation.
    tokenizer.train(input_file_path=args.data, save_path=args.output)
    logger.info(f"Successfully serialized tokenizer to {args.output}")


if __name__ == "__main__":
    main()
