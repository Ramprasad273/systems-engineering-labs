"""Shared pytest fixtures for the surprisal-gpt2 test suite.

All fixtures are scoped appropriately to avoid redundant construction:
- Model and config fixtures use ``scope="module"`` so they are built once per
  test file rather than once per test function.
- File-system fixtures use ``tmp_path`` (function-scoped) to guarantee
  isolation between tests.
"""

import os
import pytest
import torch

from src.models.gpt2 import GPT2Config, GPT2Model
from src.tokenizer.log_tokenizer import LogTokenizer


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> str:
    """Returns the active compute device string ('cuda' or 'cpu')."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def autocast_dtype(device: str) -> torch.dtype:
    """Returns the mixed-precision dtype matched to the active device."""
    return torch.bfloat16 if device == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# Tiny model — fast for unit tests, no GPU required
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_config() -> GPT2Config:
    """Minimal GPT-2 config for rapid unit testing.

    Chosen so a forward pass runs in < 100 ms on CPU with no CUDA required.
    """
    return GPT2Config(
        vocab_size=64,
        n_embd=128,
        n_layer=2,
        n_head=4,       # head_dim = 32
        block_size=32,
        d_ff=256,
        layer_norm_epsilon=1e-5,
    )


@pytest.fixture(scope="module")
def tiny_model(tiny_config: GPT2Config) -> GPT2Model:
    """Instantiated GPT2Model on CPU using ``tiny_config``."""
    return GPT2Model(tiny_config)


# ---------------------------------------------------------------------------
# Full-scale config — matches production stage1_config.yaml
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_config() -> GPT2Config:
    """Production-scale GPT-2 config (768-dim, 12-layer, 12-head)."""
    return GPT2Config(
        vocab_size=5000,
        n_embd=768,
        n_layer=12,
        n_head=12,
        block_size=512,
        d_ff=2048,
        layer_norm_epsilon=1e-5,
    )


# ---------------------------------------------------------------------------
# Tokenizer — trained on a small synthetic HDFS-like corpus
# ---------------------------------------------------------------------------

SYNTHETIC_LOG_LINES = [
    "081109 203518 3 INFO dfs.DataNode$DataXceiver: Receiving block blk_-3509323198988774369 "
    "src: /10.250.6.101:52922 dest: /10.250.6.214:50010",
    "081109 203518 3 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block "
    "blk_-3509323198988774369 terminating",
    "081109 203519 1 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: "
    "blockMap updated: 10.250.6.214:50010 is added to blk_-3509323198988774369",
    "081109 203519 3 ERROR dfs.DataNode$DataXceiver: Got exception while serving "
    "blk_-1608999687919862906 to /10.250.6.102",
    "081109 203520 2 INFO dfs.DataNode$BlockSender: Transmitted block blk_8229193803249955061 "
    "to /10.250.6.215:50010",
    "081109 203521 1 WARN dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: 0x9f31a failed",
    "081109 203521 2 INFO dfs.DataNode$DataXceiver: Receiving block blk_7503483334202473044 "
    "src: /10.250.6.130:52930 dest: /10.250.6.214:50010",
    "081109 203522 1 INFO dfs.DataNode$PacketResponder: Received block blk_7503483334202473044 "
    "of size 67108864 from /10.250.6.130",
]


@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory) -> LogTokenizer:
    """A LogTokenizer trained on a small synthetic HDFS corpus.

    Uses ``tmp_path_factory`` (module-scoped) so the tokenizer is trained once
    per test module that requests it.
    """
    corpus_dir = tmp_path_factory.mktemp("tokenizer_corpus")
    corpus_path = corpus_dir / "train.log"
    tokenizer_path = corpus_dir / "tokenizer.json"

    # Write a small corpus — enough for BPE to build a minimal vocabulary
    with open(corpus_path, "w", encoding="utf-8") as f:
        for _ in range(50):               # repeat to give BPE enough merges
            for line in SYNTHETIC_LOG_LINES:
                f.write(line + "\n")

    tok = LogTokenizer(vocab_size=200, special_tokens=[
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "<EOS>", "<IP>", "<HEX>", "<DATE>", "<TIME>",
    ])
    tok.train(str(corpus_path), str(tokenizer_path))
    return tok
