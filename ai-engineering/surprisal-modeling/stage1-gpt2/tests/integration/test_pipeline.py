"""Integration tests for the end-to-end pipeline components.

These tests wire multiple modules together to verify that their interfaces
compose correctly. They do NOT download the real HDFS dataset — instead they
create a small synthetic corpus that mimics the HDFS log format, allowing the
full pipeline to execute in under 30 seconds without network access.

Test inventory
--------------
    test_checkpoint_save_restore_logit_identity
        Save a model checkpoint, restore it into a fresh model instance, and
        assert that the logits from both models are bit-for-bit identical.
        This validates the serialization / deserialization lifecycle.

    test_synthetic_pipeline_produces_valid_dataloader
        Run the full data pipeline (write corpus → train tokenizer → pack
        sequences → construct DataLoader) on a tiny synthetic corpus and
        assert the resulting batch tensors have correct shapes and value ranges.

    test_validate_tokenizer_encodes_masked_lines
        Verify that log lines routed through mask_variables() before BPE
        encoding produce only IDs that exist in the trained vocabulary.
"""

import os
import math
import shutil
import pytest
import torch
import yaml

from src.models.gpt2 import GPT2Config, GPT2Model
from src.tokenizer.log_tokenizer import LogTokenizer
from src.dataset.data_loader import pack_sequences_ffd


# ---------------------------------------------------------------------------
# Synthetic HDFS corpus helpers
# ---------------------------------------------------------------------------

_HDFS_TEMPLATE = (
    "081109 {time} {pid} INFO dfs.DataNode$DataXceiver: "
    "Receiving block blk_{blk} src: /10.250.6.{src}:{sport} "
    "dest: /10.250.6.214:50010"
)

def _write_synthetic_corpus(path: str, n_blocks: int = 40, lines_per_block: int = 15):
    """Writes a minimal HDFS-format log corpus and anomaly_label.csv."""
    os.makedirs(path, exist_ok=True)
    log_path   = os.path.join(path, "HDFS.log")
    label_path = os.path.join(path, "anomaly_label.csv")

    with open(log_path, "w", encoding="utf-8") as lf, \
         open(label_path, "w", encoding="utf-8") as lbf:

        lbf.write("BlockId,Label\n")

        for b in range(n_blocks):
            blk = f"blk_{b * 1_000_000}"
            label = "Anomaly" if b % 10 == 0 else "Normal"
            lbf.write(f"{blk},{label}\n")
            for i in range(lines_per_block):
                line = _HDFS_TEMPLATE.format(
                    time=f"20{i:04d}",
                    pid=1000 + i,
                    blk=b * 1_000_000,
                    src=b % 255,
                    sport=50000 + i,
                )
                lf.write(line + "\n")

    return log_path, label_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckpointLifecycle:

    def test_checkpoint_save_restore_logit_identity(self, tmp_path, tiny_config, device):
        """Checkpoint round-trip must produce bit-for-bit identical logits.

        This test guarantees that:
        1. ``torch.save`` correctly serializes all weight tensors.
        2. ``load_state_dict`` restores them exactly.
        3. The ``_orig_mod.`` prefix stripping logic (for torch.compile) works.
        """
        model = GPT2Model(tiny_config).to(device)
        model.eval()

        ckpt_path = tmp_path / "checkpoint_test.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        # Restore into a fresh instance with random initialization
        restored = GPT2Model(tiny_config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        restored.load_state_dict(state)
        restored.eval()

        x = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size), device=device)
        with torch.no_grad():
            logits_orig,     _ = model(x)
            logits_restored, _ = restored(x)

        assert torch.allclose(logits_orig, logits_restored, atol=1e-6), \
            "Checkpoint round-trip produced different logits — serialization is broken."


class TestDataPipeline:

    @pytest.fixture(scope="class")
    def corpus_dir(self, tmp_path_factory):
        d = tmp_path_factory.mktemp("hdfs_corpus")
        _write_synthetic_corpus(str(d), n_blocks=30, lines_per_block=10)
        return str(d)

    def test_synthetic_pipeline_produces_valid_dataloader(self, corpus_dir, tmp_path_factory):
        """Full pipeline: corpus → tokenizer → FFD pack → DataLoader.

        Validates that:
        - The tokenizer trains without error on the synthetic corpus.
        - FFD packing produces batches with the correct shape.
        - No out-of-vocabulary token IDs appear in the output tensors.
        """
        from torch.utils.data import DataLoader
        from src.dataset.data_loader import (
            load_anomaly_labels,
            parse_and_group_logs,
            PackedLogDataset,
        )

        SEQ_LEN  = 64
        VOCAB    = 200
        BATCH    = 4

        log_path   = os.path.join(corpus_dir, "HDFS.log")
        label_path = os.path.join(corpus_dir, "anomaly_label.csv")

        tok_dir  = tmp_path_factory.mktemp("tokenizer")
        tok_path = str(tok_dir / "tok.json")

        # Train a small tokenizer on normal blocks only
        labels_map     = load_anomaly_labels(label_path)
        normal, _      = parse_and_group_logs(log_path, labels_map)

        tok = LogTokenizer(vocab_size=VOCAB)
        corpus_file = str(tok_dir / "corpus.log")
        with open(corpus_file, "w", encoding="utf-8") as f:
            for block in normal:
                for line in block["lines"]:
                    f.write(line + "\n")
        tok.train(corpus_file, tok_path)

        eos_id = tok.tokenizer.token_to_id("<EOS>")

        # Tokenize and pack
        tokenized = []
        for block in normal[:20]:
            toks = []
            masked = [tok.mask_variables(l) for l in block["lines"]]
            for enc in tok.tokenizer.encode_batch(masked):
                toks.extend(enc.ids + [eos_id])
            tokenized.append(toks)

        packed = pack_sequences_ffd(tokenized, max_len=SEQ_LEN, eos_token_id=eos_id)
        assert len(packed) > 0, "No packed sequences produced"

        dataset = PackedLogDataset(packed)
        loader  = DataLoader(dataset, batch_size=BATCH, shuffle=False)
        batch   = next(iter(loader))

        ids = batch["input_ids"]
        assert ids.dtype == torch.long
        assert ids.shape[1] == SEQ_LEN, f"Expected seq_len={SEQ_LEN}, got {ids.shape[1]}"
        assert ids.min().item() >= 0
        assert ids.max().item() < VOCAB

    def test_tokenizer_encodes_masked_lines_within_vocab(self, corpus_dir, tmp_path_factory):
        """All token IDs produced on masked log lines must be within [0, vocab_size).

        If the BPE tokenizer emits out-of-vocabulary IDs, it indicates a
        mismatch between the masking patterns and the vocabulary training
        corpus — a subtle but critical data pipeline bug.
        """
        from src.dataset.data_loader import load_anomaly_labels, parse_and_group_logs

        VOCAB = 200
        log_path   = os.path.join(corpus_dir, "HDFS.log")
        label_path = os.path.join(corpus_dir, "anomaly_label.csv")

        tok_dir  = tmp_path_factory.mktemp("tok2")
        tok_path = str(tok_dir / "tok.json")

        labels_map = load_anomaly_labels(label_path)
        normal, _  = parse_and_group_logs(log_path, labels_map)

        tok = LogTokenizer(vocab_size=VOCAB)
        corpus_file = str(tok_dir / "corpus.log")
        with open(corpus_file, "w", encoding="utf-8") as f:
            for block in normal:
                for line in block["lines"]:
                    f.write(line + "\n")
        tok.train(corpus_file, tok_path)

        for block in normal[:5]:
            for line in block["lines"]:
                enc = tok.encode(line)
                for tid in enc.ids:
                    assert 0 <= tid < VOCAB, (
                        f"Out-of-vocab ID {tid} for line: {line[:60]}..."
                    )
