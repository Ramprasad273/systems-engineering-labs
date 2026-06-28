"""Unit tests for the First-Fit Decreasing (FFD) sequence bin-packing algorithm.

FFD bin-packing is the most algorithmically novel component of this pipeline.
It replaces the standard padding-to-max-length approach used in most NLP
pipelines, achieving near-100% attention matrix density on variable-length
syslog sequences.

These tests verify the core invariants of the packing algorithm independently
of the data download or tokenization steps.

Test inventory
--------------
    test_all_sequences_exactly_max_len     — no under- or over-length bins
    test_eos_fills_residual_slack          — padding tokens fill remaining space
    test_empty_input_returns_empty         — empty list → empty list
    test_single_exact_fit_sequence         — sequence == max_len, no EOS added
    test_oversized_sequence_truncated      — seq > max_len truncated, not dropped
    test_small_sequences_packed_densely    — multiple smalls pack into single bin
    test_deterministic_output             — same input → identical output
    test_large_batch_all_correct_length    — 10 000 random sequences all correct
"""

import random
import pytest

from src.dataset.data_loader import pack_sequences_ffd


EOS_ID = 5      # matches the production config


# ---------------------------------------------------------------------------
# Core invariant: every output sequence must be exactly max_len tokens
# ---------------------------------------------------------------------------

class TestPackingInvariants:

    def test_all_sequences_exactly_max_len(self):
        max_len = 32
        sequences = [[1] * random.randint(1, 20) for _ in range(200)]
        packed = pack_sequences_ffd(sequences, max_len=max_len, eos_token_id=EOS_ID)
        assert all(len(s) == max_len for s in packed), (
            f"Found sequences with length != {max_len}: "
            f"{set(len(s) for s in packed)}"
        )

    def test_eos_fills_residual_slack(self):
        """When a bin has remaining capacity after packing, that capacity must
        be filled with EOS tokens — never with zeros or arbitrary values."""
        max_len = 16
        # One short sequence → one bin, tail filled with EOS
        packed = pack_sequences_ffd([[1, 2, 3, 4]], max_len=max_len, eos_token_id=EOS_ID)
        assert len(packed) == 1
        # Tail tokens must all be EOS
        sequence = packed[0]
        assert all(t == EOS_ID for t in sequence[4:])

    def test_empty_input_returns_empty(self):
        """Edge case: the empty list must return the empty list, not crash."""
        packed = pack_sequences_ffd([], max_len=32, eos_token_id=EOS_ID)
        assert packed == []

    def test_single_exact_fit_sequence(self):
        """A sequence that exactly fills max_len should occupy a bin by itself
        with no EOS padding appended."""
        max_len = 16
        sequence = list(range(max_len))
        packed = pack_sequences_ffd([sequence], max_len=max_len, eos_token_id=EOS_ID)
        assert len(packed) == 1
        assert packed[0] == sequence

    def test_oversized_sequence_truncated_not_dropped(self):
        """Sequences longer than max_len must be hard-truncated to max_len
        rather than silently dropped from the output."""
        max_len = 8
        long_seq = list(range(100))
        packed = pack_sequences_ffd([long_seq], max_len=max_len, eos_token_id=EOS_ID)
        assert len(packed) == 1
        assert len(packed[0]) == max_len
        assert packed[0] == list(range(max_len))

    def test_small_sequences_pack_into_fewer_bins(self):
        """FFD should pack multiple short sequences into the same bin.
        Four sequences of length max_len//4 should fit into a single bin."""
        max_len = 16
        quarter = max_len // 4
        sequences = [[i] * quarter for i in range(4)]
        packed = pack_sequences_ffd(sequences, max_len=max_len, eos_token_id=EOS_ID)
        # All four should fit in one bin
        assert len(packed) == 1

    def test_deterministic_output(self):
        """The same input must always produce the same packed output —
        FFD is a deterministic algorithm."""
        sequences = [[j for j in range(i % 10 + 1)] for i in range(50)]
        packed_a = pack_sequences_ffd(sequences, max_len=32, eos_token_id=EOS_ID)
        packed_b = pack_sequences_ffd(sequences, max_len=32, eos_token_id=EOS_ID)
        assert packed_a == packed_b

    def test_large_batch_all_correct_length(self):
        """Stress test: 10 000 random-length sequences must all produce
        exactly max_len output bins without any assertion errors."""
        max_len = 64
        rng = random.Random(0)
        sequences = [[1] * rng.randint(1, 50) for _ in range(10_000)]
        packed = pack_sequences_ffd(sequences, max_len=max_len, eos_token_id=EOS_ID)
        assert all(len(s) == max_len for s in packed)
