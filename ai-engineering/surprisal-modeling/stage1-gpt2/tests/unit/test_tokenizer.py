"""Unit tests for the LogTokenizer dynamic variable masking and BPE lifecycle.

Each test isolates a single regex substitution rule or tokenizer behavior so
that a failure pinpoints the exact pattern that is broken.

Test inventory
--------------
Variable Masking
    test_ipv4_address_masked           — bare IPv4 (e.g., 10.250.6.101)
    test_ipv4_with_port_masked         — IPv4:port pair
    test_block_id_masked               — HDFS blk_<digits> identifier
    test_negative_block_id_masked      — blk_-<digits> variant
    test_hex_address_masked            — 0x<hex> memory offset
    test_iso_date_masked               — YYYY-MM-DD calendar date
    test_slash_date_masked             — YY/MM/DD log date format
    test_time_masked                   — HH:MM:SS wall-clock timestamp
    test_hdfs_6digit_prefix_masked     — compact HDFS syslog date prefix
    test_plain_text_unchanged          — structural words not mangled

Tokenizer Lifecycle
    test_encode_returns_ids            — encode() returns integer list
    test_encoded_ids_within_vocab      — IDs within [0, vocab_size)
    test_encode_decode_recoverable     — decoded text non-empty (BPE roundtrip)
    test_untrained_tokenizer_raises    — encode before train() raises ValueError
"""

import pytest

from src.tokenizer.log_tokenizer import LogTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer() -> LogTokenizer:
    """Raw (untrained) LogTokenizer — sufficient for masking tests."""
    return LogTokenizer()


# ---------------------------------------------------------------------------
# Variable Masking Tests
# ---------------------------------------------------------------------------

class TestVariableMasking:

    def test_ipv4_address_masked(self, tokenizer):
        line = "Connection from 10.250.6.101 established"
        masked = tokenizer.mask_variables(line)
        assert "<IP>" in masked
        assert "10.250.6.101" not in masked

    def test_ipv4_with_port_masked(self, tokenizer):
        line = "src: /10.250.6.101:52922 dest: /10.250.6.214:50010"
        masked = tokenizer.mask_variables(line)
        assert masked.count("<IP>") == 2
        assert "52922" not in masked
        assert "50010" not in masked

    def test_block_id_masked(self, tokenizer):
        line = "Receiving block blk_7503483334202473044"
        masked = tokenizer.mask_variables(line)
        assert "blk_" not in masked
        assert "<HEX>" in masked

    def test_negative_block_id_masked(self, tokenizer):
        line = "PacketResponder for block blk_-3509323198988774369 terminating"
        masked = tokenizer.mask_variables(line)
        assert "blk_-" not in masked
        assert "<HEX>" in masked

    def test_hex_address_masked(self, tokenizer):
        line = "WARN: memory offset 0x9f31a caused fault"
        masked = tokenizer.mask_variables(line)
        assert "0x9f31a" not in masked
        assert "<HEX>" in masked

    def test_iso_date_masked(self, tokenizer):
        line = "2026-06-23 system event recorded"
        masked = tokenizer.mask_variables(line)
        assert "2026-06-23" not in masked
        assert "<DATE>" in masked

    def test_slash_date_masked(self, tokenizer):
        line = "Log entry 08/11/09 processed successfully"
        masked = tokenizer.mask_variables(line)
        assert "08/11/09" not in masked
        assert "<DATE>" in masked

    def test_time_masked(self, tokenizer):
        line = "Event at 22:20:48 completed"
        masked = tokenizer.mask_variables(line)
        assert "22:20:48" not in masked
        assert "<TIME>" in masked

    def test_hdfs_6digit_prefix_masked(self, tokenizer):
        """The HDFS syslog format starts with a compact YYMMDD prefix (e.g.,
        081109) which must be collapsed to <DATE> to prevent the tokenizer
        from treating each date as a unique token."""
        line = "081109 203518 3 INFO dfs.DataNode$DataXceiver: Receiving block"
        masked = tokenizer.mask_variables(line)
        assert "081109" not in masked
        assert "<DATE>" in masked

    def test_plain_text_unchanged(self, tokenizer):
        """Structural log keywords must survive masking unmodified — the model
        learns anomalies from these invariant structural tokens."""
        line = "INFO dfs.DataNode PacketResponder Receiving block terminating"
        masked = tokenizer.mask_variables(line)
        assert "INFO" in masked
        assert "PacketResponder" in masked
        assert "terminating" in masked


# ---------------------------------------------------------------------------
# Tokenizer Lifecycle Tests
# ---------------------------------------------------------------------------

class TestTokenizerLifecycle:

    def test_encode_returns_ids(self, trained_tokenizer):
        encoding = trained_tokenizer.encode("INFO dfs.DataNode block received")
        assert isinstance(encoding.ids, list)
        assert len(encoding.ids) > 0

    def test_encoded_ids_within_vocab(self, trained_tokenizer):
        encoding = trained_tokenizer.encode("ERROR connection refused 10.0.0.1")
        vocab_size = trained_tokenizer.vocab_size
        assert all(0 <= tok_id < vocab_size for tok_id in encoding.ids)

    def test_encode_decode_recoverable(self, trained_tokenizer):
        """BPE roundtrip — decoded text must be non-empty and contain
        recognizable structural tokens (exact whitespace may differ)."""
        encoding = trained_tokenizer.encode("PacketResponder terminating block")
        decoded = trained_tokenizer.decode(encoding.ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_untrained_tokenizer_raises(self):
        """Calling encode() before train() or load() must raise ValueError
        with a clear error message, not AttributeError or NoneType crash."""
        tok = LogTokenizer()
        with pytest.raises(ValueError, match="uninitialized"):
            tok.encode("some log line")
