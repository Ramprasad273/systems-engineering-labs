"""Unit tests for evaluation metric computations.

All tests operate on pure Python scalars and lists — no GPU or model
inference required. This makes them run in milliseconds and suitable for
fast CI gating.

Test inventory
--------------
calculate_perplexity
    test_zero_loss_gives_unit_perplexity     — exp(0) = 1.0
    test_unit_loss_gives_euler_number        — exp(1) ≈ e
    test_large_loss_returns_infinity         — overflow guard returns inf
    test_negative_loss_gives_subunit_value   — exp(-1) < 1.0

calculate_classification_metrics
    test_perfect_classifier                  — all correct → accuracy=1, F1=1
    test_all_normal_predicted_on_mixed       — all zero preds
    test_confusion_matrix_values             — TP/FP/TN/FN counts
    test_precision_numerics                  — TP/(TP+FP) formula
    test_recall_numerics                     — TP/(TP+FN) formula
    test_f1_harmonic_mean                    — 2PR/(P+R) formula
    test_empty_prediction_no_crash           — empty lists → zeros, no exception
    test_length_mismatch_raises              — mismatched lists → AssertionError
"""

import math
import pytest

from src.utils.metrics import calculate_perplexity, calculate_classification_metrics


# ---------------------------------------------------------------------------
# calculate_perplexity
# ---------------------------------------------------------------------------

class TestCalculatePerplexity:

    def test_zero_loss_gives_unit_perplexity(self):
        """A model with perfect predictions has loss=0, perplexity=1.0."""
        assert math.isclose(calculate_perplexity(0.0), 1.0)

    def test_unit_loss_gives_euler_number(self):
        """Cross-entropy loss of 1.0 nat corresponds to perplexity = e."""
        assert math.isclose(calculate_perplexity(1.0), math.e, rel_tol=1e-9)

    def test_large_loss_returns_infinity(self):
        """Anomalous sequences can drive loss beyond float overflow bounds.
        The function must return float('inf') rather than raising OverflowError."""
        result = calculate_perplexity(1_000_000.0)
        assert result == float("inf")

    def test_negative_loss_gives_subunit_value(self):
        """Negative losses can occur on deterministic toy datasets; the result
        should be exp(-1) < 1 without error."""
        result = calculate_perplexity(-1.0)
        assert math.isclose(result, math.exp(-1.0))


# ---------------------------------------------------------------------------
# calculate_classification_metrics
# ---------------------------------------------------------------------------

class TestCalculateClassificationMetrics:

    def _make_metrics(self, preds, targets):
        return calculate_classification_metrics(preds, targets)

    def test_perfect_classifier(self):
        preds   = [1, 0, 1, 0, 1]
        targets = [1, 0, 1, 0, 1]
        m = self._make_metrics(preds, targets)
        assert math.isclose(m["accuracy"],  1.0)
        assert math.isclose(m["precision"], 1.0)
        assert math.isclose(m["recall"],    1.0)
        assert math.isclose(m["f1"],        1.0)

    def test_all_normal_predicted_on_mixed(self):
        """When all predictions are 0 (normal) on a mixed set, recall = 0
        and precision is undefined (guarded to 0.0)."""
        preds   = [0, 0, 0, 0]
        targets = [1, 0, 1, 0]
        m = self._make_metrics(preds, targets)
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["fn"] == 2
        assert math.isclose(m["recall"],    0.0)
        assert math.isclose(m["precision"], 0.0)

    def test_confusion_matrix_values(self):
        """Hand-verified confusion matrix for a known prediction vector."""
        preds   = [1, 1, 0, 0, 1]
        targets = [1, 0, 0, 1, 1]
        m = self._make_metrics(preds, targets)
        # TP: positions 0, 4  → 2
        # FP: position 1       → 1
        # TN: position 2       → 1
        # FN: position 3       → 1
        assert m["tp"] == 2
        assert m["fp"] == 1
        assert m["tn"] == 1
        assert m["fn"] == 1

    def test_precision_numerics(self):
        preds   = [1, 1, 0, 0, 1]
        targets = [1, 0, 0, 1, 1]
        m = self._make_metrics(preds, targets)
        expected_precision = 2 / (2 + 1)   # TP / (TP + FP)
        assert math.isclose(m["precision"], expected_precision)

    def test_recall_numerics(self):
        preds   = [1, 1, 0, 0, 1]
        targets = [1, 0, 0, 1, 1]
        m = self._make_metrics(preds, targets)
        expected_recall = 2 / (2 + 1)      # TP / (TP + FN)
        assert math.isclose(m["recall"], expected_recall)

    def test_f1_harmonic_mean(self):
        preds   = [1, 1, 0, 0, 1]
        targets = [1, 0, 0, 1, 1]
        m = self._make_metrics(preds, targets)
        p, r = m["precision"], m["recall"]
        expected_f1 = 2 * p * r / (p + r)
        assert math.isclose(m["f1"], expected_f1)

    def test_empty_prediction_no_crash(self):
        """Empty inputs must return 0.0 for all metrics without raising."""
        m = self._make_metrics([], [])
        assert m["accuracy"] == 0.0
        assert m["f1"] == 0.0

    def test_length_mismatch_raises(self):
        """Mismatched prediction/target lengths indicate a data alignment bug;
        the function must raise AssertionError rather than silently truncating."""
        with pytest.raises(AssertionError):
            self._make_metrics([1, 0, 1], [1, 0])
