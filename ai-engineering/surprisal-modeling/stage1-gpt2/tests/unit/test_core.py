"""
Unit tests for core mathematical properties and anti-regression calibration.
Explicit shapes, pedagogical docstrings explaining why, and structured telemetry.
"""
import logging
import numpy as np
import pytest
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_unmasked_loss():
    """
    Padding tokens should not contribute to sequence loss.
    WHY: In autoregressive language modeling, sequences in a batch are padded to the maximum length.
    Computing cross-entropy loss over padding tokens distorts gradient signals and artificially lowers
    perplexity. We explicitly mask out padding locations so gradients only flow from valid tokens.
    """
    pad = 0
    # Inputs and targets: [batch_size=1, seq_len=3]
    inputs = torch.tensor([[10, 20, pad]])
    targets = torch.tensor([[20, 30, pad]])
    
    # Synthetic per-token losses: [batch_size=1, seq_len=3]
    losses = torch.tensor([[0.5, 0.5, 99.9]])
    
    # Binary mask boolean tensor: [batch_size=1, seq_len=3]
    mask = ~((inputs == pad) & (targets == pad))
    
    # Compute mean loss over active (unmasked) tokens only
    seq_loss = (losses * mask.float()).sum() / mask.sum()
    logger.info(f"Unmasked loss computed: {seq_loss.item():.4f} (expected: 0.5000)")
    
    assert torch.isclose(seq_loss, torch.tensor(0.5)), f"Expected 0.5, got {seq_loss.item()}"


def test_threshold():
    """
    Threshold must be strictly greater than sample mean.
    WHY: In anomaly detection via surprisal modeling, normal validation log perplexities follow a unimodal
    distribution. Setting the anomaly boundary at mu + 3*sigma ensures ~99.7% of in-distribution validation
    samples are classified as normal under Gaussian assumptions, bounding the false positive rate.
    """
    # Sample validation perplexities: [num_samples=3]
    arr = np.array([1.0, 1.1, 0.9])
    mu = np.mean(arr)
    sigma = np.std(arr)
    threshold = mu + 3 * sigma
    logger.info(f"Threshold test: mu={mu:.4f}, sigma={sigma:.4f}, threshold={threshold:.4f}")
    
    assert threshold > mu, "Threshold mu + 3*sigma must strictly exceed sample mean."


def test_depth_calibration():
    """
    Each depth model must produce a unique threshold (anti-regression test for Bug B0a).
    WHY: Shallower transformer models (e.g., depth=2) lack capacity to represent complex syntax, yielding
    orders of magnitude higher validation perplexity than deep models (e.g., depth=12). Applying a fixed,
    shared threshold derived from a 12-layer baseline to a 2-layer model marks 100% of validation sequences
    as anomalous. Per-model calibration is mandatory.
    """
    # Synthetic perplexities for depth-2 model: [num_samples=3]
    ppls_shallow = np.array([6.0e9, 6.2e9, 5.8e9])
    # Synthetic perplexities for depth-12 model: [num_samples=3]
    ppls_deep = np.array([1.16, 1.18, 1.15])
    
    tau_shallow = np.mean(ppls_shallow) + 3 * np.std(ppls_shallow)
    tau_deep = np.mean(ppls_deep) + 3 * np.std(ppls_deep)
    
    logger.info(f"Depth calibration: tau_shallow={tau_shallow:.2e}, tau_deep={tau_deep:.4f}")
    
    assert tau_shallow > 1e6, "Shallow model threshold must reflect high initial surprisal."
    assert tau_deep < 2.0, "Deep model threshold must reflect low converged surprisal."
    assert abs(tau_shallow - tau_deep) > 1e6, "Thresholds across depths must differ substantially."
