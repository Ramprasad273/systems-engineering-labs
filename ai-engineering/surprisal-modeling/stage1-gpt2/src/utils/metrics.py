"""Perplexity Evaluation, Classification Confusion Matrices, and Hardware Benchmarking Utilities.

This module provides academic evaluation primitives for unsupervised surprisal anomaly detection.
Includes numerical overflow mitigation for exponential cross-entropy loss transformations, exact
binary classification metric derivation (Precision, Recall, F1, Accuracy), and CUDA memory peak
resident allocation sweeping across arbitrary context sequence horizons.
"""

import math
import torch


def calculate_perplexity(loss: float) -> float:
    """Derives statistical perplexity from mean cross-entropy loss with overflow guard.

    Perplexity represents the exponentiated average per-token cross-entropy loss ($\exp(L)$).
    In anomalous sequences where unmasked tokens diverge sharply from the learned language
    model distribution, loss can spike sufficiently to trigger floating-point overflow.

    Args:
        loss: Scalar mean cross-entropy loss value.

    Returns:
        Perplexity score, or float('inf') if exponentiation overflows standard bounds.
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')


def calculate_classification_metrics(predictions: list[int], targets: list[int]) -> dict[str, float]:
    """Computes exact binary classification evaluation statistics from prediction vectors.

    Aggregates True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    to derive standard academic benchmark evaluation metrics.

    Args:
        predictions: Binary prediction flags (1=Anomaly, 0=Normal).
        targets: Ground-truth binary classification labels (1=Anomaly, 0=Normal).

    Returns:
        Dictionary mapping metric names (`accuracy`, `precision`, `recall`, `f1`, `tp`, `fp`, `tn`, `fn`)
        to computed floating-point values or integer counts.

    Raises:
        AssertionError: If prediction and target vectors differ in length.
    """
    assert len(predictions) == len(targets), "Evaluation error: Predictions and targets must be identical in length."
    
    # Derive exact confusion matrix elements
    tp = sum(1 for p, t in zip(predictions, targets) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, targets) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(predictions, targets) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(predictions, targets) if p == 0 and t == 1)
    
    total = len(targets)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def get_peak_vram(device: str = "cuda") -> float:
    """Queries NVIDIA CUDA runtime for peak resident memory allocated on the device.

    Args:
        device: Target hardware device string identifier.

    Returns:
        Peak memory allocation footprint in Megabytes (MB).
    """
    if not torch.cuda.is_available() or device == "cpu":
        return 0.0
        
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 * 1024)


def sweep_vram_footprint(
    model: torch.nn.Module, 
    device: str = "cuda", 
    seq_lengths: list[int] = None, 
    batch_size: int = 1, 
    vocab_size: int = 5000
) -> dict[int, float]:
    """Sweeps context sequence horizons to benchmark hardware VRAM memory scalability.

    Simulates dummy forward inference passes across scaling sequence lengths ($T \in [128, 2048]$)
    under mixed-precision `bfloat16` autocasting to record empirical peak memory consumption.

    Args:
        model: PyTorch neural network module to benchmark.
        device: Active hardware accelerator device string.
        seq_lengths: List of discrete sequence length horizons to evaluate.
        batch_size: Fixed batch size for simulated inference tensors.
        vocab_size: Upper token ID bound for dummy integer tensor sampling.

    Returns:
        Dictionary mapping sequence length integers to peak allocated VRAM in Megabytes (MB).
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]
        
    sweep_results = {}
    is_cuda = torch.cuda.is_available() and device != "cpu"
    
    model.eval()
    
    for seq_len in seq_lengths:
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            
        # Generate synthetic input sequence matching active horizon bound
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            if is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
                
        peak_mb = get_peak_vram(device) if is_cuda else 0.0
        sweep_results[seq_len] = round(peak_mb, 2)
        
    return sweep_results
