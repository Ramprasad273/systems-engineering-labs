"""Evaluation metrics, surprisal threshold derivation, and anomaly classification utilities.

- Unsupervised surprisal anomaly threshold derivation (`tau = mu + 3 * sigma`).
- Exact binary classification confusion matrix accounting (`TP, FP, TN, FN, Precision, Recall, F1`).
- Numerical overflow guards (`exp(loss)` perplexity scaling).
"""

import math
import numpy as np
import torch


def calculate_perplexity(loss: float) -> float:
    """Derives statistical perplexity (`exp(loss)`) with overflow mitigation.

    Args:
        loss: Mean per-token cross-entropy scalar loss.

    Returns:
        Exponentiated perplexity, or float('inf') if numerical overflow occurs.
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def calculate_surprisal_threshold(
    val_perplexities: list[float], k_factor: float = 3.0
) -> tuple[float, float, float]:
    """Computes Gaussian surprisal anomaly threshold (`tau = mu + k * sigma`) from validation scores.

    WHY: Healthy HDFS log sequences follow a tightly bounded perplexity distribution under a well-trained LM.
    Sequences exceeding `mu + 3 * sigma` exhibit statistical surprisal characteristic of system anomalies.

    Args:
        val_perplexities: List of per-sequence perplexity scores on healthy validation set.
        k_factor: Standard deviation multiplier (`3.0` corresponds to ~99.7% confidence interval).

    Returns:
        Tuple containing `(threshold_tau, mean_mu, std_sigma)`.
    """
    arr = np.array([p for p in val_perplexities if not math.isinf(p) and not math.isnan(p)], dtype=np.float64)
    if len(arr) == 0:
        return 10.0, 5.0, 1.0  # Fallback default if validation set is empty or all inf
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    tau = mu + float(k_factor) * sigma
    return tau, mu, sigma


def calculate_classification_metrics(predictions: list[int], targets: list[int]) -> dict[str, float]:
    """Computes exact binary classification statistics from prediction and target vectors.

    Args:
        predictions: Binary prediction flags (`1=Anomaly`, `0=Normal`).
        targets: Ground-truth binary classification labels (`1=Anomaly`, `0=Normal`).

    Returns:
        Dictionary containing `accuracy`, `precision`, `recall`, `f1`, and raw counts (`tp`, `fp`, `tn`, `fn`).

    Raises:
        AssertionError: If prediction and target vectors differ in length.
    """
    assert len(predictions) == len(targets), "Evaluation error: Predictions and targets must be identical in length."

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
        "fn": int(fn),
    }


def get_peak_vram(device: str = "cuda") -> float:
    """Queries active hardware runtime for peak resident memory allocation.

    Args:
        device: Device string (`cuda` or `cpu`).

    Returns:
        Peak VRAM allocated in Megabytes (MB).
    """
    if not torch.cuda.is_available() or device == "cpu":
        return 0.0
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 * 1024)


def sweep_vram_footprint(
    model: torch.nn.Module,
    device: str = "cuda",
    seq_lengths: list[int] | None = None,
    batch_size: int = 4,
    vocab_size: int = 5000,
) -> dict[int, float]:
    """Sweeps context sequence horizons to benchmark VRAM memory scalability (`O(1)` Mamba vs `O(T^2)` GPT-2).

    Args:
        model: PyTorch model module to benchmark.
        device: Target hardware accelerator string (`cuda` or `cpu`).
        seq_lengths: List of discrete sequence lengths (`T \in [128, 8192]`).
        batch_size: Batch size for simulated forward pass tensors.
        vocab_size: Upper token ID bound for synthetic integer inputs.

    Returns:
        Dictionary mapping sequence length integers (`T`) to peak allocated VRAM in MB.
    """
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

    sweep_results = {}
    is_cuda = torch.cuda.is_available() and device != "cpu"
    model.eval()

    for seq_len in seq_lengths:
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        try:
            dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            with torch.no_grad():
                if is_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _ = model(dummy_input)
                else:
                    _ = model(dummy_input)
            peak_mb = get_peak_vram(device) if is_cuda else 0.0
            sweep_results[seq_len] = round(peak_mb, 2)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                sweep_results[seq_len] = float("inf")
                if is_cuda:
                    torch.cuda.empty_cache()
            else:
                raise e

    return sweep_results
