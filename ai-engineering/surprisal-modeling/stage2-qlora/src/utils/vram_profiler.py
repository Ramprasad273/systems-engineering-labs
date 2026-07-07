"""VRAM footprint tracking utility for model profiling and ablation comparisons."""

import torch
import logging

logger = logging.getLogger("stage2.vram_profiler")


def get_peak_vram_mb() -> float:
    """Returns peak GPU allocated VRAM in megabytes."""
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        return round(peak_bytes / (1024 * 1024), 2)
    return 0.0


def reset_vram_peak():
    """Resets peak GPU VRAM tracking counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
