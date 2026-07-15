"""VRAM Memory Tracker and NVIDIA GPU Power Wattage Profiler.

- Monitors peak allocated VRAM (`torch.cuda.max_memory_allocated()`).
- Profiles live GPU power consumption (`nvidia-smi --query-gpu=power.draw`) across Windows/Linux.
- Computes energy efficiency metrics (`Joules / 1M logs` and `Mean/Peak Watts`).
"""

import os
import time
import subprocess
import logging
import torch

logger = logging.getLogger(__name__)


def get_current_power_watts() -> float:
    """Queries `nvidia-smi` for instantaneous GPU power draw in Watts (`W`).

    Returns:
        Current power draw (`float`), or `0.0` if `nvidia-smi` is unavailable or running on CPU.
    """
    if not torch.cuda.is_available():
        return 0.0
    try:
        cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, universal_newlines=True)
        lines = output.strip().split("\n")
        if lines and lines[0].strip():
            return float(lines[0].strip())
    except Exception:
        pass
    return 0.0


def get_peak_vram_mb(device: str = "cuda") -> float:
    """Returns peak allocated GPU memory in Megabytes (`MB`)."""
    if not torch.cuda.is_available() or device == "cpu":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


class HardwareProfiler:
    """Context manager and time-series profiler tracking latency, VRAM, and energy consumption."""

    def __init__(self, device: str = "cuda", sample_interval_sec: float = 0.05):
        """Initializes HardwareProfiler.

        Args:
            device: Active device string (`cuda` or `cpu`).
            sample_interval_sec: Sampling rate interval for power telemetry.
        """
        self.device = device
        self.sample_interval_sec = sample_interval_sec
        self.start_time = 0.0
        self.end_time = 0.0
        self.power_samples = []
        self.peak_vram_start = 0.0
        self.peak_vram_end = 0.0

    def start(self):
        """Resets CUDA memory stats and starts profiling timer."""
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            self.peak_vram_start = get_peak_vram_mb(self.device)
        self.power_samples.clear()
        self.power_samples.append(get_current_power_watts())
        self.start_time = time.perf_counter()

    def sample(self):
        """Records a instantaneous power wattage sample."""
        self.power_samples.append(get_current_power_watts())

    def stop(self, total_logs_processed: int = 0) -> dict[str, float]:
        """Stops timer, queries final VRAM stats, and computes aggregate energy/throughput metrics.

        Args:
            total_logs_processed: Number of log lines processed during the profiled interval.

        Returns:
            Dictionary containing `elapsed_sec`, `latency_ms_per_log`, `logs_per_sec`,
            `peak_vram_mb`, `mean_watts`, `peak_watts`, and `joules_per_1m_logs`.
        """
        if torch.cuda.is_available() and self.device != "cpu":
            # Synchronize CUDA queue to ensure exact timing
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.power_samples.append(get_current_power_watts())

        elapsed = max(self.end_time - self.start_time, 1e-6)
        peak_vram = get_peak_vram_mb(self.device) if torch.cuda.is_available() and self.device != "cpu" else 0.0

        mean_watts = float(sum(self.power_samples) / len(self.power_samples)) if self.power_samples else 0.0
        peak_watts = float(max(self.power_samples)) if self.power_samples else 0.0

        total_joules = mean_watts * elapsed
        logs_per_sec = total_logs_processed / elapsed if total_logs_processed > 0 else 0.0
        latency_ms = (elapsed * 1000.0) / total_logs_processed if total_logs_processed > 0 else 0.0
        joules_per_1m = (total_joules / total_logs_processed) * 1_000_000 if total_logs_processed > 0 else 0.0

        return {
            "elapsed_sec": round(elapsed, 4),
            "latency_ms_per_log": round(latency_ms, 4),
            "logs_per_sec": round(logs_per_sec, 2),
            "peak_vram_mb": round(peak_vram, 2),
            "mean_watts": round(mean_watts, 2),
            "peak_watts": round(peak_watts, 2),
            "joules_per_1m_logs": round(joules_per_1m, 2),
        }
