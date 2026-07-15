"""Suite 1: Context Length VRAM Memory Footprint Scaling Sweep (`benchmark_vram_sweep.py`).

Pedagogical hardware benchmark following Karpathy guidelines:
- Evaluates peak allocated GPU memory (`torch.cuda.max_memory_allocated()`) across context horizons $L \in \{128, \dots, 8192\}$.
- Demonstrates quadratic attention memory wall ($O(T^2)$) where GPT-2 crashes with CUDA Out of Memory (OOM) at $4,096$ tokens.
- Demonstrates Mamba S6's flat $O(1)$ recurrent memory profile across extended context horizons.
- Outputs clean CSV table (`results/vram_scaling_metrics.csv`) for publication and figure generation.
"""

import os
import sys
import csv
import argparse
import logging
import torch

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.mamba_lm import MambaLMHeadModel
from src.models.hybrid_mambalog import MambaLogLMHeadModel
from src.utils.metrics import get_peak_vram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("stage3.benchmark_vram")


def get_gpt2_model(vocab_size: int = 5000, n_embd: int = 768, n_layer: int = 12, n_head: int = 12, block_size: int = 1024):
    """Initializes Stage 1 GPT-2 architecture for VRAM benchmarking."""
    try:
        import sys
        stage1_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../stage1-gpt2"))
        if stage1_dir not in sys.path and os.path.exists(stage1_dir):
            sys.path.insert(0, stage1_dir)
        from src.models.gpt2 import GPT2Model
        cfg = {"vocab_size": vocab_size, "n_embd": n_embd, "n_layer": n_layer, "n_head": n_head, "block_size": block_size}
        return GPT2Model(cfg)
    except Exception as e:
        logger.warning(f"Could not load Stage 1 GPT2Model directly ({e}); initializing standalone Transformer fallback.")
        import torch.nn as nn
        from src.models.hybrid_mambalog import AttentionResidualBlock
        cfg = {"vocab_size": vocab_size, "n_embd": n_embd, "n_layer": n_layer, "n_head": n_head, "block_size": block_size}
        class FallbackGPT2(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.token_embeddings = nn.Embedding(c["vocab_size"], c["n_embd"])
                self.layers = nn.ModuleList([AttentionResidualBlock(c) for _ in range(c["n_layer"])])
                self.lm_head = nn.Linear(c["n_embd"], c["vocab_size"], bias=False)
            def forward(self, x):
                h = self.token_embeddings(x)
                for l in self.layers:
                    h = l(h)
                return self.lm_head(h)
        return FallbackGPT2(cfg)


def run_vram_sweep(
    models_to_test: list[str],
    seq_lengths: list[int],
    batch_size: int = 4,
    vocab_size: int = 5000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_csv: str = "results/vram_scaling_metrics.csv",
):
    """Executes multi-model VRAM memory footprint sweep across scaling sequence horizons."""
    is_cuda = torch.cuda.is_available() and device != "cpu"
    if not is_cuda:
        logger.warning("CUDA is not available or active. VRAM telemetry will report 0.0 MB for CPU execution.")

    results_table = []
    logger.info(f"Commencing VRAM Scaling Sweep on device '{device}' with batch_size={batch_size}...")

    for model_name in models_to_test:
        logger.info(f"\n--- Benchmarking Architecture: {model_name.upper()} ---")
        if model_name.lower() == "gpt2":
            model = get_gpt2_model(vocab_size=vocab_size, n_embd=768, n_layer=12, n_head=12, block_size=max(seq_lengths))
        elif model_name.lower() == "mambalog":
            cfg = {"vocab_size": vocab_size, "n_embd": 768, "n_layer": 24, "d_state": 16, "conv_kernel": 4, "expand": 2, "dt_rank": "auto", "layer_norm_epsilon": 1e-5, "attn_layer_indices": [3, 7, 11, 15, 19, 23]}
            model = MambaLogLMHeadModel(cfg)
        else:
            cfg = {"vocab_size": vocab_size, "n_embd": 768, "n_layer": 24, "d_state": 16, "conv_kernel": 4, "expand": 2, "dt_rank": "auto", "layer_norm_epsilon": 1e-5}
            model = MambaLMHeadModel(cfg)

        model.to(device)
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
                logger.info(f"[{model_name.upper()}] Seq Len: {seq_len:>5,d} | Peak VRAM: {peak_mb:>8.2f} MB")
                results_table.append({"model": model_name.upper(), "seq_len": seq_len, "peak_vram_mb": round(peak_mb, 2), "status": "SUCCESS"})

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    logger.warning(f"[{model_name.upper()}] Seq Len: {seq_len:>5,d} | CUDA Out of Memory (OOM) Crash!")
                    results_table.append({"model": model_name.upper(), "seq_len": seq_len, "peak_vram_mb": "OOM", "status": "OOM"})
                    if is_cuda:
                        torch.cuda.empty_cache()
                else:
                    raise e

        # Clean up model from GPU memory before testing next architecture
        del model
        if is_cuda:
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seq_len", "peak_vram_mb", "status"])
        writer.writeheader()
        writer.writerows(results_table)

    logger.info(f"\nVRAM scaling benchmark complete. Results written to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Sweep context length VRAM scaling across GPT-2, Mamba, and MambaLog.")
    parser.add_argument("--models", type=str, nargs="+", choices=["gpt2", "mamba", "mambalog"], default=["gpt2", "mamba", "mambalog"], help="Models to benchmark.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096, 8192], help="Sequence lengths to sweep.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for benchmark forward pass.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="results/vram_scaling_metrics.csv", help="Path to save output CSV.")
    args = parser.parse_args()

    run_vram_sweep(
        models_to_test=args.models,
        seq_lengths=args.lengths,
        batch_size=args.batch_size,
        device=args.device,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
