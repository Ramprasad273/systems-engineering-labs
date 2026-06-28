"""Results Analysis and Paper Table Generator.

Loads all experimental results from data/ and renders paper-ready formatted
tables. Run this after completing the main training run (train.py + evaluate.py)
and any ablation studies (scripts/ablation_*.py).

Outputs
-------
  stdout : all paper tables, formatted for direct copy-paste into LaTeX or Markdown
  data/paper_summary.json : structured summary of all results

Usage
-----
    # After main training:
    python scripts/analyze_results.py --config config/stage1_config.yaml

    # After all ablations:
    python scripts/analyze_results.py \\
        --config config/stage1_config.yaml \\
        --eval-results data/stage1_eval_results.json \\
        --training-log data/stage1_results.json \\
        --vocab-ablation data/ablations/ablation_vocab.json \\
        --depth-ablation data/ablations/ablation_depth.json \\
        --threshold-sensitivity data/ablations/threshold_sensitivity.json \\
        --token-stability data/ablations/token_stability.json
"""

import argparse
import json
import math
import os
import sys

import yaml


def _load_json_safe(path: str) -> dict | None:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _banner(title: str, width: int = 70) -> str:
    return f"\n{'=' * width}\n{title.center(width)}\n{'=' * width}"


def _print_main_results(eval_results: dict):
    print(_banner("MAIN EXPERIMENT RESULTS"))

    cal = eval_results.get("calibration", {})
    tst = eval_results.get("test_metrics", {})
    ckpt = eval_results.get("checkpoint_evaluated", "N/A")

    print(f"\nCheckpoint: {ckpt}")
    print(f"\nCalibration (Normal Validation Split):")
    print(f"  Mean perplexity  (μ) : {cal.get('mean_val_perplexity', 0):.6f}")
    print(f"  Std deviation    (σ) : {cal.get('std_val_perplexity', 0):.6f}")
    print(f"  Threshold        (τ) : {cal.get('threshold_tau', 0):.6f}")

    print(f"\nClassification Results (Test Split):")
    print(f"  Accuracy         : {tst.get('accuracy', 0):.4f}")
    print(f"  Precision        : {tst.get('precision', 0):.4f}")
    print(f"  Recall           : {tst.get('recall', 0):.4f}")
    print(f"  F1 Score         : {tst.get('f1', 0):.4f}")

    tp = tst.get("tp", 0)
    fp = tst.get("fp", 0)
    tn = tst.get("tn", 0)
    fn = tst.get("fn", 0)
    print(f"\nConfusion Matrix:")
    print(f"  {'':20}  {'Pred Normal':>14}  {'Pred Anomaly':>14}")
    print(f"  {'True Normal':20}  {tn:>14,} (TN)  {fp:>14,} (FP)")
    print(f"  {'True Anomaly':20}  {fn:>14,} (FN)  {tp:>14,} (TP)")

    vram = eval_results.get("vram_sweep_mb", {})
    if vram:
        print(f"\nVRAM Scalability (FlashAttention O(T) memory):")
        print(f"  {'Seq Len T':>12}  {'VRAM (MB)':>12}  {'Growth (%)':>12}")
        base = vram.get("128", 0)
        for k, v in sorted((int(k), v) for k, v in vram.items()):
            growth = (v - base) / base * 100 if base > 0 else 0.0
            print(f"  {k:>12,}  {v:>12.2f}  {growth:>11.2f}%")


def _print_training_curve(training_log: dict | list):
    """Print ASCII loss curve from training JSON."""
    print(_banner("TRAINING LOSS CURVE (ASCII)"))
    if not training_log:
        print("  No training log data available.")
        return

    # If training_log is a dict of lists (from train.py telemetry), convert to list of dicts
    if isinstance(training_log, dict):
        steps = training_log.get("steps", [])
        train_losses = training_log.get("train_loss", [])
        val_ppls = training_log.get("val_perplexity", [])
        samples = []
        for i in range(len(steps)):
            samples.append({
                "step": steps[i],
                "train_loss": train_losses[i] if i < len(train_losses) else 0,
                "val_perplexity": val_ppls[i] if i < len(val_ppls) else 0
            })
        training_log = samples

    # Sample at most 20 points
    n = len(training_log)
    stride = max(1, n // 20)
    samples = training_log[::stride]

    max_loss = max(s.get("train_loss", 0) for s in samples)
    width    = 40

    print(f"\n  {'Step':>6}  {'Train Loss':>12}  {'Val PPL':>10}  Chart")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*width}")
    for s in samples:
        step       = s.get("step", 0)
        train_loss = s.get("train_loss", 0)
        val_ppl    = s.get("val_perplexity", 0)
        bar_len    = int((train_loss / max_loss) * width) if max_loss > 0 else 0
        bar        = "█" * bar_len
        print(f"  {step:>6}  {train_loss:>12.4f}  {val_ppl:>10.4f}  {bar}")



def _print_vocab_ablation(vocab_data: list):
    print(_banner("ABLATION: Vocabulary Size"))
    print(f"\n  {'V':>8}  {'Val PPL':>9}  {'τ':>8}  {'F1':>7}  {'Precision':>10}  {'Recall':>8}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*8}")
    for r in vocab_data:
        marker = " ← main" if r["vocab_size"] == 5000 else ""
        print(
            f"  {r['vocab_size']:>8,}  {r['val_perplexity']:>9.4f}  {r['threshold_tau']:>8.4f}  "
            f"{r['f1']:>7.4f}  {r['precision']:>10.4f}  {r['recall']:>8.4f}{marker}"
        )


def _print_depth_ablation(depth_data: list):
    print(_banner("ABLATION: Model Depth"))
    print(f"\n  {'L':>4}  {'Params':>12}  {'Val PPL':>9}  {'τ':>8}  {'F1':>7}  {'Precision':>10}  {'Recall':>8}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*8}")
    for r in depth_data:
        marker = " ← main" if r["n_layer"] == 12 else ""
        print(
            f"  {r['n_layer']:>4}  {r['n_params']:>12,}  {r['val_perplexity']:>9.4f}  "
            f"{r['threshold_tau']:>8.4f}  {r['f1']:>7.4f}  "
            f"{r['precision']:>10.4f}  {r['recall']:>8.4f}{marker}"
        )


def _print_threshold_sensitivity(data: dict):
    print(_banner("ABLATION: Threshold Sensitivity (τ = μ + k·σ)"))
    results = data.get("results", [])
    mu, sig = data.get("mu", 0), data.get("sigma", 0)
    print(f"\n  μ = {mu:.4f}  σ = {sig:.4f}")
    print(f"\n  {'k':>5}  {'τ':>8}  {'F1':>7}  {'Precision':>10}  {'Recall':>8}  F1-bar")
    print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*20}")
    best_f1 = max(r["f1"] for r in results) if results else 1.0
    for r in results:
        marker  = " ← default" if abs(r["k"] - 3.0) < 0.01 else ""
        bar_len = int((r["f1"] / best_f1) * 20)
        print(
            f"  {r['k']:>5.1f}  {r['tau']:>8.4f}  {r['f1']:>7.4f}  "
            f"{r['precision']:>10.4f}  {r['recall']:>8.4f}  "
            f"{'█' * bar_len}{marker}"
        )


def _print_token_stability(data: dict):
    print(_banner("TOKEN STABILITY ANALYSIS"))
    for split in ["normal", "anomaly"]:
        d = data.get(split, {})
        unk_pct = d.get("unk_rate", 0) * 100
        status  = "✓ PASS" if d.get("total_unk", 0) == 0 else "⚠ WARN"
        print(f"\n  {split.upper():10}  {status}  |  "
              f"tokens={d.get('total_tokens',0):,}  "
              f"UNK={d.get('total_unk',0)}  "
              f"rate={unk_pct:.4f}%")


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables from experiment results")
    parser.add_argument("--config",               default="config/stage1_config.yaml")
    parser.add_argument("--eval-results",         default="data/stage1_eval_results.json")
    parser.add_argument("--training-log",         default="data/stage1_results.json")
    parser.add_argument("--vocab-ablation",       default="data/ablations/ablation_vocab.json")
    parser.add_argument("--depth-ablation",       default="data/ablations/ablation_depth.json")
    parser.add_argument("--threshold-sensitivity",default="data/ablations/threshold_sensitivity.json")
    parser.add_argument("--token-stability",      default="data/ablations/token_stability.json")
    parser.add_argument("--output",               default="data/paper_summary.json")
    args = parser.parse_args()

    eval_results  = _load_json_safe(args.eval_results)
    training_log  = _load_json_safe(args.training_log)
    vocab_data    = _load_json_safe(args.vocab_ablation)
    depth_data    = _load_json_safe(args.depth_ablation)
    threshold_data= _load_json_safe(args.threshold_sensitivity)
    token_data    = _load_json_safe(args.token_stability)

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  SURPRISAL-GPT2: COMPLETE EXPERIMENTAL RESULTS".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    if eval_results:
        _print_main_results(eval_results)
    else:
        print("\n[Eval results not found — run evaluate.py first]")

    if training_log:
        _print_training_curve(training_log)
    else:
        print("\n[Training log not found — run train.py first]")

    if vocab_data:
        _print_vocab_ablation(vocab_data)
    else:
        print(_banner("ABLATION: Vocabulary Size"))
        print("\n  [Not yet run — execute: python scripts/ablation_vocab.py]")

    if depth_data:
        _print_depth_ablation(depth_data)
    else:
        print(_banner("ABLATION: Model Depth"))
        print("\n  [Not yet run — execute: python scripts/ablation_depth.py]")

    if threshold_data:
        _print_threshold_sensitivity(threshold_data)
    else:
        print(_banner("ABLATION: Threshold Sensitivity"))
        print("\n  [Not yet run — execute: python scripts/threshold_sensitivity.py]")

    if token_data:
        _print_token_stability(token_data)
    else:
        print(_banner("TOKEN STABILITY ANALYSIS"))
        print("\n  [Not yet run — execute: python scripts/token_stability_check.py]")

    print(f"\n{'═' * 70}")
    print("Run 'python scripts/analyze_results.py' after all experiments complete")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
