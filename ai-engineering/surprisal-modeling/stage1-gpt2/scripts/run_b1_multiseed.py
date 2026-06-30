#!/usr/bin/env python3
"""Multi-Seed Statistical Significance Evaluation (B1).

WHY: A single evaluation run on a fixed data split cannot rule out that the observed F1 is due
to a lucky random seed affecting train/val/test assignment. Running 5 seeds and computing
mean ± std over the SAME checkpoint but DIFFERENT data shuffles yields defensible confidence intervals.
Idempotency: individual seed result files are cached, so interrupted runs resume from last completed seed.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("surprisal.b1_multiseed")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed F1 statistical significance evaluation.")
    parser.add_argument("--checkpoint", default="data/checkpoints/checkpoint_10000.pt")
    parser.add_argument("--output", default="data/b1_multiseed_summary.json")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of all seeds.")
    args = parser.parse_args()

    seeds = [42, 43, 44, 45, 46]
    f1_scores = []

    if os.path.exists(args.output) and not args.force:
        logger.info(f"[IDEMPOTENCY] B1 multiseed summary already exists at {args.output}. Pass --force to override.")
        return

    os.makedirs("data", exist_ok=True)

    for seed in seeds:
        seed_result_path = f"data/b1_seed_{seed}.json"
        logger.info(f"--- Evaluating Seed {seed} ---")

        if not os.path.exists(seed_result_path) or args.force:
            cmd = [
                sys.executable, "evaluate.py",
                "--checkpoint", args.checkpoint,
                "--seed", str(seed),
                "--results", seed_result_path,
                "--force"
            ]
            subprocess.run(cmd, check=True)
            time.sleep(2.0)  # Thermal cooldown between evaluations

        with open(seed_result_path, "r") as f:
            data = json.load(f)
        f1_scores.append(data["test_metrics"]["f1"])

    mean_f1 = sum(f1_scores) / len(f1_scores)
    variance = sum((x - mean_f1) ** 2 for x in f1_scores) / len(f1_scores)
    std_f1 = variance ** 0.5

    summary = {
        "seeds": seeds,
        "f1_scores": f1_scores,
        "mean_f1": mean_f1,
        "std_f1": std_f1
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("==========================================")
    logger.info(f"FINAL B1 RESULT: F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"Individual scores: {[round(s, 4) for s in f1_scores]}")
    logger.info("==========================================")


if __name__ == "__main__":
    main()
