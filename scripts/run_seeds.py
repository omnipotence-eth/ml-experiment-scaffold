"""Train + eval across multiple seeds, aggregate mean +/- std."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Multi-seed training + evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--eval-tasks", default="arc_easy,hellaswag")
    args = parser.parse_args()

    all_results: dict[int, dict] = {}

    for seed in args.seeds:
        logger.info("=== SEED %d ===", seed)

        # Train
        train_cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            args.config,
            "--seed",
            str(seed),
            "--run-name",
            f"seed-{seed}",
        ]
        logger.info("training cmd=%s", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)

        # Eval
        output_dir = f"results/seed-{seed}"
        eval_cmd = [
            sys.executable,
            "-m",
            "src.eval",
            "--model",
            "checkpoints",
            "--tasks",
            args.eval_tasks,
            "--output-dir",
            output_dir,
        ]
        logger.info("eval cmd=%s", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)

        # Collect results
        results_files = sorted(Path(output_dir).glob("**/results.json"))
        if results_files:
            with open(results_files[-1]) as f:
                all_results[seed] = json.load(f)

    if all_results:
        _print_aggregate(all_results)


def _print_aggregate(all_results: dict[int, dict]) -> None:
    """Print mean +/- std for each task across seeds."""
    task_scores: dict[str, list[float]] = {}
    for results in all_results.values():
        for task_name, task_data in results.get("results", {}).items():
            acc = task_data.get("acc,none") or task_data.get("acc_norm,none")
            if acc is not None:
                task_scores.setdefault(task_name, []).append(acc * 100)

    print("\n## Multi-Seed Results")
    print("| Task | Mean | Std | Seeds |")
    print("|------|------|-----|-------|")
    for task, scores in sorted(task_scores.items()):
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"| {task} | {mean:.1f}% | ±{std:.1f}% | {len(scores)} |")


if __name__ == "__main__":
    main()
