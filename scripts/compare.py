"""Compare two result directories and print a markdown delta table."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_results(dir_path: str) -> dict[str, float]:
    """Load lm-eval results from a directory, returning task→accuracy mapping."""
    results_files = sorted(Path(dir_path).glob("**/results.json"))
    if not results_files:
        return {}
    with open(results_files[-1]) as f:
        data = json.load(f)

    scores: dict[str, float] = {}
    for task_name, task_data in data.get("results", {}).items():
        acc = task_data.get("acc,none") or task_data.get("acc_norm,none")
        if acc is not None:
            scores[task_name] = acc * 100
    return scores


def compare(baseline_dir: str, experiment_dir: str) -> str:
    """Generate markdown delta table.  Returns the table as a string."""
    baseline = load_results(baseline_dir)
    experiment = load_results(experiment_dir)

    if not baseline and not experiment:
        return "No results found in either directory."

    lines = [
        f"\n## Comparison: {baseline_dir} vs {experiment_dir}",
        "| Task | Baseline | Experiment | Delta |",
        "|------|----------|------------|-------|",
    ]

    all_tasks = sorted(set(list(baseline.keys()) + list(experiment.keys())))
    for task in all_tasks:
        b = baseline.get(task, 0.0)
        e = experiment.get(task, 0.0)
        delta = e - b
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {task} | {b:.1f}% | {e:.1f}% | {sign}{delta:.1f}% |")

    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compare eval results")
    parser.add_argument("baseline", help="Path to baseline results dir")
    parser.add_argument("experiment", help="Path to experiment results dir")
    parser.add_argument("--format", default="markdown", choices=["markdown"])
    args = parser.parse_args()

    table = compare(args.baseline, args.experiment)
    print(table)


if __name__ == "__main__":
    main()
