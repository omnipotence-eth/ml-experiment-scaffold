"""Evaluation — lm-eval wrapper with vLLM backend support."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF name")
    parser.add_argument(
        "--tasks", type=str, default="arc_easy,hellaswag", help="Comma-separated lm-eval tasks"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default 4 — auto OOMs on generate_until tasks)",
    )
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "vllm"],
        help="Inference backend (vllm = continuous batching, faster)",
    )
    parser.add_argument(
        "--cache-requests",
        action="store_true",
        help="Cache prompts+responses to avoid re-inference",
    )
    parser.add_argument(
        "--low-cpu-mem",
        action="store_true",
        help="Use low_cpu_mem_usage to prevent doubling memory during load",
    )
    return parser.parse_args()


def run_lm_eval(
    model: str,
    tasks: str,
    batch_size: int,
    output_dir: str,
    backend: str = "hf",
    cache_requests: bool = False,
    low_cpu_mem: bool = False,
) -> dict:
    """Run lm-eval-harness and return results dict."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_args = f"pretrained={model}"
    if backend == "hf":
        if low_cpu_mem:
            model_args += ",low_cpu_mem_usage=True"
        model_args += ",dtype=bfloat16"
    elif backend == "vllm":
        model_args += ",dtype=bfloat16,gpu_memory_utilization=0.9"

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        backend,
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--device",
        "cuda",
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
    ]

    if cache_requests:
        cmd.extend(["--cache_requests", "true"])

    logger.info("running lm-eval cmd=%s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("lm-eval failed stderr=%s", result.stderr[-1000:])
        raise RuntimeError(f"lm-eval failed: {result.stderr[-500:]}")

    logger.info("lm-eval completed stdout=%s", result.stdout[-500:])

    # Parse results
    results_files = sorted(output_path.glob("**/results.json"))
    if results_files:
        with open(results_files[-1]) as f:
            return json.load(f)
    return {}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()

    results = run_lm_eval(
        model=args.model,
        tasks=args.tasks,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        backend=args.backend,
        cache_requests=args.cache_requests,
        low_cpu_mem=args.low_cpu_mem,
    )

    if results:
        logger.info("results:\n%s", json.dumps(results.get("results", {}), indent=2)[:2000])


if __name__ == "__main__":
    main()
