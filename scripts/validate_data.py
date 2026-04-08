"""Validate dataset schema before training.  Exit 1 on failure."""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "messages": ["messages"],
    "text": ["text"],
    "preference": ["prompt", "chosen", "rejected"],
    "prompt_answer": ["prompt", "answer"],
    "csv": [],
    "image_folder": [],
}


def validate(config_path: str) -> bool:
    """Check dataset schema matches format.  Returns True on success."""
    from datasets import load_dataset

    from src.config import load_config

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    fmt = data_cfg.get("format", "messages")

    # Load a small sample
    try:
        if data_cfg.get("dataset_name"):
            ds = load_dataset(
                data_cfg["dataset_name"],
                data_cfg.get("dataset_config"),
                split=f"{data_cfg.get('dataset_split', 'train')}[:10]",
            )
        elif data_cfg.get("train_file"):
            ext = data_cfg["train_file"].rsplit(".", 1)[-1]
            ds = load_dataset(ext, data_files=data_cfg["train_file"], split="train[:10]")
        else:
            logger.error("No dataset specified in config")
            return False
    except Exception:
        logger.error("Failed to load dataset", exc_info=True)
        return False

    # Check required columns
    expected_cols = REQUIRED_COLUMNS.get(fmt, [])
    actual_cols = ds.column_names
    missing = [c for c in expected_cols if c not in actual_cols]

    if missing:
        logger.error(
            "Missing columns for format=%s: %s. Got: %s",
            fmt,
            missing,
            actual_cols,
        )
        return False

    logger.info("validation passed format=%s columns=%s rows=%d", fmt, actual_cols, len(ds))
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Validate dataset schema")
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()

    if not validate(args.config):
        sys.exit(1)


if __name__ == "__main__":
    main()
