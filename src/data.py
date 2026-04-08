"""Dataset loading and format-aware processing."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_data(config: dict[str, Any], tokenizer: Any = None) -> tuple[Any, Any | None]:
    """Load and split dataset based on ``config.data.format``.

    Returns ``(train_dataset, eval_dataset_or_None)``.
    """
    from datasets import load_dataset

    data_cfg = config.get("data", {})
    fmt = data_cfg.get("format", "messages")
    eval_split = config.get("training", {}).get("eval_split", 0.1)
    seed = config.get("seed", 42)

    # Load from HuggingFace hub or local file
    if data_cfg.get("dataset_name"):
        ds = load_dataset(
            data_cfg["dataset_name"],
            data_cfg.get("dataset_config"),
            split=data_cfg.get("dataset_split", "train"),
        )
        logger.info(
            "loaded from HuggingFace name=%s split=%s rows=%d",
            data_cfg["dataset_name"],
            data_cfg.get("dataset_split", "train"),
            len(ds),
        )
    elif data_cfg.get("train_file"):
        ext = data_cfg["train_file"].rsplit(".", 1)[-1]
        ds = load_dataset(ext, data_files=data_cfg["train_file"], split="train")
        logger.info("loaded from file path=%s rows=%d", data_cfg["train_file"], len(ds))
    else:
        raise ValueError("config.data must specify dataset_name or train_file")

    # Format-specific processing
    if fmt == "messages" and tokenizer:
        ds = _format_messages(ds, tokenizer)
    elif fmt == "preference" and tokenizer:
        ds = _format_preference(ds, tokenizer)

    # Split into train / eval
    if data_cfg.get("eval_file"):
        ext = data_cfg["eval_file"].rsplit(".", 1)[-1]
        eval_ds = load_dataset(ext, data_files=data_cfg["eval_file"], split="train")
        return ds, eval_ds

    if eval_split > 0:
        splits = ds.train_test_split(test_size=eval_split, seed=seed)
        logger.info("split train=%d eval=%d", len(splits["train"]), len(splits["test"]))
        return splits["train"], splits["test"]

    return ds, None


def _format_messages(ds: Any, tokenizer: Any) -> Any:
    """Apply chat template to messages-format dataset."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def apply_template(examples: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]
        return {"text": texts}

    return ds.map(apply_template, batched=True, remove_columns=ds.column_names)


def _format_preference(ds: Any, tokenizer: Any) -> Any:
    """Format prompt/chosen/rejected for ORPO/DPO."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos = tokenizer.eos_token or "</s>"

    def format_fn(examples: dict) -> dict:
        prompts, chosens, rejecteds = [], [], []
        for prompt_msgs, chosen_msgs, rejected_msgs in zip(
            examples["prompt"], examples["chosen"], examples["rejected"], strict=True
        ):
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            chosen_text = prompt_text + chosen_msgs[0]["content"] + eos + "\n"
            rejected_text = prompt_text + rejected_msgs[0]["content"] + eos + "\n"
            prompts.append(prompt_text)
            chosens.append(chosen_text)
            rejecteds.append(rejected_text)
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    return ds.map(format_fn, batched=True, remove_columns=ds.column_names)
