"""Unified training entrypoint — supports SFT, ORPO, GRPO, vision, and tabular."""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from typing import Any

import wandb

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML experiment training")
    parser.add_argument("--config", type=str, default="configs/sft.yaml", help="YAML config path")
    parser.add_argument("--dry-run", action="store_true", help="10 steps, no W&B, log VRAM")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    """Set all seeds for reproducibility."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Platform workarounds (Windows, Blackwell, Flash Attention)
    from src.platform import setup_platform

    caps = setup_platform()

    # Load config
    from src.config import load_config

    config = load_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed

    _set_seed(config.get("seed", 42))

    # Dry-run overrides
    if args.dry_run:
        config.setdefault("training", {})
        config["training"]["max_steps"] = 10
        config["training"]["save_steps"] = 10
        config["training"]["eval_steps"] = 10
        config["training"]["logging_steps"] = 1
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("DRY RUN: 10 steps, W&B disabled")

    # Init W&B
    wandb_cfg = config.get("wandb", {})
    project = wandb_cfg.get("project") or os.getenv("WANDB_PROJECT", "ml-experiment")
    run_name = args.run_name or config.get("model", {}).get("name", "model").split("/")[-1]

    wandb_settings = wandb.Settings(
        _disable_stats=True,
        console="off",
    )
    if caps.get("windows"):
        wandb_settings._service_wait = 90

    if not args.dry_run:
        wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=wandb_cfg.get("tags", []),
            settings=wandb_settings,
        )
    else:
        wandb.init(project=project, name=f"dry-run-{run_name}", mode="disabled")

    try:
        # Load model + data
        from src.callbacks import build_callbacks
        from src.data import load_data
        from src.models import load_model

        model, tokenizer = load_model(config, caps)
        train_ds, eval_ds = load_data(config, tokenizer)
        callbacks = build_callbacks(config)

        # Dispatch to training method
        method = config.get("training", {}).get("method", "sft")

        if method == "sft":
            _train_sft(model, tokenizer, train_ds, eval_ds, config, callbacks, args.resume)
        elif method == "orpo":
            _train_orpo(model, tokenizer, train_ds, eval_ds, config, callbacks, args.resume)
        elif method == "grpo":
            _train_grpo(model, tokenizer, train_ds, eval_ds, config, callbacks, args.resume)
        elif method == "vision" or method == "tabular":
            _train_vision(model, train_ds, eval_ds, config, callbacks, args.resume)
        else:
            raise ValueError(f"Unknown training.method: {method}")
    finally:
        # Always log VRAM, even on crash
        from src.vram import log_peak_vram

        log_peak_vram(
            log=logger,
            wandb=wandb if not args.dry_run else None,
            is_dry_run=args.dry_run,
        )
        wandb.finish()


def _build_training_args(config: dict, cls: type) -> Any:
    """Build HF training arguments from config, handling common fields."""
    t = config.get("training", {})
    report_to = "wandb" if os.environ.get("WANDB_MODE") != "disabled" else "none"

    kwargs: dict[str, Any] = {
        "output_dir": config.get("output_dir", "checkpoints"),
        "num_train_epochs": t.get("num_train_epochs", 3),
        "max_steps": t.get("max_steps", -1),
        "per_device_train_batch_size": t.get("per_device_train_batch_size", 2),
        "gradient_accumulation_steps": t.get("gradient_accumulation_steps", 8),
        "learning_rate": float(t.get("learning_rate", 2e-4)),
        "warmup_steps": t.get("warmup_steps", 100),
        "save_steps": t.get("save_steps", 200),
        "logging_steps": t.get("logging_steps", 10),
        "eval_strategy": t.get("eval_strategy", "steps"),
        "eval_steps": t.get("eval_steps", 200),
        "bf16": config.get("bf16", True),
        "report_to": report_to,
        "seed": config.get("seed", 42),
        # Tier 1 optimizations
        "optim": t.get("optim", "adamw_torch_fused"),
        "gradient_checkpointing": t.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": t.get(
            "gradient_checkpointing_kwargs", {"use_reentrant": False}
        ),
        "dataloader_num_workers": t.get("dataloader_num_workers", 4),
        "dataloader_pin_memory": t.get("dataloader_pin_memory", True),
        "dataloader_persistent_workers": t.get("dataloader_persistent_workers", True),
        "dataloader_prefetch_factor": t.get("dataloader_prefetch_factor", 4),
    }

    if t.get("full_determinism"):
        kwargs["full_determinism"] = True

    # DeepSpeed (Tier 4)
    ds_config = config.get("deepspeed_config")
    if ds_config:
        kwargs["deepspeed"] = ds_config

    return cls(**kwargs)


def _train_sft(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Any,
    config: dict,
    callbacks: list,
    resume: bool,
) -> None:
    from trl import SFTConfig, SFTTrainer

    training_args = _build_training_args(config, SFTConfig)
    training_args.max_seq_length = config.get("max_seq_length", 2048)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume)


def _train_orpo(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Any,
    config: dict,
    callbacks: list,
    resume: bool,
) -> None:
    from trl import ORPOConfig, ORPOTrainer

    t = config.get("training", {})
    training_args = _build_training_args(config, ORPOConfig)
    training_args.beta = float(t.get("beta", 0.1))
    training_args.max_length = config.get("max_seq_length", 2048)

    trainer = ORPOTrainer(
        model=model,
        tokenizer=tokenizer,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume)


def _train_grpo(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Any,
    config: dict,
    callbacks: list,
    resume: bool,
) -> None:
    from trl import GRPOConfig, GRPOTrainer

    from src.rewards import build_reward_functions

    t = config.get("training", {})
    reward_fns = build_reward_functions(config.get("rewards", []))

    training_args = _build_training_args(config, GRPOConfig)
    training_args.loss_type = t.get("loss_type", "dapo")
    training_args.beta = float(t.get("beta", 0.0))
    training_args.num_generations = t.get("num_generations", 4)
    training_args.max_completion_length = t.get("max_completion_length", 512)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        reward_funcs=reward_fns,
        args=training_args,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume)


def _train_vision(
    model: Any,
    train_ds: Any,
    eval_ds: Any,
    config: dict,
    callbacks: list,
    resume: bool,
) -> None:
    """Standard PyTorch training loop for vision / tabular models."""
    import torch
    from torch.utils.data import DataLoader

    t = config.get("training", {})
    batch_size = t.get("per_device_train_batch_size", 32)
    epochs = t.get("num_train_epochs", 10)
    lr = float(t.get("learning_rate", 1e-3))
    max_steps = t.get("max_steps", -1)
    logging_steps = t.get("logging_steps", 50)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    criterion = torch.nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images, labels = batch[0].cuda().bfloat16(), batch[1].cuda()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % logging_steps == 0:
                logger.info("step=%d epoch=%d loss=%.4f", global_step, epoch, loss.item())
                wandb.log({"loss": loss.item(), "epoch": epoch}, step=global_step)

            if 0 < max_steps <= global_step:
                break
        if 0 < max_steps <= global_step:
            break

    output_dir = config.get("output_dir", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    logger.info("model saved to %s/model.pt", output_dir)


if __name__ == "__main__":
    main()
