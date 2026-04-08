"""Model factory — load LLM / vision / tabular models from config."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_model(config: dict[str, Any], caps: dict[str, bool]) -> tuple[Any, Any]:
    """Load model + tokenizer/processor based on config.

    Returns ``(model, tokenizer_or_None)``.
    For vision/tabular models, tokenizer is ``None``.
    """
    model_type = config.get("model", {}).get("type", "llm")

    if model_type == "llm":
        model, tokenizer = _load_llm(config, caps)
    elif model_type == "vision":
        model, tokenizer = _load_vision(config), None
    elif model_type == "tabular":
        model, tokenizer = _load_tabular(config), None
    else:
        raise ValueError(f"Unknown model.type: {model_type}")

    # torch.compile (Tier 3) — apply AFTER LoRA to avoid module name mismatch
    if config.get("compile", False):
        import torch

        compile_mode = config.get("compile_mode", "reduce-overhead")
        model = torch.compile(model, mode=compile_mode)
        logger.info("torch.compile applied mode=%s", compile_mode)

    return model, tokenizer


def _load_llm(config: dict, caps: dict) -> tuple[Any, Any]:
    """Load LLM via Unsloth (preferred) or plain HF transformers."""
    from src.platform import get_attn_implementation

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    model_name = model_cfg["name"]
    max_seq = config.get("max_seq_length", 2048)
    load_4bit = model_cfg.get("load_in_4bit", False)
    load_fp8 = model_cfg.get("load_in_fp8", False)

    # Try Unsloth first (2x faster, 60% less VRAM)
    try:
        from unsloth import FastLanguageModel

        kwargs: dict[str, Any] = {
            "model_name": model_name,
            "max_seq_length": max_seq,
            "load_in_4bit": load_4bit,
        }
        # Unsloth fp8 via TorchAO (Tier 2)
        if load_fp8:
            kwargs["load_in_fp8"] = True
            kwargs["load_in_4bit"] = False
            logger.info("loading model in fp8 via Unsloth TorchAO")
        else:
            kwargs["dtype"] = "bfloat16"

        model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)

        if lora_cfg.get("enabled", True):
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_cfg.get("r", 16),
                target_modules=lora_cfg.get("target_modules"),
                lora_alpha=lora_cfg.get("alpha", 32),
                lora_dropout=lora_cfg.get("dropout", 0.05),
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=config.get("seed", 42),
            )
        logger.info("loaded via Unsloth name=%s fp8=%s 4bit=%s", model_name, load_fp8, load_4bit)
        return model, tokenizer

    except ImportError:
        logger.info("Unsloth not available — falling back to HF transformers")

    # Fallback: plain HF transformers
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attn_impl = get_attn_implementation(config, caps)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    if lora_cfg.get("enabled", True):
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules"),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    logger.info("loaded via HF transformers name=%s attn=%s", model_name, attn_impl)
    return model, tokenizer


def _load_vision(config: dict) -> Any:
    """Load a vision model via timm or torchvision."""
    import torch

    model_cfg = config.get("model", {})
    model_name = model_cfg["name"]

    try:
        import timm

        model = timm.create_model(
            model_name,
            pretrained=model_cfg.get("pretrained", True),
            num_classes=model_cfg.get("num_classes", 10),
        )
        logger.info("loaded vision model via timm name=%s", model_name)
    except ImportError:
        from torchvision import models

        model_fn = getattr(models, model_name)
        model = model_fn(pretrained=model_cfg.get("pretrained", True))
        logger.info("loaded vision model via torchvision name=%s", model_name)

    if config.get("bf16", True):
        model = model.to(dtype=torch.bfloat16)
    return model.cuda()


def _load_tabular(config: dict) -> Any:
    """Load a user-defined tabular model from a module path.

    ``config.model.module``: ``"my_model.Net"`` → imports ``my_model``,
    instantiates ``Net(**kwargs)``.
    """
    import importlib

    import torch

    model_cfg = config.get("model", {})
    module_path, class_name = model_cfg["module"].rsplit(".", 1)
    mod = importlib.import_module(module_path)
    model_cls = getattr(mod, class_name)
    kwargs = model_cfg.get("kwargs", {})
    model = model_cls(**kwargs)

    if config.get("bf16", True):
        model = model.to(dtype=torch.bfloat16)
    logger.info("loaded tabular model module=%s.%s", module_path, class_name)
    return model.cuda()
