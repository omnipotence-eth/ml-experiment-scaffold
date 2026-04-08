"""Platform detection and workarounds for Windows, Blackwell, and Flash Attention."""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def setup_platform() -> dict[str, bool]:
    """Apply platform-specific workarounds.  Returns detected capabilities."""
    caps: dict[str, bool] = {
        "windows": sys.platform == "win32",
        "blackwell": False,
        "flash_attention": False,
        "cuda": False,
    }

    # Windows W&B + encoding fixes
    if caps["windows"]:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        os.environ.setdefault("WANDB__SERVICE_WAIT", "90")
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    # Deterministic cuBLAS (for full_determinism in training)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # CUDA / Blackwell detection
    try:
        import torch

        if torch.cuda.is_available():
            caps["cuda"] = True
            cap = torch.cuda.get_device_capability()
            device_name = torch.cuda.get_device_name()
            logger.info("GPU detected name=%s sm_%d%d", device_name, cap[0], cap[1])

            if cap[0] >= 12:
                caps["blackwell"] = True
                _disable_xformers()
                logger.info("Blackwell GPU (sm_%d%d) — xformers disabled", cap[0], cap[1])
    except ImportError:
        logger.info("torch not installed — skipping GPU detection")

    # Flash Attention detection
    try:
        import flash_attn  # noqa: F401

        caps["flash_attention"] = True
        logger.info("flash_attn detected")
    except ImportError:
        logger.info("flash_attn not installed — using eager/SDPA fallback")

    return caps


def _disable_xformers() -> None:
    """Disable xformers for Unsloth model modules (no Blackwell operator)."""
    module_names = [
        "unsloth.models.llama",
        "unsloth.models.qwen3",
        "unsloth.models.qwen3_5",
        "unsloth.models.mistral",
    ]
    for name in module_names:
        try:
            mod = __import__(name, fromlist=["HAS_XFORMERS"])
            mod.HAS_XFORMERS = False
        except (ImportError, AttributeError):
            pass


def get_attn_implementation(config: dict, caps: dict[str, bool]) -> str:
    """Resolve attention implementation from config + detected capabilities."""
    explicit = config.get("model", {}).get("attn_implementation")
    if explicit:
        return explicit
    if caps.get("flash_attention"):
        return "flash_attention_2"
    return "sdpa"
