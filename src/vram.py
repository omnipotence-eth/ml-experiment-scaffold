"""VRAM profiling — log peak GPU memory at end of training."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_peak_vram(
    log: logging.Logger | None = None,
    wandb: Any = None,
    is_dry_run: bool = False,
) -> float | None:
    """Log peak VRAM usage.  Returns peak_gb or ``None`` if CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    log = log or logger
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("Peak VRAM: %.1f / %.1f GB (%.0f%%)", peak_gb, total_gb, 100 * peak_gb / total_gb)

    if not is_dry_run and wandb is not None:
        try:
            wandb.log({"peak_vram_gb": round(peak_gb, 2)})
            wandb.run.summary["peak_vram_gb"] = round(peak_gb, 2)
        except Exception:
            log.debug("failed to log VRAM to wandb", exc_info=True)

    return peak_gb
