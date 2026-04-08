"""Training callbacks — VRAM tracking, health monitoring, local metrics logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Health thresholds from CLAUDE.md monitoring protocol
HEALTH_THRESHOLDS = {
    "reward_std": {"warning": 0.1, "critical": 0.05, "direction": "above"},
    "entropy": {"warning": 0.05, "critical": 0.02, "direction": "above"},
    "grad_norm": {"warning": 5.0, "critical": 10.0, "direction": "below"},
}


def _get_trainer_callback_cls() -> type:
    """Lazy import of TrainerCallback to avoid hard torch dependency in tests."""
    from transformers import TrainerCallback

    return TrainerCallback


class VRAMTracker(_get_trainer_callback_cls()):  # type: ignore[misc]
    """Log VRAM usage at configurable intervals."""

    def __init__(self, log_every_steps: int = 50) -> None:
        self.log_every = log_every_steps

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if state.global_step % self.log_every != 0:
            return
        try:
            import torch

            if torch.cuda.is_available():
                current_gb = torch.cuda.memory_allocated() / 1e9
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                logger.info(
                    "vram step=%d current=%.1fGB peak=%.1fGB",
                    state.global_step,
                    current_gb,
                    peak_gb,
                )
        except ImportError:
            pass


class TrainingHealthMonitor(_get_trainer_callback_cls()):  # type: ignore[misc]
    """Monitor training health metrics and warn on degradation.

    Checks reward_std, entropy, grad_norm against thresholds.
    """

    def __init__(self, check_steps: list[int] | None = None) -> None:
        self.check_steps = set(check_steps or [10, 50, 100, 200, 300, 400, 500])
        self._warned: set[str] = set()

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs or state.global_step not in self.check_steps:
            return

        for metric, thresholds in HEALTH_THRESHOLDS.items():
            value = logs.get(metric)
            if value is None:
                continue

            above = thresholds["direction"] == "above"
            crit_key = f"{metric}_critical"
            warn_key = f"{metric}_warning"

            if above:
                if value < thresholds["critical"] and crit_key not in self._warned:
                    logger.warning(
                        "CRITICAL %s=%.4f < %.4f at step %d",
                        metric,
                        value,
                        thresholds["critical"],
                        state.global_step,
                    )
                    self._warned.add(crit_key)
                elif value < thresholds["warning"] and warn_key not in self._warned:
                    logger.warning(
                        "WARNING %s=%.4f < %.4f at step %d",
                        metric,
                        value,
                        thresholds["warning"],
                        state.global_step,
                    )
                    self._warned.add(warn_key)
            else:
                if value > thresholds["critical"] and crit_key not in self._warned:
                    logger.warning(
                        "CRITICAL %s=%.4f > %.4f at step %d",
                        metric,
                        value,
                        thresholds["critical"],
                        state.global_step,
                    )
                    self._warned.add(crit_key)
                elif value > thresholds["warning"] and warn_key not in self._warned:
                    logger.warning(
                        "WARNING %s=%.4f > %.4f at step %d",
                        metric,
                        value,
                        thresholds["warning"],
                        state.global_step,
                    )
                    self._warned.add(warn_key)


class LocalMetricsLogger(_get_trainer_callback_cls()):  # type: ignore[misc]
    """Write metrics to a JSONL file (fallback when W&B is offline)."""

    def __init__(self, path: str = "logs/metrics.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        record = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def build_callbacks(config: dict) -> list:
    """Build callback list from config."""
    logging_steps = config.get("training", {}).get("logging_steps", 10)
    return [
        VRAMTracker(log_every_steps=max(logging_steps, 10)),
        TrainingHealthMonitor(),
        LocalMetricsLogger(),
    ]
