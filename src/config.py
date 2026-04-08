"""YAML config loader with single-level inheritance via _base key."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config with single-level inheritance.

    If the config contains ``_base: "base.yaml"``, loads the base first,
    then deep-merges the child config on top.  Only one level of
    inheritance is supported.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    base_name = cfg.pop("_base", None)
    if base_name:
        base_path = config_path.parent / base_name
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}
        cfg = deep_merge(base_cfg, cfg)

    logger.info("config loaded path=%s keys=%s", config_path, list(cfg.keys()))
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.  Override wins for leaf values."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
