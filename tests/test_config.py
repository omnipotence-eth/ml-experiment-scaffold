"""Tests for config loading and inheritance."""

from __future__ import annotations

import pytest

from src.config import deep_merge, load_config


@pytest.fixture()
def config_dir(tmp_path):
    """Create temporary config files for testing."""
    base = tmp_path / "base.yaml"
    base.write_text(
        """
seed: 42
bf16: true
model:
  name: "test-model"
  type: "llm"
training:
  learning_rate: 0.001
  batch_size: 8
"""
    )

    child = tmp_path / "child.yaml"
    child.write_text(
        """
_base: base.yaml
training:
  learning_rate: 0.0001
  warmup_steps: 10
"""
    )

    standalone = tmp_path / "standalone.yaml"
    standalone.write_text(
        """
seed: 0
model:
  name: "standalone-model"
"""
    )

    return tmp_path


def test_load_standalone(config_dir):
    cfg = load_config(config_dir / "standalone.yaml")
    assert cfg["seed"] == 0
    assert cfg["model"]["name"] == "standalone-model"


def test_load_with_inheritance(config_dir):
    cfg = load_config(config_dir / "child.yaml")
    # Child overrides
    assert cfg["training"]["learning_rate"] == 0.0001
    assert cfg["training"]["warmup_steps"] == 10
    # Base values preserved
    assert cfg["seed"] == 42
    assert cfg["bf16"] is True
    assert cfg["model"]["name"] == "test-model"
    # Base training values preserved
    assert cfg["training"]["batch_size"] == 8


def test_load_missing_config():
    with pytest.raises(FileNotFoundError, match="Config not found"):
        load_config("nonexistent.yaml")


def test_load_missing_base(tmp_path):
    child = tmp_path / "bad.yaml"
    child.write_text("_base: missing.yaml\nseed: 1\n")
    with pytest.raises(FileNotFoundError, match="Base config not found"):
        load_config(child)


def test_base_key_removed_from_result(config_dir):
    cfg = load_config(config_dir / "child.yaml")
    assert "_base" not in cfg


def test_deep_merge_nested():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10, "e": 5}, "f": 6}
    result = deep_merge(base, override)
    assert result == {"a": {"b": 10, "c": 2, "e": 5}, "d": 3, "f": 6}


def test_deep_merge_override_wins_for_leaves():
    base = {"x": [1, 2, 3]}
    override = {"x": [4, 5]}
    result = deep_merge(base, override)
    assert result["x"] == [4, 5]


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}
    deep_merge(base, override)
    assert "c" not in base["a"]


def test_load_empty_config(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    cfg = load_config(empty)
    assert cfg == {}


def test_load_real_configs():
    """Smoke test: all shipped configs load without error."""
    from pathlib import Path

    configs_dir = Path("configs")
    if not configs_dir.exists():
        pytest.skip("configs/ not found (running from different directory)")

    for yaml_file in configs_dir.glob("*.yaml"):
        cfg = load_config(yaml_file)
        assert isinstance(cfg, dict)
