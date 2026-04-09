"""Tests for the generic Registry class."""

from __future__ import annotations

import pytest

from src.registry import Registry


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry("test")

        @reg.register("add")
        def add(a, b):
            return a + b

        assert reg.get("add") is add

    def test_get_missing_raises_keyerror(self):
        reg = Registry("test")
        with pytest.raises(KeyError, match="Unknown test"):
            reg.get("nonexistent")

    def test_list_returns_sorted(self):
        reg = Registry("test")
        reg.register("zebra")(lambda: None)
        reg.register("alpha")(lambda: None)
        assert reg.list() == ["alpha", "zebra"]

    def test_contains(self):
        reg = Registry("test")
        reg.register("x")(lambda: None)
        assert "x" in reg
        assert "y" not in reg

    def test_len(self):
        reg = Registry("test")
        assert len(reg) == 0
        reg.register("a")(lambda: None)
        assert len(reg) == 1

    def test_overwrite_warns(self, caplog):
        reg = Registry("test")

        @reg.register("fn")
        def first():
            return 1

        @reg.register("fn")
        def second():
            return 2

        assert reg.get("fn") is second
        assert "overwriting" in caplog.text

    def test_repr(self):
        reg = Registry("reward")
        reg.register("a")(lambda: None)
        assert repr(reg) == "Registry('reward', keys=['a'])"

    def test_keyerror_message_shows_available(self):
        reg = Registry("model")
        reg.register("llm")(lambda: None)
        reg.register("vision")(lambda: None)
        with pytest.raises(KeyError, match="llm.*vision"):
            reg.get("tabular")

    def test_decorator_preserves_function(self):
        """Registered function should be the exact same object."""
        reg = Registry("test")

        @reg.register("my_fn")
        def my_fn(x):
            return x * 2

        assert my_fn(3) == 6
        assert reg.get("my_fn")(3) == 6


class TestRewardRegistryIntegration:
    """Verify reward_registry is populated by importing rewards module."""

    def test_reward_registry_has_builtins(self):
        from src.rewards import reward_registry

        assert "correctness" in reward_registry
        assert "format" in reward_registry
        assert len(reward_registry) == 2
