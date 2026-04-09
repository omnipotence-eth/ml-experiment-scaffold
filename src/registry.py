"""Generic typed registry for reward functions, model loaders, and training methods."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


class Registry:
    """Named function registry with decorator-based registration.

    Usage::

        rewards = Registry("reward")

        @rewards.register("correctness")
        def correctness_reward(completions, answers):
            ...

        fn = rewards.get("correctness")
        rewards.list()  # ["correctness"]
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[str, Callable] = {}

    def register(self, key: str) -> Callable[[F], F]:
        """Decorator to register a function under *key*."""

        def decorator(fn: F) -> F:
            if key in self._entries:
                logger.warning(
                    "registry %s: overwriting key=%s old=%s new=%s",
                    self.name,
                    key,
                    self._entries[key].__name__,
                    fn.__name__,
                )
            self._entries[key] = fn
            return fn

        return decorator

    def get(self, key: str) -> Callable:
        """Look up a registered function. Raises KeyError with available keys."""
        if key not in self._entries:
            available = ", ".join(sorted(self._entries))
            msg = f"Unknown {self.name}: {key!r}. Available: [{available}]"
            raise KeyError(msg)
        return self._entries[key]

    def list(self) -> list[str]:
        """Return sorted list of registered keys."""
        return sorted(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, keys={self.list()})"
