"""GRPO reward functions — correctness, format, and registry."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

RewardFn = Callable[[list[str], list[str]], list[float]]


def build_reward_functions(reward_configs: list[dict[str, Any]]) -> list[RewardFn]:
    """Build reward functions from config list.

    Each entry: ``{"name": "correctness", "weight": 1.0}``
    """
    registry: dict[str, RewardFn] = {
        "correctness": correctness_reward,
        "format": format_reward,
    }
    fns: list[RewardFn] = []
    for rc in reward_configs:
        name = rc["name"]
        weight = rc.get("weight", 1.0)
        if name not in registry:
            raise ValueError(f"Unknown reward function: {name}. Available: {list(registry.keys())}")
        base_fn = registry[name]
        if weight != 1.0:
            fns.append(_weighted(base_fn, weight))
        else:
            fns.append(base_fn)
    return fns


def _weighted(fn: RewardFn, weight: float) -> RewardFn:
    """Wrap a reward function to scale its output by *weight*."""

    def wrapped(completions: list[str], answers: list[str]) -> list[float]:
        return [r * weight for r in fn(completions, answers)]

    wrapped.__name__ = f"{fn.__name__}_w{weight}"
    return wrapped


def correctness_reward(completions: list[str], answers: list[str]) -> list[float]:
    """Check if the model's numeric answer matches the expected answer."""
    rewards: list[float] = []
    for completion, answer in zip(completions, answers, strict=True):
        extracted = extract_number(completion)
        expected = extract_number(answer)
        if extracted is not None and expected is not None:
            rewards.append(1.0 if abs(extracted - expected) < 1e-6 else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions: list[str], answers: list[str]) -> list[float]:
    """Reward for structured output format (thinking tags + boxed answer)."""
    rewards: list[float] = []
    for completion in completions:
        has_thinking = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
        has_answer = bool(re.search(r"\\boxed\{.+?\}|####\s*\S+", completion))
        if has_thinking and has_answer:
            rewards.append(1.0)
        elif has_answer:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def extract_number(text: str) -> float | None:
    """Extract the last number from text (handles ``####``, ``\\boxed{}``, commas, negatives)."""
    # #### answer format
    match = re.search(r"####\s*([-\d,]+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    # \boxed{answer}
    match = re.search(r"\\boxed\{([-\d,]+(?:\.\d+)?)\}", text)
    if match:
        return float(match.group(1).replace(",", ""))
    # Last number in text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return None
