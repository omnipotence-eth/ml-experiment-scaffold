"""Tests for GRPO reward functions."""

from __future__ import annotations

import pytest

from src.rewards import build_reward_functions, correctness_reward, extract_number, format_reward


class TestExtractNumber:
    def test_hash_format(self):
        assert extract_number("The answer is #### 42") == 42.0

    def test_boxed_format(self):
        assert extract_number("So \\boxed{123}") == 123.0

    def test_with_commas(self):
        assert extract_number("#### 1,234") == 1234.0

    def test_negative(self):
        assert extract_number("#### -5") == -5.0

    def test_decimal(self):
        assert extract_number("The result is 3.14") == 3.14

    def test_last_number_fallback(self):
        assert extract_number("Step 1: 10, Step 2: 20, Final: 30") == 30.0

    def test_no_number(self):
        assert extract_number("No numbers here") is None

    def test_empty_string(self):
        assert extract_number("") is None

    def test_boxed_with_commas(self):
        assert extract_number("\\boxed{1,000,000}") == 1000000.0

    def test_negative_decimal(self):
        assert extract_number("The answer is -3.5") == -3.5


class TestCorrectnessReward:
    def test_correct_answer(self):
        assert correctness_reward(["#### 42"], ["#### 42"]) == [1.0]

    def test_wrong_answer(self):
        assert correctness_reward(["#### 42"], ["#### 99"]) == [0.0]

    def test_no_answer_in_completion(self):
        assert correctness_reward(["I don't know"], ["#### 42"]) == [0.0]

    def test_batch_mixed(self):
        completions = ["#### 10", "#### 20", "no answer"]
        answers = ["#### 10", "#### 99", "#### 5"]
        result = correctness_reward(completions, answers)
        assert result == [1.0, 0.0, 0.0]

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            correctness_reward(["#### 1"], ["#### 1", "#### 2"])

    def test_commas_match(self):
        assert correctness_reward(["#### 1,000"], ["#### 1000"]) == [1.0]


class TestFormatReward:
    def test_full_format(self):
        text = "<think>Let me think...</think> So \\boxed{42}"
        assert format_reward([text], [""]) == [1.0]

    def test_answer_only(self):
        assert format_reward(["#### 42"], [""]) == [0.5]

    def test_no_format(self):
        assert format_reward(["The answer is forty-two"], [""]) == [0.0]

    def test_thinking_no_answer(self):
        assert format_reward(["<think>thinking</think> no boxed answer"], [""]) == [0.0]

    def test_batch(self):
        completions = [
            "<think>x</think> \\boxed{1}",
            "#### 2",
            "nothing",
        ]
        result = format_reward(completions, ["", "", ""])
        assert result == [1.0, 0.5, 0.0]


class TestBuildRewardFunctions:
    def test_build_single(self):
        fns = build_reward_functions([{"name": "correctness", "weight": 1.0}])
        assert len(fns) == 1

    def test_build_weighted(self):
        fns = build_reward_functions([{"name": "format", "weight": 0.5}])
        result = fns[0](["#### 42"], [""])
        assert result == [0.25]  # 0.5 * 0.5 (answer only)

    def test_unknown_function(self):
        with pytest.raises(ValueError, match="Unknown reward function"):
            build_reward_functions([{"name": "nonexistent"}])

    def test_empty_config(self):
        fns = build_reward_functions([])
        assert fns == []
