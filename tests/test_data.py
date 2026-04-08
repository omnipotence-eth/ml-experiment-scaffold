"""Tests for data loading validation logic."""

from __future__ import annotations

from scripts.validate_data import REQUIRED_COLUMNS


class TestRequiredColumns:
    def test_messages_format(self):
        assert "messages" in REQUIRED_COLUMNS["messages"]

    def test_preference_format(self):
        cols = REQUIRED_COLUMNS["preference"]
        assert "prompt" in cols
        assert "chosen" in cols
        assert "rejected" in cols

    def test_prompt_answer_format(self):
        cols = REQUIRED_COLUMNS["prompt_answer"]
        assert "prompt" in cols
        assert "answer" in cols

    def test_csv_format_flexible(self):
        assert REQUIRED_COLUMNS["csv"] == []

    def test_all_formats_defined(self):
        expected_formats = {
            "messages",
            "text",
            "preference",
            "prompt_answer",
            "csv",
            "image_folder",
        }
        assert set(REQUIRED_COLUMNS.keys()) == expected_formats


class TestColumnValidation:
    def test_missing_columns_detected(self):
        """Verify missing column detection logic."""
        required = ["prompt", "chosen", "rejected"]
        actual = ["prompt", "text"]
        missing = [c for c in required if c not in actual]
        assert missing == ["chosen", "rejected"]

    def test_all_columns_present(self):
        required = ["messages"]
        actual = ["messages", "extra_col"]
        missing = [c for c in required if c not in actual]
        assert missing == []

    def test_empty_required(self):
        required: list[str] = []
        actual = ["anything"]
        missing = [c for c in required if c not in actual]
        assert missing == []
