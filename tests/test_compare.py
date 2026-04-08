"""Tests for comparison script."""

from __future__ import annotations

import json

from scripts.compare import compare, load_results


def test_load_results_from_dir(tmp_path):
    """Load results from a directory with results.json."""
    results_dir = tmp_path / "eval"
    results_dir.mkdir()
    results_file = results_dir / "results.json"
    results_file.write_text(
        json.dumps(
            {
                "results": {
                    "arc_easy": {"acc,none": 0.75},
                    "hellaswag": {"acc_norm,none": 0.60},
                }
            }
        )
    )

    scores = load_results(str(results_dir))
    assert scores["arc_easy"] == pytest.approx(75.0)
    assert scores["hellaswag"] == pytest.approx(60.0)


def test_load_results_empty_dir(tmp_path):
    """Empty directory returns empty dict."""
    assert load_results(str(tmp_path)) == {}


def test_compare_output(tmp_path):
    """Compare generates valid markdown table."""
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "results.json").write_text(
        json.dumps({"results": {"arc_easy": {"acc,none": 0.70}}})
    )

    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    (exp_dir / "results.json").write_text(json.dumps({"results": {"arc_easy": {"acc,none": 0.80}}}))

    table = compare(str(baseline_dir), str(exp_dir))
    assert "arc_easy" in table
    assert "+10.0%" in table


def test_compare_negative_delta(tmp_path):
    """Negative deltas show minus sign."""
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "results.json").write_text(
        json.dumps({"results": {"task1": {"acc,none": 0.90}}})
    )

    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    (exp_dir / "results.json").write_text(json.dumps({"results": {"task1": {"acc,none": 0.85}}}))

    table = compare(str(baseline_dir), str(exp_dir))
    assert "-5.0%" in table


def test_compare_no_results(tmp_path):
    table = compare(str(tmp_path / "a"), str(tmp_path / "b"))
    assert "No results found" in table


import pytest  # noqa: E402
