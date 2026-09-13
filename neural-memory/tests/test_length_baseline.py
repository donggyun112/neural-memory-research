from __future__ import annotations

import pytest
import torch

from analyze_length_baseline import length_baseline


def test_length_baseline_detects_a_perfect_length_cue() -> None:
    lengths = torch.tensor([[10.0, 90.0, 20.0, 30.0], [80.0, 10.0, 20.0, 30.0]])
    targets = torch.tensor([1, 0])
    scores = length_baseline(lengths, targets, keep=1)
    assert scores["longest_hit"] == pytest.approx(1.0)
    assert scores["shortest_hit"] == pytest.approx(0.0)
    assert scores["random_expectation"] == pytest.approx(0.25)
    assert scores["target_mean_chars"] == pytest.approx(85.0)
    assert scores["other_mean_chars"] == pytest.approx(20.0)


def test_length_baseline_reports_recency_from_the_final_slots() -> None:
    lengths = torch.ones(2, 4)
    targets = torch.tensor([3, 0])
    scores = length_baseline(lengths, targets, keep=2)
    assert scores["recency_hit"] == pytest.approx(0.5)


def test_length_baseline_rejects_a_capacity_equal_to_the_trace_count() -> None:
    with pytest.raises(ValueError):
        length_baseline(torch.ones(1, 4), torch.tensor([0]), keep=4)
