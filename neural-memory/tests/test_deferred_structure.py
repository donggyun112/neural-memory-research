from __future__ import annotations

import pytest
import torch

from analyze_deferred_structure import gap_buckets, retention


def test_retention_detects_a_target_inside_the_top_k() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.2], [0.9, 0.1, 0.2, 0.3]])
    targets = torch.tensor([2, 2])
    # Row two ranks slot 2 third, so it survives at k=3 but not at k=2.
    assert retention(scores, targets, 2).tolist() == [1.0, 0.0]
    assert retention(scores, targets, 3).tolist() == [1.0, 1.0]


def test_retention_rejects_an_impossible_capacity() -> None:
    with pytest.raises(ValueError):
        retention(torch.zeros(2, 4), torch.zeros(2, dtype=torch.long), 5)


def test_gap_buckets_cover_every_episode_exactly_once() -> None:
    gaps = torch.arange(20)
    bands = gap_buckets(gaps, 4)
    covered = torch.zeros(20, dtype=torch.bool)
    for _, _, mask in bands:
        assert not bool((covered & mask).any())
        covered |= mask
    assert bool(covered.all())


def test_gap_buckets_keep_the_bands_ordered() -> None:
    bands = gap_buckets(torch.arange(20), 4)
    lows = [low for low, _, _ in bands]
    assert lows == sorted(lows)
