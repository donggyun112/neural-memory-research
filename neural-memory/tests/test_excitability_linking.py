from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from analyze_excitability_linking import by_separation, cross_recall, excitable_codes


def test_no_gain_leaves_the_codes_untouched_by_history() -> None:
    torch.manual_seed(3)
    projected = torch.randn(10, 64)
    plain, _ = excitable_codes(projected, 8, gain=0.0, decay=0.5)
    for index, row in enumerate(projected):
        alone, _ = excitable_codes(row[None], 8, gain=0.0, decay=0.5)
        assert torch.equal((plain[index] != 0), (alone[0] != 0))


def test_excitability_makes_neighbours_share_units() -> None:
    torch.manual_seed(3)
    projected = torch.randn(12, 64)
    _, without = excitable_codes(projected, 8, gain=0.0, decay=0.5)
    _, with_gain = excitable_codes(projected, 8, gain=0.5, decay=0.5)
    assert float(with_gain.mean()) > float(without.mean())


def test_every_code_keeps_its_budget() -> None:
    torch.manual_seed(3)
    codes, _ = excitable_codes(torch.randn(12, 64), 8, gain=0.5, decay=0.5)
    assert torch.equal((codes != 0).sum(dim=1), torch.full((12,), 8))


def test_decay_must_be_a_fraction() -> None:
    with pytest.raises(ValueError):
        excitable_codes(torch.randn(4, 16), 4, gain=0.1, decay=1.5)


def test_cross_recall_is_strongest_on_the_diagonal() -> None:
    torch.manual_seed(7)
    keys = torch.eye(5)
    values = F.normalize(torch.randn(5, 8), dim=1)
    matrix = values.T @ keys
    agreement = cross_recall(matrix, keys, values)
    assert torch.allclose(agreement.diagonal(), torch.ones(5), atol=1e-5)


def test_separation_averages_both_directions() -> None:
    agreement = torch.tensor(
        [[1.0, 0.5, 0.1], [0.3, 1.0, 0.7], [0.2, 0.4, 1.0]]
    )
    scores = by_separation(agreement, 2)
    assert scores[1] == pytest.approx((0.5 + 0.7 + 0.3 + 0.4) / 4)
    assert scores[2] == pytest.approx((0.1 + 0.2) / 2)
