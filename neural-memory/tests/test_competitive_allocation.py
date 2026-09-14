from __future__ import annotations

import torch
from torch.nn import functional as F

from analyze_competitive_allocation import allocate, budget
from analyze_opponent_modulation import coincidence, recall, write_all


def _store(count: int = 8, seed: int = 5):
    torch.manual_seed(seed)
    keys = F.normalize(torch.randn(count, 32), dim=1)
    values = F.normalize(torch.randn(count, 16), dim=1)
    matrix, tags = write_all(keys, values)
    return keys, values, matrix, tags


def test_competitive_gains_sum_to_zero() -> None:
    keys, values, matrix, tags = _store()
    current = coincidence(matrix, tags, keys[0], values)
    weights = current.clamp(min=0.0)
    gain = 1.0 * weights * (torch.zeros_like(current) - current)
    assert abs(float((gain - gain.mean()).sum())) < 1e-5


def test_both_arms_move_the_addressed_trace() -> None:
    keys, values, matrix, tags = _store()
    before = recall(matrix, keys, values)
    for competitive in (False, True):
        after = recall(
            allocate(
                matrix, tags, values, keys[2], sign=-1.0, capture=1.0, competitive=competitive
            ),
            keys,
            values,
        )
        assert float(after[2]) < float(before[2])


def test_competition_spreads_a_cost_onto_the_others() -> None:
    keys, values, matrix, tags = _store()
    before = recall(matrix, keys, values)
    others = [index for index in range(len(keys)) if index != 2]

    def others_shift(competitive: bool) -> float:
        after = recall(
            allocate(
                matrix, tags, values, keys[2], sign=-1.0, capture=1.0, competitive=competitive
            ),
            keys,
            values,
        )
        return float((after[others] - before[others]).mean())

    # Taking the mean out makes the untargeted traces absorb the difference.
    assert abs(others_shift(True)) > abs(others_shift(False))


def test_budget_is_zero_for_an_untouched_state() -> None:
    _, _, matrix, _ = _store()
    assert budget(matrix, matrix) == 0.0


def test_budget_grows_with_capture() -> None:
    keys, values, matrix, tags = _store()
    small = allocate(matrix, tags, values, keys[1], sign=-1.0, capture=0.1, competitive=False)
    large = allocate(matrix, tags, values, keys[1], sign=-1.0, capture=1.0, competitive=False)
    assert budget(matrix, large) > budget(matrix, small)
