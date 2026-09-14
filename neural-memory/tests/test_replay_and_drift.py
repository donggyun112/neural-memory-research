from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from analyze_opponent_modulation import recall, write_all
from analyze_replay_and_drift import drift_cycle, reconsolidate, replay


def _store(count: int = 10, seed: int = 5):
    torch.manual_seed(seed)
    keys = F.normalize(torch.randn(count, 32), dim=1)
    values = F.normalize(torch.randn(count, 16), dim=1)
    matrix, tags = write_all(keys, values)
    return keys, values, matrix, tags


def test_zero_rounds_changes_nothing() -> None:
    _, _, matrix, tags = _store()
    selection = torch.ones(len(tags))
    assert torch.equal(replay(matrix, tags, selection, 0, 0.2), matrix)


def test_replay_only_touches_the_chosen_traces() -> None:
    keys, values, matrix, tags = _store()
    selection = torch.zeros(len(tags))
    selection[:3] = 1.0
    before = recall(matrix, keys, values)
    after = recall(replay(matrix, tags, selection, 2, 0.2), keys, values)
    moved = (after - before).abs()
    assert float(moved[:3].mean()) > float(moved[3:].mean())


def test_rounds_cannot_be_negative() -> None:
    _, _, matrix, tags = _store()
    with pytest.raises(ValueError):
        replay(matrix, tags, torch.ones(len(tags)), -1, 0.2)


def test_reconsolidation_returns_what_it_read() -> None:
    keys, _, matrix, _ = _store()
    _, retrieved = reconsolidate(matrix, keys[0], 1.0)
    assert torch.allclose(retrieved, matrix @ keys[0], atol=1e-6)


def test_reading_a_trace_changes_the_state() -> None:
    keys, _, matrix, _ = _store()
    updated, _ = reconsolidate(matrix, keys[1], 1.0)
    assert float((updated - matrix).norm()) > 0.0


def test_a_re_deposit_leaves_the_recalled_direction_alone() -> None:
    keys, values, matrix, _ = _store()
    before = recall(matrix, keys, values)
    updated, _ = reconsolidate(matrix, keys[0], 1.0)
    after = recall(updated, keys, values)
    # The key is a unit vector and the state already answers along it, so
    # depositing that same direction cannot rotate the read.
    assert abs(float(after[0] - before[0])) < 1e-5


def test_repeated_recall_corrupts_the_neighbours() -> None:
    keys, values, matrix, _ = _store()
    mine, theirs = drift_cycle(matrix, keys, values, 2, 5, 1.0)
    assert abs(mine[-1] - mine[0]) < 1e-4
    assert theirs[-1] < theirs[0]
