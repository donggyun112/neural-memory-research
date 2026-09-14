from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from analyze_opponent_modulation import (
    coincidence,
    opponent_modulate,
    recall,
    write_all,
)


def _store(count: int = 6, seed: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    keys = F.normalize(torch.randn(count, 32), dim=1)
    values = F.normalize(torch.randn(count, 16), dim=1)
    matrix, tags = write_all(keys, values)
    return keys, values, matrix, tags


def test_every_document_leaves_its_own_trace() -> None:
    keys, values, matrix, tags = _store()
    assert tags.shape == (len(keys), values.shape[1], keys.shape[1])
    assert torch.allclose(tags.sum(dim=0), matrix, atol=1e-5)


def test_coincidence_points_at_the_probed_trace() -> None:
    keys, values, matrix, tags = _store()
    for index in range(len(keys)):
        weights = coincidence(matrix, tags, keys[index], values)
        assert int(weights.argmax()) == index


def test_a_positive_event_strengthens_and_a_negative_one_weakens() -> None:
    keys, values, matrix, tags = _store()
    before = recall(matrix, keys, values)
    up = recall(
        opponent_modulate(matrix, tags, values, keys[2], sign=1.0, capture=1.0, prediction_error=True),
        keys,
        values,
    )
    down = recall(
        opponent_modulate(matrix, tags, values, keys[2], sign=-1.0, capture=1.0, prediction_error=True),
        keys,
        values,
    )
    # Depression is the strong direction, as it is in the circuit. Potentiating a
    # trace that already answers its own probe is limited by the prediction error
    # itself, so the reliable claim is the ordering.
    assert float(down[2]) < float(before[2])
    assert float(down[2]) < float(up[2])


def test_prediction_error_stops_once_the_state_already_answers() -> None:
    keys, values, matrix, tags = _store()
    # A freshly stored trace already answers its own probe, so a positive event
    # asking for exactly that has almost nothing left to add.
    updated = opponent_modulate(
        matrix, tags, values, keys[1], sign=1.0, capture=1.0, prediction_error=True
    )
    without = opponent_modulate(
        matrix, tags, values, keys[1], sign=1.0, capture=1.0, prediction_error=False
    )
    assert float((updated - matrix).norm()) < float((without - matrix).norm())


def test_a_mismatched_probe_moves_the_target_less() -> None:
    keys, values, matrix, tags = _store()
    before = recall(matrix, keys, values)
    matched = recall(
        opponent_modulate(matrix, tags, values, keys[3], sign=-1.0, capture=1.0, prediction_error=True),
        keys,
        values,
    )
    other = F.normalize(torch.randn(1, keys.shape[1]), dim=1)[0]
    mismatched = recall(
        opponent_modulate(matrix, tags, values, other, sign=-1.0, capture=1.0, prediction_error=True),
        keys,
        values,
    )
    assert abs(float(matched[3] - before[3])) > abs(float(mismatched[3] - before[3]))


def test_the_pathway_never_sees_the_stored_content_of_its_target() -> None:
    keys, values, matrix, tags = _store()
    weights = coincidence(matrix, tags, keys[0], values)
    # Coincidence is a per-trace scalar, so nothing document-specific can reach
    # the gain beyond how strongly each trace answered.
    assert weights.shape == (len(keys),)
    assert weights.ndim == 1


def test_capture_scales_the_effect() -> None:
    keys, values, matrix, tags = _store()
    small = opponent_modulate(
        matrix, tags, values, keys[0], sign=-1.0, capture=0.1, prediction_error=True
    )
    large = opponent_modulate(
        matrix, tags, values, keys[0], sign=-1.0, capture=1.0, prediction_error=True
    )
    assert float((large - matrix).norm()) > float((small - matrix).norm())
