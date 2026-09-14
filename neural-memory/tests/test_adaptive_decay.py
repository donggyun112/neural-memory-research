from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from analyze_adaptive_decay import fidelity, store


def _episode(count: int = 12, seed: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    keys = F.normalize(torch.randn(count, 48), dim=1)
    values = F.normalize(torch.randn(count, 16), dim=1)
    return keys, values


def test_no_decay_reproduces_plain_storage() -> None:
    keys, values = _episode()
    matrix = store(keys, values, decay=1.0, adaptive=False, sensitivity=0.0)
    plain = values.new_zeros(values.shape[1], keys.shape[1])
    for key, value in zip(keys, values, strict=True):
        plain = plain + torch.outer(value - plain @ key, key)
    assert torch.allclose(matrix, plain, atol=1e-5)


def test_zero_sensitivity_leaves_the_state_undecayed() -> None:
    keys, values = _episode()
    assert torch.allclose(
        store(keys, values, decay=1.0, adaptive=True, sensitivity=0.0),
        store(keys, values, decay=1.0, adaptive=False, sensitivity=0.0),
        atol=1e-5,
    )


def test_adaptive_decay_erodes_the_crowded_units_most() -> None:
    torch.manual_seed(3)
    # Every document lands on the same first eight units, so those crowd and the
    # rest stay quiet.
    keys = torch.zeros(10, 32)
    keys[:, :8] = torch.randn(10, 8)
    keys = F.normalize(keys, dim=1)
    values = F.normalize(torch.randn(10, 16), dim=1)
    matrix = store(keys, values, decay=1.0, adaptive=True, sensitivity=0.2)
    crowded = float(matrix[:, :8].abs().mean())
    quiet_reference = float(
        store(keys, values, decay=1.0, adaptive=False, sensitivity=0.0)[:, :8].abs().mean()
    )
    assert crowded < quiet_reference


def test_sensitivity_must_not_be_negative() -> None:
    keys, values = _episode()
    with pytest.raises(ValueError):
        store(keys, values, decay=1.0, adaptive=True, sensitivity=-1.0)


def test_decay_must_be_a_valid_fraction() -> None:
    keys, values = _episode()
    with pytest.raises(ValueError):
        store(keys, values, decay=0.0, adaptive=False, sensitivity=0.0)


def test_fidelity_is_one_for_orthogonal_keys() -> None:
    keys = torch.eye(4)
    values = F.normalize(torch.randn(4, 8), dim=1)
    matrix = store(keys, values, decay=1.0, adaptive=False, sensitivity=0.0)
    assert fidelity(matrix, keys, values) == pytest.approx(1.0, abs=1e-5)
