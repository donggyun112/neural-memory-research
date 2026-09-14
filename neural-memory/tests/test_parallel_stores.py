from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from analyze_parallel_stores import run_conflict, write


def _episode(width: int = 40, seed: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    keys = F.normalize(torch.randn(width, 48), dim=1)
    values = F.normalize(torch.randn(width, 16), dim=1)
    return keys, values


def _run(delay: int, single: bool = False) -> tuple[float, float]:
    keys, values = _episode()
    return run_conflict(
        keys[0],
        values[1],
        values[2],
        keys[3:],
        values[3:],
        delay=delay,
        fast=(1.0, 0.75),
        slow=(0.25, 0.99),
        single=single,
    )


def test_a_single_write_is_recalled_exactly() -> None:
    key = F.normalize(torch.randn(8), dim=0)
    value = F.normalize(torch.randn(4), dim=0)
    matrix = write(torch.zeros(4, 8), key, value, 1.0, 1.0)
    assert torch.allclose(F.normalize(matrix @ key, dim=0), value, atol=1e-5)


def test_decay_must_be_a_valid_fraction() -> None:
    with pytest.raises(ValueError):
        write(torch.zeros(4, 8), torch.zeros(8), torch.zeros(4), 1.0, 0.0)


def test_the_fast_store_dominates_at_first() -> None:
    early, late = _run(0)
    assert early > late


def test_the_expressed_answer_flips_with_delay() -> None:
    early_now, late_now = _run(0)
    early_later, late_later = _run(32)
    assert early_now > late_now
    assert late_later > early_later


def test_one_store_cannot_cross_over() -> None:
    # One matrix holding both answers keeps whichever was written last, and a
    # single decay rescales the whole read, so the ordering cannot reverse.
    margins = [_run(delay, single=True) for delay in (0, 8, 32)]
    assert all(late > early for early, late in margins)


def test_delay_cannot_exceed_the_filler_supply() -> None:
    keys, values = _episode(width=8)
    with pytest.raises(ValueError):
        run_conflict(
            keys[0],
            values[1],
            values[2],
            keys[3:],
            values[3:],
            delay=32,
            fast=(1.0, 0.75),
            slow=(0.25, 0.99),
        )
