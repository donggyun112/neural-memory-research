from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from analyze_tagging_window import run_episode


def _episode(width: int = 20, seed: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    keys = F.normalize(torch.randn(width, 48), dim=1)
    values = F.normalize(torch.randn(width, 16), dim=1)
    return keys, values


def _run(**overrides: float | int) -> float:
    keys, values = _episode()
    settings: dict[str, float | int] = {
        "weak_strength": 0.2,
        "intervening": 2,
        "tag_decay": 0.9,
        "capture": 0.25,
        "event_strength": 2.0,
    }
    settings.update(overrides)
    return run_episode(keys, values, **settings)  # type: ignore[arg-type]


def test_capture_improves_a_weakly_written_trace() -> None:
    assert _run(capture=0.25) > _run(capture=0.0)


def _recovered(gap: int, tag_decay: float) -> float:
    """How much of the plasticity a full-strength write would have expressed is
    recovered by capture. An absolute gain is not comparable across gaps, because
    the control falls as interference accumulates."""
    settings = {"intervening": gap, "tag_decay": tag_decay}
    without = _run(capture=0.0, **settings)
    ceiling = _run(capture=0.0, weak_strength=1.0, **settings)
    return (_run(**settings) - without) / (ceiling - without)


def test_rescue_shrinks_as_the_tag_fades() -> None:
    assert _recovered(1, 0.7) > _recovered(8, 0.7)


def test_an_undecaying_tag_removes_the_window() -> None:
    # With nothing fading, waiting longer cannot cost the target its rescue.
    assert _recovered(8, 1.0) > 0.5 * _recovered(1, 1.0)


def test_a_silent_event_rescues_nothing() -> None:
    assert _run(event_strength=0.0) == pytest.approx(_run(event_strength=0.0, capture=0.0))


def test_tag_decay_must_be_a_valid_fraction() -> None:
    with pytest.raises(ValueError):
        _run(tag_decay=0.0)


def test_episode_needs_room_for_the_event() -> None:
    keys, values = _episode(width=4)
    with pytest.raises(ValueError):
        run_episode(
            keys,
            values,
            weak_strength=0.2,
            intervening=8,
            tag_decay=0.9,
            capture=0.25,
            event_strength=2.0,
        )
