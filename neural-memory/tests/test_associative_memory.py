from __future__ import annotations

import pytest
import torch

from neural_memory.context_memory import AssociativeDeferredMemory


def _model(**kwargs: object) -> AssociativeDeferredMemory:
    torch.manual_seed(7)
    return AssociativeDeferredMemory(16, 12, **kwargs)  # type: ignore[arg-type]


def test_persistent_state_is_the_matrix_and_nothing_else() -> None:
    model = _model()
    state = model.write(torch.randn(3, 8, 16), torch.ones(3, 8, dtype=torch.bool))
    assert state.matrix.shape == (3, 12, 12)
    assert state.matrix[0].numel() == 144


def test_every_candidate_is_written_with_a_graded_strength() -> None:
    model = _model()
    state = model.write(torch.randn(4, 8, 16), torch.ones(4, 8, dtype=torch.bool))
    # No selection anywhere: strengths are continuous and no trace is zeroed out.
    assert state.write_strengths.shape == (4, 8)
    assert bool(((state.write_strengths > 0.0) & (state.write_strengths < 1.0)).all())


def test_masked_slots_contribute_nothing() -> None:
    model = _model()
    masks = torch.ones(2, 8, dtype=torch.bool)
    masks[:, 6:] = False
    candidates = torch.randn(2, 8, 16)
    padded = candidates.clone()
    padded[:, 6:] = 5.0
    assert torch.allclose(
        model.write(candidates, masks).matrix, model.write(padded, masks).matrix, atol=1e-6
    )


def test_consolidation_starts_inert_and_never_reads_content() -> None:
    model = _model()
    written = model.write(torch.randn(3, 8, 16), torch.ones(3, 8, dtype=torch.bool))
    consolidated = model.consolidate(written, torch.randn(3, 16))
    # The gain head is zero-initialised, so an untrained event changes nothing.
    assert torch.allclose(consolidated.matrix, written.matrix, atol=1e-6)
    assert torch.allclose(consolidated.event_gains, torch.zeros(3, 8), atol=1e-6)


def test_event_gain_reads_only_summary_scalars() -> None:
    model = _model()
    assert model.event_gain[0].in_features == 3


def _saturate(model: AssociativeDeferredMemory) -> None:
    torch.nn.init.normal_(model.event_gain[-1].weight, std=10.0)
    torch.nn.init.normal_(model.event_gain[-1].bias, std=10.0)


def test_gain_is_bounded_without_competition() -> None:
    model = _model(event_gain_bound=0.5, competitive_gain=False)
    _saturate(model)
    written = model.write(torch.randn(3, 8, 16), torch.ones(3, 8, dtype=torch.bool))
    gains = model.consolidate(written, torch.randn(3, 16)).event_gains
    assert float(gains.detach().abs().max()) <= 0.5 + 1e-6


def test_competitive_gains_sum_to_zero() -> None:
    model = _model(competitive_gain=True)
    _saturate(model)
    written = model.write(torch.randn(3, 8, 16), torch.ones(3, 8, dtype=torch.bool))
    gains = model.consolidate(written, torch.randn(3, 16)).event_gains
    # Strengthening one trace has to come out of another, so a uniform gain -
    # which could only rescale the matrix - is not expressible.
    assert torch.allclose(gains.detach().sum(dim=-1), torch.zeros(3), atol=1e-5)
    assert float(gains.detach().abs().max()) > 1e-3


def test_recall_scores_every_candidate() -> None:
    model = _model()
    candidates = torch.randn(3, 8, 16)
    logits, state = model(candidates, torch.randn(3, 16), torch.randn(3, 16), torch.ones(3, 8, dtype=torch.bool))
    assert logits.shape == (3, 8)
    assert state.matrix.shape == (3, 12, 12)


def test_decay_must_be_a_valid_fraction() -> None:
    with pytest.raises(ValueError):
        AssociativeDeferredMemory(16, 12, decay=0.0)
