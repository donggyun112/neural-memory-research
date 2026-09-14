from __future__ import annotations

import pytest
import torch

from neural_memory.context_memory import DeferredConsolidationMemory, hard_top_k


def test_hard_top_k_selects_a_fraction_of_eligible_slots_only() -> None:
    logits = torch.tensor([[3.0, 1.0, 2.0, 0.0]])
    eligible = torch.tensor([[True, True, False, True]])
    selected, strengths = hard_top_k(logits, eligible, 0.5)
    # Three slots are eligible, so two survive. Slot 2 outranks slot 1 on logits
    # but is ineligible, which is exactly what the mask has to prevent.
    assert selected.tolist() == [[True, True, False, False]]
    assert strengths[0, 2] == pytest.approx(0.0)


def test_hard_top_k_keeps_a_gradient_path_through_the_discrete_choice() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]], requires_grad=True)
    _, strengths = hard_top_k(logits, torch.ones(1, 4, dtype=torch.bool), 0.5)
    strengths.sum().backward()
    assert logits.grad is not None and bool((logits.grad != 0).any())


def test_hard_top_k_rejects_an_invalid_ratio() -> None:
    with pytest.raises(ValueError):
        hard_top_k(torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool), 0.0)


def test_consolidation_cannot_recover_a_trace_the_write_stage_dropped() -> None:
    torch.manual_seed(7)
    model = DeferredConsolidationMemory(16, 8, provisional_ratio=0.5, keep_ratio=0.25)
    candidates = torch.randn(4, 8, 16)
    masks = torch.ones(4, 8, dtype=torch.bool)
    written = model.write(candidates, masks)
    final = model.consolidate(written, torch.randn(4, 16))
    assert bool((final.selected <= written.provisional).all())
    assert written.provisional.sum(dim=1).tolist() == [4, 4, 4, 4]
    assert final.selected.sum(dim=1).tolist() == [2, 2, 2, 2]


def test_recall_suppresses_every_discarded_trace() -> None:
    torch.manual_seed(7)
    model = DeferredConsolidationMemory(16, 8, provisional_ratio=0.5, keep_ratio=0.25)
    logits, state = model(
        torch.randn(3, 8, 16),
        torch.randn(3, 16),
        torch.randn(3, 16),
        torch.ones(3, 8, dtype=torch.bool),
    )
    assert logits.shape == (3, 8)
    kept = logits.detach()[state.selected]
    dropped = logits.detach()[~state.selected]
    assert float(dropped.max()) < float(kept.min())


def test_write_context_widens_the_gate_and_still_selects() -> None:
    torch.manual_seed(3)
    plain = DeferredConsolidationMemory(16, 8, write_context=False)
    contextual = DeferredConsolidationMemory(16, 8, write_context=True)
    assert contextual.write_gate[0].in_features == plain.write_gate[0].in_features * 4
    masks = torch.ones(4, 8, dtype=torch.bool)
    state = contextual.write(torch.randn(4, 8, 16), masks)
    assert state.provisional.sum(dim=1).tolist() == [4, 4, 4, 4]


def test_write_context_ignores_masked_slots_in_the_episode_mean() -> None:
    torch.manual_seed(3)
    model = DeferredConsolidationMemory(16, 8, write_context=True)
    candidates = torch.randn(2, 8, 16)
    masks = torch.ones(2, 8, dtype=torch.bool)
    masks[:, 5:] = False
    padded = candidates.clone()
    padded[:, 5:] = 1e3
    # Padding beyond the mask must not move the episode summary, and so must not
    # change which slots win.
    assert torch.equal(
        model.write(candidates, masks).provisional, model.write(padded, masks).provisional
    )


def test_correction_bound_limits_the_head_contribution() -> None:
    torch.manual_seed(5)
    model = DeferredConsolidationMemory(16, 8, similarity="residual", correction_bound=0.5)
    candidates = torch.randn(3, 8, 16) * 50.0
    event = torch.randn(3, 16)
    similarity = model.encoder_similarity(candidates, event)
    state = model.write(candidates, torch.ones(3, 8, dtype=torch.bool))
    logits = model.consolidate(state, event, similarity).consolidation_logits
    residual = model.similarity_scale * (similarity - similarity.mean(dim=-1, keepdim=True))
    assert float((logits - residual).detach().abs().max()) <= 0.5 + 1e-5


def test_ratios_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        DeferredConsolidationMemory(16, 8, provisional_ratio=0.25, keep_ratio=0.5)


def test_similarity_is_required_once_declared() -> None:
    for mode in ("feature", "residual"):
        model = DeferredConsolidationMemory(16, 8, similarity=mode)
        state = model.write(torch.randn(2, 8, 16), torch.ones(2, 8, dtype=torch.bool))
        with pytest.raises(ValueError):
            model.consolidate(state, torch.randn(2, 16))


def test_unknown_similarity_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        DeferredConsolidationMemory(16, 8, similarity="cosine")


def test_feature_mode_widens_the_head_while_residual_does_not() -> None:
    plain = DeferredConsolidationMemory(16, 8, similarity="none")
    feature = DeferredConsolidationMemory(16, 8, similarity="feature")
    residual = DeferredConsolidationMemory(16, 8, similarity="residual")
    assert feature.consolidation_head[0].in_features == plain.consolidation_head[0].in_features + 1
    assert residual.consolidation_head[0].in_features == plain.consolidation_head[0].in_features


def test_residual_mode_starts_as_the_cosine_rule() -> None:
    torch.manual_seed(11)
    model = DeferredConsolidationMemory(16, 8, provisional_ratio=1.0, keep_ratio=0.25,
                                        similarity="residual")
    candidates = torch.randn(6, 8, 16)
    event = torch.randn(6, 16)
    similarity = model.encoder_similarity(candidates, event)
    state = model.write(candidates, torch.ones(6, 8, dtype=torch.bool))
    consolidated = model.consolidate(state, event, similarity)
    # The head is zero-initialised, so the untrained model ranks purely by cosine.
    expected = similarity.topk(2, dim=1).indices.sort(dim=1).values
    actual = consolidated.consolidation_logits.topk(2, dim=1).indices.sort(dim=1).values
    assert torch.equal(actual, expected)
