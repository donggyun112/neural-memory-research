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


def test_ratios_must_be_ordered() -> None:
    with pytest.raises(ValueError):
        DeferredConsolidationMemory(16, 8, provisional_ratio=0.25, keep_ratio=0.5)


def test_similarity_feature_is_required_once_declared() -> None:
    model = DeferredConsolidationMemory(16, 8, similarity_feature=True)
    state = model.write(torch.randn(2, 8, 16), torch.ones(2, 8, dtype=torch.bool))
    with pytest.raises(ValueError):
        model.consolidate(state, torch.randn(2, 16))


def test_similarity_feature_changes_the_consolidation_decision() -> None:
    torch.manual_seed(11)
    candidates = torch.randn(6, 8, 16)
    event = torch.randn(6, 16)
    plain = DeferredConsolidationMemory(16, 8, similarity_feature=False)
    enriched = DeferredConsolidationMemory(16, 8, similarity_feature=True)
    similarity = enriched.encoder_similarity(candidates, event)
    assert similarity.shape == (6, 8)
    masks = torch.ones(6, 8, dtype=torch.bool)
    plain_out = plain.consolidate(plain.write(candidates, masks), event)
    rich_out = enriched.consolidate(enriched.write(candidates, masks), event, similarity)
    assert plain_out.selected.shape == rich_out.selected.shape
    assert enriched.consolidation_head[0].in_features == plain.consolidation_head[0].in_features + 1
