from __future__ import annotations

import pytest
import torch

from train_deferred_consolidation import (
    VARIANTS,
    event_tensor,
    resolve_variants,
    stratified_folds,
)


def test_named_variants_resolve_to_their_definitions() -> None:
    plan = resolve_variants("single_stage,deferred", "", "correct", 0.25)
    assert plan == {"single_stage": VARIANTS["single_stage"], "deferred": VARIANTS["deferred"]}


def test_unknown_named_variant_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_variants("nope", "", "correct", 0.25)


def test_sweep_crosses_capacities_with_event_modes() -> None:
    plan = resolve_variants("", "0.25,0.5", "correct,shuffled", 0.25)
    assert plan == {
        "provisional0.25_correct": (0.25, "correct", "none"),
        "provisional0.25_shuffled": (0.25, "shuffled", "none"),
        "provisional0.5_correct": (0.5, "correct", "none"),
        "provisional0.5_shuffled": (0.5, "shuffled", "none"),
    }


def test_sweep_can_select_a_similarity_mode() -> None:
    plan = resolve_variants("", "0.5", "correct", 0.25, similarity="residual")
    assert plan == {"provisional0.5_correct": (0.5, "correct", "residual")}


def test_sweep_rejects_a_capacity_below_the_final_one() -> None:
    with pytest.raises(ValueError):
        resolve_variants("", "0.125", "correct", 0.25)


def test_shuffled_event_comes_from_another_episode() -> None:
    consolidation = torch.arange(12.0).reshape(4, 3)
    shuffled = event_tensor(consolidation, "shuffled")
    assert not torch.equal(shuffled, consolidation)
    assert torch.equal(shuffled.sort(dim=0).values, consolidation.sort(dim=0).values)
    assert torch.equal(event_tensor(consolidation, "blank"), torch.zeros_like(consolidation))


def test_stratified_folds_balance_each_question_type() -> None:
    question_type = torch.tensor([0] * 10 + [1] * 5)
    assignment = stratified_folds(question_type, 5)
    for value in (0, 1):
        counts = assignment[question_type == value].bincount(minlength=5)
        assert int(counts.max()) - int(counts.min()) <= 1
