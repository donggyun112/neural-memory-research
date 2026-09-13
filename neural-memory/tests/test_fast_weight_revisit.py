from __future__ import annotations

import torch

from neural_memory.context_memory import FastWeightRecallMemory


def test_fast_weight_memory_updates_bounded_parametric_state() -> None:
    torch.manual_seed(4)
    model = FastWeightRecallMemory(feature_dim=6, memory_dim=4, keep_ratio=0.25)
    candidates = torch.randn(3, 8, 6)
    queries = torch.randn(3, 6)
    masks = torch.ones(3, 8, dtype=torch.bool)
    logits, state = model(candidates, queries, masks)
    recalled = model.read(state, queries)
    assert state.matrix.shape == (3, 4, 4)
    assert state.selected.sum(dim=1).tolist() == [2, 2, 2]
    assert state.surprise.shape == (3, 8)
    assert state.retention.shape == (3, 8)
    assert logits.shape == (3, 8)
    assert recalled.shape == (3, 4)


def test_fast_weight_adaptive_retention_is_input_dependent() -> None:
    model = FastWeightRecallMemory(
        feature_dim=6, memory_dim=4, keep_ratio=0.25, adaptive_retention=True
    )
    assert model.retention_gate is not None
    assert all(parameter.requires_grad for parameter in model.retention_gate.parameters())
