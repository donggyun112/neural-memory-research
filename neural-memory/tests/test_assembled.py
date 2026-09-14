from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from neural_memory.assembled import AssembledMemory, MemoryConfig, sparsify


def _documents(count: int = 12, seed: int = 5):
    torch.manual_seed(seed)
    config = MemoryConfig()
    projected = torch.randn(count, config.key_dim)
    values = F.normalize(torch.randn(count, config.value_dim), dim=1)
    return projected, values


def _fill(memory: AssembledMemory, projected, values, weak_first: float = 1.0):
    state = memory.empty()
    keys = [
        memory.write(state, row, value, weak_first if index == 0 else 1.0)
        for index, (row, value) in enumerate(zip(projected, values, strict=True))
    ]
    return state, torch.stack(keys)


def test_sparsify_keeps_a_budget_and_normalises() -> None:
    coded = sparsify(torch.tensor([0.1, -0.9, 0.4, 0.0]), 2)
    assert int((coded != 0).sum()) == 2
    assert float(coded.norm()) == pytest.approx(1.0)


def test_an_empty_state_holds_nothing() -> None:
    memory = AssembledMemory()
    state = memory.empty()
    assert float(state.fast.norm()) == 0.0
    assert state.eligibility == []


def test_adaptive_sparsity_widens_the_code_as_the_state_grows() -> None:
    memory = AssembledMemory()
    projected, values = _documents(count=40)
    state, _ = _fill(memory, projected, values)
    assert sum(state.active_counts[-8:]) > sum(state.active_counts[:8])


def test_fixed_sparsity_never_moves() -> None:
    memory = AssembledMemory(replace(MemoryConfig(), adaptive_sparsity=False))
    projected, values = _documents(count=20)
    state, _ = _fill(memory, projected, values)
    assert set(state.active_counts) == {MemoryConfig().fixed_active}


def test_only_weak_writes_leave_a_tag() -> None:
    memory = AssembledMemory()
    projected, values = _documents()
    state, _ = _fill(memory, projected, values, weak_first=0.4)
    assert len(state.eligibility) == 1
    strong, _ = _fill(memory, projected, values, weak_first=1.0)
    assert strong.eligibility == []


def test_disabling_tags_removes_the_modulatory_pathway() -> None:
    memory = AssembledMemory(replace(MemoryConfig(), tags=False))
    projected, values = _documents()
    state, keys = _fill(memory, projected, values, weak_first=0.4)
    before = state.fast.clone()
    memory.reinforce(state, keys[0], sign=1.0)
    assert torch.equal(state.fast, before)


def test_the_parallel_store_contributes_to_the_read() -> None:
    projected, values = _documents()
    with_slow = AssembledMemory()
    without = AssembledMemory(replace(MemoryConfig(), parallel_stores=False))
    state_a, keys = _fill(with_slow, projected, values)
    state_b, _ = _fill(without, projected, values)
    assert not torch.allclose(
        with_slow.read(state_a, keys[0]), without.read(state_b, keys[0]), atol=1e-5
    )


def test_discrimination_is_perfect_when_one_document_is_stored() -> None:
    memory = AssembledMemory()
    projected, values = _documents(count=1)
    state, keys = _fill(memory, projected, values)
    assert float(memory.discriminate(state, keys, values).mean()) == 1.0


def test_replay_moves_only_the_selected_traces() -> None:
    memory = AssembledMemory()
    projected, values = _documents()
    state, _ = _fill(memory, projected, values, weak_first=0.4)
    before = state.fast.clone()
    memory.replay(state, torch.zeros(len(state.eligibility)))
    assert torch.equal(state.fast, before)
    memory.replay(state, torch.ones(len(state.eligibility)))
    assert not torch.equal(state.fast, before)
