from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from neural_memory.trainable import TrainableMemory, hard_sparse


def test_hard_sparse_keeps_a_budget_and_normalises() -> None:
    coded = hard_sparse(torch.tensor([0.1, -0.9, 0.4, 0.0]), 2)
    assert int((coded != 0).sum()) == 2
    assert float(coded.norm()) == pytest.approx(1.0)


def test_hard_sparse_passes_gradient_to_the_survivors() -> None:
    projected = torch.tensor([0.1, -0.9, 0.4, 0.05], requires_grad=True)
    hard_sparse(projected, 2).sum().backward()
    assert projected.grad is not None
    assert float(projected.grad[0]) == 0.0 and float(projected.grad[3]) == 0.0
    assert float(projected.grad[1].abs()) > 0.0


def test_hard_sparse_rejects_an_impossible_budget() -> None:
    with pytest.raises(ValueError):
        hard_sparse(torch.zeros(4), 5)


def test_a_single_document_is_scored_against_itself() -> None:
    torch.manual_seed(3)
    model = TrainableMemory(16, key_dim=64, value_dim=8)
    assert model.discrimination(F.normalize(torch.randn(1, 16), dim=1)) == 1.0


def test_logits_are_square_and_differentiable() -> None:
    torch.manual_seed(3)
    model = TrainableMemory(16, key_dim=64, value_dim=8)
    logits = model(F.normalize(torch.randn(6, 16), dim=1))
    assert logits.shape == (6, 6)
    logits.diagonal().sum().backward()
    assert model.key_projection.weight.grad is not None
    assert float(model.key_projection.weight.grad.abs().sum()) > 0.0


def test_training_reduces_the_loss_it_is_given() -> None:
    torch.manual_seed(3)
    model = TrainableMemory(16, key_dim=64, value_dim=8, adaptive=False, fixed_active=8)
    block = F.normalize(torch.randn(10, 16), dim=1)
    target = torch.arange(10)
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = float(F.cross_entropy(model(block), target))
    for _ in range(30):
        loss = F.cross_entropy(model(block), target)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    assert float(F.cross_entropy(model(block), target)) < first


def test_a_probe_reads_the_store_with_a_cue_that_was_never_written() -> None:
    torch.manual_seed(3)
    model = TrainableMemory(16, key_dim=64, value_dim=8, adaptive=False, fixed_active=8)
    block = F.normalize(torch.randn(6, 16), dim=1)
    logits = model.probe(block, F.normalize(torch.randn(2, 16), dim=1))
    assert logits.shape == (2, 6)


def test_probing_with_a_stored_document_matches_its_own_row() -> None:
    torch.manual_seed(3)
    model = TrainableMemory(16, key_dim=64, value_dim=8, adaptive=False, fixed_active=8)
    block = F.normalize(torch.randn(6, 16), dim=1)
    # The cue path has to be the same key path the write used, so a document
    # handed back as its own cue must land where it was written.
    assert torch.allclose(model.probe(block, block), model(block), atol=1e-5)


def test_probes_must_be_a_matrix() -> None:
    model = TrainableMemory(16, key_dim=64, value_dim=8)
    with pytest.raises(ValueError):
        model.probe(F.normalize(torch.randn(4, 16), dim=1), torch.randn(16))


def test_adaptive_width_grows_with_the_state() -> None:
    torch.manual_seed(3)
    model = TrainableMemory(16, key_dim=256, value_dim=8)
    small = model._active(torch.zeros(8, 256))
    large = model._active(torch.randn(8, 256) * 3.0)
    assert large > small


def test_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TrainableMemory(0, key_dim=64, value_dim=8)
