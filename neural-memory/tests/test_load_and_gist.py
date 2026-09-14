from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from analyze_load_and_gist import gist_basis, recovery, split_components, store


def test_no_decay_stores_orthogonal_documents_exactly() -> None:
    keys = torch.eye(4)
    values = F.normalize(torch.randn(4, 8), dim=1)
    read = keys @ store(keys, values, 1.0).T
    assert torch.allclose(F.normalize(read, dim=1), values, atol=1e-5)


def test_decay_erodes_what_was_written_first() -> None:
    keys = torch.eye(3)
    values = F.normalize(torch.randn(3, 8), dim=1)
    read = keys @ store(keys, values, 0.5).T
    # With orthogonal keys decay only scales a read, so the loss shows up in
    # magnitude rather than direction.
    assert float(read[0].norm()) < float(read[2].norm())


def test_decay_must_be_a_valid_fraction() -> None:
    with pytest.raises(ValueError):
        store(torch.eye(2), torch.randn(2, 4), 0.0)


def test_components_split_a_document_without_loss() -> None:
    torch.manual_seed(5)
    values = F.normalize(torch.randn(20, 16), dim=1)
    shared, residual = split_components(values, gist_basis(values, 3))
    assert torch.allclose(shared + residual, values, atol=1e-5)
    # The two parts live in orthogonal subspaces.
    assert float((shared * residual).sum(dim=1).abs().max()) < 1e-4


def test_shared_structure_dominates_when_documents_are_alike() -> None:
    torch.manual_seed(5)
    anchor = F.normalize(torch.randn(1, 16), dim=1)
    values = F.normalize(anchor + 0.05 * torch.randn(24, 16), dim=1)
    shared, residual = split_components(values, gist_basis(values, 3))
    assert float(shared.norm(dim=1).mean()) > float(residual.norm(dim=1).mean())


def test_gist_rank_must_fit_the_documents() -> None:
    values = F.normalize(torch.randn(2, 16), dim=1)
    with pytest.raises(ValueError):
        gist_basis(values, 8)


def test_recovery_skips_documents_with_no_such_component() -> None:
    read = F.normalize(torch.randn(3, 8), dim=1)
    component = torch.zeros(3, 8)
    component[0] = read[0]
    assert len(recovery(read, component)) == 1
    assert float(recovery(read, component)[0]) == pytest.approx(1.0, abs=1e-5)
