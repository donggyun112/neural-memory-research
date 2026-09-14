from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from analyze_sparse_capacity import (
    dissimilar_set,
    similar_set,
    sparse_code,
    store_and_recall,
)


def test_sparse_code_keeps_only_the_strongest_units() -> None:
    projected = torch.tensor([[0.1, -0.9, 0.4, 0.0]])
    coded = sparse_code(projected, 2)
    assert int((coded != 0).sum()) == 2
    assert coded[0, 1] != 0.0 and coded[0, 2] != 0.0
    assert coded[0, 0] == 0.0 and coded[0, 3] == 0.0
    assert float(coded.norm()) == pytest.approx(1.0)


def test_sparse_code_rejects_an_impossible_budget() -> None:
    with pytest.raises(ValueError):
        sparse_code(torch.zeros(1, 4), 5)


def test_a_single_document_is_recalled_exactly() -> None:
    torch.manual_seed(3)
    keys = F.normalize(torch.randn(1, 16), dim=1)
    values = F.normalize(torch.randn(1, 8), dim=1)
    fidelity, discrimination = store_and_recall(keys, values)
    assert float(fidelity[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(discrimination[0]) == 1.0


def test_orthogonal_keys_do_not_interfere() -> None:
    keys = torch.eye(4)
    values = F.normalize(torch.randn(4, 8), dim=1)
    fidelity, discrimination = store_and_recall(keys, values)
    assert torch.allclose(fidelity, torch.ones(4), atol=1e-5)
    assert torch.allclose(discrimination, torch.ones(4))


def test_discrimination_falls_when_values_collide() -> None:
    torch.manual_seed(9)
    keys = F.normalize(torch.randn(8, 6), dim=1)
    base = F.normalize(torch.randn(1, 8), dim=1)
    # Near-identical values cannot be told apart however well they are stored.
    values = F.normalize(base + 0.01 * torch.randn(8, 8), dim=1)
    _, discrimination = store_and_recall(keys, values)
    assert float(discrimination.mean()) < 1.0


def test_interference_grows_with_load() -> None:
    torch.manual_seed(5)
    keys = F.normalize(torch.randn(32, 12), dim=1)
    values = F.normalize(torch.randn(32, 8), dim=1)
    light = float(store_and_recall(keys[:4], values[:4])[0].mean())
    heavy = float(store_and_recall(keys, values)[0].mean())
    assert heavy < light


def test_similar_and_dissimilar_sets_differ_in_spread() -> None:
    torch.manual_seed(11)
    documents = F.normalize(torch.randn(60, 16), dim=1)
    close = documents[similar_set(documents, 6, 0)]
    far = documents[dissimilar_set(documents, 6, 0)]

    def mean_similarity(block: torch.Tensor) -> float:
        gram = block @ block.T
        off = ~torch.eye(len(block), dtype=torch.bool)
        return float(gram[off].mean())

    assert mean_similarity(close) > mean_similarity(far)


def test_both_set_builders_return_distinct_documents() -> None:
    torch.manual_seed(11)
    documents = F.normalize(torch.randn(40, 16), dim=1)
    for builder in (similar_set, dissimilar_set):
        chosen = builder(documents, 8, 3)
        assert len(set(chosen.tolist())) == 8
