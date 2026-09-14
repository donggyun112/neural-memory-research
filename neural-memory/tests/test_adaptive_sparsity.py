from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from analyze_adaptive_sparsity import adaptive_store, crowding_signal, state_size


def _inputs(count: int = 24, units: int = 64, seed: int = 5):
    torch.manual_seed(seed)
    return torch.randn(count, units), F.normalize(torch.randn(count, 16), dim=1)


def test_an_empty_state_reports_no_crowding() -> None:
    projected, _ = _inputs()
    assert crowding_signal(torch.zeros(16, 64), projected[0]) == 0.0


def test_crowding_grows_as_documents_accumulate() -> None:
    projected, values = _inputs()
    matrix = torch.zeros(16, 64)
    early = None
    for index in range(12):
        key = F.normalize(projected[index], dim=0)
        if index == 4:
            early = crowding_signal(matrix, projected[20])
        matrix = matrix + torch.outer(values[index] - matrix @ key, key)
    late = crowding_signal(matrix, projected[20])
    assert early is not None and late > early


def test_the_first_document_gets_the_base_count() -> None:
    projected, values = _inputs()
    _, _, counts = adaptive_store(projected, values, base=12.0, exponent=1.4, lowest=4, highest=48)
    assert counts[0] == 12


def test_state_size_grows_with_what_is_stored() -> None:
    projected, values = _inputs()
    matrix = torch.zeros(16, 64)
    sizes = []
    for index in range(8):
        key = F.normalize(projected[index], dim=0)
        matrix = matrix + torch.outer(values[index] - matrix @ key, key)
        sizes.append(state_size(matrix))
    assert sizes == sorted(sizes)


def test_the_chosen_count_rises_with_load() -> None:
    projected, values = _inputs(count=40)
    _, _, counts = adaptive_store(projected, values, base=12.0, exponent=1.4, lowest=4, highest=48)
    assert sum(counts[-10:]) / 10 > sum(counts[:10]) / 10


def test_counts_stay_inside_their_bounds() -> None:
    projected, values = _inputs(count=40)
    _, _, counts = adaptive_store(projected, values, base=12.0, exponent=4.0, lowest=4, highest=20)
    assert all(4 <= count <= 20 for count in counts)


def test_every_key_keeps_the_count_it_was_given() -> None:
    projected, values = _inputs()
    keys, _, counts = adaptive_store(projected, values, base=12.0, exponent=1.4, lowest=4, highest=48)
    assert [int((key != 0).sum()) for key in keys] == counts


def test_bounds_must_be_ordered() -> None:
    projected, values = _inputs()
    with pytest.raises(ValueError):
        adaptive_store(projected, values, base=12.0, exponent=1.4, lowest=20, highest=16)
