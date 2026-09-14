"""Smoke test: guards the ROI/type-filter logic against a live neuPrint fetch.

Skipped without NEUPRINT_TOKEN, since it needs a real registered account.
"""
import os

import pytest
from neuprint import Client

from fetch_mushroom_body import DATASET, SERVER, cell_class, fetch_circuit_neurons

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEUPRINT_TOKEN"),
    reason="requires NEUPRINT_TOKEN (register at https://neuprint.janelia.org)",
)


def test_mushroom_body_cell_counts_are_in_expected_range():
    client = Client(SERVER, dataset=DATASET, token=os.environ["NEUPRINT_TOKEN"])
    neuron_df = fetch_circuit_neurons(client)
    counts = neuron_df["type"].map(cell_class).value_counts()

    # Loose sanity bounds from the published hemibrain mushroom body counts
    # (Li et al. 2020; see RESEARCH.md), not exact per-ROI figures -- this
    # only has to fail if the ROI/type filter breaks, not pin the biology down.
    assert 1000 <= counts.get("KC", 0) <= 3000
    assert 10 <= counts.get("PN", 0) <= 150
