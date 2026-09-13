from __future__ import annotations

import torch

from neural_memory.repair_chains import RepairCandidate, RepairEpisode
from prepare_repair_write_recall_features import build_payload, unique_episode_texts


def test_real_repair_payload_separates_write_from_later_outcome() -> None:
    episode = RepairEpisode(
        project="repo/a",
        session_id="session",
        family="pytest",
        candidates=(
            RepairCandidate("old", "old edit", "ignore", 1),
            RepairCandidate("bad", "bad edit", "revise", 2),
            RepairCandidate("good", "good edit", "strengthen", 3),
        ),
        outcome_context="failed then passed",
        failed_attempts=1,
    )
    texts = unique_episode_texts([episode])
    cache = {text: torch.arange(4, dtype=torch.float32) for text in texts}
    payload = build_payload([episode], cache, 4)
    assert payload["all_candidates"].shape == (1, 3, 4)
    assert payload["all_write_queries"].count_nonzero() == 0
    assert payload["all_recall_queries"].shape == (1, 4)
    assert payload["all_targets"].sum() == 2
    assert sorted(payload["all_action_targets"][0].tolist()) == [0, 1, 2]
