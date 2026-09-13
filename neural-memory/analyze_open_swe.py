from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from neural_memory.open_swe import iter_open_swe_repair_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Open-SWE repair episodes")
    parser.add_argument("root", type=Path, nargs="?", default=Path("local-data/open-swe-v1.jsonl"))
    parser.add_argument("--max-per-action", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=2)
    args = parser.parse_args()
    episodes = list(
        iter_open_swe_repair_episodes(
            args.root,
            max_per_action=args.max_per_action,
            distractors=args.distractors,
        )
    )
    actions = Counter(
        candidate.action for episode in episodes for candidate in episode.candidates
    )
    summary = {
        "episodes": len(episodes),
        "sessions": len({episode.session_id for episode in episodes}),
        "projects": len({episode.project for episode in episodes}),
        "labels": dict(sorted(actions.items())),
        "multi_action": sum(len(episode.candidates) > 1 for episode in episodes),
        "families": dict(sorted(Counter(x.family for x in episodes).items())),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
