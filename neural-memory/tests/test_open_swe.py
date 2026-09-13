from __future__ import annotations

import json
from pathlib import Path

from neural_memory.open_swe import (
    load_open_swe_repair_episodes,
    repair_episodes_from_open_swe_record,
)


def _assistant(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": name,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _tool(text: str) -> dict[str, object]:
    return {"role": "tool", "content": text}


def _record(resolved: int = 1) -> dict[str, object]:
    return {
        "repo": "example/project",
        "trajectory_id": "trajectory-1",
        "resolved": resolved,
        "messages": [
            _assistant(
                "str_replace_editor",
                {"command": "str_replace", "path": "/testbed/x.py", "new_str": "bad"},
            ),
            _tool("done"),
            _assistant("bash", {"command": "cd /testbed && python -m pytest -q"}),
            _tool("1 failed"),
            _assistant(
                "str_replace_editor",
                {"command": "str_replace", "path": "/testbed/x.py", "new_str": "fixed"},
            ),
            _tool("done"),
            _assistant("bash", {"command": "cd /testbed && pytest tests/test_x.py"}),
            _tool("3 passed"),
        ],
    }


def test_resolved_failure_repair_pass_becomes_episode() -> None:
    episode = repair_episodes_from_open_swe_record(_record())[0]
    assert episode.project == "example/project"
    assert episode.session_id == "trajectory-1"
    assert [candidate.action for candidate in episode.candidates] == [
        "revise",
        "strengthen",
    ]
    assert "trajectory resolved=true" in episode.outcome_context


def test_unresolved_trajectory_is_not_weakly_relabeled() -> None:
    assert repair_episodes_from_open_swe_record(_record(resolved=0)) == []


def test_jsonl_loader_skips_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "open-swe.jsonl"
    path.write_text("not-json\n" + json.dumps(_record()) + "\n")
    assert len(load_open_swe_repair_episodes(path)) == 1
