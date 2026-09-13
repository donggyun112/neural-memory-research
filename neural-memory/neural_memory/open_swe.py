from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from .repair_chains import RepairCandidate, RepairEpisode, verification_family
from .tool_outcomes import _command, _input_text, _is_mutation, classify_verification


_EDITOR_MUTATIONS = {"create", "insert", "str_replace"}
_SHELL_MUTATION = re.compile(
    r"(?:\bapply_patch\b|\bsed\s+-i\b|\bperl\s+-pi\b|"
    r"\bgit\s+apply\b|\bpatch\s+-p\d+\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Mutation:
    action_id: str
    context: str
    order: int


@dataclass(frozen=True)
class _ToolCall:
    name: str
    arguments: dict[str, object]
    command: str
    family: str | None
    order: int


@dataclass
class _FailureChain:
    failure_result: str
    bad_actions: list[_Mutation]
    last_failure_order: int
    failed_attempts: int = 1


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _deduplicate(actions: list[_Mutation]) -> list[_Mutation]:
    seen: set[str] = set()
    unique: list[_Mutation] = []
    for action in actions:
        if action.action_id in seen:
            continue
        seen.add(action.action_id)
        unique.append(action)
    return unique


def _is_mutating_call(name: str, arguments: Mapping[str, object], command: str) -> bool:
    editor_command = arguments.get("command")
    if name == "str_replace_editor" and editor_command in _EDITOR_MUTATIONS:
        return True
    return _is_mutation(name, command) or bool(_SHELL_MUTATION.search(command))


def _candidate_episode(
    *,
    project: str,
    session_id: str,
    family: str,
    chain: _FailureChain,
    repair_actions: list[_Mutation],
    older_actions: list[_Mutation],
    pass_result: str,
    max_per_action: int,
    distractors: int,
) -> RepairEpisode | None:
    bad = _deduplicate(chain.bad_actions)[-max_per_action:]
    repaired = _deduplicate(repair_actions)[-max_per_action:]
    if not bad or not repaired:
        return None
    used = {action.action_id for action in [*bad, *repaired]}
    ignored = [item for item in reversed(older_actions) if item.action_id not in used]
    ignored = list(reversed(_deduplicate(ignored)[:distractors]))
    candidates = [
        *(RepairCandidate(x.action_id, x.context, "ignore", x.order) for x in ignored),
        *(RepairCandidate(x.action_id, x.context, "revise", x.order) for x in bad),
        *(RepairCandidate(x.action_id, x.context, "strengthen", x.order) for x in repaired),
    ]
    candidates.sort(key=lambda item: item.order)
    return RepairEpisode(
        project=project,
        session_id=session_id,
        family=family,
        candidates=tuple(candidates),
        outcome_context=(
            f"failed verification\n{chain.failure_result}\n"
            f"passed verification\n{pass_result}\ntrajectory resolved=true"
        ),
        failed_attempts=chain.failed_attempts,
    )


def repair_episodes_from_open_swe_record(
    record: Mapping[str, object],
    *,
    max_per_action: int = 4,
    distractors: int = 2,
) -> list[RepairEpisode]:
    """Convert one resolved Open-SWE trajectory into explicit fail→repair→pass episodes."""
    if max_per_action < 1 or distractors < 0:
        raise ValueError("candidate limits must be non-negative and max_per_action positive")
    if record.get("resolved") not in (1, True):
        return []
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    project = record.get("repo")
    session_id = record.get("trajectory_id") or record.get("instance_id")
    if not isinstance(project, str) or not isinstance(session_id, str):
        return []

    pending: deque[_ToolCall] = deque()
    mutations: deque[_Mutation] = deque(maxlen=64)
    chains: dict[str, _FailureChain] = {}
    episodes: list[RepairEpisode] = []
    order = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for raw_call in calls:
                if not isinstance(raw_call, dict):
                    continue
                function = raw_call.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    continue
                name = function["name"]
                arguments = _mapping(function.get("arguments"))
                command = _command(arguments)
                family = verification_family(command)
                order += 1
                pending.append(_ToolCall(name, arguments, command, family, order))
                if family is None and _is_mutating_call(name, arguments, command):
                    digest_input = f"{session_id}:{order}:{name}:{_input_text(arguments)}"
                    action_id = hashlib.sha256(digest_input.encode()).hexdigest()[:16]
                    mutations.append(
                        _Mutation(action_id, f"{name} {_input_text(arguments)}", order)
                    )
            continue
        if role != "tool" or not pending:
            continue

        call = pending.popleft()
        if call.family is None:
            continue
        content = message.get("content")
        result = content if isinstance(content, str) else ""
        result = result[:8_000]
        passed = classify_verification(call.command, result)
        if passed is None:
            continue
        current = [item for item in mutations if item.order < call.order]
        chain = chains.get(call.family)
        if not passed:
            lower_bound = chain.last_failure_order if chain else -1
            recent = [item for item in current if item.order > lower_bound]
            if chain is None:
                chains[call.family] = _FailureChain(
                    result, recent[-max_per_action:], call.order
                )
            else:
                chain.bad_actions.extend(recent[-max_per_action:])
                chain.failure_result = result
                chain.last_failure_order = call.order
                chain.failed_attempts += 1
            continue
        if chain is None:
            continue
        repair_actions = [item for item in current if item.order > chain.last_failure_order]
        oldest_bad = min((item.order for item in chain.bad_actions), default=call.order)
        older = [item for item in current if item.order < oldest_bad]
        episode = _candidate_episode(
            project=project,
            session_id=session_id,
            family=call.family,
            chain=chain,
            repair_actions=repair_actions,
            older_actions=older,
            pass_result=result,
            max_per_action=max_per_action,
            distractors=distractors,
        )
        if episode is not None:
            episodes.append(episode)
        del chains[call.family]
    return episodes


def load_open_swe_repair_episodes(
    path: Path,
    *,
    max_per_action: int = 4,
    distractors: int = 2,
) -> list[RepairEpisode]:
    episodes: list[RepairEpisode] = []
    try:
        lines = path.open(encoding="utf-8")
    except OSError:
        return episodes
    with lines:
        for line in lines:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(record, dict):
                episodes.extend(
                    repair_episodes_from_open_swe_record(
                        record,
                        max_per_action=max_per_action,
                        distractors=distractors,
                    )
                )
    return episodes


def iter_open_swe_repair_episodes(
    root: str | Path,
    *,
    max_per_action: int = 4,
    distractors: int = 2,
) -> Iterator[RepairEpisode]:
    root = Path(root).expanduser()
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    for path in paths:
        yield from load_open_swe_repair_episodes(
            path,
            max_per_action=max_per_action,
            distractors=distractors,
        )
