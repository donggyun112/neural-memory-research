from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContextCandidate:
    context: str
    active: bool


@dataclass(frozen=True)
class ContextEpisode:
    project: str
    task_id: str
    query_context: str
    candidates: tuple[ContextCandidate, ...]


@dataclass(frozen=True)
class _Span:
    task_id: str
    file: str
    start_line: int
    end_line: int
    content: str

    @property
    def identity(self) -> tuple[str, int, int, str]:
        return self.file, self.start_line, self.end_line, self.content

    @property
    def context(self) -> str:
        return (
            f"file {self.file}\nlines {self.start_line}-{self.end_line}\n"
            f"{self.content[:8_000]}"
        )


def _spans(record: dict[str, object]) -> list[_Span]:
    task_id = record.get("instance_id")
    encoded = record.get("gold_context")
    if not isinstance(task_id, str) or not isinstance(encoded, str):
        return []
    try:
        items = json.loads(encoded)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(items, list):
        return []
    spans: list[_Span] = []
    seen: set[tuple[str, int, int, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        file = item.get("file")
        start = item.get("start_line")
        end = item.get("end_line")
        content = item.get("content")
        if (
            not isinstance(file, str)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue
        span = _Span(task_id, file, start, end, content)
        if span.identity not in seen:
            seen.add(span.identity)
            spans.append(span)
    return spans


def load_contextbench_episodes(
    path: str | Path,
    *,
    positives_per_episode: int = 4,
) -> list[ContextEpisode]:
    """Build same-repository positive/negative trace sets from human gold contexts."""
    if positives_per_episode < 1:
        raise ValueError("positives_per_episode must be positive")
    records: list[dict[str, object]] = []
    with Path(path).expanduser().open(encoding="utf-8") as lines:
        for line in lines:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(record, dict):
                records.append(record)

    by_project: dict[str, list[tuple[dict[str, object], list[_Span]]]] = {}
    for record in records:
        project = record.get("repo")
        spans = _spans(record)
        if isinstance(project, str) and spans:
            by_project.setdefault(project, []).append((record, spans))

    episodes: list[ContextEpisode] = []
    for project, tasks in sorted(by_project.items()):
        project_spans = [span for _, spans in tasks for span in spans]
        for record, positives in tasks:
            task_id = record.get("instance_id")
            query = record.get("problem_statement")
            if not isinstance(task_id, str) or not isinstance(query, str) or not query.strip():
                continue
            positive_ids = {span.identity for span in positives}
            negative_pool = [
                span
                for span in project_spans
                if span.task_id != task_id and span.identity not in positive_ids
            ]
            if not negative_pool:
                continue
            for chunk_index, start in enumerate(range(0, len(positives), positives_per_episode)):
                chunk = positives[start : start + positives_per_episode]
                seed_text = f"{project}:{task_id}:{chunk_index}"
                seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
                rng = random.Random(seed)
                negative_count = min(len(chunk), len(negative_pool))
                negatives = rng.sample(negative_pool, negative_count)
                candidates = [
                    *(ContextCandidate(span.context, True) for span in chunk),
                    *(ContextCandidate(span.context, False) for span in negatives),
                ]
                rng.shuffle(candidates)
                episodes.append(
                    ContextEpisode(
                        project=project,
                        task_id=task_id,
                        query_context=query[:8_000],
                        candidates=tuple(candidates),
                    )
                )
    return episodes
