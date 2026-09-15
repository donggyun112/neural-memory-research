from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def render(part: dict) -> str:
    """One line describing an action, without dragging its whole payload along."""
    name = str(part.get("name", "unknown"))
    arguments = part.get("input")
    if not isinstance(arguments, dict):
        return name
    # The fields that say what an action was about. Everything else is payload
    # that would swamp the embedding and carry content we have no reason to keep.
    for key in ("command", "file_path", "path", "pattern", "query", "prompt", "description"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return f"[{name}] {value.strip()[:400]}"
    return f"[{name}]"


def session_streams(root: Path, minimum: int) -> tuple[list[list[str]], list[list[int]]]:
    """Ordered action text per session, with a flag for actions that failed."""
    streams, failures = [], []
    for path in sorted(root.rglob("*.jsonl")):
        actions, failed, pending = [], [], None
        try:
            handle = path.open()
        except OSError:
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = row.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "tool_use":
                        actions.append(render(part))
                        failed.append(0)
                        pending = len(actions) - 1
                    elif part.get("type") == "tool_result" and pending is not None:
                        # The result arrives after the call, so the flag belongs
                        # to the action that produced it.
                        failed[pending] = int(bool(part.get("is_error")))
                        pending = None
        if len(actions) >= minimum:
            streams.append(actions)
            failures.append(failed)
    return streams, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode the model's own action stream, for transfer testing only"
    )
    parser.add_argument("--root", type=Path, default=Path.home() / ".claude" / "projects")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/claude_tool_stream.pt")
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--min-calls", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    streams, failures = session_streams(args.root, args.min_calls)
    if not streams:
        raise RuntimeError("no session carries enough tool calls")
    flat = [text for stream in streams for text in stream]
    encoder = SentenceTransformer(args.model, device=args.device)
    encoded = encoder.encode(
        flat,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).cpu()

    offsets = [0]
    for stream in streams:
        offsets.append(offsets[-1] + len(stream))
    output = {
        "source": "local Claude Code session logs, action stream",
        "model": args.model,
        # Named to match the turn artifacts so the same evaluation runs on both.
        "turns": encoded,
        "offsets": torch.tensor(offsets, dtype=torch.long),
        # No annotation exists for this corpus, so the target column is a
        # sentinel; only the self-supervised objective can be run here.
        "targets": torch.full((len(streams),), -1, dtype=torch.long),
        "failed": torch.tensor(
            [flag for stream in failures for flag in stream], dtype=torch.bool
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        f"saved sessions={len(streams)}, actions={len(flat)}, "
        f"failures={int(output['failed'].sum())}, dim={encoded.shape[-1]}, "
        f"raw_text=false to {args.output}"
    )


if __name__ == "__main__":
    main()
