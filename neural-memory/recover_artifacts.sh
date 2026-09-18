#!/usr/bin/env bash
# Rebuild everything an `rsync --delete` removed from this repo.
#
# The corpora and every derived feature file were destroyed when a cleanup
# synced the old path over this one after the data had already been moved here:
# rsync saw them missing at the source and deleted them at the destination.
# Only local-data/ and artifacts/ were lost — code, results and history survived,
# which is the only reason this is a rerun rather than a restart.
#
# Every stage skips work it already has, so this can be re-invoked after an
# interruption without repeating the expensive encodes.
set -uo pipefail
cd "$(dirname "$0")"

say() { printf '\n=== %s ===\n' "$1"; }

# The resolved-only corpus is the default; the experiments from phase 74 onward
# need the mixed one, because a memory that helps failures as much as successes
# is exactly what the outcome split exists to detect.
if [ ! -s local-data/open-swe-v1.jsonl ]; then
  say "downloading open-swe (resolved only)"
  uv run python download_open_swe.py || exit 1
fi

if [ ! -s local-data/open-swe-mixed.jsonl ]; then
  say "downloading open-swe (mixed outcomes)"
  uv run python download_open_swe.py --no-resolved-only \
    --output local-data/open-swe-mixed.jsonl || exit 1
fi

if [ ! -s local-data/contextbench.jsonl ]; then
  say "downloading contextbench"
  uv run python download_contextbench.py || true
fi

# The action stream in BGE space: cheap, and most evaluations read it.
if [ ! -s artifacts/open_swe_stream.pt ]; then
  say "encoding the action stream (BGE)"
  uv run python prepare_claude_tool_stream.py \
    --format open_swe --root local-data/open-swe-mixed.jsonl \
    --output artifacts/open_swe_stream.pt || true
fi

# The expensive one: 46,113 actions through Qwen's own hidden states. Hours.
if [ ! -s artifacts/open_swe_qwen_stream.pt ]; then
  say "encoding the action stream (Qwen hidden states) -- this is the slow one"
  uv run python prepare_generator_stream.py || exit 1
fi

say "recovered"
ls -la local-data/*.jsonl artifacts/*.pt 2>/dev/null | awk '{printf "%10.1f MB  %s\n", $5/1048576, $NF}'
