#!/usr/bin/env bash
# The whole repair experiment, end to end, unattended.
#
# Round one solves one set of modules with no memory and keeps what the agent
# did. Round two solves a disjoint set of modules four ways: with nothing, and
# with notes chosen by the three rules phases 72 to 76 ranked on next-action
# likelihood. Nothing here decides whether a repair happened; pytest does.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRATCH="${SCRATCH:?set SCRATCH to a working directory}"
REPO="${REPO:-$SCRATCH/repos/packaging-clean}"
MODEL="${MODEL:-haiku}"
FIRST="${FIRST:-20}"      # round one tasks: enough actions for a note block
SECOND="${SECOND:-15}"    # round two tasks: staged, extend only if it looks alive
KEEP="${KEEP:-12}"        # lines in a note block
AGENT="claude -p \"{prompt}\" --model $MODEL --max-turns 25 \
  --output-format stream-json --verbose \
  --allowedTools Read Edit Write Grep Glob \"Bash(uv run:*)\""

cd "$ROOT"
say() { printf '\n=== %s ===\n' "$1"; }

say "splitting tasks so no module appears in both rounds"
uv run python split_repair_tasks.py \
  --tasks artifacts/repair_tasks.jsonl \
  --first "$SCRATCH/round1.jsonl" --second "$SCRATCH/round2.jsonl" || exit 1

say "round one: $FIRST tasks, no memory"
uv run python run_repair_trials.py \
  --tasks "$SCRATCH/round1.jsonl" --repo "$REPO" \
  --workspace "$SCRATCH/w1" --condition round1 --limit "$FIRST" \
  --agent "$AGENT" --output "$SCRATCH/round1.out.jsonl" || exit 1

say "building note blocks from what round one did"
for rule in spread random similarity; do
  uv run python build_repair_memory.py \
    --trials "$SCRATCH/round1.out.jsonl" --workspace "$SCRATCH/w1" \
    --condition "$rule" --keep "$KEEP" \
    --per-task "$SCRATCH/round2.jsonl" --output "$SCRATCH/notes/$rule" || exit 1
done

# `none` first: if the baseline cannot be measured the rest is not worth paying
# for. `random` before `spread` because it is the control that overturned the
# similarity result three phases running.
for condition in none random spread similarity; do
  say "round two: $condition"
  memory=(--memory "$SCRATCH/notes/$condition")
  [ "$condition" = none ] && memory=()
  uv run python run_repair_trials.py \
    --tasks "$SCRATCH/round2.jsonl" --repo "$REPO" \
    --workspace "$SCRATCH/w2-$condition" --condition "$condition" --limit "$SECOND" \
    "${memory[@]}" \
    --agent "$AGENT" --output "$SCRATCH/round2.$condition.jsonl" || exit 1
done

say "comparing"
uv run python analyze_repair_trials.py "$SCRATCH"/round2.*.jsonl \
  --baseline none --output artifacts/repair_experiment.json

say "done"
