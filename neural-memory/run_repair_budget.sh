#!/usr/bin/env bash
# The same round two, under a turn budget the agent cannot always meet.
#
# At 25 turns every condition repaired every task, so `resolved` was a constant
# and only `actions` carried any signal. That is a real measurement — does the
# memory shorten the search — but it cannot answer whether the memory lets the
# agent finish work it otherwise would not.
#
# Round one answers what budget to pick: its successes took a median of 14
# actions and every one of its failures ran out of turns rather than reaching a
# wrong conclusion. A budget near that median puts roughly half the tasks out of
# reach, which is where a difference between conditions is easiest to see.
#
# Nothing else changes. Same tasks, same note blocks, same repository.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRATCH="${SCRATCH:?set SCRATCH to the directory the main experiment used}"
REPO="${REPO:-$SCRATCH/repos/packaging-clean}"
MODEL="${MODEL:-haiku}"
TURNS="${TURNS:-12}"
SECOND="${SECOND:-15}"
AGENT="claude -p {prompt} --model $MODEL --max-turns $TURNS \
  --output-format stream-json --verbose \
  --allowedTools Read Edit Write Grep Glob \"Bash(uv run:*)\""

cd "$ROOT"
for condition in none random spread similarity; do
  out="$SCRATCH/budget$TURNS.$condition.jsonl"
  [ -s "$out" ] && continue
  printf '\n=== budget %s turns: %s ===\n' "$TURNS" "$condition"
  memory=""
  [ "$condition" != none ] && memory="--memory $SCRATCH/notes/$condition"
  # shellcheck disable=SC2086
  uv run python run_repair_trials.py \
    --tasks "$SCRATCH/round2.jsonl" --repo "$REPO" \
    --workspace "$SCRATCH/wb$TURNS-$condition" --condition "$condition" \
    --limit "$SECOND" $memory \
    --agent "$AGENT" --output "$out" || exit 1
done

printf '\n=== comparing at %s turns ===\n' "$TURNS"
uv run python analyze_repair_trials.py "$SCRATCH"/budget"$TURNS".*.jsonl \
  --baseline none --output "artifacts/repair_budget$TURNS.json"
