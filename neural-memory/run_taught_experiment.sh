#!/usr/bin/env bash
# board/003: does write-time selection (teach from failure, not discard it) beat
# the current solved-only note pipeline, and does either beat no memory at all.
#
# Round one solves a disjoint module set with no memory, same as the main
# experiment. `taught` is the only new condition: it keeps round one's failed
# runs instead of dropping them, and distils them into named dead ends
# alongside the usual "what worked" block. `none` is re-run fresh in this batch
# rather than reused from an earlier one, because the harness gained a
# refusal() guard since the last time it ran.
#
# First run of this script used the main experiment's 25-turn round-one budget
# and got 20/20 resolved: no failures to mine, so `taught` degenerated to a
# do-only note, and round two also hit ceiling (resolved constant) at the same
# budget.
#
# The two rounds want opposite things from the budget, per keymem-20's own
# budget ladder on this harness/model/repo: round one needs enough failure to
# mine (their 12-turn number: 90% resolved, thin but nonzero), round two needs
# the baseline off the ceiling so an improvement has room to show (their
# numbers: 12 turns still ~93%, 9 turns is where it bites at 60%). Matching the
# two budgets is what coupled the two failure modes in the first place, so they
# are separate knobs here. Cross-pool transfer is not assumed -- this task pool
# is freshly generated at a different max-broken setting -- so round two runs
# `none` alone first; look at that number before spending anything on `taught`.
#
# Second run of this script confirmed the pool doesn't transfer: 20/20 at 12
# turns, 15/15 at 9. Median action count was 9-14 out of a 9-12 turn budget --
# not turn-starved, just solved directly, because `run_repair_trials.py` hands
# the agent the failing test names by default and this pool's mutations are
# localised the moment those are read. `--blind` withholds them (the harness
# already supports this; it is an existing flag, not new difficulty invented
# for this run), which targets the actual reason these tasks are easy instead
# of guessing at a smaller turn count against a pool that clearly is not turn
# bound.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRATCH="${SCRATCH:?set SCRATCH to a working directory}"
REPO="${REPO:-$SCRATCH/repos/packaging-clean}"
MODEL="${MODEL:-haiku}"
FIRST="${FIRST:-20}"
SECOND="${SECOND:-15}"
KEEP="${KEEP:-12}"
ROUND1_TURNS="${ROUND1_TURNS:-12}"
ROUND2_TURNS="${ROUND2_TURNS:-9}"
ROUND2_CONDITIONS="${ROUND2_CONDITIONS:-none taught}"
BLIND="${BLIND:---blind}"
# POOL suffixes every output file, so a second run against a different task
# pool (e.g. the cross-module one) doesn't overwrite the first run's files --
# both stay on the record rather than one silently replacing the other.
POOL="${POOL:-}"
TASKS="${TASKS:-$SCRATCH/repair_tasks$POOL.jsonl}"
agent() {
  echo "claude -p {prompt} --model $MODEL --max-turns $1 \
  --safe-mode \
  --output-format stream-json --verbose \
  --allowedTools Read Edit Write Grep Glob \"Bash(uv run:*)\""
}
ROUND1_AGENT="$(agent "$ROUND1_TURNS")"
ROUND2_AGENT="$(agent "$ROUND2_TURNS")"

cd "$ROOT"
say() { printf '\n=== %s ===\n' "$1"; }

if [ ! -s "$SCRATCH/round2$POOL.jsonl" ]; then
  say "splitting tasks so no module appears in both rounds"
  uv run python split_repair_tasks.py \
    --tasks "$TASKS" \
    --first "$SCRATCH/round1$POOL.jsonl" --second "$SCRATCH/round2$POOL.jsonl" || exit 1
fi

if [ ! -s "$SCRATCH/round1$POOL.out.jsonl" ]; then
  say "round one: $FIRST tasks, no memory, $ROUND1_TURNS turns"
  # shellcheck disable=SC2086
  uv run python run_repair_trials.py \
    --tasks "$SCRATCH/round1$POOL.jsonl" --repo "$REPO" \
    --workspace "$SCRATCH/w1$POOL" --condition round1 --limit "$FIRST" $BLIND \
    --agent "$ROUND1_AGENT" --output "$SCRATCH/round1$POOL.out.jsonl" || exit 1
fi

if [ ! -d "$SCRATCH/notes$POOL/taught" ]; then
  say "building taught note blocks: what worked, what did not"
  uv run python build_taught_memory.py \
    --trials "$SCRATCH/round1$POOL.out.jsonl" --workspace "$SCRATCH/w1$POOL" \
    --keep "$KEEP" --per-task "$SCRATCH/round2$POOL.jsonl" \
    --output "$SCRATCH/notes$POOL/taught" | tee "$SCRATCH/taught_notes$POOL.log" || exit 1
fi

# none re-run fresh in this batch, not reused from an earlier harness version.
# ROUND2_CONDITIONS defaults to both, but can be limited to "none" to check the
# baseline is off the ceiling before spending anything on `taught`.
for condition in $ROUND2_CONDITIONS; do
  [ -s "$SCRATCH/round2$POOL.$condition.jsonl" ] && continue
  say "round two: $condition, $ROUND2_TURNS turns"
  memory=""
  [ "$condition" != none ] && memory="--memory $SCRATCH/notes$POOL/$condition"
  # shellcheck disable=SC2086
  uv run python run_repair_trials.py \
    --tasks "$SCRATCH/round2$POOL.jsonl" --repo "$REPO" \
    --workspace "$SCRATCH/w2$POOL-$condition" --condition "$condition" --limit "$SECOND" \
    $memory $BLIND \
    --agent "$ROUND2_AGENT" --output "$SCRATCH/round2$POOL.$condition.jsonl" || exit 1
done

if [ -s "$SCRATCH/round2$POOL.none.jsonl" ] && [ -s "$SCRATCH/round2$POOL.taught.jsonl" ]; then
  say "comparing"
  uv run python analyze_repair_trials.py "$SCRATCH"/round2$POOL.none.jsonl "$SCRATCH"/round2$POOL.taught.jsonl \
    --baseline none --output "$SCRATCH/taught_experiment$POOL.json"
else
  say "round two partial (ROUND2_CONDITIONS=$ROUND2_CONDITIONS) -- not comparing yet"
fi

say "done"
