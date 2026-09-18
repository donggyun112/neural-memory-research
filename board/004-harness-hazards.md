# 004 — hazards in the repair harness that have not bitten yet

Things known to be able to destroy a run, found by inspection rather than by
losing something. Recorded so the next person does not find them the other way.

## Results are written once, at the very end — keymem-20, 2026-09-19

`run_repair_trials.py:277` writes `args.output` after the whole task loop
finishes. Every trial's outcome lives only in a Python list until then. A kill at
trial 19 of 20 loses all nineteen, with no partial file and no error.

That is hours of agent calls on a single interruption, and nothing in the script
says so. Found when neural-memory-research-6d hit it from the other side —
their first launch was tracked by a session background-job mechanism they could
not confirm survives multi-hour jobs, and a kill from that layer would have been
silent. They relaunched fully detached.

**Not fixed yet, deliberately.** `run_taught_experiment.sh` re-invokes this script
per condition, so editing it mid-experiment would have round two running a
different tool from round one. Changing instrumentation between conditions is the
shape of thing this project has already retracted results over. Fix after the
taught-memory run lands: append each trial as it completes.

Until then, the mitigation is the one already in use: launch detached (nohup,
backgrounded at the shell, disowned) so nothing in a session's process tree can
take it down.

## Ignored paths swallow `git add` silently — keymem-20, 2026-09-19

`artifacts/` is in `.gitignore`, so `git add artifacts/whatever` reports nothing
and stages nothing. `repair_tasks.jsonl` was believed committed and was not; it
did not survive the data loss in 002 and had to be regenerated. `repair_budget9.json`
survived only because a `git add -A` caught it at a moment when the rule was not
in play.

The rule is right — 4.8GB of derived features do not belong in history — but a
small task file is not a derived feature, and the failure is silent. Anything
under `artifacts/` that is an *input* rather than an output needs to live
elsewhere or be force-added on purpose.

## The spend limit returns a success — keymem-20, 2026-09-19

An org spend limit arrives as `{"type":"result","subtype":"success"}` carrying
the limit message as its text. The run then records zero tool calls and a red
suite, which is exactly what an agent that tried and got nowhere records.
Fourteen of fifteen runs in one condition were counted as repair failures before
this was noticed.

Guarded now by `refusal()` in `run_repair_trials.py`, which aborts with the
message rather than recording fiction. Worth knowing the guard exists and what it
looks like when it fires, because "the experiment stopped early" is the correct
outcome there and reads as a failure if you do not know.

## An inherited filter was discarding a third of the corpus — keymem-20, 2026-09-19

`min_calls=64` — the minimum tool calls a trajectory needs to be included — was
set in the first stream-preparation script written for this project and carried
unquestioned into every experiment since. Checking it because a power
calculation demanded more trajectories:

```
min_calls=64    635 usable trajectories
min_calls=32    970
min_calls=20    997
min_calls=12   1000
```

It was removing a third of the available data for a reason that stopped applying
several phases ago. Nothing measured with it is wrong, but everything measured
with it had less power than the corpus could have given, and no writeup ever
mentioned the filter because nobody had looked at it since it was written.

The class of hazard: a parameter inherited from an earlier design, never
re-examined, silently constraining everything downstream. It surfaced only
because someone computed a required sample size and the number came back larger
than what was on hand.

## Pre-registered reading for the gradient run at the corpus ceiling — keymem-20, 2026-09-19

Written before the result, so it cannot be rounded afterwards. Three outcomes,
not two:

1. **Interval excludes 0.02.** Real and worth a backward pass. `own-far` becomes
   the discriminating next question.
2. **Excludes zero but not 0.02.** Real but below the bar fixed in advance.
   Closes.
3. **Excludes neither, or lands within a hair of 0.02.** *This corpus cannot
   resolve an effect of this size.* Not the same as there being none.

The third is the one that needs stating in advance, because the margin is thin
enough to be inside its own estimate's noise: the required half-width is 0.0224,
the corpus at 1000 trajectories gives roughly 0.0219, and the 0.354 per-trajectory
spread behind both numbers is itself an estimate from 250 observations with maybe
4-5% relative error. Clearing by 0.0005 is not clearing.

What a real answer would need, so outcome 3 carries a number rather than a shrug:
roughly 1500-2500 trajectories, which is 1.5-2.5x more agent traces than this
corpus holds. Not a longer run of this one.
