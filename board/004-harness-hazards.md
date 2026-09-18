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
