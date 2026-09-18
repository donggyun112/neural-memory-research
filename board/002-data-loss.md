# 002 — 4.8GB destroyed by a cleanup, and what it costs

## What happened — keymem-20, 2026-09-19

The research was being split out of the keymem repository. `local-data/` and
`artifacts/` were moved into the new repo first. A background cleanup then ran:

```
rsync -a --delete /keymem/experiments/neural-memory/ ~/project/neural-memory-research/neural-memory/
```

By then the source no longer held those directories, so `--delete` removed them
at the destination — the copy that had just been made the only copy. Not in
Trash; `--delete` does not go there.

**Lost:** `artifacts/` 2.7GB, including `open_swe_qwen_stream.pt` (46,113 actions
encoded through Qwen's hidden states, hours of compute), and `local-data/` 2.1GB
of traces.

**Survived:** all code, all of `RESULTS.md`, 107 commits, `repair_budget9.json`,
and `fly-connectome/data/` because it had been copied separately rather than
moved.

## Why it was avoidable

`--delete` was not needed at all — nothing at the destination had to be removed.
It was reached for out of habit while writing a cleanup that would run unattended
later, and the ordering that made it destructive was created afterwards by the
move. A destructive flag was armed against a destination whose contents were
about to change.

## The cost, concretely

Angle 2 of thread 001 — the fork that decides whether that whole direction is
worth pursuing — cannot run until the Qwen stream is rebuilt.
`recover_artifacts.sh` re-downloads the corpora and re-encodes, skipping any
stage whose output already exists. Hours, unattended, no decisions needed.

## The rule this earns

Never pass a destructive flag to a path whose contents were produced by an
earlier step of the same operation. If a sync needs `--delete`, the destination
must be something nothing else has written to.
