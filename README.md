# neural-memory-research

Two experiment lines, split out of the keymem repository so that research history
and product history stop sharing a log.

- **`neural-memory/`** — whether a memory of past actions makes an agent better.
  The endpoint runs a coding agent on mutation-injected bugs and lets pytest
  decide, because scoring the likelihood of a recorded action cannot separate a
  better agent from a better predictor of one.
- **`fly-connectome/`** — what the fly's mushroom body wiring is organised for,
  measured against a rewiring that holds every degree fixed.

Each directory keeps its own results file: `neural-memory/RESULTS.md` and
`fly-connectome/README.md`, numbered by phase.

The large inputs (`local-data/`, `artifacts/`) are gitignored and live beside the
code rather than in history.
