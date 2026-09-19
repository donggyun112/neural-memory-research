# neural-memory-research

Whether a memory of past work makes an LLM agent better. Split out of the keymem
repository so that research history and product history stop sharing a log.

**Start with [FINDINGS.md](FINDINGS.md)** — what this repository establishes, what
it does not, and the limit on each. The results files below are chronological and
run to seven thousand lines between them; the findings file is the hundred-line
version for someone who needs the conclusions rather than the path that produced
them.

The short version: the central question is **not answered**. What is established is
a family of approaches that does not work, and why — a memory built on judging
whether the present resembles something seen before loses to a signal the model
already emits for free. Explicit procedural knowledge is a separate case and does
help, slightly.

## Layout

- **`neural-memory/`** — the agent experiments. The endpoint runs a coding agent on
  mutation-injected bugs and lets pytest decide, because scoring the likelihood of
  a recorded action cannot separate a better agent from a better predictor of one.
  Results in `RESULTS.md`, numbered by phase.
- **`fly-connectome/`** — what the fly's mushroom body wiring is organised for,
  measured against a rewiring that holds every degree fixed. Results in
  `README.md`, numbered by phase.
- **`board/`** — a shared notice board from a stretch where two agent sessions
  worked this repository at once. `claims.md` holds what is believed and what
  measured it; `004-harness-hazards.md` is the list of measurement defects found
  the hard way, and is probably more transferable than any result here.

## Regenerating the data

Every corpus and derived feature is gitignored and has been deleted. Nothing here
depends on them being present — the scripts rebuild them:

```bash
cd neural-memory && ./recover_artifacts.sh      # corpora, then the encoded streams
cd fly-connectome && uv run python fetch_mushroom_body.py
```

The Qwen encode is hours. Everything else is a download or minutes.

## One thing worth knowing before reusing the harness

Task difficulty in the repair harness is almost entirely a property of *which
module* a task mutates, and no measured property of a module predicts it — not
size, not structure, not how many tests break. A pool drawn without stratifying by
module produces condition effects that are module effects wearing condition
labels. Phases 87 and 88.
