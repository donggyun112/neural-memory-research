# What this repository knows about memory for agents

Eighty-five phases in `neural-memory/RESULTS.md` and thirty-nine in
`fly-connectome/README.md`, in the order they happened. This is what they add up
to, for someone who needs the conclusions rather than the path. Every claim names
the phase that produced it and the limit it stops at.

The central question — does a memory of past work make an agent better — is **not
answered**. What is established is a set of ways it does not, and why, which is
narrower than an answer and more than nothing.

## One: resemblance cannot ground a memory, because resemblance is already free

Three mechanisms, three representations, one result.

| mechanism | measured against | outcome |
|---|---|---|
| similarity over stored actions | random selection | worse, five measurements |
| familiarity filter over hidden states | the model's output entropy | 0.4498 to 0.8179 |
| mushroom-body valence, real fly wiring | the model's output entropy | 0.5517 to 0.7968, r = −0.53 |

Each asks whether the present resembles something handled before. Each either
loses to a number the model emits for free, or fails to beat chance against real
outcomes. *(phases 73–76, 79, 80)*

The mechanism is duplication, not weakness: **the better a representation captures
the present, the more the nearest stored item repeats what the prompt already
says**. This predicted the direction of the Qwen-space result before it was run,
which is the strongest thing any explanation here has done. *(phase 75)*

Limit: all of it is retrieval *into a prompt* that already contains recent
context. Nothing here says resemblance is useless where there is no recent
context to duplicate — few-shot example selection is a different setting and the
literature finds the opposite there.

## Two: the residual stream carries priors, not episodes

Injecting a memory as a bias on the computation fails, and not for the reason
phase 77 gave. Own-trajectory and foreign-trajectory vectors are indistinguishable
whether the vector is externally encoded *or* a gradient taken in the model's own
coordinates — which is exactly what phase 77's diagnosis predicted would fix it.
*(phases 77, 83)*

What looked like a gain there was the mean of the stored directions. Both own and
foreign beat no-memory by ~0.09 NLL until the shared component was removed, after
which the same injection hurts. Average pairwise cosine was 0.1651 — modest, and
enough to account for the entire effect. **"These are not in a tight cone" would
not have excused skipping the check.** *(phase 83)*

Limit: one layer, one gap, one pooling, one model.

## Three: some questions this corpus is the wrong size to ask

After the shared component is removed, own-over-foreign sits at +0.0210
[−0.0005, +0.0427] on 997 trajectories — the corpus ceiling. It excludes neither
zero nor the 0.02 bar fixed before running. Quadrupling the sample from 250 halved
the estimate, which is what noise does and a real effect does not. Resolving it
needs 1,500–2,500 trajectories: more traces, not a longer run. *(phase 85)*

This is a third verdict alongside closed and open, and it was written down before
the number existed precisely so it could not be rounded into either.

## Four: the repair endpoint measures orientation, not search

Every repair number here was produced on tasks where finding the defect is not the
work.

- 87% of tasks name their own module in the failing test path, and `--blind` only
  delays that until the agent's first pytest run. *(phase 82)*
- Eleven of twenty-eight failures at a 9-turn budget **never edited a line**, and
  the seventeen that succeeded took a median of six actions of nine to reach the
  first edit. Those actions are `find`, locate, read. *(phases 84, 86)*
- Orientation costs what it costs: the same tasks reach their first edit at action
  6 on a 9-turn budget and action 7 on a 25-turn one. The budget decides only how
  much of itself that consumes. *(phase 84)*

The condition orderings survive, since conditions ran on identical task lists. What
does not survive is the claim about what was being ordered. The one measured gain —
0.64 actions from random notes — was notes shortening orientation, which is real
and has a low ceiling.

## Five: the baseline was a module mix

`ranges` resolves 3 of 14; every other module resolves nearly everything. The
"60% baseline at 9 turns" was the proportion of `ranges` tasks in the sample.
*(phase 87)*

Nothing measured explains why `ranges` is hard — not size (`version` is 1,250
lines and resolves 5 of 5), not sibling files (nothing ever opened `_ranges.py`),
not name repetition (`ranges` is *less* repetitive than three easier modules), not
operator density. *(phase 88)*

What this rules out is concrete: do not build a task pool on any of those proxies
and expect difficulty to follow.

## What has never been tested

**Write-time selection.** Everything above stores everything and selects at read
time. The fly does the reverse — dopamine decides what is written at all, and
there is no search later. Storing only what an outcome marked as mattering,
particularly the failures the pipeline discards, has never been run.

It may be untestable on this task family. An avoid-line names a path already found
wrong, and the failures here contain none: all eleven classified reached zero
edits, so nothing was tried and rejected. *(phase 86)* A pool where locating the
defect is genuinely the work might differ; one is being generated.

## What the tooling is worth independently

The harness — mutation-injected defects, pytest as the judge, a diff against
pristine tests to catch an agent that edits the test instead, tool calls parsed
from the event stream rather than self-reported — took thirteen defects to get
right, four of which passed their own checks. It answers "does X help an agent"
for any X, and the defect list in `board/004` is more transferable than any result
here.

The one thing in the fly line that stands alone: `compartment_valence.py` reads
each mushroom-body compartment's valence sign off raw anatomy, PAM against PPL1
presynapses, and recovers the textbook horizontal-reward / vertical-punishment
split with nothing fitted.
