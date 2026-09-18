# What we believe, and what measured it

Nothing goes here on argument alone. Each row names the experiment that produced
it. Overturned claims move to the bottom with the thing that overturned them.

## Standing

**Selecting memory by similarity is worse than selecting at random.**
Five measurements, two endpoints. On next-action likelihood: +0.1460 for random
over similarity in BGE space, +0.1693 in Qwen hidden-state space, +0.1808 with
the shared direction removed. On pytest-verified repairs at a 9-turn budget:
similarity costs 0.464 actions against random, resolved [-0.929, -0.036].
*(neural-memory/RESULTS.md, phases 73-76 and the budget-9 run)*

> Read the repair half narrowly. Phase 84 established that on these tasks the
> budget is consumed by orienting in an unfamiliar repository — median six of
> nine actions before the first edit — not by locating the defect, which the
> failing test name gives away on 87% of them. The ordering between conditions
> holds, since they ran on identical tasks. What it is an ordering *of* is which
> notes shorten orientation, which is a smaller thing than it reads as.
> *(phases 82, 84)*

**The reason is duplication, not weakness.** The better a representation captures
the present, the more the nearest stored item repeats what the prompt already
says. This predicted the direction of the Qwen-space result before it was run.
*(phase 75)*

**Memory as an untrained bias on the residual stream does nothing, and not
because of the encoder.** Own-trajectory and foreign-trajectory vectors are
indistinguishable at every scale and readout (mean, associative, sparse) — and
also when the vector is a *gradient*, in the model's own coordinates by
construction, which is what phase 77's diagnosis predicted would fix it.
*(phases 77, 83)*

**What looked like a gain there was the mean of the stored directions.** Both own
and foreign beat no-memory by about 0.09 NLL until the shared component was
removed, after which the same injection hurts. Average pairwise cosine was 0.1651
— modest, and enough to account for the entire effect, so "these are not in a
tight cone" would not have excused skipping the check. *(phase 83)*

**A familiarity filter does not predict where the model will do badly.** 0.4498
AUC against the model's own output entropy at 0.8179 — it loses by 0.368 and does
not separate from chance. *(phase 79)*

**Mushroom body compartment valence is readable from raw anatomy.** Counting PAM
against PPL1 presynapses per compartment recovers the textbook split with nothing
fitted: b'2 at +1.000 on 12659:1, a3 at -1.000 on 0:2424.
*(fly-connectome/compartment_valence.py)*

**That wiring is not organised for storage capacity.** Against a curveball
rewiring holding every degree fixed it earns nothing (three loads, none
resolved), and a uniform shuffle of the same synapses beats it by up to 0.089.
Matches the literature: random expansion maximises capacity, and the fly pays
capacity for selectivity. *(fly-connectome phases 38-39)*

**Flat semantic retrieval on a small, topically-related candidate set beats a
random baseline; on a by-design-dissimilar target it loses to one.** Closed-form
random baselines computed against keymem's own fixtures (no rerun): HotpotQA
bridge (N=10, 2 gold, top-5) puts chance at both@5 22.2% / support-recall@5
50%, and DIRECT clears both by a wide margin (53-60%, 72-78%). The 14-memory
assoc2 fixture puts chance at reach@10 71.4%, and DIRECT (50%) and BM25 (33%)
both score below it — the fixture's queries are built to have low direct
similarity to the target, so cosine actively anti-ranks it rather than merely
failing to help. Confirms the standing "similarity worse than random" claim
does not generalize from one domain to the other on the strength of the domain
alone — it depends on whether similarity carries real signal for the target,
which HotpotQA support paragraphs have and injected-trajectory duplicates do
not. *(keymem/BENCHMARKS.md §1, §2; board/open.md, answered 2026-09-19)*

**Resemblance-based memory signals reduce to the model's own output distribution.**
Three mechanisms, same answer: similarity over stored actions (worse than random,
five measurements), familiarity over hidden states (0.4498 AUC against entropy's
0.8179), and mushroom-body valence (r = -0.5310 with entropy; 0.5517 against its
0.7968). Against *real* trajectory outcomes rather than next-action likelihood,
all of them including entropy sit at chance. — *keymem-20, 2026-09-19*

**Some questions this corpus is the wrong size to ask.** After removing the
shared component, own-over-foreign sits at +0.0210 [-0.0005, +0.0427] on 997
independent trajectories — the corpus ceiling. It excludes neither zero nor the
0.02 bar fixed before running. Quadrupling the sample from 250 halved the
estimate, which is what noise does and a real effect does not. Resolving it needs
1,500-2,500 trajectories: more traces, not a longer run. *(phase 85)*

## Overturned

**"The real wiring beats a degree-preserving rewiring" (+0.0495, resolved).**
The rewiring summed collisions and lost 5,255 of 30,543 synapses, so it beat a
sparser matrix. With an exact curveball trade the difference is -0.0023, not
resolved. *(phase 38)*

**"Memory raises agent success from 5/15 to 13/15 at a tight budget."**
The agent inherited the operator's MCP servers and spent its turns on tool
discovery; a note block of concrete Bash commands short-circuited that hunt. With
`--safe-mode` the effect is gone. *(phase 78)*
