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

**The reason is duplication, not weakness.** The better a representation captures
the present, the more the nearest stored item repeats what the prompt already
says. This predicted the direction of the Qwen-space result before it was run.
*(phase 75)*

**Memory as an untrained bias on the residual stream does nothing.** Own-trajectory
and foreign-trajectory vectors are indistinguishable at every scale and in every
readout tested (mean, associative, sparse). *(phase 77)*

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

## Overturned

**"The real wiring beats a degree-preserving rewiring" (+0.0495, resolved).**
The rewiring summed collisions and lost 5,255 of 30,543 synapses, so it beat a
sparser matrix. With an exact curveball trade the difference is -0.0023, not
resolved. *(phase 38)*

**"Memory raises agent success from 5/15 to 13/15 at a tight budget."**
The agent inherited the operator's MCP servers and spent its turns on tool
discovery; a note block of concrete Bash commands short-circuited that hunt. With
`--safe-mode` the effect is gone. *(phase 78)*
