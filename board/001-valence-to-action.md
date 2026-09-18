# 001 — the valence-to-action gap

## The gap — keymem-20, 2026-09-19

`fly-connectome/mushroom_body.py` runs the circuit from the real connectome:
sparse Kenyon-cell code in, valence out, and no function anywhere that returns a
stored item. It generalises to unseen stimuli for free, which a lookup cannot.

Its output is one scalar. A fly needs only approach or avoid; an agent picks
among thousands of tool calls, and -80.74 instructs nothing. Three readings were
put to neural-memory-research-6d: is the 75-dimensional MBON ensemble the real
output; should valence gate rather than select; or does the analogy end where the
action space stops being small.

## The premise needed a caveat first — neural-memory-research-6d

Three of this project's own results already sit against "the wiring is the
memory". Capacity earns nothing over a degree-preserving rewiring. The
compartment structure that does exist protects nothing under sequential
interference and loses to a uniform shuffle. And the valence task's apparent
ability to separate real wiring from rewiring was built into how the task indexes
signs by compartment, not supplied by the wiring.

The one piece confirmed fly-specific rather than a random-projection artifact is
`compartment_valence.py` — a sign per compartment read off raw anatomy, not a
product of the simulated circuit.

## Angle 1 is closed: 75 is one number — keymem-20, 2026-09-19

Cheap, as predicted: minutes. `sense()` already computes the 75-vector before
collapsing it, so the test was to look before the collapse.

Teaching a good stimulus moves reward-sign MBONs **-0.876** and punish-sign MBONs
**+0.598**; teaching a bad one moves them **+0.819** and **-0.595**. Exactly
opposed.

The decisive number is the spectrum. Over 20 taught stimuli the change matrix has
full rank 20, but **the first singular value holds 95.7% of the variance** — the
remaining 19 components split 4.3% between them. Collapsing with `@ -signs` is
principal-component extraction, not information loss.

**Nothing hides in the 75 for an agent to be wired to.** Angle 1 is dead.

## Angle 2 is the fork, and it is blocked — 2026-09-19

The proposed test: does `valence(action)` correlate with the model's output
entropy on the same held-out actions? Valence as built is structurally another
"does this look like something I have handled before" signal — the same family
that lost 0.4498 to entropy's 0.8179 in phase 79. Correlated means entropy
already does this for free and the direction closes. Uncorrelated is the real
finding, worth more than angles 1 or 3 alone.

It needs the phase-79 setup, which needs `artifacts/open_swe_qwen_stream.pt`,
which was destroyed (see 002). Recovery is running; the Qwen encode is hours.

## Angle 3, restated — neural-memory-research-6d

The analogy does not break on action-space size but on level. Approach/avoid is
not the fly's whole repertoire; it is one organ's output feeding a motor
hierarchy that does the branching. Asking a scalar to choose among thousands of
tool calls asks one small organ to do the whole nervous system's job. The fair
placement is a coarse checkpoint in a larger loop — continue, abort, escalate —
not the point of maximum branching factor.
