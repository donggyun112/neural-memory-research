# What a Bounded Memory Layer Can and Cannot Do Before the Query Arrives

## Status

Working preprint and reproducibility plan, 2026-09-14. The evidence supports two phenomenon claims
and one substantial negative result; it does not support a general solution to autonomous memory.
All negative branches and controls are retained in `RESULTS.md`, which is the authoritative ledger.

The project ran in two tracks. Track A asked whether a small layer can decide *what to keep* before
the query exists. Track B abandoned that framing and asked whether the network itself, as an
associative store, can *be* the memory. Both tracks reached a measured limit, and the limits are
different. Readers should not carry a Track A result into a Track B claim.

## Track A: query-hidden capacity selection

### Research question

Can a small trainable layer attached to a frozen encoder decide, from the current input alone, which
information should occupy a bounded memory, and later activate the relevant state from an ordinary
future utterance without an explicit search command?

The decisive event was assumed to occur before the future query exists: the layer must discard most
candidates and cannot recover a discarded trace later.

**Phase 17 measured that assumption and it is wrong.** Sweeping the provisional capacity from an
immediate two-of-eight commitment to holding all eight until a later event arrives moves retention
from 0.5281 to 0.6456. Over the same span, ranking those candidates by raw cosine to the same event
reaches 0.9342. The dominant cost is the learned selection mechanism, not the irreversible discard.

### Model under test

`SelectiveWriteRecallMemory`, an independent 106,946-parameter layer over a frozen encoder, with
three learned operations: a hard capacity decision taken without the future query, a projection of
survivors into persistent 64-dimensional traces, and a recall step driven by an ordinary later
utterance. No graph traversal, database search, prompt-level memory instruction, or backbone
fine-tuning.

### What Track A established

On the natural Claude corpus at two-of-eight capacity the staged model retains the future-relevant
turn in 0.5906 +/- 0.0121 of held-out-project episodes against a 0.25 random-capacity expectation,
with recall top-1 0.5627 +/- 0.0164 falling to 0.3369 +/- 0.0050 under a shuffled query. Conditional
on survival, top-1 is about 0.9527. Causal position, novelty and prediction-surprise features raise
five-seed retention to 0.6279 +/- 0.0097, reproduced on CPU at 0.6195 +/- 0.0101.

Phase 16 established that this private margin is semantic rather than a length correlate: a
length-only write arm reaches 0.4056 +/- 0.0128 against 0.5920 +/- 0.0099 for the frozen embedding,
and making length explicit alongside the embedding does not improve it.

A gated frozen-generator endpoint supplied task-level evidence for this track. Scoring the gold
answer with a frozen `Qwen/Qwen2.5-1.5B-Instruct`, the layer reaches 2.1508 answer NLL against an
oracle 2.2135 and a no-memory 2.7587 on the 29 of 72 episodes where its write succeeded; on the 43
where it failed the injected memory is near-inert at 2.9195 against 2.9593 while the oracle still
reaches 2.3937. The whole task-level shortfall is write selection, not recall.

### Where Track A stopped, and why

Phase 15 withdrew the external claim: on LongMemEval-S the training-free longest-two rule retains
0.4159 against a learned 0.4069, so the public benchmark cannot distinguish this layer from a length
heuristic. Phase 19 then measured the ceiling directly. An unconstrained listwise scorer on the raw
384-dimensional features, with no bottleneck and no staged freezing, reaches 0.5044 on the deferred
corpus against the gate's 0.5380, and 0.6069 against 0.5920 on Claude.

The write decision is **information-limited, not capacity-limited**. The same layer that cannot
exceed about 0.54 at write time reaches 0.9500 once the later event arrives. Further architecture
work on stage one is unjustified, which is what ended this track.

## Track B: the store itself as the memory

### Redesign

Track A kept treating memory as a selection problem over slots. Track B rebuilt the question: the
network *is* the memory. `RESEARCH.md` records ten numbered measurements drawn from the biological
literature — sparse expanded coding in the mushroom body, synaptic tagging and capture, opponent
modulatory channels from the fly DAN/MBON circuit, parallel stores with different time constants,
competitive allocation, replay and reconsolidation — each naming its source finding and its
falsification criterion before it was run.

The store is delta-rule associative: `W <- λW + s·(v − W k)kᵀ`, with keys formed by a sparse expanded
code and values read back by comparison against what was stored.

### What Track B established (Phases 20–30)

Six of the ten predictions survived, three were rejected, one split. The rejections cluster: three of
them are explained by the error-correcting write rule alone, which is a single mechanism rather than
three independent failures.

The surviving results, all on held-out discrimination — whether a read returns its own value ahead of
every other stored value:

- Sparse expanded coding raises discrimination, and the advantage is 30–60x larger for *similar*
  documents, as the sparsening literature predicts. Measured on fidelity the effect appears
  reversed; discrimination is the quantity the prediction is about.
- Decay does not help. Established three independent ways: a sweep of constants (Phase 21), a
  genuine per-unit adaptive rule (Phase 27), and gradient descent driving both decay constants to
  0.9995 and 0.9964 without being told (Phase 30).
- Training the assembled store lifts held-out discrimination from 0.8774 to 0.9513.

Phase 30 also overturned Phase 29's own hand ablation. The ablation measured the tag pathway at
0.0007 and found the parallel store harmful; restored as trainable quantities, every constant moved
*up* and none switched off, with capture going from a hand-set 0.25 to 0.9445. A hand ablation
answers whether a component helps *at the setting it was given*, and cannot separate a dead component
from one configured shut.

### Track B's negative result

**Discrimination is a proxy, and it does not transfer.** Phase 31 gave the store the task the
benchmark actually asks: store a question's candidate sessions, probe with the question embedding —
a cue that was never written — and return the evidence session.

| Reader | LongMemEval deferred | LongMemEval revisit |
| --- | ---: | ---: |
| Chance | 0.1250 | 0.1250 |
| Trained store | 0.5578 +/- 0.0347 | 0.6337 +/- 0.0200 |
| Trained key projection, nothing written | 0.8178 +/- 0.0401 | 0.8416 +/- 0.0245 |
| **Cosine on the frozen encoder** | **0.8622 +/- 0.0295** | **0.8465** |

Writing the sessions down *costs* about 0.21–0.26 against reading them with the very projection the
store was trained to use. Three defences were tested and none survived:

- **Undertraining.** Twelve thousand steps gives 0.5741 +/- 0.0105, not materially different from
  three thousand.
- **Sparsity.** A question's code already shares half its active units with the evidence session's at
  width 8, and at width 512 the code is dense with total overlap by construction — the store still
  reads 0.63 against cosine's 0.86. Widening the value bottleneck peaks at 0.7074.
- **Load.** Padding each question's candidate set with sessions from other questions widens the gap
  rather than closing it: 0.16 behind at eight sessions, 0.35 behind at a hundred and twenty-eight.

The mechanism is legible. A delta-rule store is fitted to satisfy one equation per document — stored
key returns stored value — and sparsening makes those equations more nearly independent. That is why
every Track B measurement that improved discrimination did so, and why Phase 30's constants all ran
toward preserving what was written. None of it produces generalisation from a cue that was never
written, and the retrieval the benchmark asks for already lives in the encoder's geometry.

This does not retract the Track B measurements. They were about the store's ability to hold what it
was given, and they stand. What does not follow — and what the project assumed for twelve phases — is
that a store which holds documents well is a store that answers questions well.

## Evaluation design

The primary Track A outcome is whether the future-relevant item survives the capacity bottleneck and
is then activated by the correct later context; the primary Track B outcome is hit rate on the real
question against the frozen encoder it reads from. Answer quality is a gated secondary endpoint in
both.

Required controls, applied throughout:

- random capacity, untrained model, and randomly permuted write teacher;
- recency, causal novelty, causal surprise, and **length** heuristics;
- full frozen-embedding cosine with the query visible — in Track B this is the primary reference,
  not an upper bound to aspire to;
- a projection-only arm that trains the same projection but writes nothing, so a gain from the
  projection cannot be read as a gain from the store;
- shuffled later query; held-out project or question groups; at least five initialization seeds.

The frozen-generator endpoint carries a validity gate (`instrument_usable`): the oracle must beat
both no-memory and random, and its prompts must be untruncated. Two runs in Phase 31 failed that
gate and were not interpreted — correctly, since the deferred episode construction excludes the
second evidence session from the candidate pool, so no reader selecting from that pool can supply a
complete fact.

## Datasets

- **Natural Claude revisit corpus.** Eight real earlier user turns precede one naturally recurring
  turn; weak label is lexical reuse; entire projects held out. 775 training / 574 evaluation
  episodes. Raw text is not persisted in the repository.
- **ContextBench issue split.** Larger and project-grouped, but synthetically separates one document,
  so it is supporting rather than decisive.
- **LongMemEval-S.** Official cleaned histories with evidence-session IDs, each mixed with seven
  deterministic distractors, question hidden until recall, question IDs held out, target position
  deterministically randomized. Two episode constructions are used and they are **not
  interchangeable**: `revisit` episodes carry complete evidence in one session (890 episodes,
  688/202 split), while `deferred` episodes split evidence across two sessions and place only the
  first in the candidate pool (300 episodes).

## Related design lineage

- The [Differentiable Neural Computer](https://www.nature.com/articles/nature20101) established a
  learned controller with read, write, allocation, and free operations over bounded external state.
- [LongMem](https://arxiv.org/abs/2306.07174) motivates keeping the backbone frozen and training a
  decoupled side network.
- [TTT layers](https://arxiv.org/abs/2407.04620) treat the hidden state as a model updated by an
  inner self-supervised objective.
- [Titans](https://arxiv.org/abs/2501.00663) prioritizes surprising inputs and combines momentum with
  adaptive decay. Our causal surprise feature is a cheap diagnostic precursor, not an implementation.
  Phases 21, 27 and 30 independently find that decay does not help in an error-correcting store,
  which is a boundary on where adaptive-decay designs apply.
- [MIRAS](https://arxiv.org/abs/2504.13173) separates memory structure, attentional-bias objective,
  retention gate, and update algorithm — the four axes of the ablation matrix here.
- [MEMORYLLM](https://arxiv.org/abs/2402.04624) studies a fixed-size self-updatable latent memory pool
  inside a transformer; this layer instead remains independently trainable and attachable.
- [LongMemEval](https://arxiv.org/abs/2410.10813) supplies the public validation source.
- Johnson–Lindenstrauss is the relevant negative prior for Track B: random projections already
  preserve the geometry the retrieval needs, which is what the projection-only control measures.

## Claims that are not supported

The experiments do not show intrinsic importance, human-like consolidation, indefinite online
learning, robust forgetting, or **retrieval improvement over the frozen encoder**. That last one is
now a measured negative rather than an untested gap, and it is the strongest constraint on how this
work can be framed.

Two further limits carry from Track A. The layer's advantage over a training-free length rule exists
only under a tight budget — at three or four of eight slots it no longer beats that rule. And
deferral was tested where the later event is as informative as the query itself, so the result shows
a bounded memory exploiting clear later evidence, not hard temporal credit assignment.

A paper from this work should be framed as **a measured ceiling on query-hidden write selection, plus
a negative result on associative storage as retrieval**, with the biological component measurements
as supporting evidence about what such a store can hold. It should not be framed as a memory system
that improves question answering.

## Next decisive experiments

1. Establish where an associative store *does* pay, if anywhere. Phase 31 rules out retrieval against
   the same encoder at loads up to 128 sessions. The remaining candidates are regimes cosine cannot
   serve at all: composition across traces, updates that must overwrite an obsolete fact in place,
   and cues that are not embeddings of text.
2. Re-run the Track B component measurements under the Phase 30 constants rather than the hand-set
   ones. The Phase 22 rescue window in particular was measured at a tag decay of 0.9, and the learned
   value is 0.9964, which should widen it materially.
3. Find a public corpus whose evidence is not length-correlated and whose later evidence degrades
   with distance. Phase 16 shows the Claude margin is semantic and that LongMemEval-S cannot
   corroborate it; Phase 17 adds that LongMemEval-S has no measurable deferral window.
4. Test the remaining teacher correlates the way length was tested: turn position, vocabulary rarity,
   and question form each need an explicit-feature arm before the weak teacher can be called clean.
5. Extend the generator endpoint to several seeds and a second generator, keeping the activation
   metrics as the mechanistic primary endpoint and the validity gate as a precondition for reading
   any of it.
