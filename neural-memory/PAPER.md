# Can a Small Independent Memory Layer Learn What to Store Before the Query?

## Status

Working preprint and reproducibility plan, 2026-09-13. The present evidence supports a
phenomenon claim, not a general solution to autonomous memory. All negative branches and controls
are retained in `RESULTS.md`.

## Research question

Can a small trainable layer attached to a frozen language encoder decide, from the current input
alone, which information should occupy a bounded memory, update that memory state, and later
activate the relevant state from an ordinary future utterance without an explicit search command?

This is deliberately different from asking whether vector search can retrieve a relevant passage.
The decisive event occurs before the future query exists: the layer must discard most candidates
and cannot recover a discarded trace later.

## Model under test

The frozen encoder maps text to semantic features. `SelectiveWriteRecallMemory` is an independent
106,946-parameter layer with three learned operations:

1. **Write:** score each current candidate without the future query and execute a hard capacity
   decision.
2. **State update:** project selected candidates into persistent 64-dimensional traces; discarded
   traces are absent from the later state.
3. **Recall:** use an ordinary later utterance to activate a surviving trace.

There is no Keymem integration, graph traversal, database search, prompt-level memory instruction,
or LLM fine-tuning in this experiment.

## Training curriculum

Pure delayed end-to-end credit did not teach the hard write decision. The current curriculum is:

1. imitate a weak write teacher;
2. freeze the writer and train recall from the later event;
3. optionally unfreeze both operations for low-rate joint optimization while retaining a teacher
   regularizer.

The current ablation compares independent BCE and listwise ranking supervision. A second ablation
adds three online-only signals available at write time: normalized position, novelty against earlier
traces, and prediction surprise against an exponential moving summary. This surprise is an embedding
proxy, not the gradient-based surprise defined by Titans.

## Evaluation design

The primary outcome is not generated-answer quality. It is whether the future-relevant item survives
the earlier capacity bottleneck and is then activated by the correct later context.

Primary metrics:

- target retention at 25% and 75% memory capacity;
- write balanced accuracy;
- recall top-1, top-2, and mean reciprocal rank;
- correct-query minus shuffled-query top-1;
- position-macro retention and top-1 to remove recency-distribution advantages.

Required controls:

- random capacity and untrained writer;
- randomly permuted write teacher;
- recency, causal novelty, and causal surprise heuristics;
- full frozen-embedding cosine with the query visible, as a retrieval upper reference;
- shuffled later query;
- held-out project or held-out question groups;
- at least five initialization seeds for the adopted configuration.

## Datasets

### Natural Claude revisit corpus

Eight real earlier user turns precede one naturally recurring user turn by at least two turns. The
weak label is lexical reuse. Entire projects are held out. The current artifact contains 775 training
and 574 evaluation episodes; raw conversation text is not persisted in the repository.

### ContextBench issue split

An early title-like context drives write and a later issue body drives recall. This is larger and
project-grouped but synthetically separates one document, so it is supporting evidence rather than
the strongest temporal test.

### LongMemEval-S external validation

The official cleaned LongMemEval-S histories provide evidence-session IDs. Each evidence session is
mixed with seven deterministic distractor sessions, while the question is hidden until recall. The
split holds out complete question IDs and target position is deterministically randomized. This
tests cross-source transfer of the phenomenon on a public benchmark, although its conversations are
constructed and evidence sessions may contain generation artifacts.

## Current result

On the natural Claude corpus at two-of-eight capacity, the staged model retains the future-relevant
turn in 0.5906 +/- 0.0121 of held-out-project episodes versus a 0.25 random-capacity expectation.
Recall top-1 is 0.5627 +/- 0.0164 and falls to 0.3369 +/- 0.0050 when the later query is shuffled.
Conditional on the target surviving, top-1 is approximately 0.9527. This localizes the remaining
bottleneck to write selection.

Adding causal position, novelty, and prediction-surprise signals raises five-seed retention to
0.6279 +/- 0.0097 and top-1 to 0.5882 +/- 0.0085 on MPS; an independent CPU repeat reaches
0.6195 +/- 0.0101 and 0.5878 +/- 0.0075, preserving the direction of both gains. On the adapted
public LongMemEval-S test, the same
staged layer retains 0.4030 +/- 0.0263 versus a 0.25 capacity expectation; position-macro retention is
0.4058 +/- 0.0285 and every seed has a positive correct-query advantage. A 14,463-parameter
fast-weight alternative with a 144-scalar matrix state reaches 0.6066 +/- 0.0250 retention and 0.5575
+/- 0.0211 top-1 on the natural Claude corpus, showing a parameter-efficient parametric-memory
effect but not surpassing the adopted trace layer.

## Related design lineage

- The [Differentiable Neural Computer](https://www.nature.com/articles/nature20101) established a
  learned controller with read, write, allocation, and free operations over bounded external state.
- [LongMem](https://arxiv.org/abs/2306.07174) motivates keeping the backbone frozen and training a
  decoupled side network, avoiding representation staleness and full-model adaptation.
- [TTT layers](https://arxiv.org/abs/2407.04620) treat the hidden state as a model updated by an
  inner self-supervised objective; this motivates the planned teacher-free state-update branch.
- [Titans](https://arxiv.org/abs/2501.00663) prioritizes surprising inputs and combines momentum with
  adaptive decay. Our causal surprise feature is a cheap diagnostic precursor, not an implementation
  of Titans.
- [MIRAS](https://arxiv.org/abs/2504.13173) separates memory structure, attentional-bias objective,
  retention gate, and update algorithm. Those four axes define the ablation matrix here.
- [MEMORYLLM](https://arxiv.org/abs/2402.04624) studies a fixed-size, self-updatable latent memory
  pool inside a transformer; our layer instead remains independently trainable and attachable.
- [LongMemEval](https://arxiv.org/abs/2410.10813) supplies the public long-term conversational-memory
  validation source.

## Claims that are not yet supported

The experiments do not yet show intrinsic importance, human-like consolidation, indefinite online
learning, answer-quality improvement, or robust forgetting. The weak teachers reveal future utility
during training, and frozen embeddings already contain substantial semantic geometry. A paper must
therefore frame the contribution as **query-hidden capacity selection plus later spontaneous
activation**, and treat teacher-free importance discovery as the next hypothesis rather than a
completed result.

## Next decisive experiments

1. Finish five-seed listwise, causal-surprise, and joint-training ablations on natural logs.
2. Repeat the adopted model and controls on LongMemEval-S with position-macro reporting.
3. Add a parametric fast-weight state whose inner reconstruction gradient supplies genuine
   test-time surprise, then compare it with the trace-state layer at equal persistent-state budget.
4. Add explicit retention/forget gates and evaluate knowledge-update examples, where obsolete
   evidence must lose activation to newer evidence.
5. Measure whether injected memories improve a frozen generator while preserving the activation
   metrics as the mechanistic primary endpoint.
