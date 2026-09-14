# Research reuse map

This experiment deliberately combines established mechanisms instead of proposing a new
architecture before the basic phenomenon is measured.

| Prior work | Mechanism reused here | Deferred from phase 0 |
| --- | --- | --- |
| [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620) | A model-valued hidden state updated online; slow parameters learn the update behavior | Full TTT language-model stack and optimized mini-batch scan |
| [Titans](https://arxiv.org/abs/2501.00663) | Prediction residual as surprise; explicit retention/decay | Momentum, deep memory MLP, and integration into a transformer backbone |
| [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) | Residual delta-rule update, which can replace an existing key/value association | Kernelized multi-head fast weights |
| [MIRAS](https://arxiv.org/abs/2504.13173) | Separate the choices of memory architecture, attentional objective, retention, and optimizer | Alternative objectives and retention rules |
| [Fast Weight Memory](https://arxiv.org/abs/2011.07831) | Evaluate compositional associative behavior as a learned recurrent state | Its LSTM controller and task-specific reasoning stack |
| [Synaptic plasticity-dependent competition rule influences memory formation](https://doi.org/10.1038/s41467-021-24269-4) | Treat memory allocation as competition among plastic parameter groups, not only a global write/no-write decision | Any claim that mouse fear-memory allocation is a general model of human memory |

The official [TTT PyTorch implementation](https://github.com/test-time-training/ttt-lm-pytorch)
is a useful equation and tensor-shape reference, but its authors describe it as a slow tutorial
implementation and do not recommend it for training. Copying the full language-model stack would
therefore increase the first experiment's cost without testing the write-selection hypothesis.

## Frozen design choices for phase 0

- **Memory representation:** one matrix-valued fast-weight model state per episode.
- **Write rule:** normalized residual delta update.
- **Attentional objective:** later key-to-value recall cross-entropy.
- **Retention:** fixed multiplicative decay.
- **Write selection:** learned scalar gate conditioned on the observation, noisy present-time
  evidence, and prediction surprise.
- **Outer supervision:** future recall only. `retained` is used for episode construction and
  evaluation, never as a direct write-gate label.
- **No graph or external store:** the changing parameter tensor is the memory.

These choices isolate one question: can delayed future utility teach a model when to change its own
fast weights? They do not yet establish semantic text memory.

## Decision gates

1. **Mechanism gate:** learned write gating must beat an identically trained always-write model on
   held-out random bindings, while assigning higher mean gates to useful events.
2. **Dynamics gate:** add correction, delayed decay, and multi-timescale consolidation only after
   gate 1 passes.
3. **Language gate:** replace symbolic IDs with frozen text encoder representations and add a small
   decoder only after the state-update mechanism passes.
4. **Agent gate:** connect the memory output to a frozen LLM and measure spontaneous recall without
   memory instructions only after the standalone model passes.

Moving in this order prevents a capable language decoder from hiding a failed memory mechanism.

## Biological allocation hypothesis for phase 1

Jeong et al. manipulated potentiation and depression at auditory inputs to the lateral amygdala in
mice. Potentiated inputs were preferentially recruited into the fear-memory engram; post-training
depression could reverse that allocation. The total recall-active population remained approximately
constant, which the authors interpret as evidence for competitive allocation under limited
plasticity rather than independent binary selection by every neuron.

This suggests a specific computational ablation, not a biological equivalence claim:

```text
single global write gate              competitive block allocation
g = sigmoid(controller(x))     vs.    a = sparse competition(score_1 ... score_n)
W <- W + g * delta                    W_i <- W_i + a_i * delta_i
```

The competition should have a fixed plasticity budget (`sum(a) = 1` or top-k allocation). A recent
event can still be reassigned during a short eligibility window: strengthening one block competes
with or depresses neighboring blocks. The testable question is whether this improves capacity,
correction, and memory linking over the scalar-gate phase-0 model at an equal parameter and update
budget.

Do not use this study as evidence that the same rule governs all human memory. It examined a
specific fear-conditioning circuit in mice with artificial optogenetic LTP/LTD.

# Survey, 2026-09-14: should the write decision be deferred?

Phases 12 to 16 train a write gate to predict future utility from the present observation. Phase 7
already measured the ceiling of that framing on real logs: 0.543 balanced accuracy from
observation-time evidence against 0.980 once delayed feedback arrives. This survey asks whether the
deferred alternative has support beyond that single measurement, and what shape it should take.

The boundary from phase 1 still applies, but it applies to one use only. Animal results are a source
of *architecture* to test, never evidence for a claim about human memory or about our own numbers.
Phase 1 is the precedent: competitive allocation came from Jeong et al. and beat a single matrix
0.9243 to 0.2347, which is a result about our model, not about mice.

## Deferred consolidation in biology

Three independent mechanisms converge on writing cheaply from a local signal and letting a later
event decide what survives. What they add beyond the principle is a time constant.

| Mechanism | Deferral window | Source |
| --- | --- | --- |
| Excitability-biased allocation (CREB); excitability decays over hours | hours | [Josselyn & Frankland 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262914/); [J Neurosci 2024](https://www.jneurosci.org/content/44/21/e0846232024) |
| Synaptic tagging and capture: a weak tag captures later plasticity-related proteins | classically under 3 h; one 2026 report extends capture to 9 h | [Frey & Morris, Nature 1997](https://www.nature.com/articles/385533a0); [Commun Biol 2026](https://www.nature.com/articles/s42003-025-07998-w) |
| Replay selection biased by reward prediction, consolidated across sleep | hours to overnight | [Nat Commun 2026](https://www.nature.com/articles/s41467-025-65354-2); [eLife 2015](https://elifesciences.org/articles/07903) |
| Reconsolidation: retrieval returns a trace to a labile state | about 6 h | [Nader et al., Nature 2000](https://pubmed.ncbi.nlm.nih.gov/35145138/) |

The design consequence is that the deferral window is finite. "Hold every candidate until the query
arrives" is retrieval, not memory, and has no biological counterpart. The measurable quantity is how
far a bounded delay closes the gap between deciding at write time and deciding with the query
visible, which on the Claude corpus is 0.5686 against 0.8206 top-1.

## The mushroom body as a specified circuit

The fly connectome makes a complete memory circuit available as architecture rather than principle:
139,255 neurons and about 50 million synapses in the adult female brain
([Dorkenwald et al., Nature 2024](https://www.nature.com/articles/s41586-024-07558-y);
[Schlegel et al. 2024](https://www.nature.com/articles/s41586-024-07686-5)). Connectivity alone,
with neurotransmitter identity, already predicts behaviour in a whole-brain leaky integrate-and-fire
model ([Shiu et al., Nature 2024](https://pubmed.ncbi.nlm.nih.gov/37205514/)).

Four properties bear on our design.

- **Sparse expansion.** About 50 projection neurons fan out to roughly 2,000 Kenyon cells, each
  integrating 6 to 8 inputs across 5 or 6 claws. Reciprocal inhibition through the single GABAergic
  APL neuron keeps only 5 to 10 percent of Kenyon cells active, and removing that sparsening impairs
  discrimination of similar but not dissimilar odours
  ([Lin et al., Nat Neurosci 2014](https://www.nature.com/articles/nn.3660)). Our `keep_ratio` of
  0.25 is several times denser than this.
- **The write signal does not read the content.** Kenyon cells carry odour identity; dopaminergic
  neurons carry reinforcement and gate plasticity at the Kenyon-cell-to-output-neuron synapse. The
  same dopaminergic neuron writes aversive memory when it fires within about 30 s after odour onset
  and appetitive memory when it precedes odour by 20 to 60 s, so sign comes from the modulatory
  channel alone. A second activation minutes after training, **with no odour present**, nearly
  abolishes the conditioned response ([Aso & Rubin, eLife 2016](https://elifesciences.org/articles/16135)).
  `SelectiveWriteRecallMemory` does the opposite: its gate scores the content itself. Phase 7's
  `apply_outcome()` is the component that already matches this shape and is currently unused by the
  phase 12 to 16 line.
- **Compartments are parallel, not a promotion pipeline.** Output neurons tile the lobes into 15
  compartments with their own dopaminergic input and their own decay, capacity, and flexibility.
  γ1pedc holds strong immediate memory that is largely gone by 24 h and has a capacity of one,
  retaining only the most recently trained odour, while α1 has weak immediate memory, supports
  long-term consolidation, and holds at least two. Trained with opposing signals, both traces form
  simultaneously and the expressed valence flips as the fast compartment decays. This favours
  several stores with different decay rates read as a sum, over a provisional buffer that promotes
  into a durable one.
- **Derived algorithms exist but not for this.** Fly-inspired locality-sensitive hashing
  ([Dasgupta, Stevens & Navlakha, Science 2017](https://www.science.org/doi/10.1126/science.aam9868)),
  Bloom-filter novelty detection ([PNAS 2018](https://www.pnas.org/content/115/51/13093)), and sparse
  coding against catastrophic forgetting ([arXiv:2107.07617](https://arxiv.org/pdf/2107.07617)) all
  come from this circuit. None of them learns a write gate.

## What forgetting research says about capacity

Forgetting is an active, default-on process rather than passive leakage: ongoing dopaminergic
activity erases labile memory in Drosophila unless salience marks it for retention
([Berry et al., Neuron 2012](https://www.cell.com/neuron/fulltext/S0896-6273(12)00338-8)), and
transience is argued to be functional, preventing overfitting to specific episodes
([Richards & Frankland, Neuron 2017](https://www.cell.com/fulltext/S0896-6273(17)30365-3)). Decay
rate itself appears to adapt to update frequency, existing to limit interference
([Altmann & Gray, Psych Science 2002](https://journals.sagepub.com/doi/10.1111/1467-9280.00405)).

Competition bounds *relative* excitability, not a counted slot budget, and Landauer explicitly
disclaimed his capacity estimate as a bound on storage
([Landauer, Cognitive Science 1986](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1004_4)).
A hard top-k at write time therefore collapses three separable knobs — encoding strength, decay
rate, and retrieval threshold — into one irreversible switch. Phase 14 already screened an
input-dependent retention gate and rejected it, attributing the failure to 775 training episodes
being too few to identify a richer retention policy rather than to the formulation.

## Where our novelty now sits

| Prior work | Relation to this project |
| --- | --- |
| [Learning to Evict from Key-Value Cache](https://arxiv.org/abs/2602.10238) (KVP, ICML 2026) | Per-head reinforcement-learned agents rank tokens by predicted usefulness. This is already a learned future-utility write gate, so "learned write gating" is no longer a differentiator. Differs by acting on KV tokens inside the model rather than semantic traces beside a frozen encoder, and by being single-stage. |
| [RecMem](https://arxiv.org/abs/2605.16045) (ACL 2026 Findings) | Closest two-stage system: every interaction enters a subconscious embedding layer and an LLM consolidates only when sustained recurrence appears among semantically similar interactions, cutting construction tokens up to 87%. Promotion-only and never discards, so it does not face irreversible capacity loss. |
| [What Eviction Destroys](https://arxiv.org/abs/2609.08279) | Independent support rather than competition: a restore-counterfactual audit attributing 60 to 73 percent of corrected errors to irreversible eviction at an 80k budget, and all of them at 8k. Corroborates the phase 15 finding that the whole task-level shortfall is write selection. |
| [Agentic Memory](https://arxiv.org/abs/2601.01885) (ACL 2026) | Learns unified long- and short-term memory management with three-stage reinforcement learning and a step-wise GRPO variant for sparse memory-action rewards. Whether the backbone language model is itself updated is not stated in the abstract and was not verified. |

The remaining differentiators are the two-stage deferral and the irreversible-discard semantic-trace
setting, not learned write gating as such.

## Contested — do not present as settled

- Post-encoding reversal of allocation rests mainly on Jeong et al. 2021. General allocation reviews
  do not establish it.
- Human retroactive behavioural tagging has explicit null results, and reconsolidation has a direct
  failed replication ([Sci Rep 2022](https://www.nature.com/articles/s41598-022-06119-5)).
- Parallel compartments versus the older sequential consolidation model (γ to α′β′ to αβ) remains
  open; trace migration must not be presented as fact.
- Whether forgetting erases or only removes access is disputed
  ([Ryan & Frankland, Nat Rev Neurosci 2022](https://pubmed.ncbi.nlm.nih.gov/35027710/)).
- Kenyon cell connectivity is mostly but not entirely random; food-responsive projection neurons are
  a reported exception ([eLife 2022](https://elifesciences.org/articles/77578)).
- [Neuromem](https://arxiv.org/abs/2602.13967) is a streaming-memory testbed that decomposes the
  lifecycle into data structure, normalization, consolidation policy, query formulation, and context
  integration. It treats consolidation as a distinct dimension, but whether any stage carries learned
  parameters is not stated in the abstract and was not verified.

All arXiv identifiers in this survey were checked against the listing pages on 2026-09-14.

# Redesign, 2026-09-14

## Why the previous design has to be abandoned

Phases 12 to 19 asked a bounded layer to pick which of eight candidates a later query would need.
Every result in that line collapsed to the same thing once controlled. On the deferred corpus an
untrained random projection of the same width already reaches 0.4033 top-1 with no matrix and no
learning; the trained associative model reaches 0.4956; a blank event scores 0.4956 against 0.4971
for the correct one; and four different write policies land between 0.4825 and 0.5000. The slot
model's large event effect came from feeding cosine-to-event in as a residual score rather than from
anything the memory computed.

The cause is the task shape. Presenting the candidates at recall time lets encoder cosine stand in
for memory, and cosine is better at it than any bounded state we can train. Two further points
sharpen this. Importance is not utility, so scoring a write decision against a future-utility oracle
measures the wrong thing by construction. And memory is gist-preserving and detail-losing, so a
metric rewarding exact reconstruction of a stored embedding rewards verbatim storage, which is not
what memory does.

None of the survey's findings are about choosing from a candidate list. Every one of them is about
how a bounded memory degrades: transience as a function, decay adapting to interference, compartment
time constants, sparsening that helps only for similar items. The experiments must measure that
instead.

## What the survey licenses, as constraints

1. **The write signal must not read content.** Dopaminergic neurons carry reinforcement, not odour
   identity, and sign comes from the modulatory channel alone. Write strength may depend on the
   memory's own prediction error or on an external outcome, never on a judgement about the content.
2. **Modulation must work with the item absent.** A second activation minutes after training, with
   no odour present, nearly abolishes the response. Consolidation therefore acts on an eligibility
   trace addressed by coincidence, not by re-presenting the item.
3. **Stores run in parallel with different time constants and are read as a sum.** Compartments do
   not promote into one another; γ1pedc holds one item and is gone by 24 h while α1 is weak
   immediately and durable later, and the expressed answer flips as the fast store decays.
4. **Codes are sparse and expanded, and sparsening matters only for similar items.** Fifty inputs
   fan out to two thousand cells with 5 to 10 percent active, and removing the inhibition impairs
   discrimination of similar but not dissimilar odours.
5. **Forgetting is active and its rate adapts to interference.** Decay is a trained dynamic, not a
   constant, and encoding strength, decay rate, and retrieval threshold stay three separate knobs.
6. **Allocation is competitive under a roughly fixed budget.** Potentiated inputs are preferentially
   recruited while the recall-active population stays about constant, so strengthening one trace
   should cost others rather than being free.
7. **What is allocated depends on current excitability, which decays over hours.** Items encountered
   close together are therefore biased onto shared substrate, which predicts linking between them and
   not only interference.
8. **A weak trace can be rescued later, inside a window.** Tagging and capture give a weak event
   access to plasticity a later strong event supplies, classically under three hours. The window is
   finite, which is the part the corpora could not show us.
9. **Consolidation continues offline and selectively.** Replay is biased by reward and keeps working
   across sleep, so retention can improve with no new input.
10. **Reading a memory changes it.** Retrieval returns a trace to a labile state before it
    restabilises, so recall is not a passive operation on the state.

## The replacement task

Store N documents in a bounded state, probe with a cue, and measure what comes back. No candidate
list exists at recall, so similarity between the cue and the stored items cannot substitute for the
memory. The state is the only thing carried from write to read.

Two scores per probe, because the survey says surface and gist should not be treated alike:

- **gist recovery** — agreement between what is recalled and the fact the document carried;
- **surface recovery** — agreement between what is recalled and the document's own form.

A memory that behaves like the literature describes should lose surface faster than gist as load
rises. A verbatim store loses both together; a retriever loses neither until it fails entirely.

## Measurements, one per mechanism the survey established

Every row names the finding it comes from, the prediction that follows, and what would falsify it.
Elapsed time in the animal work maps onto intervening writes here, since interference rather than
the clock is what the state actually experiences.

| # | From | Measurement | Prediction | Fails if |
| --- | --- | --- | --- | --- |
| 1 | Transience is functional (Richards & Frankland 2017); decay adapts to interference (Altmann & Gray 2002) | Fidelity against number of documents stored, at fixed state size | Graceful degradation, and a trained decay beats a fixed one only as load rises | Flat, or a cliff, or trained decay never separates from fixed |
| 2 | Sparse expansion and APL inhibition; removing it impairs similar but not dissimilar odours (Lin et al. 2014) | The same curve for sparse expanded codes against dense low-dimensional ones, run separately on similar and dissimilar document sets | Sparse degrades more slowly, and the gap appears only for similar documents | Sparsity helps uniformly, or not at all |
| 3 | Memory is gist-preserving and detail-losing | Gist against surface recovery across the same load | Surface decays first | Both decay together |
| 4 | Compartments run in parallel with their own decay and capacity; expressed valence flips as the fast one fades (Aso & Rubin 2016) | Two stores with different decay read as a sum, probed after varying numbers of intervening writes | What is expressed changes with delay, and one store cannot reproduce the curve | A single store matches the two-store readout |
| 5 | Competitive allocation under limited plasticity; total engram population roughly constant (Jeong et al. 2021) | Whether strengthening one trace measurably costs others, at fixed state size | Gains and losses trade off; forcing that trade-off helps rather than hurts | Traces strengthen independently, or the constraint only hurts |
| 6 | Excitability biases allocation and decays over hours (Josselyn & Frankland 2015) | Whether documents written close together share substrate | Items written adjacently are recalled together more than distant ones, a linking effect distinct from interference | No difference between adjacent and distant pairs |
| 7 | Synaptic tagging and capture: a weak tag is rescued by a later strong event inside a window (Frey & Morris 1997) | Write an item weakly, apply a strong unrelated event after k intervening writes, measure rescue against k | Rescue falls off with k, giving the deferral window the corpora could not supply | Rescue is flat in k, or absent |
| 8 | Replay selection is biased by reward and consolidates offline (eLife 2015; Nat Commun 2026) | An offline phase that re-applies stored eligibility with no new input | Selected traces retain better than unselected ones after the same load | Offline replay changes nothing |
| 9 | Retrieval returns a trace to a labile state (Nader et al. 2000) | Probe an item, then re-write, and measure drift | Recalled items drift, and repeated recall compounds it | The state is unchanged by being read |
| 10 | The dopaminergic write signal never reads content, and a second activation with the item absent abolishes the response (Aso & Rubin 2016) | A modulatory signal applied after writing, with the item absent | Recall changes, and a mismatched signal does not produce that change | The signal is inert, as it was in every attempt today |

Order matters. Measurement 2 runs first: it is the sharpest circuit-derived prediction available, it
fails in a specific direction, and nothing in this project depends on it yet. Measurements 1 and 3
come with it, since they share the load curve. Measurement 7 is the one worth the most, because it
recovers the deferral window that LongMemEval-S could not show — there the later event stayed equally
informative at every distance, whereas here the delay is ours to set. Measurement 10 has already
failed twice in this project and should not be attempted again until 1 to 3 establish that the memory
holds anything at all.

Three of these are marked contested above and must be reported as tests of a disputed claim rather
than as confirmations: 5 rests mainly on Jeong et al., 9 has a direct failed replication, and 4 sits
against the older sequential-consolidation model.

## What has to be built

Controlled document sets rather than found episodes, because measurements 1 and 2 need the number of
stored items and the similarity between them as independent variables. LongMemEval sessions supply
the documents and their facts; grouping them by encoder similarity supplies the similarity axis. The
teacher disappears entirely: the objective is reconstruction of what was stored from its own cue, so
no label about which document will matter later enters training at any point.

## What this design deliberately gives up

It stops asking whether the layer can predict future utility, which Phase 19 measured as
information-limited anyway, and it stops competing with retrieval on retrieval's own ground. If the
memory cannot beat cosine at picking a candidate, that is no longer a finding about the memory. The
claim available at the end of this program is narrower and about capacity: how much a bounded state
holds, how it fails, and whether it fails the way the biology says it should.
