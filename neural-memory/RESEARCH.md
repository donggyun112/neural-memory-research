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
