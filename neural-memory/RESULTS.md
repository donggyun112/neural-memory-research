# Phase 0 results

Date: 2026-09-12

Command, repeated with seeds 7, 17, and 29:

```bash
uv run python train.py --steps 600 --eval-batches 50 --batch-size 64 --seed <seed>
```

Hardware selected by PyTorch: Apple M4 Pro via MPS.

| Seed | Learned-gate accuracy | Always-write accuracy | Accuracy lift | Useful gate | Distractor gate | Gate separation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.9255 | 0.6507 | +0.2748 | 0.6238 | 0.3920 | 0.2318 |
| 17 | 0.9235 | 0.6504 | +0.2731 | 0.6177 | 0.3929 | 0.2248 |
| 29 | 0.9252 | 0.6493 | +0.2760 | 0.6242 | 0.3947 | 0.2295 |
| **Mean** | **0.9248** | **0.6501** | **+0.2746** | **0.6219** | **0.3932** | **0.2287** |

Chance recall accuracy is 0.0078 (1/128). All three seeds clear the phase-0 mechanism gate:
the learned updater both beats the always-write control and assigns larger updates to observations
that will be useful later.

## Interpretation boundary

This is positive evidence for the narrow mechanism, not yet for human-like or semantic memory.
The present-time evidence feature is intentionally correlated with future usefulness, although its
distributions overlap. The experiment shows that delayed recall loss can train a differentiable
write gate to exploit such evidence and reduce interference. It does not show that a language model
can discover the relevant evidence from raw conversation.

The next experiment should remove the scalar evidence shortcut. It should derive write strength
from frozen text embeddings containing novelty, repetition, correction, and goal-relevance cues,
while retaining the same always-write control.

# Phase 1: competitive allocation

Date: 2026-09-12

Command, run with seeds 7, 17, and 29:

```bash
uv run python train_competition.py \
  --steps 200 --eval-batches 50 --batch-size 64 --seed <seed> --summary
```

All variants had the same 576-scalar fast-state capacity and unit total update budget per write.
The task used four contexts and eight shared keys per context. For every shared key, each context
assigned a different target value, creating deliberate overwrite pressure.

| Variant | Recall accuracy | Wrong-context intrusion | One-to-one assignment |
| --- | ---: | ---: | ---: |
| Single matrix | 0.2347 +/- 0.0012 | 0.7042 +/- 0.0036 | N/A |
| Uniform four blocks | 0.1956 +/- 0.0004 | 0.5868 +/- 0.0013 | 0.2500 |
| **Learned competition** | **0.9243 +/- 0.0004** | **0.0038 +/- 0.0002** | **0.9982 +/- 0.0006** |
| Oracle routing | 0.9190 +/- 0.0005 | 0.0039 +/- 0.0002 | 1.0000 |

Values are mean +/- population standard deviation over three seeds. Chance recall accuracy was
0.0156. Each seed took about 25 seconds on an Apple M4 Pro using MPS.

## What this establishes

With no block labels, delayed recall loss reliably taught the router to send four conflicting
contexts to four different fast-weight blocks. The allocation was almost exactly one-to-one and
removed nearly all wrong-context intrusions. Episode values were randomized, so the slow weights
could learn the allocation policy but could not memorize the recalled associations themselves.

This is a positive mechanism result, not a semantic-memory result. Context was supplied as an
explicit categorical input at both write and recall time. The experiment therefore does not yet
show that a model can infer latent context from raw text, decide whether an observation deserves
storage, allocate more contexts than available blocks, or forget over long horizons. The learned
variant slightly exceeding oracle accuracy is not evidence against the reference: models were
trained independently, and learned routing can retain tiny soft cross-block mixtures.

The next falsifiable step is to replace the context ID with frozen text embeddings and include
novelty, repetition, and correction episodes. A learned encoder/router must then recover useful
latent contexts without receiving context or write labels.

# Phase 2: semantic joint write and allocation

Date: 2026-09-12

Frozen encoder: `BAAI/bge-small-en-v1.5` (384 dimensions). The encoder was used only once to
precompute features and was never updated by the memory experiment. Training and evaluation used
disjoint natural-language paraphrases.

Final command, run with seeds 7, 17, and 29:

```bash
uv run python train_semantic.py \
  --embeddings artifacts/bge_small_en.pt \
  --steps 600 --eval-batches 50 --batch-size 64 --seed <seed> \
  --variants single_gate competitive_always joint oracle --summary
```

| Variant | Recall | Context intrusion | Distractor overwrite | Useful / distractor gate | Assignment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate only, single matrix | 0.2497 +/- 0.0000 | 0.7490 +/- 0.0001 | 0.0001 +/- 0.0000 | 0.122 / 0.010 | N/A |
| Allocation only, always write | 0.0341 +/- 0.0018 | 0.0652 +/- 0.0028 | 0.4289 +/- 0.0100 | 1.000 / 1.000 | 0.4765 +/- 0.0163 |
| **Joint learned memory** | **0.9757 +/- 0.0006** | **0.0010 +/- 0.0000** | **0.0030 +/- 0.0002** | **0.658 / 0.105** | **0.9988 +/- 0.0003** |
| Oracle write and routing | 0.9823 +/- 0.0003 | 0.0009 +/- 0.0000 | 0.0003 +/- 0.0000 | 1.000 / 0.000 | 1.0000 |

Values are mean +/- population standard deviation over three seeds. Chance recall was 0.0156.
All variants again had 576 fast-state scalars. The joint model reached 99.3% of oracle recall while
receiving neither write labels nor context IDs in its learned path.

## Ablation reading

The gate-only model correctly weakened temporary observations but suffered the expected 3/4
wrong-context intrusion because one matrix could not distinguish four colliding contexts. The
allocation-only model could not recover the stable value because the later temporary value was
always written. Only the joint model learned both necessary operations.

An earlier MiniLM run exposed a useful failure mode. It embedded the held-out sentence "one-off
occurrence rather than my usual choice" closer to durable preferences, apparently underweighting
the negation. The joint model then plateaued at 0.6528 recall with a 0.3210 overwrite rate. BGE-small
separated all held-out cue paraphrases and restored oracle-level recall. This isolates frozen input
representation as a real bottleneck rather than hiding it inside the memory update rule.

## Interpretation boundary and next test

This establishes a stronger synthetic phenomenon: frozen natural-language features can drive a
separate model-state memory to decide both update strength and memory region using only delayed
recall loss. It still uses four globally recurring semantic domains, short episodes, fixed
key/value vocabularies, and hand-authored durability contrasts. It does not yet demonstrate
open-world context creation, capacity eviction, correction chains, or integration with a frozen
generative LLM.

The next experiment should randomize which semantic domains co-occur per episode and provide more
domains than blocks. That forces online allocation, reuse, and eviction instead of allowing the
slow router to learn a permanent domain-to-block map.

# Phase 3: episode-local online allocation

Date: 2026-09-12

Eight global semantic domains were represented by frozen BGE-small embeddings. Each episode drew
a new random subset of four domains for four available blocks, randomized the order of first
arrival, and used held-out context phrasings at evaluation. The same local keys had conflicting
values across all four active contexts.

Final command, run with seeds 7, 17, and 29:

```bash
uv run python train_online.py \
  --steps 300 --eval-batches 50 --batch-size 64 --seed <seed> --summary
```

| Variant | Total runtime state | Recall | Context intrusion | Unique blocks used | Collision-free episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static semantic router | 576 | 0.6174 +/- 0.0186 | 0.2668 +/- 0.0106 | 0.7643 +/- 0.0139 | 0.1907 +/- 0.0239 |
| **Online prototype router** | **552** | **0.8809 +/- 0.0004** | **0.0054 +/- 0.0001** | **1.0000** | **1.0000** |
| Episode-local oracle | 484 | 0.8809 +/- 0.0004 | 0.0054 +/- 0.0001 | 1.0000 | 1.0000 |

Values are mean +/- population standard deviation over three seeds. The online router exactly
matched the oracle at every seed even though the order and identity of the four active domains
changed per episode. Its learned novelty threshold moved from 0.9000 to 0.9627 +/- 0.0027.

The capacity comparison includes the online model's routing metadata. It used four 11 x 11
association matrices (484 scalars), four 16-dimensional projected prototypes, and four counts
(68 routing scalars), totaling 552. The static baseline had four 12 x 12 matrices (576) and no
episode-local routing state.

## Interpretation boundary

This rules out the simplest permanent domain-to-block shortcut. It shows that a memory layer can
create an episode-specific permutation, reuse it consistently, and recover the right region at
query time without using the domain ID. However, the architectural prior still explicitly says
that a novel context should claim an empty block; delayed recall learns the novelty threshold, not
the whole allocation algorithm. Context features are intentionally constant within each episode,
and active contexts never exceed block count, so semantic drift and eviction remain untested.

The next experiment must allow five or six simultaneously active contexts with four blocks and
vary future query utility. Success then requires retaining useful contexts while evicting or
weakening lower-impact ones rather than merely filling empty slots.

# Phase 4: utility-aware eviction

Date: 2026-09-12

Each episode contained six semantic contexts competing for four blocks. Four randomly selected
contexts were queried later; usefulness was independent of arrival order. Training and evaluation
used disjoint frozen BGE cue phrasings. The learned model never consumed the retention mask; that
mask was reserved for query construction, metrics, and the oracle branch.

Final command, run with seeds 7, 17, and 29:

```bash
uv run python train_eviction.py \
  --steps 600 --eval-batches 50 --batch-size 64 --seed <seed> --summary
```

| Variant | Runtime state | Recall | Useful-context coverage | Useful blocks at end | Useful / temporary priority |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recency/least-used eviction | 552 | 0.5921 +/- 0.0015 | 0.6682 +/- 0.0015 | 0.6682 +/- 0.0015 | N/A |
| **Learned priority eviction** | **556** | **0.8926 +/- 0.0007** | **1.0000** | **1.0000** | **0.342 / 0.188** |
| Oracle priority eviction | 556 | 0.8927 +/- 0.0009 | 1.0000 | 1.0000 | 1.000 / 0.000 |

Values are mean +/- population standard deviation over three seeds. The learned model accepted
some temporary contexts while empty capacity remained, but later replaced them: temporary-context
acceptance was 0.7103 while final useful-block purity was 1.0000. This distinction matters because
successful forgetting is a state transition, not simply refusing every low-value input.

The priority scorer learned only a relative ordering. Its absolute outputs remained moderate, but
useful contexts ranked above temporary ones reliably enough to match oracle retention and recall.
Association state used four 11 x 11 matrices (484 scalars); four compressed prototypes, counts, and
priorities added 72, for 556 total runtime scalars.

## Interpretation boundary

This is positive evidence for joint write, online allocation, and eviction in a separate
model-state memory. It is still a controlled curriculum: usefulness is encoded by hand-authored
durable-versus-temporary sentences, every context arrives as one contiguous chunk, context
features do not drift within an episode, and only the priority comparison is learned. Real
conversation may provide ambiguous or delayed evidence, revisit evicted contexts, and mix
corrections with preference changes.

The next stress test should remove chunked arrivals, paraphrase the same context over time, and
delay utility evidence until after the first write. That requires an eligibility trace or a
retroactive consolidation update instead of a priority score available at initial allocation.

# Phase 5: delayed utility and eligibility traces

Date: 2026-09-12

All six context streams were fully interleaved and carried no usefulness cue during observation.
Six model-state trace slots temporarily held their associations. After observation ended, one
frozen natural-language outcome cue arrived for each context; only then did the memory consolidate
four traces and suppress two. Learned mode did not consume the retention mask.

Final command, run with seeds 7, 17, and 29:

```bash
uv run python train_delayed.py \
  --steps 600 --eval-batches 50 --batch-size 64 --seed <seed> --summary
```

| Consolidation | Transient state | Recall | Useful-trace coverage | Useful / temporary priority | Active traces after feedback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recency top-4 | 594 | 0.5784 +/- 0.0018 | 0.6656 +/- 0.0005 | 0.501 / 0.498 | 4.0 |
| **Learned delayed top-4** | **594** | **0.8630 +/- 0.0014** | **1.0000** | **0.971 / 0.909** | **4.0** |
| Oracle delayed top-4 | 594 | 0.8626 +/- 0.0017 | 1.0000 | 1.000 / 0.000 | 4.0 |

Values are mean +/- population standard deviation over three seeds. Learned consolidation matched
the oracle at all seeds. Its priority outputs were not calibrated probabilities: both classes had
high absolute scores, but the useful traces consistently ranked above the temporary traces. The
competitive top-k ordering, not a hand-set threshold, produced perfect retention selection.

The transient state comprises six 9 x 9 association matrices plus six 16-dimensional compressed
context prototypes, counts, and priorities: `6 * (81 + 16 + 2) = 594` scalars. After consolidation,
only four association matrices remain nonzero, although all six prototypes remain available to
route a query to either a retained matrix or a suppressed trace.

## Interpretation boundary

This shows retroactive consolidation in model state: associations can be written provisionally
before their utility is known, then strengthened or forgotten after delayed semantic evidence.
It does not yet solve unconstrained temporal credit assignment. The feedback event explicitly
contains the corresponding context representation, the trace has room for every candidate, and
there is no time decay. A real conversation may reveal utility indirectly through task success,
corrections, or later behavior without restating the original context.

The next experiment should remove the explicit context pointer from feedback. A global outcome
must then update the responsible recent traces through learned eligibility weights, which is the
first condition that genuinely tests temporal credit assignment rather than delayed ranking.

# Phase 6: global feedback on local Claude conversation logs

Date: 2026-09-12

The user authorized local use of the full `~/.claude` conversation history. No raw prompt text is
copied into this repository or persisted in the generated tensor artifact. The global history was
sessionized per project with a six-hour idle boundary. After filtering short/internal turns it
contained 75 projects, 495 conversations, and 16,622 user turns.

A weak delayed-revisit target was emitted when a later user turn uniquely reused specific lexical
anchors from one of eight candidates ending at least two turns earlier. This produced 1,349
examples. A stable project hash reserved entire projects for evaluation: 775 train and 574 eval.
Median target delay was five turns and the 90th percentile was eight turns.

Commands:

```bash
uv run python analyze_claude_logs.py --summary
HF_HUB_OFFLINE=1 uv run python prepare_claude_log_embeddings.py
uv run python train_global_feedback.py --steps 1000 --seed <seed> --summary
```

| Selector | Stored trace width | Top-1 | Top-2 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Random candidate | N/A | 0.1376 | 0.2485 | 0.3471 |
| Recency | N/A | 0.2178 | 0.3885 | 0.4299 |
| Untrained random projection | 64 | 0.7863 +/- 0.0109 | 0.8728 +/- 0.0079 | 0.8616 +/- 0.0073 |
| **Learned shared projection** | **64** | **0.8002 +/- 0.0036** | **0.8722 +/- 0.0030** | **0.8676 +/- 0.0022** |
| Full frozen BGE cosine | 384 | 0.8206 | 0.8990 | 0.8851 |

Values are means and population standard deviations over seeds 7, 17, and 29. The learned model
has 24,577 trainable parameters. It compresses each candidate from 384 to 64 scalars, so eight
retained traces require 512 rather than 3,072 scalars. Training improved top-1 by 1.39 percentage
points over the untrained compressed projection, and the learned trace remained 2.03 points below
full-dimensional cosine.

## Interpretation boundary

This is the first experiment in the series where a single later event must choose among multiple
old traces without receiving the responsible context or trace ID. Its large advantage over recency
shows that relevant past state can be activated reliably from natural, delayed conversation data,
and most of the frozen semantic geometry survives sixfold trace compression.

It is not yet evidence that the layer discovers what is intrinsically worth remembering. The weak
teacher defines relevance using lexical recurrence, full frozen cosine still wins, and training
adds only a modest gain over a random shared projection. The next experiment should label memory
utility using downstream events that are not semantic restatements: corrections, accepted versus
reverted actions, or later task success. That would separate delayed causal credit from ordinary
similarity retrieval.

# Phase 7: natural delayed outcome gating

Date: 2026-09-12

The full global history was re-sessionized while retaining short acknowledgements and corrections.
Across 557 conversations and 23,240 usable turns, high-precision weak rules found 2,940 explicit
outcomes: 1,677 accept/strengthen and 1,263 correct/revise. Stable project-level splitting yielded
1,721 train and 1,219 held-out-project evaluation examples. No raw text is stored in the feature
artifact; a frozen 512-dimensional signed character n-gram hash supports both Korean shorthand and
English feedback without downloading another encoder.

Commands, repeated with seeds 7, 17, and 29:

```bash
uv run python analyze_claude_outcomes.py --summary
uv run python prepare_claude_outcome_features.py
uv run python train_outcome_gate.py --steps 1000 --seed <seed> --summary
```

| Available evidence at decision time | Accuracy | Balanced accuracy | Accept recall | Correction recall |
| --- | ---: | ---: | ---: | ---: |
| Majority accept | 0.5874 | 0.5000 | 1.0000 | 0.0000 |
| Request at initial write only | 0.5562 +/- 0.0158 | 0.5429 +/- 0.0073 | 0.6192 | 0.4665 |
| Later feedback only | 0.9773 +/- 0.0004 | 0.9790 +/- 0.0003 | 0.9693 | 0.9887 |
| **Request trace + later feedback** | **0.9776 +/- 0.0010** | **0.9796 +/- 0.0009** | **0.9679** | **0.9914** |

The joint gate assigned mean retention 0.9652 to accepted traces and 0.0110 to correction traces.
The full training module has 66,310 parameters; its persistent per-interaction state is one
64-dimensional request trace plus one strength scalar. `apply_outcome()` uses the learned retention
as an actual model-state update rather than returning a diagnostic class alone.

## Interpretation boundary

This experiment answers the original storage bottleneck more directly: these conversations do not
contain enough evidence at initial observation to predict future utility reliably. Request-only
balanced accuracy was just 54.3%. Once delayed user feedback arrived, the same small gate made a
stable strengthen/revise decision at 98.0%. The useful architecture is therefore provisional write
followed by retroactive consolidation, not a supposedly omniscient write gate at input time.

The result is still imitation learning from explicit surface forms. The weak-label rules define the
classes, feedback usually belongs to the immediately preceding interaction, and a character hash
can expose those forms easily. A stronger next test must use implicit outcomes—whether a code change
survived, was reverted, passed tests, or was reused later—and jointly train trace selection plus the
strengthen/revise/forget action. That is the point where delayed utility becomes causal rather than
linguistic.

# Phase 8: implicit tool outcomes and file survival

Date: 2026-09-13

## Test and build outcomes

The project-session logs contained 917 high-confidence verification outcomes across 11 projects
and 33 sessions: 496 passes and 421 failures. Of these, 525 followed one or more Edit, Write,
apply-patch, or formatter mutations. Stable project-level splitting produced 835 train and 82 eval
events. Raw commands, code, and tool output were used only to build ignored 512-dimensional hashed
feature artifacts and were not persisted in git.

Five-seed results on all held-out-project events:

| Evidence | Accuracy | Balanced accuracy | Pass / fail strength |
| --- | ---: | ---: | ---: |
| Majority pass | 0.6707 | 0.5000 | 0.731 / 0.731 |
| Action and command before result | 0.6268 | 0.6370 | varies by seed |
| Test result only | 0.9829 | 0.9741 | 0.956 / 0.068 |
| **Action trace + test result** | **0.9927** | **0.9889** | **0.994 / 0.034** |

On the mutation-only subset (471 train / 54 eval), feedback-only balanced accuracy was 0.8761 and
joint accuracy was 0.8708. The action context did not add reliable causal attribution once the
dataset was restricted to real mutations. Test output is therefore a strong update signal, but this
experiment alone cannot identify which one of several changes caused the result.

## File-change survival

Claude file history contained 430 multi-version file chains from 35 sessions and 1,679 versions.
There were only 11 exact `A -> B -> A` restorations. Line-survival scoring expanded this to 672
high-confidence next-version outcomes: 637 retained and 35 reverted, with revert labels spread over
13 sessions. A stratified whole-session split yielded 566 train examples (23 reverted) and 106 eval
examples (12 reverted).

Commands, repeated with seeds 7, 17, 29, 41, and 53:

```bash
uv run python analyze_file_survival.py --summary
uv run python prepare_file_survival_features.py
uv run python train_survival_gate.py --steps 2000 --seed <seed> --summary
```

| Decision path | Balanced accuracy | Revert recall | Revert precision | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Change at observation only | 0.4798 | 0.0000 | 0.0000 | 0.520 |
| Future file delta only | 0.5567 | 0.1667 | 0.2857 | 0.561 |
| Generic concatenation gate | 0.5716 | 0.1667 | about 0.48 | 0.659 |
| Raw 512d cosine threshold | 0.6711 | 0.4167 | 0.4167 | 0.8954 |
| **Learned 64d relational gate** | **0.8340 +/- 0.0021** | **1.0000** | **0.2778** | **0.9021 +/- 0.0012** |
| Relational gate, train-F1 threshold | 0.6090 | 0.2500 | 0.5000 | 0.9021 |

The relational gate has a fixed 512-to-64 projection and only six trainable decision parameters.
Its persistent state is a 64-dimensional change trace plus one strength scalar. Freezing the
projection removed seed collapse. The large gap between request-only, outcome-only, and relational
comparison is positive evidence that the delayed state relation—not a surface outcome token—is the
useful signal.

## Interpretation boundary

This is the strongest phenomenon result so far: an old model-state trace is necessary to interpret
a later external event, and the learned policy ranks all 12 held-out reverts above most retained
changes. It is nevertheless not ready to delete memories. At the high-recall threshold it flags too
many retained changes, while a precision-oriented threshold misses most reverts. Only 35 negative
examples and 13 negative-bearing sessions exist, and next-version line survival is a proxy rather
than end-task utility.

The next investment should collect richer causal episodes prospectively: record each candidate
memory/action ID, test outcome, subsequent fix, final accepted diff, and later reuse. Training can
then assign a global outcome across multiple traces and distinguish revise from permanent forget
without relying on file adjacency heuristics.

# Phase 9: prospective causal logging

Date: 2026-09-13

An observation-only project hook is now installed for future Claude Code sessions. `PreToolUse` and
the two post-tool events join candidate mutations and verifications by `tool_use_id`; `Stop` records
the turn-final workspace fingerprint; the following `UserPromptSubmit` can attach explicit
accept/revise evidence to the previous turn's candidate set. A failed verification also becomes the
parent of the next workspace-changing action, preserving a direct repair edge.

The replay builder emits multi-candidate `strengthen`/`revise` examples from tests and feedback. If
a later post-action or turn snapshot exactly equals a candidate's pre-action workspace digest, it
also emits a single-candidate `forget` transition at that later time. This prevents permanent
forgetting from being conflated with the earlier, recoverable request to revise an action.

Privacy reduction happens before persistence: only keyed opaque IDs and content digests, file
sizes, and keyed quantized 256-dimensional signed character n-gram sketches reach disk. These
sketches remain trainable signals rather than a cryptographic anonymization guarantee, but the raw
hook payload is never logged. The event stream, salt, and mutable join ledger are ignored by git
and created with user-only permissions. Unit tests use embedded secret markers and private
filenames to verify neither appears in the JSONL output.

Current status: instrumentation only, with no prospective scientific metric claimed. The full suite has 70
passing tests. The next decision gate is empirical: wait for multi-session prospective data, report
label counts and multi-action coverage, then train a trace selector/action head with whole-session
holdout. Until then, the Phase 8 relational result remains the last model result.

## Retrospective repair-chain bootstrap: negative control

Because the prospective stream initially contained zero events, a separate bootstrap extractor
searched the authorized historical transcripts for verification-failure, mutation, and subsequent
same-family verification-pass chains. It found 119 episodes across 17 sessions and six projects;
all 119 had more than two candidate actions. The 930 candidate labels comprised 220 ignore, 237
strengthen, and 473 revise examples. Pytest dominated with 92 episodes, followed by 16 JavaScript
tests, eight builds, and three lint runs.

Features were frozen 512-dimensional signed character n-gram hashes; raw commands, code, and tool
output were not stored in the artifact. Candidate order was deterministically shuffled. Evaluation
used six-fold leave-one-project-out predictions, covering every episode out of project. Each result
below is the mean and population standard deviation over seeds 7, 17, 29, 41, and 53 with 600
training steps per fold.

| Architecture / evidence | Balanced accuracy |
| --- | ---: |
| Majority revise | 0.3333 |
| Coupled head, candidate only | 0.3519 +/- 0.0059 |
| Coupled head, outcome only | 0.3337 |
| **Coupled head, candidate + outcome** | **0.3650 +/- 0.0062** |
| Coupled joint, shuffled outcomes | 0.3552 +/- 0.0045 |
| Decoupled activation/action, candidate only | 0.3578 +/- 0.0049 |
| Decoupled activation/action, candidate + outcome | 0.3632 +/- 0.0078 |
| Decoupled joint, shuffled outcomes | 0.3489 +/- 0.0045 |

The small joint gain is real enough to be shuffle-sensitive but not useful. Coupled joint accuracy
averaged about 0.413 versus 0.509 for the majority class, its activation recall fell relative to
candidate-only, and no coupled seed classified every action in any held-out episode correctly.
Separating activation from strengthen/revise did not rescue the task.

This is a negative control, not a failed prospective experiment. Temporal position created the
teacher labels, so mutations on either side of a test can be unrelated to its outcome, and several
mutations share the same bag-level result. The outcome contains weak relational information, but
historical transcripts cannot tell which edit actually caused the fix. Further architecture tuning
on these labels would optimize teacher noise. The next valid investment remains accumulation of
prospective action IDs, repair edges, accepted final states, and exact later reverts.

## Claude + Codex history expansion

The user explicitly authorized training use of both their local Claude and Codex conversation
histories. A Codex rollout adapter added structured `function_call` and `custom_tool_call` events,
including `exec` wrappers and direct `apply_patch` calls. Working directories from session metadata
canonicalize project identity across the two log formats.

The combined extractor found 370 episodes across 61 sessions and 18 projects: 119 from Claude and
251 from Codex. The 2,871 candidate labels comprised 637 ignore, 796 strengthen, and 1,438 revise.
There were 293 pytest, 28 JavaScript-test, 35 lint, eight build, and six type-check episodes. Of the
370 episodes, 365 had more than two candidates.

The feature preparation and evaluation protocol remained unchanged: no raw text in the artifact,
deterministically shuffled candidate slots, 512-dimensional character hashes, 64-dimensional
persistent traces, and leave-one-project-out evaluation over every episode. Five seeds used 600
steps per fold.

| Architecture / evidence | Balanced accuracy |
| --- | ---: |
| Majority revise | 0.3333 |
| Coupled, candidate only | 0.3553 +/- 0.0028 |
| **Coupled, candidate + outcome** | **0.3748 +/- 0.0046** |
| Coupled joint, shuffled outcomes | 0.3630 +/- 0.0085 |
| Decoupled, candidate only | 0.3635 +/- 0.0037 |
| **Decoupled, candidate + outcome** | **0.3757 +/- 0.0072** |
| Decoupled joint, shuffled outcomes | 0.3606 +/- 0.0088 |

The larger corpus changes the confidence more than the absolute ceiling. Joint evidence beat the
matching candidate-only control in both architectures, and shuffling outcomes removed most of the
gain. In the decoupled model, mean selected-action accuracy rose from about 0.263 to 0.394 and
activation recall from 0.454 to 0.669 when the outcome was added. This is clearer evidence that a
small memory layer can use a later global event to alter old action traces.

It is still below a deployment gate. Coupled joint accuracy averaged about 0.409 versus 0.501 for
the majority class, and exact whole-episode accuracy remained below 1%. Temporal repair labels
remain ambiguous about which mutation caused a pass. The combined result supports the existence of
the delayed relational phenomenon, not autonomous memory deletion or trustworthy causal credit.

# Phase 10: public Open-SWE trajectory scaling

Date: 2026-09-13

NVIDIA Open-SWE-Traces provides more than 200,000 public software-agent trajectories, including
67,153 trajectories in the selected v1.0 SWE-agent split. A deterministic downloader sampled 5,000
rows that were resolved by the hidden task tests, had no Dataset Viewer truncation, and declared one
of MIT, Apache-2.0, BSD-2-Clause, or BSD-3-Clause. The ignored local JSONL is 1.2 GB; its manifest
records the selection seed, per-license counts, and SHA-256 digest.

The public adapter deliberately uses a stricter teacher than final success alone. It emits an
episode only when a trajectory contains an explicit failing verification, one or more intervening
editor mutations, a later passing verification from the same test family, and final hidden-test
resolution. This produced 611 episodes from 471 trajectories and 254 repositories, with 3,784
labels: 645 ignore, 997 strengthen, and 2,142 revise. All 611 episodes contain multiple actions.

For a fair comparison across the much larger repository vocabulary, every condition below uses the
same balanced five-fold grouped-project split. Values are mean and population standard deviation
over seeds 7, 17, 27, 37, and 47 with the decoupled 64-dimensional model and 600 steps per fold.

| Corpus | Episodes / labels / projects | Candidate-only balanced accuracy | Candidate + outcome | Shuffled outcome |
| --- | ---: | ---: | ---: | ---: |
| Local Claude + Codex | 370 / 2,871 / 18 | 0.3621 +/- 0.0044 | **0.3692 +/- 0.0125** | 0.3523 +/- 0.0053 |
| Public Open-SWE | 611 / 3,784 / 254 | 0.3647 +/- 0.0035 | **0.3672 +/- 0.0043** | 0.3596 +/- 0.0050 |
| Combined | 981 / 6,655 / 272 | 0.3613 +/- 0.0060 | **0.3693 +/- 0.0087** | 0.3624 +/- 0.0097 |

Public-only joint evidence beat shuffled outcomes by 0.0077 balanced-accuracy points on average and
did so in four of five seeds. The combined model also beat its shuffle control in four of five
seeds. However, public data did not raise absolute joint balanced accuracy: local-only, public-only,
and combined all remained near 0.37. Adding 611 stricter public episodes increased sample size and
repository diversity without breaking the ceiling.

This rejects the simplest quantity-only explanation. Public test logs repeat generic failure/pass
language, several mutations still share one outcome, and frozen character hashes preserve lexical
form better than code-level causal semantics. The next public-data investment should target
candidate-level utility annotations or use a stronger frozen code/text representation; merely
adding more final-success trajectories is unlikely to resolve action credit.

# Phase 11: ContextBench human-gold activation

Date: 2026-09-13

ContextBench contains 1,136 issue-resolution tasks and human-verified gold code spans. Each task's
natural problem statement became the activation context; its gold spans became positive candidate
traces. Gold spans belonging to other tasks in the same repository became hard negatives. Splitting
by repository prevents the model from memorizing repository identity, and shuffling queries only
within each held-out repository provides a conservative relation control.

Tasks without another annotated task in their repository cannot supply same-project negatives and
were excluded. The resulting artifact contains 3,177 multi-trace episodes, 22,051 candidate labels,
11,029 positives, and 57 repository groups. The learned layer stores each candidate as a
64-dimensional trace. The BGE version has 65,923 trainable memory-layer parameters; the frozen BGE
encoder receives no gradient.

Five-fold grouped-project evaluation used seeds 7, 17, 27, 37, and 47 and 600 steps per fold.

| Frozen representation / evidence | Balanced accuracy | Top-1 gold activation |
| --- | ---: | ---: |
| Character hash, candidate + query | 0.5113 +/- 0.0039 | 0.4697 +/- 0.0085 |
| Character hash, same-project shuffled query | 0.5064 +/- 0.0034 | 0.4609 +/- 0.0079 |
| BGE, candidate only | 0.4975 +/- 0.0014 | 0.4311 |
| BGE, query only | approximately 0.5000 | 0.5024 |
| **BGE, candidate + natural query** | **0.6046 +/- 0.0024** | **0.6624 +/- 0.0081** |
| BGE, same-project shuffled query | 0.5658 +/- 0.0021 | 0.5746 +/- 0.0084 |

The BGE joint model beats the same-repository shuffled-query control by 3.88 balanced-accuracy
points and 8.78 top-1 points. Candidate-only and query-only controls remain at chance, so neither
static memorability nor query priors explain the result. The shuffled control stays above chance
because another issue from the same repository often shares APIs and subsystem vocabulary; the
additional gain from the correct issue is therefore the task-specific associative signal.

This is stronger evidence for the original recall hypothesis than the repair-credit experiments:
a natural context, without an explicit retrieval instruction, selects human-annotated relevant
traces stored in a small independent memory layer. It is not yet a complete memory system. Candidate
traces are supplied offline, exact all-candidate accuracy is only about 7.7%, and this phase does not
learn write, revise, or forget. The next architectural step is a shared trace projection with the
ContextBench activation head and the Open-SWE delayed action head trained as separate objectives.

## Frozen Gemma representation follow-up

The feature pipeline now also accepts `--encoder gemma`. It extracts normalized final-token hidden
states or masked mean-pooled states from the pretrained `google/gemma-3-270m` backbone. The model is
frozen and the grouped-project protocol is unchanged. Float16 inference on MPS produced NaNs, so the
final extractor uses float32 and fails explicitly on non-finite features.

Five-seed results show that pooling is decisive:

| Frozen representation / training budget | Balanced accuracy | Shuffled query | Top-1 | Shuffled Top-1 |
| --- | ---: | ---: | ---: | ---: |
| Gemma last token, 64d / 600 steps | 0.5015 +/- 0.0018 | 0.4963 +/- 0.0057 | 0.4806 +/- 0.0055 | 0.4728 +/- 0.0051 |
| Gemma masked mean, 64d / 600 steps | 0.5681 +/- 0.0048 | 0.5427 +/- 0.0044 | 0.5761 +/- 0.0034 | 0.5126 +/- 0.0023 |
| **Gemma masked mean, 128d / 2,000 steps** | **0.5862 +/- 0.0044** | **0.5550 +/- 0.0036** | **0.6154 +/- 0.0076** | **0.5408 +/- 0.0103** |
| BGE, 128d / 2,000 steps | 0.6140 +/- 0.0028 | 0.5710 +/- 0.0020 | 0.6805 +/- 0.0063 | 0.5850 +/- 0.0052 |

The best Gemma condition beats its same-repository shuffled-query control by 3.12 balanced-accuracy
points and 7.46 top-1 points, so a frozen small generative model does carry usable task-conditioned
association signal. It remains weaker than the retrieval-trained BGE encoder under the same memory
head budget. Raw cosine explains part of the gap: BGE positive/negative similarity is 0.7258/0.6603
with 0.8562 top-1, while Gemma mean pooling is 0.7604/0.7484 with 0.5458 top-1. Gemma's frozen space
is more anisotropic and less retrieval-separated; the learned memory head improves it but does not
fully recover the BGE margin.

## Gemma ablations and lightweight adaptation

The follow-up kept the ContextBench split and memory interface fixed while testing inexpensive
ways to improve Gemma. A five-seed alignment-loss result is directly comparable to the 128d / 2,000
step row above; the other rows are screening runs used to reject branches before a full sweep.

| Change | Evaluation | Balanced accuracy | Shuffled query | Top-1 | Shuffled Top-1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Final mean + cosine alignment weight 1 | 5 seeds, full 5-fold | **0.5946 +/- 0.0024** | 0.5597 +/- 0.0028 | **0.6270 +/- 0.0046** | 0.5504 +/- 0.0042 |
| Echo second-copy mean + alignment 1 | 5 seeds, full 5-fold | 0.5938 +/- 0.0010 | 0.5601 +/- 0.0010 | 0.6255 +/- 0.0040 | 0.5486 +/- 0.0062 |
| Layer 15 mean + centering | seed 7, full 5-fold | 0.5824 | 0.5492 | 0.5971 | 0.5307 |
| BGE + Gemma normalized concatenation | seed 7, full 5-fold | 0.6081 | 0.5666 | 0.6654 | 0.5681 |
| Dual-encoder memory head | seed 7, full 5-fold | 0.5725 | 0.5463 | 0.5817 | 0.5203 |

Alignment is the only branch adopted: versus the matching unaligned five-seed Gemma result it adds
0.0084 balanced-accuracy and 0.0116 top-1 while retaining a clear correct-query advantage. Echo
embeddings, feature centering, a dual head, larger capacity, extra steps, intermediate layers, and
BGE/Gemma concatenation did not beat their relevant simple baseline. Intermediate layer 15 had
better raw cosine top-1 (0.6396) than the final layer (0.5458), but its trained head still lost to
the final-layer head. This is a useful warning that raw nearest-neighbor quality and learnable
activation quality are not interchangeable.

The echo implementation follows the second-occurrence pooling idea in
[Echo Embeddings](https://arxiv.org/abs/2402.15449). It does not implement the bidirectional
attention, masked-next-token adaptation, or contrastive stages of
[LLM2Vec](https://arxiv.org/abs/2404.05961), so its negative result only rejects the cheap
repeat-and-pool variant.

A final pilot inserted rank-4 LoRA adapters into `q_proj` and `v_proj` of Gemma's last four layers:
40,960 trainable backbone parameters. The independent 230,147-parameter memory layer was initialized
on train-project frozen features and then held fixed. Because full raw-text re-encoding is slow on
the local MPS machine, this screen used the first 200 episodes of each held-out fold rather than a
publishable full cross-validation.

| Held-out fold / condition | Balanced accuracy | Shuffled query | Top-1 | Shuffled Top-1 |
| --- | ---: | ---: | ---: | ---: |
| Fold 0 frozen | 0.5233 | 0.5092 | 0.5600 | 0.5200 |
| Fold 0 LoRA, 20 steps, 2e-4 | 0.5360 | 0.5181 | 0.5650 | 0.5200 |
| Fold 0 LoRA, 50 steps, 2e-4 | **0.5377** | 0.5184 | 0.5600 | 0.5050 |
| Fold 0 LoRA, 50 steps, 5e-5 | 0.5234 | 0.5048 | 0.5600 | 0.5150 |
| Fold 0 LoRA, 50 steps, 5e-4 | 0.5318 | 0.5125 | **0.5750** | 0.5200 |
| Fold 1 frozen | 0.6334 | 0.6018 | **0.7450** | 0.6700 |
| Fold 1 LoRA, 50 steps, 2e-4 | 0.6375 | 0.6098 | 0.7350 | 0.6650 |

LoRA therefore has a real but weak screening signal: balanced accuracy improved on both tested
folds, but top-1 declined on the easier fold and the query-control gap did not improve consistently.
Training all candidates from one episode together also failed to improve balanced accuracy on fold
0 (0.5234, top-1 0.5650). This branch is not adopted yet. A worthwhile next run would save adapters,
evaluate all held-out episodes, and repeat every fold/seed; simply scaling adapter steps is not
justified by the present evidence.

# Phase 12: capacity-constrained write then recall

Date: 2026-09-13

This phase turns trace creation into a model action. Each ContextBench issue is split into an early
write context (the first non-empty line, or the first half of a one-line issue) and a later recall
context (the remaining body). Across the 3,177 episodes, the early context averages 20.1 words and
the later context 151.3 words; none are identical. Frozen Gemma produces separate embeddings for
the two contexts.

`SelectiveWriteRecallMemory` has 377,730 trainable parameters. It scores each candidate using only
the early context, executes hard top-k selection, and persists selected candidates as 128d model
traces. A separate head later sees the trace state plus the later context. No key/value graph or
external retrieval call is involved.

Pure end-to-end delayed credit failed. With 50% capacity and only final activation loss, write
selection balanced accuracy was 0.5054 and shuffling the write context changed almost nothing. A
simultaneous weak-teacher auxiliary loss also remained poor: weight 0.1 produced 0.5097 write
selection and weight 1 produced 0.5238. The usable curriculum was staged imitation: pretrain the
write gate for 1,000 steps from human-gold utility, freeze it, then train the recall head for 1,000
steps from final activation loss.

At 50% capacity the staged model achieved 0.5880 write selection and 0.5844 final activation, but
shuffling the later recall context did not hurt. The gate had enough information and exactly enough
slots to solve most of the task at write time. Raising capacity to 75% deliberately retained some
distractors and made later recall necessary.

Five-fold grouped-project results below are mean and population standard deviation over seeds 7,
17, 27, 37, and 47. Shuffle controls replace only one context with another issue from the same
held-out repository.

| 75% capacity condition | Balanced accuracy | Top-1 |
| --- | ---: | ---: |
| **Correct early write + later recall** | **0.5917 +/- 0.0064** | **0.5882 +/- 0.0041** |
| Same-project shuffled write context | 0.5757 +/- 0.0056 | 0.5803 +/- 0.0075 |
| Same-project shuffled recall context | 0.5809 +/- 0.0052 | 0.5642 +/- 0.0072 |
| Hard write selection itself | 0.5871 +/- 0.0029 | 0.5592 +/- 0.0046 |

Correct context beat shuffled write by 0.0160 +/- 0.0013 balanced-accuracy and shuffled recall by
0.0108 +/- 0.0017. Both gaps were positive in every seed. This is the first natural-data condition
in this project where the learned layer makes a capacity-limited write action and a separate later
context materially changes recall. It nearly matches the full-issue aligned Gemma activation score
(0.5946) despite separating the information across two hooks.

The boundary matters: this does not show that the model discovers intrinsic importance without a
teacher. It shows that the proposed roadmap works—weak-label write imitation followed by delayed
recall learning—and that pure end-to-end credit is currently insufficient. The temporal split is
also constructed from one issue rather than two independently occurring events. The next stronger
test needs real multi-event conversations where an early observation is reused or corrected later.

# Phase 13: real conversation write-before-revisit

Date: 2026-09-13

This phase uses the existing privacy-local Phase 6 corpus rather than splitting one event. Each
example contains eight actual earlier user turns and one natural user turn occurring at least two
turns later. A weak teacher identifies the earlier turn whose lexical anchors are uniquely reused;
the write gate sees only each candidate turn and must make a hard capacity decision before the
later turn is available. The later turn then recalls among surviving 64d traces. The artifact has
775 train and 574 held-out-project examples, and no raw conversation text.

The model has 106,946 parameters. Training follows the successful curriculum from Phase 12:
future reuse weakly supervises the write gate, the gate is frozen, and final top-1 loss trains the
recall head only on episodes where the target survived. At inference there is no explicit memory
instruction, trace ID, future query, or teacher label.

Five-seed results use seeds 7, 17, 27, 37, and 47:

| Stored capacity | Target retained | Write balanced accuracy | Top-1 recall | Shuffled recall Top-1 |
| --- | ---: | ---: | ---: | ---: |
| 6 / 8 traces (75%) | 0.8923 +/- 0.0085 | 0.5813 +/- 0.0049 | **0.6725 +/- 0.0044** | 0.3491 +/- 0.0039 |
| 2 / 8 traces (25%) | **0.5906 +/- 0.0109** | **0.6946 +/- 0.0062** | 0.5627 +/- 0.0147 | 0.3369 +/- 0.0045 |

At 25% capacity the writer retained the future target 59.1% of the time versus a 25% random
capacity expectation. Conditional on the target surviving, recall top-1 was approximately
0.9527 +/- 0.0086; most remaining failures therefore came from write selection rather than later
retrieval. Correct later context beat shuffled context by 0.2258 +/- 0.0131 top-1 in every seed.
A seed-7 midpoint kept four traces, retained 0.7631 of targets, and reached 0.6568 top-1.

Controls reject a random-capacity explanation. At 25% capacity an untrained writer retained 0.2125
and reached 0.1794 top-1. Training the writer on uniformly random target positions raised those only
to 0.3293 and 0.3101, while the real future-reuse teacher produced 0.5976 and 0.5749 for the same
seed. The earlier unlimited BGE selector remains stronger at 0.8002 top-1, quantifying the cost of
discarding six of eight turns before the query exists.

This is the clearest POC so far for the requested model layer: it decides what to store from input,
persists a compact bounded state, and later emits associated traces from an ordinary conversational
context. It is still imitation rather than autonomous importance discovery. The teacher defines
utility through lexical recurrence, and the corpus contains only Claude history rather than the
newer Codex sessions. The next data step is a cross-format revisit extractor and non-lexical utility
labels from corrections, accepted actions, and downstream task success.

## Real repair update-action negative branch

A separate experiment encoded 371 genuine Claude/Codex repair chains with Gemma: 2,879 mutation
traces across 62 sessions and 18 projects. The writer saw a mutation before the combined later
failure/pass outcome; the after-model predicted `ignore`, `strengthen`, or `revise`. Binary active
recall stayed near chance (0.5079 balanced accuracy versus 0.5041 with shuffled outcomes). Coupled
and decoupled three-action variants also lacked a stable shuffle-sensitive gain across seeds.

The failure is informative rather than a model-size result. Repair labels are constructed from
whether a mutation occurs before or after the failing verification, but the encoded candidate omits
that observable temporal boundary. Without it, visually similar edit payloads can receive opposite
actions. Further tuning this branch is not justified until candidate features include the state
available at the actual write moment.

# Phase 14: objective, causal-signal, and public-data ablations

Date: 2026-09-13

This phase turns the Phase 13 result into a falsifiable comparison matrix. Five seeds use the same
775/574 held-out-project split, two-of-eight capacity, 1,000 write steps, and 1,000 recall steps.
Reported dispersions are sample standard deviations. Position-macro metrics average the eight target
positions equally, so the corpus's recency skew cannot inflate them. The first objective sweep ran
on MPS; a complete CPU repeat and causal-component dissection follows it.

| Claude condition | Target retained | Position-macro retained | Top-1 | Position-macro Top-1 | Query-shuffle gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Staged BCE | 0.5906 +/- 0.0121 | 0.6064 +/- 0.0112 | 0.5627 +/- 0.0164 | 0.5800 +/- 0.0162 | 0.2258 +/- 0.0146 |
| Listwise write loss | 0.5812 +/- 0.0135 | - | 0.5502 +/- 0.0187 | - | 0.2094 +/- 0.0132 |
| **Position + novelty + surprise, BCE** | **0.6279 +/- 0.0097** | **0.6246 +/- 0.0102** | **0.5882 +/- 0.0085** | **0.5897 +/- 0.0086** | **0.2331 +/- 0.0063** |
| Causal signals + listwise | 0.6028 +/- 0.0115 | - | 0.5589 +/- 0.0149 | - | 0.1920 +/- 0.0098 |
| Causal listwise + 500 joint steps | 0.6063 +/- 0.0084 | - | 0.5690 +/- 0.0120 | - | 0.2195 +/- 0.0225 |

| CPU causal dissection | Target retained | Position-macro retained | Top-1 | Position-macro Top-1 | Query-shuffle gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Staged BCE | 0.5920 +/- 0.0110 | 0.6086 +/- 0.0111 | 0.5686 +/- 0.0205 | 0.5861 +/- 0.0214 | 0.2254 +/- 0.0077 |
| Causal position only | 0.6077 +/- 0.0141 | 0.6073 +/- 0.0117 | 0.5742 +/- 0.0115 | 0.5771 +/- 0.0120 | 0.2310 +/- 0.0077 |
| Causal surprise only | 0.6014 +/- 0.0221 | 0.6173 +/- 0.0216 | 0.5774 +/- 0.0128 | 0.5934 +/- 0.0127 | 0.2380 +/- 0.0062 |
| **Position + novelty + surprise, BCE** | **0.6195 +/- 0.0101** | **0.6203 +/- 0.0104** | **0.5878 +/- 0.0075** | 0.5894 +/- 0.0082 | 0.2328 +/- 0.0036 |

The adopted causal BCE layer has 107,522 parameters. Versus staged BCE, target retention improves
by 0.0373 on MPS and 0.0275 on CPU; top-1 improves by 0.0254 and 0.0192. Position-macro retention
also rises on both backends, rejecting an explanation based only on the non-uniform target-position
distribution. Direct heuristics are much weaker: at 25% capacity recency retains 0.3885, causal
novelty 0.3571, and causal surprise 0.3275.
The neural gain therefore requires the learned interaction between frozen semantic features and the
causal signals. Listwise supervision and the tested joint schedule are clean negative results.
Using causal surprise directly as a teacher-free write rule retains 0.3275 and reaches 0.3223 BGE
top-1 on Claude; on LongMemEval both are 0.2327, below random capacity. Surprise is therefore a
useful auxiliary signal after weak supervision, but it does not yet solve what-to-store by itself.

The public validation adapts the official cleaned LongMemEval-S histories. Each non-abstention
evidence session becomes a positive write target mixed with seven deterministic non-evidence
sessions. The question remains hidden until recall, target position is deterministically randomized,
and complete question IDs are held out. This yields 688 training and 202 evaluation episodes from
890 evidence targets across information extraction, preference, multi-session, temporal reasoning,
and knowledge-update questions. This is not the official LongMemEval QA protocol or score.

| LongMemEval-S adapted condition | Target retained | Position-macro retained | Top-1 | Position-macro Top-1 | Query-shuffle gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random capacity expectation | 0.2500 | 0.2500 | - | - | - |
| Recency / novelty / surprise heuristics | 0.2277 / 0.2327 / 0.2327 | 0.2500 / 0.2120 / 0.2146 | - | - | - |
| **Staged BCE, five seeds** | **0.4030 +/- 0.0263** | **0.4058 +/- 0.0285** | 0.3198 +/- 0.0254 | 0.3227 +/- 0.0270 | **0.0455 +/- 0.0187** |
| Causal BCE, five seeds | 0.4069 +/- 0.0137 | 0.4104 +/- 0.0165 | **0.3287 +/- 0.0090** | **0.3336 +/- 0.0085** | **0.0614 +/- 0.0103** |
| Untrained writer, seed 7 | 0.2723 | - | 0.1683 | - | 0.0149 |
| Random write teacher, seed 7 | 0.2673 | - | 0.1782 | - | 0.0099 |
| Query-visible full BGE cosine | 0.9406 at two slots | - | 0.8465 | - | - |

All five staged-BCE seeds have a positive correct-query advantage. The external retention result is
substantially above random and survives position macro-averaging; untrained and random-teacher
controls remain near capacity chance. The result is weaker than on the private natural corpus, and
causal features stabilize more than they improve retention. This is useful evidence against a
single-corpus artifact, but not proof of autonomous importance: oracle evidence IDs remain weak
training teachers and the benchmark sessions are LLM-simulated and human-edited.

**Correction, 2026-09-14.** The heuristic controls above omit candidate length. Phase 15 shows that
keeping the two longest sessions retains the LongMemEval evidence in 0.4159 of the same held-out
episodes, which is above this phase's learned 0.4069. The correct reference for the public
validation is therefore 0.4159, not the 0.25 random-capacity expectation, and the LongMemEval
retention claim does not survive it. The private Claude corpus is unaffected: its longest-two
baseline is 0.3868 against a learned 0.6195.

The literature-directed next branch is `FastWeightRecallMemory`. Following the parametric-state view
of TTT and Titans, it predicts a candidate value from the current associative matrix, exposes the
online prediction error as surprise, and delta-updates the matrix only for hard-selected candidates.
At memory width 12, its 12-by-12 matrix uses 144 persistent scalars, close to the two selected 64d
traces' 128 scalars. It has only 14,463 trainable parameters. Across five Claude seeds it retains
0.6066 +/- 0.0250, reaches 0.5575 +/- 0.0211 top-1, and has a 0.2178 +/- 0.0121 shuffle gap. The
matrix therefore learns genuine query-dependent association at roughly equal state budget and with
far fewer parameters, but it does not beat the trace model's recall. On LongMemEval seed 7 it reaches
0.3911 retention and 0.2525 top-1, also below the trace layer's 0.3960 and 0.3218. It remains an
architecturally interesting efficiency branch, not the adopted accuracy model.

Increasing the fast-weight width from 12 to 64 raises persistent state from 144 to 4,096 scalars and
parameters from 14,463 to 90,435, but seed-7 retention/top-1 fall from 0.6307/0.5732 to
0.6063/0.5662. More matrix capacity is therefore not a justified next step without a better update
objective or retention rule.

An input-dependent retention/forget gate was then trained through recall loss at width 12. Its seed-7
retention was 0.6115, but top-1 and the shuffle gap fell to 0.5557 and 0.1951 versus 0.5732 and
0.2213 for fixed learned decay. This adaptive-forgetting formulation is rejected at screening stage;
the negative result suggests that 775 training episodes do not identify a richer retention policy.

# Phase 15: frozen generator utility and the missing length control

Date: 2026-09-14

## Does the written memory change what a frozen generator predicts?

Every earlier phase scored the memory layer against its own activation metrics. This phase asks a
task-level question: does the layer's capacity decision change a frozen generator's mean negative
log-likelihood of the human-written gold answer? The generator is never trained, never sees a label,
and is used only as an instrument.

The instrument is gated before it is trusted. `instrument_usable` is false unless the oracle prompts
survive the context window intact, beat the no-memory reference, and beat a random selection of the
same size. Every capacity-limited condition holds exactly two slots, including the oracle, so prompt
length cannot explain a difference between them.

Three configurations failed that gate and none of them says anything about the writer. Frozen
`google/gemma-3-270m` is a base model and ranked random above oracle. `HuggingFaceTB/SmolLM2-135M-Instruct`
recovered the oracle-over-random ordering but never beat the no-memory reference. Most importantly, an
early 1,200-character session cap failed the gate by deleting the evidence from the oracle prompt
itself: LongMemEval-S sessions average 10,870 characters and evidence sessions average 13,979, so the
cap retained about a tenth of the text the oracle condition exists to supply. Conditions now report
`truncated_prompt_rate`, and an intact oracle prompt is part of the gate.

`Qwen/Qwen2.5-1.5B-Instruct` in bfloat16 passes. Results below use seed 7, the adopted causal-BCE
writer at two-of-eight capacity, and the first 72 held-out LongMemEval-S episodes that carry a gold
answer.

> **Correction, Phase 33.** The gate that passed here is a comparison of means over a heavy-tailed
> quantity, read on a 72-episode prefix. Re-measured on the full 202 held-out episodes it fails, and
> even on these 72 the oracle is better than random on exactly 50.0% of episodes (paired t = 1.74).
> The instrument never had a per-episode preference for the evidence session. The task-level claim
> below is withdrawn; see Phase 33.

| Condition | Answer NLL | Gain over no memory | Evidence present | Prompt truncated |
| --- | ---: | ---: | ---: | ---: |
| **Oracle two slots** | **2.3211** | **+0.5574** | 1.000 | 0.00 |
| Length-matched control | 2.4632 | +0.4153 | 0.431 | 0.00 |
| Random two slots | 2.5467 | +0.3318 | 0.236 | 0.00 |
| Learned writer | 2.6099 | +0.2686 | 0.403 | 0.00 |
| All eight traces | 2.6577 | +0.2208 | 1.000 | 1.00 |
| Recency two slots | 2.7182 | +0.1603 | 0.194 | 0.00 |
| No memory | 2.8785 | 0.0000 | 0.000 | 0.00 |

Injecting all eight sessions is worse than any two-slot condition because it overflows the context
window in every episode. Capacity-limited selection is therefore not only cheaper than supplying the
whole history, it is more accurate, which is the premise the earlier phases assumed rather than
measured.

The aggregate learned row is not the informative number. Splitting the same 72 episodes by whether
the writer actually kept the evidence session, and reading every condition on each subset, separates
writer quality from episode difficulty:

| Subset | Episodes | Learned | No memory | Oracle |
| --- | ---: | ---: | ---: | ---: |
| Writer kept the evidence | 29 | **2.1508** | 2.7587 | 2.2135 |
| Writer discarded it | 43 | 2.9195 | 2.9593 | 2.3937 |

When the write succeeds the learned layer delivers the whole oracle benefit: 2.1508 against an
oracle 2.2135 and a no-memory 2.7587 on the identical episodes. When it fails, the injected memory is
close to inert rather than harmful, moving 2.9593 to 2.9195. The discarded-evidence episodes are not
intrinsically hard, because the oracle still reaches 2.3937 on them. The task-level shortfall is
entirely write selection, which independently confirms the conditional top-1 of about 0.95 reported
in Phase 13 and 14.

## The training-free length control

The heuristic sweeps in Phase 14 compared recency, causal novelty, and causal surprise, but never
candidate length. Text that will be revisited is also longer than text that will not, in both
corpora, so a learned writer must beat "keep the longest" before its retention counts.
`analyze_length_baseline.py` recomputes candidate lengths from each source and persists only
aggregates.

| Corpus, held-out split | Longest two | Recency two | Shortest two | Random | Learned writer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claude revisit, 574 episodes | 0.3868 | 0.3885 | 0.1185 | 0.2500 | **0.6195** |
| LongMemEval-S, 202 episodes | **0.4159** | 0.2277 | 0.0594 | 0.2500 | 0.4069 |

Target candidates average 72.2 characters against 47.9 for non-targets on Claude, and 13,979 against
10,426 on LongMemEval-S. The private natural corpus survives the control with a wide margin: 0.6195
against 0.3868. The public validation does not. On LongMemEval-S the adopted model retains 0.4069
while an untrained longest-two rule retains 0.4159, so Phase 14's comparison against a 0.25
random-capacity expectation used the wrong reference and the external retention claim is withdrawn.
The Phase 14 section carries a correction pointing here.

## Interpretation boundary

Two results stand. The first is positive and new: a task-level endpoint, gated so that it provably
detects a known-good memory, shows that a correct capacity decision by this layer buys essentially
the full oracle gain on a frozen generator it was never trained against. The second is negative and
retracts an earlier claim: the layer's advantage over training-free rules exists on the private
Claude corpus and does not exist on the public benchmark once length is controlled.

Neither result establishes autonomous importance. The generator comparison uses one seed, 72
episodes, and one 1.5-billion-parameter model, so its margins are illustrative rather than
statistically settled. The length finding does not show that the writer is only a length detector on
Claude; it shows that the public benchmark cannot distinguish the two hypotheses. The decisive next
experiment is to supply candidate length as an explicit input feature and measure whether the learned
layer still improves on the residual. If it does, the Claude margin is semantic; if it does not, much
of the reported retention is a length correlate and the weak teacher needs redesigning before any
further architecture work.

# Phase 16: is the margin semantic or a length correlate?

Date: 2026-09-14

Phase 15 left one question that gates every further design decision. `train_length_residual.py`
answers it by giving the same layer three different write inputs while holding the model, the weak
teacher, the split, and the training budget fixed. Candidate lengths are recomputed from each source
and checked against the feature artifact's targets before training, so an unnoticed misalignment
cannot produce the result.

The length arm receives two channels: standardized log length and within-episode length rank. The
rank channel is exactly what a "keep the longest" rule reads. Five seeds, two-of-eight capacity,
1,000 write and 1,000 recall steps on CPU.

| Write input | Claude retention | Claude top-1 | LongMemEval retention | LongMemEval top-1 |
| --- | ---: | ---: | ---: | ---: |
| Length only | 0.4056 +/- 0.0128 | 0.2209 +/- 0.0126 | **0.4158 +/- 0.0000** | 0.2277 +/- 0.0000 |
| **Frozen embedding** | **0.5920 +/- 0.0099** | **0.5686 +/- 0.0184** | 0.4030 +/- 0.0236 | **0.3198 +/- 0.0227** |
| Embedding plus length | 0.5812 +/- 0.0104 | 0.5578 +/- 0.0098 | 0.3980 +/- 0.0240 | 0.3218 +/- 0.0266 |

Reference points are the training-free longest-two rule, 0.3868 on Claude and 0.4158 on
LongMemEval-S, and the 0.25 random-capacity expectation. The embedding arm reproduces the Phase 14
staged-BCE CPU retention of 0.5920 exactly, which confirms the harness did not change the adopted
configuration.

## What this settles

On the natural Claude corpus the margin is semantic. The embedding arm beats the length arm by
0.1864 retention and 0.3477 top-1, and making length explicit does not improve it: the combined arm
is 0.0108 retention lower, inside seed noise but with no sign of a gain. A length cue the layer could
not already read would have raised the combined arm, so the frozen embedding evidently carries
whatever length information is useful and the remaining advantage is content.

On LongMemEval-S it is not. The length arm converges to the longest-two rule with zero seed variance
and beats the embedding arm on retention. Adding the embedding does not recover the gap. The public
benchmark therefore cannot separate this layer from a trivial rule at the write step, which is
consistent with the Phase 15 withdrawal rather than a new failure.

One nuance keeps the public result from being wholly negative. Even where length retains more
evidence, it recalls far less: 0.2277 top-1 against the embedding arm's 0.3198. Length-only traces
carry nothing a later query can match, so the recall mechanism still requires semantics. What
LongMemEval-S fails to validate is write selection specifically, not the memory layer as a whole.

## Interpretation boundary

This does not show that the Claude writer is free of shortcuts, only that length is not the shortcut.
The weak teacher still defines utility through lexical recurrence, and other correlates — turn
position, vocabulary rarity, question form — remain untested in the same way length was until
yesterday. The result also does not transfer: it establishes the phenomenon on one private corpus
whose public counterpart cannot currently corroborate it.

The next investment is therefore a public corpus whose evidence is not length-correlated, rather than
a teacher redesign, since the teacher has now survived the specific confound that motivated
questioning it.

# Phase 17: deferring the write decision

Date: 2026-09-14

Phase 7 measured the ceiling of predicting future utility at write time: 0.543 balanced accuracy
from observation-time evidence against 0.980 once delayed feedback arrives. Phases 12 to 16 kept the
write-time framing anyway. The literature survey in `RESEARCH.md` found three biological mechanisms
converging on cheap local write plus later revision, and prior art that makes two-stage deferral the
only remaining differentiator of this project. This phase tests it on real data.

## The corpus was already on disk

63.8% of non-abstention LongMemEval-S questions carry evidence in two or more sessions, a median of
20 sessions apart, but `iter_longmemeval_revisits` emits one single-evidence episode per evidence
session and discards that structure. `iter_longmemeval_deferred` keeps it: the earliest evidence
session becomes the write target, the next becomes an event that arrives before the query, and every
evidence session is excluded from the distractor pool so the later event is never reachable as a
candidate. This yields 300 episodes, one per question — 121 multi-session, 107 temporal-reasoning,
and 72 knowledge-update. Evaluation is five-fold grouped cross-validation stratified by question
type, so every episode is predicted out of fold.

`DeferredConsolidationMemory` stages the decision. Stage one sees only the candidates and commits to
a provisional set; stage two receives the later event and may narrow it but cannot recover anything
stage one dropped. Setting the provisional ratio equal to the final one makes stage two a no-op, so
the single-stage baseline is the same class with the same parameter count.

## What the corpus supports before any training

| Training-free rule | All | knowledge-update | multi-session | temporal-reasoning |
| --- | ---: | ---: | ---: | ---: |
| Random capacity | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| Recency, last two | 0.2433 | 0.3056 | 0.1818 | 0.2710 |
| Longest two | 0.4767 | 0.3472 | 0.4380 | 0.6075 |
| Cosine to the later event | 0.9400 | 0.9583 | 0.9752 | 0.8879 |
| Cosine to the query | 0.9367 | 0.9583 | 0.9421 | 0.9159 |

Two readings matter. Recency sits below random because candidate order is hash-randomised, which
confirms position carries no cue. And the later event is not weak evidence: ranking by raw cosine to
it retains the target slightly more often than the query itself does. The deferred variants must be
read against that 0.94 ceiling, not against random capacity.

The corpus also has no measurable deferral window. Bucketing by the distance between the write
target and the later event, cosine-to-event retention runs 0.9296, 0.9241, 0.9324, and 0.9737 from
the closest band to the furthest. The finite window the biological survey describes is not
observable here, so this phase can ask whether deferring helps but not how long a decision may be
deferred.

## Deferral helps, and the help is event-specific

Five seeds, two-of-eight final capacity, four-of-eight provisional capacity for the deferred arms.

| Variant | Retention | Margin over longest-two | Top-1 |
| --- | ---: | ---: | ---: |
| **Deferred, event similarity exposed** | **0.6220 +/- 0.0225** | **+0.1453** | **0.3453** |
| Deferred | 0.5373 +/- 0.0168 | +0.0607 | 0.3273 |
| Deferred, blank event | 0.5087 +/- 0.0072 | +0.0320 | 0.3120 |
| Single stage | 0.4947 +/- 0.0086 | +0.0180 | 0.3147 |
| Deferred, shuffled event | 0.4920 +/- 0.0078 | +0.0153 | 0.3273 |
| Deferred similarity, shuffled event | 0.4887 +/- 0.0129 | +0.0120 | 0.3113 |

Deferring beats the single-stage writer by 0.0426, and the gain is specific to the event rather than
to the extra selection step: a blank event recovers only part of it and a real event belonging to
another question recovers none, landing below the single-stage baseline. The same ordering holds on
the 228-episode subset that excludes knowledge-update, where deferral reaches 0.5526 +/- 0.0133
against 0.5281 +/- 0.0106 single-stage and 0.5167 +/- 0.0064 shuffled.

## The second stage was the bottleneck, and why

The write stage retains the target in 0.7860 of episodes at four-of-eight capacity, and ranking that
same provisional set by raw cosine to the event would retain 0.7573. The learned consolidation head
reached only 0.5373, so roughly a third of the available signal was being used.

The likely cause is structural rather than a capacity limit. Stage one trains
`candidate_projection` for write-worthiness and then freezes it, so a trace need not retain whatever
would let stage two match it against a later event. Handing the consolidation head the untouched
frozen-encoder similarity as one extra scalar input tests that directly, and it recovers 0.0847
retention. The control settles the interpretation: with a shuffled event the same enriched head
gains nothing and scores lowest of all six variants, so the improvement comes from the correct
event's similarity rather than from the extra input width.

Per-type margins over the longest-two baseline show why this matters beyond the pooled average:

| Variant | knowledge-update | multi-session | temporal-reasoning |
| --- | ---: | ---: | ---: |
| Single stage | +0.0583 | +0.0347 | -0.0280 |
| Deferred | +0.0917 | +0.0810 | +0.0168 |
| **Deferred, event similarity exposed** | **+0.1722** | **+0.1950** | **+0.0710** |

The single-stage writer loses to a training-free length rule on temporal-reasoning, the type with
the strongest length bias. Only deferral combined with the exposed similarity clears the baseline on
all three types.

## knowledge-update is constructed wrong for this test

Trained on its own 72 episodes, every variant lands at about 0.333, below that subset's own
longest-two baseline of 0.3472, and the deferred and shuffled arms agree to four decimal places.
That is the expected outcome rather than a failure: for these questions the later evidence supersedes
the write target, and the adapter deliberately excludes it from the candidate pool, so the session
the query actually needs can never be stored. Retaining the write target is not the right objective
there. The subset does improve when trained alongside the 228 well-posed episodes, which is transfer
rather than evidence about supersession. Measuring supersession needs an episode construction where
the later evidence is itself written, not merely consulted.

## Interpretation boundary

This is positive evidence for two-stage deferral on real multi-evidence data, with the gain
attributable to the later event by two independent controls. It is not evidence that weak delayed
signals suffice: the event here is as informative as the query, so the result shows that a bounded
memory can exploit clear later evidence, not that it can perform hard temporal credit assignment.
The absolute numbers also remain far below what the same event supports — 0.6220 against a 0.9400
cosine reference — so most of the available signal is still unused even after the fix.

The corpus limits what can follow. Without a measurable deferral window there is no curve of
retention against delay, and knowledge-update cannot test forgetting as constructed. The next data
requirement is therefore a corpus whose later evidence degrades with distance, which is a stronger
condition than the length-neutrality Phase 16 asked for.

## Twenty-seed confirmation

Repeating the headline at twenty seeds moves nothing. Pooled retention is 0.6255 +/- 0.0141 with the
similarity exposed, 0.5373 +/- 0.0164 deferred, 0.4958 +/- 0.0107 single-stage, 0.4873 +/- 0.0103
with a shuffled event: the five-seed deferred figure reproduces to four decimals. On the
228-episode subset the blank-event arm lands at 0.5285 +/- 0.0186 against a single-stage
0.5283 +/- 0.0162, so there the extra selection step contributes nothing at all and every point of
the gain is attributable to the event.

## The remaining gap is structural, not undertraining

| Consolidation steps | Single stage | Deferred | Similarity exposed |
| ---: | ---: | ---: | ---: |
| 150 | 0.5281 | 0.5254 | 0.5605 |
| 300 | 0.5281 | 0.5430 | 0.6061 |
| 600 | 0.5281 | 0.5526 | 0.6289 |
| 1,200 | 0.5281 | 0.5526 | 0.6307 |
| 2,400 | 0.5281 | 0.5561 | 0.6289 |

Both deferred arms saturate by roughly 600 steps against an unchanged 0.8132 cosine reference, so
sixteen times the training budget does not approach it. Trace width does not explain it either:
32, 64, and 128 dimensions give 0.5474, 0.5526, and 0.5711 deferred. The similarity arm is the
exception, falling to 0.6018 at 128 dimensions, which is consistent with one scalar being diluted
inside a wider relation vector.

## Deferral buys less than the premise assumed

Sweeping the provisional capacity traces the axis between committing at write time and waiting for
the event. Each capacity carries its own shuffled-event control.

| Provisional capacity | Deferred | Similarity exposed | Shuffled control |
| --- | ---: | ---: | ---: |
| 2 of 8, no deferral | 0.5281 | 0.5281 | 0.5281 |
| 3 of 8 | 0.5509 | 0.5982 | 0.5202 |
| 4 of 8 | 0.5526 | 0.6289 | 0.5167 |
| 6 of 8 | 0.5596 | 0.6439 | 0.5263 |
| 8 of 8, nothing discarded early | 0.5632 | **0.6456** | 0.5325 |

Three readings. Most of the benefit arrives at the first step: holding one extra slot reaches 0.5982
of an eventual 0.6456 in the similarity arm, about three quarters of the total gain. The shuffled
control is flat near 0.52 at every capacity, so nothing here comes from capacity itself.

The third reading revises the project's premise. At full provisional capacity there is no
irreversible early discard at all, and retention still stops at 0.6456 against the 0.9342 achieved by
ranking the same candidates by raw cosine to the same event. The decisive loss is therefore not the
early discard that `PAPER.md` frames as the central difficulty; it is the learned selection
mechanism. Removing the irreversibility entirely buys 0.0351 in the plain arm, while the gap to a
training-free rule with identical information is roughly 0.29.

## The learned advantage needs a tight budget

| Final slots | Longest-k | Write-time decision | Full deferral | Deferral gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2895 | 0.3018 | 0.3395 | +0.0377 |
| 2 | 0.5175 | 0.5281 | 0.5632 | +0.0351 |
| 3 | 0.7237 | 0.6930 | 0.7184 | +0.0254 |
| 4 | 0.8333 | 0.8307 | 0.8465 | +0.0158 |

The deferral gain shrinks monotonically as capacity loosens, which matches the competitive-allocation
reading in `RESEARCH.md`: deferring matters most when the budget binds. It also bounds the claim. At
three and four slots the learned model no longer beats the training-free length rule, so the
advantage reported throughout this phase exists only under tight capacity, where a length heuristic
has little room to work.

## Revised boundary

The two-stage result stands and is robust to seeds, training budget, and trace width. What does not
stand is the framing that made it interesting. The irreversible write-time discard was supposed to be
the hard part; measured directly, it costs about 0.035 retention, while the learned selection
mechanism costs about 0.29 against a rule with the same inputs. Any further work on deferral
schedules optimises the smaller term. The larger one is the relation head, and the fact that exposing
one untransformed scalar recovers 0.0882 of it suggests the frozen projections, not the objective,
are where the information is lost.

# Phase 18: the training procedure was erasing the signal

Date: 2026-09-14

Phase 17 ended with the learned consolidation head about 0.29 retention behind a cosine rule holding
the same inputs. Scoring residually makes that gap measurable rather than inferred: the head's output
layer is zero-initialised and the encoder similarity is added directly to the logit, so an untrained
model reproduces the cosine ranking exactly. A test pins that property.

The untrained model scores 0.9342. After the usual 600 write, 600 consolidation, and 600 recall
steps it scores 0.6491. **Training destroys 0.285 retention.**

## Localising the damage

The first explanation was that encoder cosines are all positive, so the residual makes every
candidate score high and a binary objective forces the head to learn a large negative offset that
takes the ranking with it. Both repairs for that failed. Centring the similarity per episode, which
is rank-preserving and leaves the untrained model unchanged, gives 0.6246. Replacing the binary
objective with a listwise softmax, which is invariant to a constant offset by construction, gives
0.6105. Neither helps and both are slightly worse, so the offset explanation is wrong.

Enabling one training stage at a time localises it instead:

| Write | Consolidation | Recall | Retention |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0.9342 |
| 600 | 0 | 0 | 0.9342 |
| 0 | 0 | 600 | 0.9342 |
| 0 | 600 | 0 | 0.8772 |
| **600** | **600** | 0 | **0.6096** |

Write training alone is harmless and recall training alone is harmless. Consolidation alone costs
0.057. The two together cost 0.325, far more than their sum. Write training reshapes
`candidate_projection` around write-worthiness, and a consolidation head reading those reshaped
traces learns a correction large enough to bury the similarity term it was supposed to refine. The
interaction, not either stage, is the fault.

## Bounding the correction

Passing the head's output through a scaled tanh keeps the similarity in charge and leaves the head
able only to adjust. The bound matters sharply:

| Correction bound | 0.25 | **0.5** | 1.0 | 2.0 | 4.0 | unbounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retention | 0.9298 | **0.9474** | 0.9123 | 0.8246 | 0.7763 | 0.6491 |

Five-seed results at a bound of 0.5, against a 0.9342 training-free cosine reference:

| Variant | Retention | Top-1 |
| --- | ---: | ---: |
| **Full deferral, residual** | **0.9500 +/- 0.0059** | **0.4816** |
| Full deferral, shuffled event | 0.5421 +/- 0.0138 | 0.3298 |
| Deferred at four of eight, residual | 0.8132 +/- 0.0094 | 0.4605 |

Training now adds to the signal instead of erasing it: 0.9500 is above the cosine rule the model
starts from, and top-1 rises from 0.3526 before the fix to 0.4816. The shuffled control collapses to
0.5421, so the result stays attributable to the correct event rather than to the residual structure.

At four-of-eight provisional capacity the write stage leaves 0.8307 and consolidation converts
0.8132 of it, or 97.9 percent, against 78.6 percent before the fix. The consolidation stage is
therefore close to exhausted and the binding constraint returns to the write gate, which is where
Phase 13 and 15 located it on the other corpora.

## Interpretation boundary

This is a fix to a training procedure, not evidence about memory. The lesson generalises only as far
as the setup: when a later stage reads representations that an earlier stage optimised for a
different objective, an unbounded correction can erase a good prior, and that failure is invisible
unless the architecture is arranged so the untrained model has a measurable baseline. The residual
formulation earns its place mainly by making the loss observable.

It also does not rescue the Phase 17 premise. The write-time discard still costs little, the later
event is still as informative as the query, and the corpus still has no deferral window. What changed
is that the learned layer now matches and slightly exceeds a training-free rule with the same inputs,
which it previously could not do.

# Phase 19: the write gate is information-limited

Date: 2026-09-14

Phase 18 returned the binding constraint to stage one, so the obvious next move is to improve it.
Two attempts and one measurement establish that there is almost nothing to improve.

Scoring each candidate against the episode mean rather than in isolation is the comparative form that
fixed consolidation, since choosing k of n is a comparison and a per-candidate gate cannot express
"more promising than the rest of this episode". It helps slightly, moving two-of-eight retention from
0.5278 +/- 0.0136 to 0.5380 +/- 0.0126, about one standard deviation.

Giving the write gate a length prior in the residual form that worked for consolidation was not
attempted, for a reason worth recording: the prior is worse than what the gate already learns. The
longest-two rule retains 0.5175 against the gate's 0.5278, so initialising at the prior would start
below the current solution rather than above it, which is the opposite of the consolidation case
where cosine sat at 0.9342 against a learned 0.6491.

## How much write-time signal exists

`analyze_write_ceiling.py` fits an unconstrained listwise scorer directly on the raw
384-dimensional encoder features under the same teacher, folds, and capacity. It has no memory
bottleneck, no projection to 64 dimensions, and no staged freezing, so it bounds what any model
could extract from a candidate's own content at write time.

| Model | LongMemEval deferred, 228 episodes | Claude revisit, 574 held-out episodes |
| --- | ---: | ---: |
| Random capacity | 0.2500 | 0.2500 |
| Longest two | 0.5175 | 0.3868 |
| Unconstrained scorer, 98,817 parameters | 0.5044 | **0.6069** |
| Unconstrained scorer, 197,633 parameters, 3x steps | 0.4971 | not run |
| **Learned write gate** | **0.5380** | 0.5920 |

On the deferred corpus no unconstrained model beats the 64-dimensional gate, and tripling capacity
and training makes it worse rather than better, which on 228 episodes is overfitting. On the Claude
corpus the unconstrained scorer is ahead by 0.0149, so a little headroom exists there, but not the
kind that a better architecture would obviously capture.

The write decision is therefore information-limited rather than capacity-limited. This reproduces
Phase 7's finding — 0.543 balanced accuracy from observation-time evidence — from a different angle,
under a different teacher, on two different corpora.

## Why this matters for the design

The same layer that cannot exceed about 0.54 at write time reaches 0.9500 once the later event
arrives. That gap is not a modelling result, it is a statement about when the information exists.
Every phase from 12 onward spent effort making the write moment smarter; the measurement says that
effort had a ceiling of roughly 0.015 and the project reached it.

What follows is a design conclusion rather than a new experiment: a bounded memory should hold
provisionally and consolidate against later evidence, not attempt to be selective at encoding time.
That is what `RESEARCH.md` found three biological mechanisms converging on, and it is now measured on
two corpora here.

## Interpretation boundary

The ceiling is measured under one teacher and one frozen encoder. A different notion of usefulness,
or a representation carrying information BGE-small discards, could move it. The claim is not that
write-time prediction is impossible in general, only that on these corpora with this supervision the
gate has extracted what is available, so further architecture work on stage one is unjustified.

# Phase 20: sparse expansion, under the redesigned protocol

Date: 2026-09-14

`RESEARCH.md` records why the phase 12 to 19 task had to be abandoned: presenting candidates at
recall lets encoder cosine stand in for memory, and every result in that line reduced to it. The
replacement stores documents in a bounded state, probes with a cue, and never shows a candidate list.
This is the first measurement from that protocol, and the one the survey makes the sharpest
prediction about.

## Setup

Documents are the 2,400 LongMemEval session embeddings. A similar set is the nearest neighbours of a
seed document; a dissimilar set is grown greedily to spread them apart. Keys and values are random
projections, there is no training anywhere, and storage is the plain delta rule, so the only thing
degrading a read is interference.

The three arms hold the comparison honest. Sparse expansion buys substrate, which is not free, so
matching total scalars would define the effect away. Matching the plasticity spent per write is the
biologically meaningful constraint, and it is what competition under limited plasticity bounds.

| Arm | Key width | Active units | Substrate | Plasticity per write |
| --- | ---: | ---: | ---: | ---: |
| Dense, small | 32 | 32 | 2,048 | 2,048 |
| Dense, large | 512 | 512 | 32,768 | 32,768 |
| **Sparse, large** | 512 | **32** | 32,768 | **2,048** |

Comparing sparse against dense-large separates sparsity from substrate size, since those two arms
differ only in how much of the substrate each write touches.

## The metric had to be corrected first

The obvious score is fidelity: the cosine between what the state returns for a document's own key
and what was written there. On that score the sparse advantage looked *larger* for dissimilar
documents, 0.0626 against 0.0480 at load 64, which points the opposite way to the prediction and
would have been reported as a falsification.

It was the wrong score. The sparsening result is about discrimination — removing inhibition impairs
telling similar odours apart — and fidelity never asks whether a read could be confused with another
stored document. Scoring instead on whether a read is closer to its own value than to any other
stored value answers the prediction as stated.

## Result

Discrimination, forty episodes at each load, three seeds:

| Condition | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Similar, dense small | 0.4750 | 0.2604 | 0.1729 | 0.1224 | 0.1346 | 0.1618 |
| Similar, dense large | 0.4667 | 0.2562 | 0.1656 | 0.1484 | 0.2344 | 0.4270 |
| **Similar, sparse large** | **0.7958** | **0.7792** | **0.7250** | **0.7417** | **0.7701** | **0.7996** |
| Sparse advantage, similar | +0.3292 | +0.5229 | +0.5594 | +0.5932 | +0.5357 | +0.3727 |
| Dissimilar, dense small | 0.9750 | 0.9042 | 0.7646 | 0.6453 | 0.5188 | 0.3458 |
| Dissimilar, dense large | 1.0000 | 1.0000 | 0.9969 | 0.9880 | 0.9747 | 0.9819 |
| Dissimilar, sparse large | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 |
| Sparse advantage, dissimilar | 0.0000 | 0.0000 | +0.0031 | +0.0120 | +0.0253 | +0.0180 |

The prediction holds and holds sharply. Sparse expansion is worth between 0.33 and 0.59
discrimination on similar documents and between 0.000 and 0.025 on dissimilar ones, a difference of
roughly thirty to sixty times, while spending sixteen times less plasticity per write than the dense
arm it beats. Dense-large and dense-small are nearly identical on similar sets, so the effect is
sparsity rather than substrate.

## Interpretation boundary

This is a property of the code, not of a trained memory. Nothing here learns, so it says what a
sparse expanded representation makes possible, not that a trained layer will exploit it. The
documents are real session embeddings but the similarity axis is constructed by nearest-neighbour
selection, which produces sets more uniformly alike than naturally co-occurring sessions would be.
And the result transfers a Drosophila olfactory finding to text embeddings by analogy; the agreement
is evidence that the mechanism generalises, not that the circuits correspond.

The immediate consequence for this project is concrete. Every memory built in phases 12 to 19 used a
dense low-dimensional code, 12 to 128 units with everything active, which is the arm that cannot tell
similar documents apart. That is a plausible contributor to why those layers never exceeded encoder
cosine, and it is now a specific thing to change rather than a guess.

# Phase 21: load, decay, and what survives it

Date: 2026-09-14

Measurements 1 and 3 of the redesign share a load curve, so they run together. The store is the
sparse expanded code Phase 20 adopted: 512 units with 32 active, 64-wide values, random projections,
plain delta rule, no training. Documents are the same 2,400 session embeddings, and both the similar
and the dissimilar regime are reported.

## Two metric corrections, both caught by tests

Shared structure was first estimated by centring the stored values before the decomposition. Centring
makes the basis describe how the documents *differ*, which is the opposite of what is shared, so the
gist and surface labels were swapped. The basis is now uncentred.

Estimating that basis from the stored set was the second fault. A fixed-rank basis fitted to N
documents explains less of each one as N grows, so the split moved with the load and surface recovery
appeared to *rise* from 0.128 at load 8 to 0.308 at 128 — an artefact of the metric, not a property of
the memory. The basis is now estimated once from a thousand documents and held fixed across every
load, which makes the loads comparable.

A third correction went the other way. The first decay test asserted that an early write should
degrade in direction; with orthogonal keys decay only scales a read, so the loss is in magnitude. The
test was wrong, not the code.

## Result

Twenty-five episodes per load, three seeds.

| Similar documents | 4 | 8 | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fidelity | 0.9704 | 0.9551 | 0.9433 | 0.9365 | 0.9276 | 0.9166 |
| Best decay | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Gist recovery | 0.8276 | 0.8395 | 0.8497 | 0.8523 | 0.8428 | 0.8341 |
| Surface recovery | 0.5063 | 0.4651 | 0.4294 | 0.4119 | 0.4083 | 0.4007 |

| Dissimilar documents | 4 | 8 | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fidelity | 0.9871 | 0.9725 | 0.9571 | 0.9413 | 0.9213 | 0.8950 |
| Best decay | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Gist recovery | 0.7504 | 0.7576 | 0.7513 | 0.7339 | 0.7284 | 0.7302 |
| Surface recovery | 0.6325 | 0.6083 | 0.5963 | 0.5922 | 0.5696 | 0.5289 |

**Measurement 1, first half: confirmed.** Fidelity falls gradually over a thirty-two-fold increase in
load, 0.9704 to 0.9166 on similar sets and 0.9871 to 0.8950 on dissimilar ones. There is no cliff.

**Measurement 1, second half: falsified.** The best decay is 1.0 at every load in both regimes, and
the ordering is monotone: 0.99 beats 0.95 beats 0.9 beats 0.8, everywhere. Decay never helps and
helping more under load was the prediction. The reason is mechanical. The delta rule already
subtracts what the state returns before writing, so interference is corrected at write time, and a
multiplicative decay only erodes what was stored earlier without reducing the collision. The claim
that decay rate adapts to interference does not transfer to an error-correcting store.

**Measurement 3: confirmed.** Surface recovery falls monotonically with load in both regimes, by
0.106 on similar sets and 0.104 on dissimilar ones, while gist recovery stays flat — 0.8276 to 0.8341
on similar, 0.7504 to 0.7302 on dissimilar, with no trend. Superposition removes what is
idiosyncratic and leaves what is shared, which is the mechanism the gist-preserving description of
memory asks for, obtained here without anything being trained to produce it.

## Interpretation boundary

Gist is operationalised as a four-dimensional shared subspace of the encoder's geometry, not as
meaning. That the shared component survives superposition while the residual does not is a statement
about linear storage of correlated vectors; calling the survivor "gist" is an interpretation the
measurement does not establish.

The decay result is narrower than it looks. It falsifies decay as a *capacity* mechanism in an
error-correcting store, which is the only role tested here. Decay in the biological account also
implements transience over time and enables a later event to act on what is still labile, and neither
of those is measured by a load curve.

# Phase 22: the rescue window

Date: 2026-09-14

This is the measurement worth the most in the redesign, because it recovers what no corpus in this
project could supply. Phase 17 wanted a curve of retention against delay and LongMemEval-S had none:
its later evidence stayed equally informative at every distance. Here the delay is ours to set.

Tagging and capture says a weak event leaves a tag rather than a lasting change, and a later strong
event supplies plasticity that the tag captures, provided it arrives inside a window. The strong
event need not be related to what it rescues.

## Modelling it took two corrections

The first implementation tagged every write and let capture re-apply all recent deltas. Capture then
made things worse, because the full-strength intervening writes have deltas five times larger than
the weak target's and re-applying them buries it. The missing piece is the asymmetry that makes the
mechanism interesting: a strong input is already consolidated and has nothing left to capture. A
write of strength s now expresses s of its update and tags the unrealised `1 - s`, so a full-strength
write leaves no tag and only the weak trace stands to gain.

The second correction was the metric. Absolute rescue, recalled-with-capture minus
recalled-without, *grows* with the gap, because the control falls as interference accumulates and
leaves more room to improve. That is the baseline moving, not a window. Rescue is now reported
against what the same trace would have reached had it been written at full strength at the same gap,
so 1.0 means the lost plasticity was fully recovered and 0.0 means none of it was.

## Result

One hundred and fifty episodes per point, three seeds. Recovered fraction against the number of
writes between the weak trace and the strong event:

| Tag decay | 0 | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0, no fading | 0.8125 | 0.8064 | 0.8067 | 0.8129 | 0.8263 | 0.8458 |
| 0.9 | 0.8125 | 0.7620 | 0.7141 | 0.6217 | 0.4553 | 0.2237 |
| 0.7 | 0.8125 | 0.6572 | 0.5087 | 0.2803 | 0.0733 | 0.0044 |
| 0.5 | 0.8125 | 0.5248 | 0.2972 | 0.0805 | 0.0051 | 0.0000 |

The window exists and the tag's fading is what creates it. With no fading the recovered fraction is
flat across a sixteen-fold change in delay, so nothing about waiting is intrinsically costly; the
loss is the tag, not the interval. With fading, rescue falls monotonically, and the half-life tracks
the decay rate: roughly seven intervening writes at 0.9, two at 0.7, one at 0.5.

The underlying numbers show the mechanism rather than a rescaling. At tag decay 0.7 and no delay the
weak trace reads 0.7405 without capture and 0.9019 with it, against a 0.9392 ceiling. At sixteen
writes of delay the same comparison is 0.7523 against 0.7530, with the ceiling still at 0.8982:
the plasticity is still available and the trace can no longer take it.

## Interpretation boundary

Content-freedom here is built in, not measured. The capture term multiplies the surviving tags by the
event's magnitude and never reads its key or value, so the pathway cannot depend on content by
construction. The only content dependence left is indirect, through the interference the strong
event's own write adds, and that applies equally to the control.

The window is measured in intervening writes, which is interference rather than time. That is the
right variable for a state that has no clock, but it means this does not reproduce the hours-long
windows in the animal work; it reproduces their shape. Tag decay is also a free parameter set by
hand rather than learned or derived, so the result establishes that a fading tag produces a rescue
window of a tunable width, not what width a real system should have.

# Phase 23: the modulatory pathway, rebuilt from the reinforcement circuit

Date: 2026-09-14

Measurement 10 was held back until the memory was shown to hold something, which Phases 20 to 22
did. It had also failed twice before: in Phase 18 and again in the associative attempt, the gain
saturated at its bound and a blank event scored within noise of the correct one.

A survey of the reinforcement side of the same circuit, recorded in `RESEARCH.md`, named two things
we had omitted. The drive is a prediction error — the return path from the output neurons to the
dopaminergic neurons implements reinforcement minus what the memory already answers — so a gain with
nothing subtracted has no reason to stop growing. And the compartments are opponent pairs read as a
difference, whereas a single positive channel can only rescale the state, which is rank-preserving
and therefore provably unable to reorder what is retrieved. Modelling work collapses the fifteen
compartments to two channels differing in sign alone and still recovers conditioning and blocking,
so two is both the minimum the biology licenses and the maximum that work shows is needed.

## Build

Every write leaves its own update as an eligibility trace. A reinforcement arrives as a probe and a
sign, and reaches the traces only through coincidence: how strongly each still-eligible trace answers
that probe, a single scalar per trace. Nothing document-specific is available to the pathway. The
drive is what the sign asks for minus what the state already delivers, split into an approach and an
avoidance channel whose difference is applied.

One bug and one sign error had to be found first. Normalising the coincidence response destroyed the
address: the response magnitude *is* the coincidence, since a trace written on a key near the probe
answers strongly while one written elsewhere barely answers, and normalising leaves only a direction
every trace shares. Separately, multiplying the drive by the sign before clamping made a negative
event strengthen its target, because the requested direction is already carried by the residual and
the clamping only splits it into channels.

## Result

Sixteen stored documents, one hundred and fifty episodes, three seeds. Coincidence addresses the
intended trace in 0.8889 of episodes.

| Condition | Change in the target's recall |
| --- | ---: |
| Strengthen, prediction error removed | +0.01902 |
| Strengthen | +0.00401 |
| Strengthen, mismatched probe | +0.00170 |
| Weaken, mismatched probe | -0.00185 |
| **Weaken** | **-0.10249** |

The pathway is no longer inert. Weakening moves the addressed trace by 0.10249 and the same event
sent to a mismatched probe moves it by 0.00185, a factor of fifty-five, so the effect is specific to
what the probe addresses rather than to the event occurring.

Two asymmetries in the table are the circuit's own, not artefacts. Depression is much the stronger
direction, which is what the fly does: dopamine depresses the output that signals the opposite
valence rather than potentiating agreement. And strengthening is weak precisely because the
prediction error limits it — a freshly written trace already answers its own probe, so there is
little left to add, and removing the subtraction raises the same effect nearly fivefold. That is
blocking, and it is the property that stopped the gain saturating.

## Interpretation boundary

Content-freedom is structural here, as it was in Phase 22: the pathway receives one scalar per trace
and cannot reach anything else, so this demonstrates the design is expressible rather than measuring
that a learned version would stay content-free. Nothing is trained; capture and the channel split are
fixed. And the two-channel reduction is a modelling claim about conditioning assays, which is a
different question from whether two channels suffice for memory capacity or longevity.

# Phase 24: two time constants, and the answer that changes

Date: 2026-09-14

The reinforcement survey recommended differing decay per channel as the next step after the opponent
pair, and it meets measurement 4 of the redesign. The phenomenon to reproduce is specific: in the fly,
compartments hold their own traces with their own decay, opposing training writes both at once, and
the expressed valence flips as the fast compartment fades. Nothing migrates between compartments, so
a flip is evidence of parallel stores rather than of consolidation moving a trace.

## Setup

One cue is taught two different answers at the same moment. The fast store writes at full strength
and decays at 0.75 per write; the slow store writes at a quarter strength and decays at 0.99. The
readout is their sum. Delay is counted in intervening writes, which both stores receive.

The control is one matrix holding both answers, written in sequence on the same key. That is the
honest comparison: it asks whether a single store with a single decay can produce the same change of
expressed answer.

## Result

One hundred and twenty episodes per point, three seeds. Margin is agreement with the early answer
minus agreement with the late one, so a sign change is a change in what the memory expresses.

| Delay | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Two stores, margin** | **+0.3034** | +0.1947 | +0.1155 | +0.0106 | **-0.0742** | -0.0852 | -0.0644 |
| Two stores, early answer | 0.9832 | 0.9277 | 0.8786 | 0.8004 | 0.7191 | 0.6958 | 0.7001 |
| Two stores, late answer | 0.6799 | 0.7331 | 0.7631 | 0.7898 | 0.7933 | 0.7810 | 0.7645 |
| One store, margin | -0.6470 | -0.5189 | -0.4161 | -0.2684 | -0.1025 | -0.0132 | +0.0013 |

The crossover is real and it holds. With two stores the early answer leads by 0.3034 immediately,
the margin passes through zero between four and eight intervening writes, and the late answer stays
ahead from there to the end of the range. Neither answer is deleted: both agreements sit near 0.7 at
long delay, and what changes is which one leads.

The single store fails in two distinct ways. It never expresses the early answer at all — starting at
-0.6470 because the second write on the same key supersedes the first — so one matrix cannot hold two
answers to one cue however long you wait. And its margin then decays monotonically towards zero, from
-0.6470 to +0.0013, which is the cue's trace being destroyed by interference until the two answers are
indistinguishable. Converging to indifference is not a crossover, and the +0.0013 at the longest delay
is noise, not a flip.

## Interpretation boundary

The two decay rates are chosen, not derived, so this shows that parallel stores with different time
constants can produce a changing expressed answer, not that any particular pair of constants is
right. The flip's timing follows directly from the fast store's decay, so it is a property of the
parameter rather than a prediction.

The stronger claim the result does support is structural, and it is about what a single store cannot
do. Two answers to one cue are not representable in one delta-rule matrix at all, because the later
write corrects the earlier one away. Every memory this project built before Phase 23 was a single
store, which means that whole line could not have represented a revised belief alongside the one it
replaced — the failure of the knowledge-update subset in Phase 17 now has a mechanical explanation
rather than only a construction-level one.

# Phase 25: competitive allocation, and where the trade-off does not appear

Date: 2026-09-14

Measurement 5 tests the allocation rule this project started from. Jeong et al. report that
potentiated inputs are preferentially recruited while the recall-active population stays roughly
constant, which implies allocation is zero-sum: what one trace takes should come out of others.

The competitive form was tried once before, in the associative model, and it made things worse. That
attempt is not evidence, because its addressing pathway was broken — the coincidence response was
normalised, which removed the address entirely. Phase 23 fixed that, so the constraint can now be
tested on a pathway that works.

## Setup

Sixteen documents stored, a weakening event addressed to one of them through coincidence. The free
arm applies the gains as computed; the competitive arm subtracts their mean so they sum to zero,
making recruitment zero-sum at the level of the parameter.

A constraint that simply spends less plasticity is not a fair win, and one that spends more is not a
real one, so both arms are also compared after rescaling each to the same total update norm.

## Result

One hundred and fifty episodes, three seeds.

| | Free | Competitive |
| --- | ---: | ---: |
| Change in the addressed trace | -0.10249 | -0.08815 |
| Mean change in the others | -0.00570 | -0.00499 |
| Selectivity | 0.39975 | 0.39554 |
| Change in total recall | -0.18798 | -0.16294 |
| **At equal plasticity budget** | **-0.08098** | **-0.08815** |

**The constraint does not hurt.** At matched budget the competitive arm moves its target by 0.08815
against 0.08098, about nine percent more, so forcing the trade-off buys a little efficiency rather
than costing anything. On the stated criterion — fails if the constraint only hurts — the prediction
survives.

**The trade-off itself does not appear.** Zeroing the sum of gains does not produce a zero-sum
outcome: the untargeted traces still fall, by 0.00499 against 0.00570 in the free arm, rather than
rising to compensate. Total recall shifts by -0.16294 rather than staying put. The constraint holds
at the level of the parameter and does not survive the translation to what the memory returns,
because the traces are not orthogonal and a gain redistributed across overlapping traces does not
redistribute their readouts.

**Selectivity is unchanged**, 0.39975 against 0.39554. Competition does not make the same plasticity
more discriminating; it only spends it on the target more efficiently.

## Interpretation boundary

The constant-population property is what the biology actually reports, and it is measured here in
recall rather than in anything corresponding to a population of active units. A store whose traces
overlap has no clean analogue of "how many units are active", so the mismatch found here may be a
fact about that translation rather than about allocation.

The effect is also small. Nine percent at matched budget is a real ordering — it held across all
three seeds — but it is not the kind of margin that would justify adopting the constraint on
performance grounds alone. Its better justification remains the one from Phase 23: a pathway that
can only add is rank-preserving, and a constraint forcing gains to trade off is one way to guarantee
it can reorder.

# Phase 26: linking, replay, and what reading does

Date: 2026-09-14

The last three measurements of the redesign. Two are falsified in ways that name the write rule as
the cause, and one holds.

## Measurement 6: excitability links nothing, and the write rule is why

Excitability is modelled as a bias that makes recently used units easier to recruit and fades
afterwards, so documents written close together compete for the same units. It does exactly that:
consecutive codes share 0.2257 of their active units at zero gain, 0.7376 at 0.05 and 0.9806 at 0.15.

Sharing substrate does not produce linking. Cross-recall by write-order separation, at gain 0.05:
0.5971 at distance one, rising to 0.6228 by distance three and flat after. Adjacent documents are the
*least* linked, not the most. At gain 0.15, where codes overlap almost completely, every distance
collapses to about 0.554 and the distinctions are gone.

The delta rule explains it. Writing a document subtracts what the state already returns for its key,
and when the previous document shares that key, the subtraction removes precisely the neighbour's
contribution. Error-correcting storage converts shared substrate into anti-linking.

Replacing the write with a plain Hebbian deposit reverses the sign, which is the check that isolates
the cause. At gain 0.05 cross-recall runs 0.7465 at distance one down to 0.7430 at distance six, a
small but monotone gradient in the predicted direction. The cost is visible in the same row: Hebbian
cross-recall sits near 0.74 everywhere against 0.55 to 0.62 for the delta rule, because without error
correction everything blurs into everything.

## Measurement 8: offline replay works, and is the cleanest of the three

Twenty-four documents are stored, eight are rehearsed offline by re-applying their eligibility with
no new input, and sixteen further documents are then written to create interference to survive.

| Replay rounds | 0 | 1 | 2 | 4 |
| --- | ---: | ---: | ---: | ---: |
| Rehearsed traces | 0.9164 | 0.9244 | 0.9293 | 0.9323 |
| Skipped traces | 0.9162 | 0.9151 | 0.9138 | 0.9100 |
| **Gap** | **0.0002** | **0.0093** | **0.0155** | **0.0223** |

The gap is 0.0002 without replay, which is the control working, and grows monotonically with
rehearsal. Rehearsed traces also gain in absolute terms while skipped ones lose, so this is
selective consolidation rather than a uniform strengthening: the same total plasticity is being moved
towards the rehearsed subset and away from the rest.

## Measurement 9: reading does not change what was read

The prediction was that recalled traces drift and repeated recall compounds it. Two facts about the
store refuse it.

An error-correcting restabilisation is exactly a no-op: a delta-rule update towards the retrieved
value is zero, because the state already returns that value. And a Hebbian re-deposit along the
retrieved direction cannot rotate the read either, since the key is a unit vector and the deposit is
parallel to what was already there. The recalled trace sits at 0.9390 after one recall and 0.9390
after six, unchanged to four decimals.

What does change is everything else. The other traces fall from 0.9172 to 0.8018 over six recalls of
one trace, because the deposit reaches them through the overlap between their keys and its. So in
this store, reading a memory does not distort that memory — it distorts its neighbours.

## Interpretation boundary

Measurements 6 and 9 are falsified for the same underlying reason, and it is worth stating plainly:
both predictions describe a memory whose writes are not error-correcting. The delta rule was adopted
in Phase 0 and has been assumed throughout, and it is what prevents linking and what makes
reconsolidation a no-op. Whether that makes the delta rule wrong or the predictions inapplicable is
not decided here; what is established is that these two phenomena and error-correcting storage cannot
coexist.

Measurement 8's result is the narrowest kind of positive. Replay works because re-applying a stored
delta is straightforwardly additive, so the finding is that selective rehearsal can be expressed in
this store, not that any rule for choosing what to rehearse has been tested. Nothing here selects;
the rehearsed subset is chosen at random.

# Phase 27: what should actually adapt

Date: 2026-09-14

Phase 21 reported adaptive decay as falsified. That was an overclaim. What it measured was a sweep
over *constant* decays, one per load, and no constant helping is a weaker statement than no adaptive
rule helping. The claim in the literature is that the decay rate responds to interference, which a
sweep cannot test.

## Testing it properly

`analyze_adaptive_decay.py` gives each key unit a running count of how much has been written onto it
and decays the state's columns in proportion, so forgetting happens where the interference is rather
than everywhere at once. Compared against the best constant at each load, over similar sets:

| Load | 8 | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Best constant | 0.9518 | 0.9406 | 0.9330 | 0.9251 | 0.9126 |
| Best adaptive | 0.9517 | 0.9405 | 0.9326 | 0.9238 | 0.9086 |
| **Adaptive margin** | -0.0000 | -0.0001 | -0.0004 | -0.0013 | **-0.0040** |

The adaptive rule loses, and loses more as load rises. Its best setting is the lowest sensitivity
tested, which is the setting closest to not decaying at all. The Phase 21 conclusion survives the
stronger test, and now with the mechanism stated correctly: the delta rule subtracts what the state
already returns before writing, so interference is corrected at write time and any subsequent decay
removes signal without removing collisions. This holds for forgetting that responds to local
crowding, not only for forgetting at a fixed rate.

## Where adaptation does belong

Nothing else in Phases 20 to 26 adapts either. The projections are random, the capture rate is fixed,
and the sparsity is a constant 32 active units of 512 throughout. The sparsity turns out to be the
one that should not have been.

Discrimination at three loads, sweeping the active count:

| Load | 8 active | 16 | 32 | 64 | 128 | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.7417 | **0.7458** | 0.7125 | 0.6167 | 0.4250 | 16 |
| 32 | 0.6708 | 0.7552 | **0.7812** | 0.7312 | 0.5677 | 32 |
| 128 | 0.5245 | 0.6896 | 0.7883 | **0.8193** | 0.8047 | 64 |

The optimum moves with load, and it moves *denser* as load rises, which is the opposite of the
intuition that a crowded store should spread its writes more thinly. A sparse code has few units to
carry each item, so under load each of those units is overloaded; a denser code distributes the
collision across more of the substrate.

The fixed choice used throughout costs real ground at both ends: 0.0333 at load 8 and 0.0310 at load
128 against the per-load optimum. That is larger than the entire competitive-allocation effect
measured in Phase 25.

## Interpretation boundary

One adaptive decay rule was tested, not all of them. Crowding per key unit is the most direct reading
of "decay adapts to interference", but a rule keyed on prediction error, on time since last write, or
on the value side rather than the key side remains untried. The claim is that this form does not
help in an error-correcting store, and that two quite different forms now agree.

The sparsity result is a sweep, not an adaptive mechanism. It establishes that the optimum moves and
by how much, which is what makes an online rule worth building; it does not show that a rule tracking
load online would capture that margin, since the load is given here and would have to be inferred
there.

## Building the rule, and which signal it can use

The obvious signal is the crowding a new document meets: read the state with the dense projection
before sparsifying, and see how much comes back. It is available at the right moment and zero on an
empty state, and it fails. Across a hundred-and-twenty-eight-fold change in load it only runs from
0.4523 to 1.0096, so a rule reading it moved its active count from 18.4 to 21.4 where the sweep asks
for 16 to 64. That arm beat the fixed choice at load 8 by 0.0479 and lost by 0.0405 at load 128.

The state's own magnitude does not saturate: 1.0000, 1.9754, 3.0128, 5.3144 at loads of one, eight,
thirty-two and a hundred and twenty-eight. The active counts the sweep prefers sit close to a fixed
power of it, and fitting the two endpoints gives roughly `6.1 * norm^1.4`. This is internal — the
memory is reading how large it has become, not being told how many documents it holds.

| Load | 8 | 32 | 128 |
| --- | ---: | ---: | ---: |
| Fixed 32 active | 0.6937 | **0.7792** | 0.7866 |
| **Adaptive** | **0.7438** | 0.7740 | **0.7979** |
| Oracle over the grid | 0.7271 | 0.7792 | 0.8151 |
| Oracle's active count | 16 | 32 | 64 |
| Mean count the rule chose | 11.65 | 20.14 | 39.33 |

The rule tracks the load: its chosen count moves more than threefold across the range. Against the
fixed choice it gains 0.0500 at load 8 and 0.0113 at 128, and loses 0.0052 at 32 where the fixed
value happens to be the optimum. At load 8 it also beats the oracle by 0.0167, because the oracle is
restricted to the powers of two on the sweep grid and the rule is not.

It does not reach the oracle at high load, falling 0.0172 short at 128, and its counts sit below the
oracle's throughout. The power law was fitted to two points and applies one exponent everywhere, so
there is margin left in the rule rather than in the idea.

## What this settles about adaptation

The memory's state adapts online and always did; that is what the delta rule is. What was fixed
everywhere in Phases 20 to 26 was the memory's own dynamics, and the two tested here separate
cleanly. Forgetting should not adapt, and should not happen at all, because error-correcting storage
has already handled the interference that decay would be responding to. Sparsity should adapt, can be
driven by a signal the state computes about itself, and doing so is worth more than any other single
change measured in this phase group.

# Phase 29: assembling the parts, and finding two of them do not pay

Date: 2026-09-14

Every component of the redesign was validated in isolation and none had been combined. `AssembledMemory`
puts them in one store — adaptive sparsity from the state's magnitude, no decay, a slower parallel
store, fading eligibility tags, and a coincidence-addressed opponent modulation — so each can be
removed while the rest stay.

The task is discrimination at three loads, on similar document sets, with one weakly written document
per episode so the tags and the later event have something to act on.

| Configuration | Load 8 | Load 32 | Load 128 | Mean | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Full** | 0.7458 | 0.7375 | 0.7191 | 0.7342 | — |
| Fixed sparsity | 0.7063 | 0.7141 | 0.6987 | 0.7063 | -0.0278 |
| No parallel store | 0.7063 | 0.7391 | **0.7966** | **0.7473** | **+0.0132** |
| No tags | 0.7438 | 0.7375 | 0.7191 | 0.7335 | -0.0007 |
| With decay at 0.95 | 0.6937 | 0.4708 | 0.1609 | 0.4418 | -0.2923 |
| Dense code, 128 active | 0.4583 | 0.5109 | 0.6626 | 0.5440 | -0.1902 |

## Three parts earn their place

Not decaying is the largest single effect in the whole redesign: adding a 0.95 decay costs 0.2923 on
average and collapses the highest load from 0.7191 to 0.1609. Sparsity is next at 0.1902, and making
that sparsity adaptive adds a further 0.0278 over a fixed budget. All three hold at every load.

## Two do not

Removing the parallel store *improves* the mean by 0.0132, and by 0.0775 at the highest load. That is
not a contradiction of Phase 24, which showed a slow store is what makes two answers to one cue
representable and lets the expressed answer change with delay. This task has one answer per document
and never asks anything to change, so the second store contributes nothing it can use and adds its
interference anyway. A component validated against the phenomenon it exists for can still be dead
weight against a task that never presents that phenomenon.

The tags are worth 0.0007, which is nothing. The reason is in the setup rather than the mechanism:
one document per episode is written weakly, so there is a single tag to rescue and its effect is
diluted across every other document's discrimination score. Phase 22 measured the rescue on the
rescued trace itself and found it large; this measures the store as a whole and finds it invisible.

## Interpretation boundary

This is an ablation of one configuration on one task, and the two negative results are both explained
by the task rather than by the components. That is the finding worth keeping: validating a mechanism
against the phenomenon it was designed for does not establish that it pays in a system, and only the
assembly shows which parts are carrying the result. The three that pay here are the ones whose
benefit is about interference, which is what this task measures.

Nothing is trained. The sparsity exponent, the tag decay, the capture rate and the slow store's
constants are all fitted or chosen by hand, so the ablation compares designs rather than learned
solutions.

# Phase 30: training the assembled store

Date: 2026-09-14

Everything from Phase 20 onward used random projections. `TrainableMemory` keeps the structure the
ablation kept — a sparse expanded code whose width follows the state's magnitude, delta-rule storage,
no decay — and learns the key and value projections against the only thing the store is asked to do.
The loss is a listwise cross-entropy over which stored value each probe returns, which is the
differentiable surrogate for discrimination. The width is read off the state and detached, so the
projection is trained through the values at the chosen units rather than through how many there are.

Documents are split into disjoint pools and episodes are drawn from one or the other, so a projection
that memorised particular documents shows it.

## The training curve inverts the conclusion

| Steps | Gain on seen documents | Gain on held-out documents |
| ---: | ---: | ---: |
| 50 | -0.2172 | -0.2177 |
| 150 | -0.1185 | -0.1221 |
| 400 | +0.0060 | -0.0203 |
| 1,200 | +0.0932 | **+0.0490** |
| 3,000 | +0.1154 | **+0.0523 +/- 0.0164** |

Read at four hundred steps this looks exactly like overfitting: the seen pool improves and the
held-out pool degrades, and that is what was concluded before the longer runs existed. It was
undertraining. By three thousand steps held-out discrimination goes from 0.8909 to 0.9432, a gain of
0.0523 that holds across three seeds.

The shape of the early curve is itself the finding. Training does not start from nothing and climb;
it starts from a random projection that is already a strong solution, destroys it — losing 0.2177
within fifty steps — and has to climb back before it can do better. A random projection preserves the
geometry this task depends on, so gradient descent begins by discarding something it must then
rediscover.

## What remains overfitted

The seen-pool gain is 0.1154 against 0.0523 held out, so more than half of what training buys does
not transfer. The learned projections are picking up document-specific structure alongside whatever
generalises. That is a real limit on the result, not a reason to discount it: the transferred half is
measured on documents the projections never saw.

## Learning the width too, and why it costs more than it buys

The logit scale was already learned; the note that it was hand-set was wrong. What remained fixed was
the width rule, and its active count is an integer chosen by a top-k, so nothing about how many units
to keep can reach a gradient through it. Making it learnable means replacing the counted budget with
a graded gate: each unit is kept in proportion to how far its magnitude exceeds a threshold that
rises with the log of the state's size, with the offset, slope and temperature all trained.

| Sparsification | Held-out before | Held-out after | Gain | Mean width |
| --- | ---: | ---: | ---: | ---: |
| **Counted budget, hand-fitted rule** | **0.9128** | **0.9357** | +0.0229 +/- 0.0176 | 27.8 |
| Graded gate, learned threshold | 0.7812 | 0.8042 | +0.0229 +/- 0.0733 | 17.3 |

The two gain the same amount and the gate starts 0.1316 lower, with four times the seed variance.
The width is not what costs it. A graded gate keeps every one of the five hundred and twelve units
and merely attenuates most of them, so the reported width of 17.3 is the sum of the gate rather than
a count of what is stored — all the units are still in the key. Phase 20 established that sparse
expansion wins by keeping few units active so that writes collide less, and a gate that leaves the
tail in place gives that up.

This is a structural conflict rather than a failed implementation. The property that makes the
sparsity valuable is the discreteness of the selection, and the discreteness is exactly what blocks
the gradient. Making the width learnable *this way* removes the thing the width was worth having.

## Keeping the discreteness and training the width anyway

The conflict is only with relaxation. A sampled width keeps top-k exactly as it was and trains the
distribution it is drawn from by policy gradient against the episode's own loss, so the forward pass
never sees a soft gate.

| Sparsification | Held-out before | Held-out after | Gain | Mean width |
| --- | ---: | ---: | ---: | ---: |
| Counted budget, hand-fitted rule | 0.9128 | 0.9357 | +0.0229 +/- 0.0176 | 27.8 |
| Graded gate, learned threshold | 0.7812 | 0.8042 | +0.0229 +/- 0.0733 | 17.3 |
| **Sampled width, policy gradient** | 0.8937 | **0.9245** | **+0.0307 +/- 0.0257** | 22.1 |

The sampled arm starts where the hand-fitted rule starts rather than 0.13 below it, which is the
point: the relaxation's cost was the softness, not the learning.

### Ten seeds settle it, and against the three-seed reading

Both arms share a seed, and so a split and an evaluation set, which makes the comparison paired.

| Seed | Hand-fitted | Sampled | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 0.9547 | 0.9430 | -0.0117 |
| 2 | 0.9328 | 0.9250 | -0.0078 |
| 3 | 0.9055 | 0.9484 | +0.0430 |
| 4 | 0.9453 | 0.9391 | -0.0062 |
| 5 | 0.9492 | 0.9469 | -0.0023 |
| 6 | 0.9578 | 0.9656 | +0.0078 |
| 7 | 0.9609 | 0.9664 | +0.0055 |
| 8 | 0.9523 | 0.9469 | -0.0055 |
| 9 | 0.9320 | 0.9219 | -0.0102 |
| 10 | 0.9695 | 0.9648 | -0.0047 |

The paired difference is +0.0008 with a standard error of 0.0051, t = 0.15, and the sampled arm wins
on three seeds of ten. **The two are indistinguishable.** Held-out gains are +0.0487 and +0.0503.

Two claims from the three-seed run do not survive. The sampled arm does not gain the most; it gains
the same. And its policy does not find a sparser rule: over ten seeds the base moves from 6.100 to
5.960 and the exponent from 1.400 to 1.373, giving a mean width of 27.1 — the same as the hand-fitted
rule's 27.1. The base of 5.394 and width of 22.1 reported at three seeds were seed variation.

What survives is narrower and still worth having. The width can be trained without giving up the
discreteness, which the graded gate could not do, and the rule it converges to is the one that
fitting two points of the Phase 27 sweep already produced. Learning does not improve on that rule; it
confirms it.

## Training the remaining constants, and overturning the Phase 29 ablation

The parallel store and the tags were removed from the trained model because Phase 29's ablation found
them worthless: removing the slow store *improved* the mean by 0.0132 and the tags were worth 0.0007.
Both were tested at the values that phase set by hand. Restoring them as trainable quantities asks a
different question — whether they are useless, or were merely sized wrong.

They were sized wrong. Ten seeds, three thousand steps, every constant starting where Phase 29 put it:

| Constant | Start | Learned | Deviation |
| --- | ---: | ---: | ---: |
| Slow-store strength | 0.250 | **0.4790** | 0.0104 |
| Slow-store decay | 0.990 | **0.9995** | 0.0000 |
| Tag decay | 0.900 | **0.9964** | 0.0003 |
| Capture rate | 0.250 | **0.9445** | 0.0051 |

Every one moves *up*, none is switched off, and the seed deviations are between 0.0000 and 0.0104 —
ten independent runs land in the same place. Held-out discrimination gains 0.0739 +/- 0.0189, against
0.0487 for the model without these components.

The capture rate is the clearest case: 0.25 to 0.9445 means the hand-set value was nearly four times
too small, so the modulatory pathway had almost no authority in the ablation. Its measured
contribution of 0.0007 was not a dead component but a component held shut.

The two decay constants both run to the edge, 0.9995 and 0.9964, which is the same answer Phases 21
and 27 reached by sweeping and by an adaptive rule: in an error-correcting store, forgetting does not
help. That is now three independent routes to it, the third one arrived at by gradient descent
without being told.

## What this says about ablation

A hand ablation answers whether a component helps *at the setting it was given*. It cannot separate a
component that does nothing from one that was configured not to. Phase 29 read the second as the
first for both the tags and the slow store, and the error was invisible until the constants were
allowed to move. Its three positive findings — sparsity, adaptive sparsity, and not decaying — stand,
and are strengthened, since decay is exactly what training also refuses.

## Interpretation boundary

Only the projections and the logit scale are learned. The width rule stays a hand-fitted top-k, for
the reason above, and the components the Phase 29 ablation found unhelpful on this task — the
parallel store and the tags — are absent from the trained model rather than trained and found
wanting.

The absolute numbers are also not comparable with Phase 29's. Episodes here are drawn from a pool
that is a fraction of the corpus, so the nearest-neighbour sets are less alike and the task is
easier: untrained discrimination is 0.89 here against 0.74 in the ablation at the same load. Only the
gains within this phase should be read.

# Phase 31: the real task, and what the proxy was hiding

Every phase since the redesign has been scored on discrimination: store a set of documents, probe with
each document's own key, and count how often the store returns that document's value ahead of the
others. Phase 30 pushed that to 0.9513 held out. The benchmark never asks that question. It asks a
question that was never written down, and expects the session holding the answer.

`eval_question_retrieval.py` closes the gap. The artifact already carries the question embedding, so
the only thing missing was a read path for a cue that is not a stored key: `TrainableMemory.probe`
projects the question through the same key projection, codes it against the state it is about to
address, and scores the read against the stored values. Training uses the same objective the task
does — cross-entropy of the question's read against the evidence slot — on a disjoint set of
questions.

## The store loses to doing nothing

Three hundred questions, eight candidate sessions each, seventy/thirty split, five seeds:

| Reader | Held-out hit rate | Top-2 |
| --- | ---: | ---: |
| Chance | 0.1250 | 0.2500 |
| Untrained store | 0.4200 +/- 0.0215 | — |
| Trained store | 0.5578 +/- 0.0347 | 0.7689 +/- 0.0163 |
| Trained key projection, nothing written | 0.8178 +/- 0.0401 | 0.9333 +/- 0.0365 |
| **Cosine on the frozen encoder** | **0.8622 +/- 0.0295** | **0.9356 +/- 0.0147** |

Training helps the store by 0.1378 and never comes close to closing the 0.30 gap to a plain dot
product against the same embeddings. The projection-only control is the one that matters: writing the
sessions down costs 0.26 against reading them with the very projection the store was trained to use.
The store is not adding retrieval; it is destroying it.

Twelve thousand steps instead of three thousand gives 0.5741 +/- 0.0105 — this is not the
undertraining that Phase 30 turned out to be. The full component set (weak first write, tags,
capture, parallel store) gives 0.5422 +/- 0.0514, no better.

## It is not the sparsity

The obvious suspect was the sparse code: a question's active units may simply miss the evidence
session's. Measured directly, they do not. Sweeping the width at three seeds:

| Code width | Cue/evidence unit overlap | Hit rate | Top-2 |
| ---: | ---: | ---: | ---: |
| 8 | 0.501 | 0.5704 | 0.7333 |
| 32 | 0.522 | 0.5926 | 0.7630 |
| 128 | 0.519 | 0.6296 | 0.8259 |
| 256 | 0.578 | 0.6704 | 0.8370 |
| 512 (dense) | 1.000 | 0.6259 | 0.8481 |

Half the cue's units already land on the evidence session at width 8, and at width 512 the code is
dense, the overlap is total by construction, and the store still reads 0.63 against cosine's 0.86.
Widening the value bottleneck does not rescue it either: 32/64/128/256/384 give 0.6481, 0.6704,
0.7074, 0.6778, 0.6852. The best configuration found anywhere in these sweeps is width 256 with a
128-dimensional value, at 0.7074 — still 0.15 behind doing nothing.

## The load argument does not survive either

The defence for a compressive store is that eight sessions is too small to show its worth: a linear
scan is cheap at eight and expensive at eight thousand. Padding each question's set with sessions
drawn from other questions tests that directly, at the best configuration above:

| Sessions | Chance | Cosine | Store | Cosine top-2 | Store top-2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.1250 | 0.8630 | 0.7037 +/- 0.0532 | 0.9370 | 0.8630 +/- 0.0319 |
| 32 | 0.0312 | 0.6815 | 0.2778 +/- 0.0327 | 0.7926 | 0.4852 +/- 0.0105 |
| 64 | 0.0156 | 0.6037 | 0.2926 +/- 0.0457 | 0.7519 | 0.4370 +/- 0.0500 |
| 128 | 0.0078 | 0.5370 | 0.1852 +/- 0.0292 | 0.6630 | 0.3259 +/- 0.0604 |

The gap widens with load rather than closing: 0.16 behind at eight sessions, 0.35 behind at a hundred
and twenty-eight. Both readers degrade, but the store degrades faster, which is the opposite of the
regime the design was defended by.

## What the proxy was measuring

Discrimination asks a stored key to return its own value. A delta-rule store is fitted to satisfy
exactly that, one equation per document, and sparsening the keys makes the equations more nearly
independent — which is why every redesign measurement that improved discrimination did so, and why
Phase 30's constants all ran toward preserving what was written. None of that produces generalisation
from a cue that was never written. The retrieval the benchmark asks for lives in the encoder's
geometry, and a write-read round trip through a rank-limited matrix can only lose some of it.

This does not retract the redesign measurements. Sparse coding really does raise discrimination, the
advantage really is larger for similar documents, decay really does not help, and the constants
really were sized wrong. Those statements were about the store's ability to hold what it was given,
and they stand. What does not follow, and what this project assumed for twelve phases, is that a
store which holds documents well is a store that answers questions well.

## Boundary

Measured on LongMemEval-S deferred episodes with BGE-small-en-v1.5 embeddings, at the retrieval step
only. The cosine baseline is the same frozen encoder the store reads from, so this is a claim about
what the store adds to that encoder, not a claim about retrieval methods in general.

# Phase 32: the rescue window was a property of a hand-set constant

Phase 22 reported that a weakly written trace can be rescued by a later strong event, and that the
window closes after a handful of intervening writes. That was measured at a tag decay of 0.9, which
was chosen by hand. Phase 30 trained the same constant and it went to 0.9964. Re-measuring the window
across decays, with capture held at Phase 22's 0.25 so only the decay moves:

| Intervening writes | decay 0.9 | decay 0.99 | **decay 0.9964** | decay 1.0 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8148 | 0.8148 | 0.8148 | 0.8148 |
| 8 | 0.4571 | 0.7895 | **0.8149** | 0.8293 |
| 32 | 0.0463 | 0.7145 | **0.8210** | 0.8831 |
| 128 | 0.0000 | 0.4071 | **0.7996** | 1.0996 |
| 256 | 0.0000 | 0.1518 | **0.6875** | 1.3690 |

Values are the fraction of the full-strength ceiling that capture recovers. Taking the half-window as
the last gap where recovery still exceeds half its value at zero:

| Tag decay | Half-window |
| ---: | ---: |
| 0.9 (hand-set) | 8 intervening writes |
| 0.99 | 64 |
| **0.9964 (learned)** | **beyond 256** |
| 1.0 | beyond 256 |

The window is thirty-two times wider at the learned constant, and on any episode length this project
actually runs it does not close at all. Phase 22's "the window closes" was a statement about the
number 0.9, not about the mechanism.

## What the learned value sits next to

At a decay of exactly 1.0 the recovered fraction passes 1.0 and keeps climbing — 1.0996 at 128 gaps
and 1.3690 at 256. A tag that never fades keeps capturing, so the rescued trace ends up written more
strongly than it would have been at full strength in the first place. That is over-consolidation, not
rescue, and it is what the ceiling reference exists to expose.

The learned 0.9964 sits just below that boundary: wide enough that the window stays open across every
realistic gap, short enough that recovery never exceeds what a full-strength write would have
achieved. Nothing in the training objective mentioned a ceiling, so this is a third property the
optimiser found rather than was given.

## A transplant that does not work

The first attempt at this re-measurement moved both learned constants across, tag decay 0.9964 and
capture 0.9445, and produced recovered fractions between 1.1970 and 1.6802 at every gap. That is not
a wider window; it is the wrong parameterisation. `analyze_tagging_window.py` scales capture by an
explicit `event_strength` of 2.0, while `TrainableMemory` applies a gain of
`capture * current * (1 - current)`, which cannot exceed a quarter of the capture value. The two
numbers named `capture` are not the same quantity, and only the decay transplants directly.

## Interpretation boundary

Untrained random projections, sparse codes of width 32, and the same delta-rule store as Phase 22, so
the comparison against that phase is clean. What transplants from Phase 30 is the decay constant
alone; the capture rate would need the trained parameterisation to be carried over honestly, which
would mean re-running this measurement inside `TrainableMemory` rather than beside it.

# Phase 33: the generator endpoint never worked, and the gate said so late

Phase 31 needed the answer-likelihood endpoint to close the loop from retrieval to the task, and it
failed its validity gate twice. The first failure was mine: deferred episodes split the evidence
across two sessions and place only the first in the candidate pool, so no reader selecting from that
pool can supply a complete fact. `evidence_and_consolidation` beats no-memory by 0.0604 while
`evidence_only` is the worst condition measured at -0.7161 — half a fact is actively harmful, and it
is the shortest memory prompt, so that is not a length effect.

Moving to revisit episodes removes that defect: evidence is complete in one session, and this is the
corpus and the instrument Phase 15 validated. On 202 held-out episodes, every memory condition now
beats no-memory, and the gate still fails:

| Condition | Answer NLL | Gain over no memory | Evidence present |
| --- | ---: | ---: | ---: |
| Random two slots | **2.7629** | +0.1793 | 0.282 |
| Oracle two slots | 2.7849 | +0.1573 | 1.000 |
| Cosine two slots | 2.8321 | +0.1101 | 0.941 |
| Recency two slots | 2.8659 | +0.0763 | 0.228 |
| Store two slots | 2.8821 | +0.0601 | 0.886 |
| Evidence session alone | 2.9287 | +0.0136 | 1.000 |
| No memory | 2.9422 | 0.0000 | 0.000 |

A random pair of sessions beats the annotated evidence. Two explanations were checked and neither
holds: prompts are truncated in 1% of episodes, not enough to matter, and the raw episodes align with
the feature rows exactly — 688/688 train, 202/202 eval, 300/300 deferred target offsets agree, so the
oracle really is receiving the evidence session.

## Why Phase 15 passed

The same script reproduces Phase 15 on the same prefix. Reading the identical per-episode scores over
growing prefixes of the held-out split:

| Episodes | Oracle | Random | No memory | Gate |
| ---: | ---: | ---: | ---: | --- |
| 36 | 2.3230 | 2.5560 | 3.1268 | passes |
| 72 | 2.3188 | 2.5001 | 2.8785 | passes |
| 101 | 2.3018 | 2.3945 | 2.8152 | passes |
| 144 | 2.5974 | 2.6070 | 2.8347 | passes |
| **202** | **2.7849** | **2.7629** | 2.9422 | **fails** |

Phase 15 reported oracle 2.3211, random 2.5467 and no-memory 2.8785 on the first 72; this run gives
2.3188, 2.5001 and 2.8785 on the same episodes. The instrument is the same, the corpus is the same,
and the difference is only how many episodes the gate was read over.

## What the gate was actually measuring

The paired form settles it, and it is available at every prefix:

| Episodes | Oracle minus random | Oracle better on | Paired t |
| ---: | ---: | ---: | ---: |
| 72 | +0.1813 | **0.500** | +1.74 |
| 202 | -0.0220 | 0.391 | -0.25 |

At the prefix where Phase 15 declared the instrument usable, the oracle is better than random on
exactly half of the episodes. The mean difference came from the tail, not from a preference. Answer
NLL is heavy-tailed, a mean-based gate cannot see that, and the endpoint was never detecting the
evidence session — it was detecting a few episodes where a long prompt happened to help.

`instrument_usable` now also requires the paired win rate to exceed chance.

## What this retracts

Phase 15's positive task-level result is withdrawn: "a correct capacity decision buys essentially the
full oracle gain on a frozen generator" rests on a gate that a paired test does not support. The
subset split that produced it — 2.1508 learned against 2.2135 oracle on 29 episodes — is a mean over
twenty-nine heavy-tailed values and cannot carry that weight.

This leaves the project with no working task-level endpoint. What survives is the mechanistic
measurement: whether the evidence session is surfaced at all, which is where Phase 31's answer lives
and which does not depend on a generator's likelihood at all.

## Interpretation boundary

One generator, one seed, two-of-eight capacity. The finding is not that frozen generators cannot be
used as instruments; it is that this one, on these episodes, does not prefer the annotated evidence
per episode, and that a gate comparing means was unable to report that. A generator with a longer
usable context, or a corpus whose evidence is shorter relative to the window, might behave
differently — but that has to be demonstrated before any answer-likelihood number from this project
is quoted.

# Phase 34: the one regime that should favour the store, and two lines that beat it

Phase 31 showed the store losing to cosine at retrieval and named the reason: a delta-rule store is
fitted to return what was written, and retrieval by a novel cue lives in the encoder's geometry
already. That argument has an exception. The delta rule subtracts what the state already returns for
a key before adding, so a second document landing on the same units *replaces* the first. Nothing in
a cosine score can express that. If the store pays anywhere, it pays where a fact was stated and then
superseded.

LongMemEval has exactly that category. `iter_longmemeval_updates` builds 78 episodes from the
knowledge-update questions: both evidence sessions stay in the candidate pool, candidates keep their
original temporal order, and the target is the *later* statement. A reader that only matches topic
has no way to choose between the two.

## The mechanism is real

Untrained random projections, sparse codes of width 32, five seeds:

| Reader | Hit rate | Prefers the current statement |
| --- | ---: | ---: |
| Chance | 0.1250 | 0.5000 |
| Cosine | 0.4487 | 0.4744 |
| Recency (last slot) | 0.4231 | 1.0000 (by construction) |
| **Store, untrained** | **0.6333 +/- 0.0310** | **0.7949 +/- 0.0229** |
| Store, write order reversed | — | **0.2128 +/- 0.0418** |

Cosine sits at 0.4744 on the pairwise question, which is below chance: it cannot tell the two
statements apart and leans very slightly toward the older one. The store reaches 0.7949 without being
trained at all.

The reversal is the control that makes this causal. Writing the identical documents in the opposite
order changes no embedding, so anything order-free must be unmoved — and the store's preference flips
from 0.7949 to 0.2128. The two statements share 45.7% of their active units, which is the overwrite
happening.

## And it does not matter

The store's win over plain cosine is a win over the wrong baseline. Cosine knows nothing about order,
and neither does it have to: order is free to read off the candidate list.

| Reader | Hit rate |
| --- | ---: |
| Cosine | 0.4487 |
| Store, untrained | 0.6333 |
| Cosine plus a recency tilt, best of nine weights | 0.7821 |
| **Cosine top-2, take the later one** | **0.8974** |

The second heuristic has no fitted parameter at all. It shortlists by topic, then picks the later of
the two, and it beats the store by 0.2641. Both statements land in the cosine top-2 in 79.5% of
episodes, which is the whole trick: cosine is excellent at finding the *pair* and useless at ordering
it, and the ordering is an integer comparison.

The recency-tilt sweep is resolved in the baseline's favour — the best of nine weights, chosen on the
same data — and that one is therefore optimistic. The top-2 rule is not; it has nothing to tune.

## What this settles

Overwriting is a genuine property of the write rule, demonstrated causally by the reversal control,
and it is the only thing in this project that a cosine score structurally cannot do. It is still not
worth having here, because expressing "later supersedes earlier" through partial overlap of sparse
codes is a lossy way to say something that an index already says exactly.

Combined with Phase 31, the design has now been measured in the regime it was built for and in the
regime that most favours it, and it loses both. The remaining untested regimes — composition across
several traces, cues that are not embeddings of text — are hypotheses, not defences, and neither is
supported by anything measured so far.

## Interpretation boundary

78 episodes, one corpus, untrained projections. Training was not applied because the overwrite
property belongs to the write rule rather than to a learned projection, and 78 episodes cannot
support a held-out split of any useful size; a trained variant could close part of the 0.2641 gap but
would have to close all of it against a parameter-free rule. The heuristics compared here have access
to candidate order, which the store also has — nothing is being given to the baseline that the store
was denied.

# Phase 38: a different question, and the first mechanism that nearly holds

Every measurement so far asked the store to *retrieve*, and Phase 31 explained why it cannot win
there: a linear associative read is at best the similarity structure it was handed, so writing can
only lose. Familiarity is not that quantity. Marking the cells a document activates and asking what
share of a cue's cells are already marked measures accumulated evidence across every stored item at
once, where maximum cosine measures the single best match. A cue matching one item at 0.6 and a cue
matching twenty items at 0.3 order differently under the two, and neither approximates the other.

The corpus has the right question for this and the project had been discarding it. Thirty of the five
hundred LongMemEval questions are abstention questions — "what is the name of my hamster?", answered
"you mentioned your cat Luna but not a hamster" — and every builder here passes
`exclude_abstention=True`.

## Baselines fixed before measuring

| Reader | State | What it is |
| --- | --- | --- |
| `question_length` | none | abstention questions average 62.5 characters against 86.4, so the confound is a reader |
| `max_cosine` | O(N) | the obvious gate: is anything stored close to the cue? |
| `topk_cosine_sum` | O(N) | the obvious accumulator, its `k` swept and resolved in its own favour |
| `cell_overlap` | **O(1)** | the proposal |

Falsification, stated in advance: if the fixed-size filter does not beat the accumulating baseline,
it buys a smaller state at the cost of accuracy and adds no capability.

## The first attempt could not decide anything

On the thirty real abstention questions against ninety answerable ones, the filter placed last at
0.6704 AUC, behind `max_cosine` 0.7333, `topk_cosine_sum_4` 0.7259, and `question_length` 0.6831. But
bootstrapping every comparison returns intervals that all straddle zero — `max_cosine` minus
`question_length` is +0.0505 [-0.0941, +0.1907]. **Thirty negatives cannot resolve differences of this
size.** The correct reading was not "rejected" but "this test decides nothing", which is the reading
Phase 15 failed to make about its own 72 episodes.

## A design that can decide

Every answerable question becomes its own negative: write the same haystack with its evidence
sessions removed, and drop an equal number of random distractors from the positive side so both
stores hold the same count. Identical words, identical question length, identical store size; the
only difference is whether the answer is in there. That gives 200 matched pairs instead of 30
negatives, and makes the comparison paired.

| Reader | State | Win rate | 95% interval |
| --- | --- | ---: | --- |
| `topk_cosine_sum_48` | O(N) | **0.9793** | [0.9650, 0.9907] |
| `topk_cosine_sum_32` | O(N) | 0.9782 | [0.9629, 0.9907] |
| **`cell_overlap_graded`** | **O(1)** | **0.9700** | [0.9536, 0.9836] |
| `topk_cosine_sum_8` | O(N) | 0.9596 | [0.9382, 0.9786] |
| `max_cosine` | O(N) | 0.8839 | [0.8532, 0.9129] |
| `cell_overlap`, binary | O(1) | 0.8311 | [0.8004, 0.8604] |

Paired differences, which resolve far tighter than the overlap of two intervals suggests:

| Comparison | Difference | 95% interval |
| --- | ---: | --- |
| graded minus `topk_cosine_sum_48` | **-0.0093** | [-0.0193, -0.0007] |
| graded minus `max_cosine` | **+0.0861** | [+0.0611, +0.1129] |
| graded minus binary | **+0.1389** | [+0.1143, +0.1639] |

## What this says

**Accumulation is real.** The baseline itself rises monotonically with `k`, 0.8839 at one item to
0.9793 at forty-eight. Reading only the best match throws away most of the signal, and that is worth
0.095 — more than anything else measured in this project.

**Binarising is what broke the fly mechanism here.** The binary filter is the Bloom filter the
fly-connectome line's phase 37 identified, and it loses 0.1389 to the same filter accumulating
magnitudes instead of clamping. The saturating mark that *won* on recognition-under-load is the wrong
rule for this task, because here the question is how strongly the cue is supported, not whether its
cells were touched.

**By the criterion fixed in advance, this fails.** The graded filter does not beat the best
accumulating baseline; it loses by 0.0093, and that difference is resolved, if barely.

It fails differently from everything else here, and the difference is worth recording. Phase 31 lost
by 0.26 and Phase 34 by 0.26, both at comparable cost. This loses by 0.009 while holding a state that
does not grow with the number of documents, against a reader that must keep every embedding, and it
beats the natural O(N) gate by 0.086. That is an engineering trade rather than a failed idea — but a
trade is not the capability claim the criterion asked for, and it is recorded as a loss.

## Interpretation boundary

The filter's shape (8,192 cells, 256 active) is the best of five swept on this same data, so that
figure is optimistic in exactly the way `topk`'s best `k` is; both sides were tuned the same way and
on the same pairs. At three seeds the best shape tied the baseline exactly, and at seven the gap
opened to a resolved 0.0093 — the tie was seed noise, and the seven-seed number is the one to quote.
Nothing here rescues retrieval: this is a gate, a cheaper job than the one Phase 31 measured, and the
content store is still beaten by cosine at that.

# Phase 39: composition is real, and nobody here can reach it

Phase 38 found accumulation across weak matches worth 0.095 on the gate. The obvious next question
is whether it is worth anything on retrieval, which Phase 31 measured as picking one evidence
session — a task with no composition in it. LongMemEval's multi-evidence questions have the
structure instead: 126 of the 200 answerable questions need two or more sessions.

## The structure exists, and it is large

Ranking the *harder* evidence session with the easier one removed from the ranking:

| Cue | Top-1 | Top-4 | Top-4, restricted to the 34 questions where it starts outside top-4 |
| --- | ---: | ---: | ---: |
| The question | 0.484 | 0.778 | **0.176** |
| The easy evidence session | 0.627 | 0.873 | 0.618 |
| Question plus easy evidence | **0.659** | **0.913** | **0.676** |

On the questions where the second session is genuinely hard to find, expanding the cue with the
first evidence session moves top-4 from 0.176 to 0.676. The session the question cannot reach is
reachable through the session it can. This is the composition case, and it is not small.

## Nobody can pick the hop

Every reader returns the same number of sessions, so none can win by returning more. Recall of the
full evidence set, five seeds, 126 questions:

| Reader | State | @2 | @4 | @8 |
| --- | --- | ---: | ---: | ---: |
| Cosine, single shot | O(N) | 0.6853 | 0.8569 | 0.9142 |
| Pseudo-relevance feedback | O(N) | 0.7197 | 0.8661 | 0.9210 |
| PRF, blended over the top four | O(N) | 0.7171 | 0.8595 | 0.9306 |
| PRF, gated on the top-1 margin | O(N) | 0.7204 | 0.8734 | 0.9194 |
| **PRF with an oracle first hop** | O(N) | **0.8229** | **0.9376** | **0.9812** |
| Delta-rule associative store | O(1) | 0.4581 | 0.6470 | 0.8123 |

Paired differences at two slots: `prf_soft` minus cosine +0.0317 [+0.0040, +0.0595] resolved,
`prf_gated` minus cosine +0.0351 [+0.0079, +0.0628] resolved, `prf_soft` minus plain `prf` -0.0026
[-0.0364, +0.0317] not resolved.

An oracle first hop buys +0.1376 at two slots. Every method that has to choose its own hop captures
about +0.03 of that, and the two fixes aimed directly at the failure — blending the top four so a
wrong hop is diluted, and expanding only when the top-1 stands clear of the runner-up — do not
improve on taking the top-1 blindly. The cosine top-1 is an evidence session 82.5% of the time, so
the loss is concentrated: a wrong hop costs far more than a right one gains, and none of the
available signals separate the two.

## The store loses a third time, by the same amount

The hypothesis worth testing here was that Phase 31 measured the right mechanism on the wrong task.
An associative read is a sum of stored values weighted by key similarity, so anything close to what
the cue retrieves is pulled along — smearing, which is noise for a single target and might be
spreading activation for a set. It is not. The store reads 0.4581 against cosine's 0.6853 at two
slots, -0.2272 [-0.2695, -0.1840], and stays 0.10 to 0.23 behind at every k.

That is the third independent regime, after retrieval (Phase 31) and in-place update (Phase 34), in
which the delta-rule store loses by about 0.22 to a baseline that costs nothing.

## What is actually blocked

The shape of this result is Phase 19's. There, the write gate could not exceed about 0.54 because the
information deciding what to keep does not exist at write time. Here, +0.1376 of recall sits behind a
decision — which retrieved session to expand with — made at a moment when nothing available
identifies the right one.

The difference is that Phase 19 earned its claim by fitting an unconstrained model to bound what was
extractable, and that has not been done here. Three hand-built selectors failing is not the same as
no selector existing. **Whether a learned hop-selector can capture the oracle gap is the open
question, and it is now the only live thread in this line.**

## Interpretation boundary

126 questions, one encoder, one corpus. The oracle first hop uses the corpus's own evidence
annotation, so +0.1376 is the ceiling for a *single* expansion step and says nothing about what
several would buy. The store was given untrained random projections, matching Phase 31's setting;
training moved that measurement by +0.14 and closed none of a 0.26 gap, so it is not expected to
close 0.22 here, but it was not run.

# Phase 40: the store was never fixed-size, and storing in the function has a hard boundary

Two things were assumed rather than checked for twenty phases, and both are wrong.

## The associative store carried O(N) state all along

`TrainableMemory._score` is two lines:

```python
reads = F.normalize(keys @ state.T, dim=-1)
return reads @ values.T * self.log_scale.exp()
```

Scoring needs `values`, which is one row per stored document. The state that was supposed to be a
fixed-size associative matrix cannot be read without the list of everything written into it. So every
comparison from Phase 31 onward was an O(N) store losing to an O(N) cosine — not a smaller state
bought at some accuracy cost, just a worse reader at the same cost. That reframes those results: they
were never a trade.

The one mechanism here that *is* fixed-size is Phase 38's familiarity filter, and it is also the one
that came within 0.009 of its baseline. The pattern is consistent.

## A network stores by changing what it computes, and that has a dimension budget

The alternative is to put the memory in the function rather than in a list beside it. The minimal
version has a closed form and no hyperparameter: fit a linear map to pass the stored items through
unchanged, and the least-squares solution is the orthogonal projector onto their span,
`W = S(SᵀS)⁺Sᵀ`. Once `W` exists the items can be discarded — it is d-by-d whatever N was — and
reading is `Wq`, the part of the cue the stored material can account for.

**It has a boundary that is arithmetic, not empirical.** `rank(W) = min(N, d)`. Below the embedding
dimension the map is a genuine bottleneck; at or above it, the span is the whole space, `W` is the
identity, and the memory stores nothing at all. With 384-dimensional embeddings:

- eight sessions per episode: `W` is rank 8, a severe bottleneck
- 491 turns per episode, the granularity where an answer actually fits in one vector: `W = I`

So this form of storage is only a memory in the regime where it discards most of the input, and stops
being one exactly where there is enough material to be worth remembering.

## And the endpoint that would have measured it does not work

Testing `Wq` needs a task that does not rank a list, since ranking requires the list. The natural one
is to ask whether the read lands on the episode's own answer embedding, scored against all 300. The
gate says no:

| Reader | State at read | Top-1 |
| --- | --- | ---: |
| The bare question | none | **0.2367** |
| Question plus the annotated evidence session | O(N) | 0.2167 |
| The annotated evidence session | O(N) | 0.1000 |
| Best similarity-weighted list | O(N) | 0.0900 |
| `Wq` | O(1) | 0.0867 |

Chance is 0.0033, so the bare question at 0.2367 is doing real work — and that is the problem.
BGE-small places a short question and its short answer in the same neighbourhood, while a session
embedding averages ten thousand characters in which the answer is a sentence. Handing the reader the
*correct* evidence session makes it worse than handing it nothing. `endpoint_usable` is false and no
row in that table is readable, which is the check Phase 15 did not make about its own numbers.

The diagnosis generalises: at session granularity the answer is not recoverable from the embedding at
all. That is the same reason the generator endpoint failed in Phase 33, and it is a fact about the
representation rather than about any architecture built on top of it.

## What this leaves

Turn-granularity features were built to remove the dilution — `prepare_longmemeval_turns.py`, 52
questions whose gold answer can be located verbatim in an evidence turn, 25,552 turns, 491 per
question. At that granularity the answer does fit in one vector. It is also the granularity at which
`W` becomes the identity, so the closed-form version of storing-in-the-function has nothing to say
there, and a bottlenecked variant (a rank-r sketch) would be compression of the list rather than
removal of it: O(Nr) instead of O(Nd).

**List-free retrieval appears to be impossible in principle, not merely unachieved.** Ranking N items
requires N comparisons against something. The tasks a function-shaped memory can do without a list
are familiarity, which Phase 38 measured to within 0.009 of an O(N) reader, and generation, which
needs a decoder this project does not have. That is the boundary, and it is where this line of
questions ends rather than continues.

# Phase 41: the level was the wrong measurement

Every verdict in this project has been a level: does the mechanism beat the baseline at the load it
was measured at. Applied to backpropagation in 1988 that standard returns "rejected" — a three-layer
network lost to hand-crafted features on nearly every benchmark it was tried on. What eventually
separated the two was not the level but the slope: the network kept improving with more data and
compute while the baseline saturated, and that was visible as a trend long before it was visible as a
win.

The slope is measurable here, and it has been available all along. Phase 31 already reported one:
the store's gap to cosine grew from 0.16 at eight documents to 0.35 at a hundred and twenty-eight.
That is the wrong direction, and it is a stronger reason to abandon that design than any single-load
comparison was.

Phase 38's filter is the one mechanism whose slope had not been measured, and it is the one whose
state does not grow with the load. Turn granularity gives 491 items per question against the 48 it
was first measured on. Same paired design: one memory holds the turn that answers the question, one
holds a distractor in its place, both hold exactly `load` items.

| Load | max cosine | top-4 sum | top-16 sum | top-64 sum | **Filter, O(1)** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.9006 | 0.9647 | 0.9679 | 0.9679 | **0.9744** |
| 64 | 0.7821 | 0.9231 | 0.9679 | 0.9615 | **0.9679** |
| 128 | 0.7244 | 0.9006 | 0.9647 | 0.9808 | **0.9744** |
| 256 | 0.6442 | 0.8333 | 0.9359 | 0.9679 | **0.9679** |
| 491 | 0.5714 | 0.8214 | 0.9107 | 0.9821 | **0.9643** |

**The filter is almost flat.** Thirty times the load costs it 0.0101, from 0.9744 to 0.9643, while
maximum cosine falls from 0.9006 to 0.5714.

| Filter minus | At load 16 | At load 491 | Direction |
| --- | ---: | ---: | --- |
| max cosine | +0.0737 | **+0.3929** | improving |
| top-4 sum | +0.0096 | **+0.1429** | improving |
| top-16 sum | +0.0064 | **+0.0536** | improving |
| top-64 sum | +0.0064 | -0.0179 | worsening |

Against every reader that holds its budget fixed, the fixed-size filter's advantage *grows* with the
load. The single reader it loses to is the one whose budget grows with N — at 491 items `top-64`
sums a sixty-fourth of the haystack, and keeping that up means keeping a fixed fraction of everything
stored, which is the cost the filter exists to avoid.

## What this changes and what it does not

It does not overturn Phases 31, 34 or 39. Those measured a store that carried O(N) state anyway
(Phase 40), and their slope, where it was measured, ran the wrong way. Nothing here rescues
retrieval.

It does change what Phase 38 established. Reported as a level, the filter lost by 0.0093 and was
recorded as a failure against its pre-registered criterion. That verdict stands at that load. But the
criterion asked the wrong question of a mechanism whose whole claim is constant state, and the slope
says the comparison it lost was against a baseline that cannot be held to its budget as the load
grows.

## Interpretation boundary

52 questions, one encoder, three seeds, turn granularity from a crude string match of the gold answer
into an evidence turn — 48 of 100 sampled questions were dropped because no turn contained the answer
verbatim, and those may not be a random half. The loads run to 491 because that is one question's
haystack; nothing here says what happens at 10,000, which is the regime the argument is really about.
The top-64 row is the honest competitor and it is ahead; the claim is about how each side's cost
behaves as N grows, not that the filter is more accurate today.

# Phase 42: at the load the argument was about, the fixed-size memory wins

Phase 41 stopped at 491 items because that is one question's haystack. Pooling every question's turns
gives 25,552 to draw from, which is the regime the scaling claim was actually about. Each question
keeps all of its own turns in both memories, so the distractors that genuinely resemble the answer
are always present and only the distant ones scale.

| Load | max cosine | top-16 sum | top-64 sum | top-256 sum | **Filter, O(1)** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 491 | 0.6042 | 0.9167 | 0.9792 | 1.0000 | **1.0000** |
| 2,000 | 0.5673 | 0.8590 | 0.9487 | 0.9808 | 0.9744 |
| 8,000 | 0.5481 | 0.8237 | 0.8750 | 0.9519 | **0.9936** |
| 25,000 | 0.5385 | 0.7596 | 0.8333 | 0.8974 | **0.9744** |

Fifty times the load costs the filter 0.0256, and it is not monotone — 0.9936 at 8,000 against 0.9744
at 2,000 — so the honest reading is that it sits at about 0.98 throughout and wobbles by 0.02. Every
list reader falls. Maximum cosine goes to 0.5385, which is nearly chance. Even top-256, holding one
percent of everything stored, loses 0.10 across the range.

Paired over the same questions, at 25,000 items:

| Comparison | Difference | 95% interval |
| --- | ---: | --- |
| Filter minus max cosine | +0.4359 | [+0.3846, +0.4776] |
| Filter minus top-16 | +0.2147 | [+0.1474, +0.2821] |
| Filter minus top-64 | +0.1410 | [+0.0801, +0.2019] |
| Filter minus top-256 | **+0.0769** | [+0.0288, +0.1282] |

All four resolved. Every slope now runs in the filter's favour, including the one against top-256
that was flat at 491 and is +0.0769 at 25,000.

## What this is

It is the first result in this project that beats its baselines on a real task, and it took changing
the question from *level* to *slope*. Read at 48 items in Phase 38 the same mechanism lost by 0.0093
and was recorded as a failure against a pre-registered criterion. That criterion was not wrong about
what it measured; it measured a mechanism whose only claim is constant state at a load where constant
state is worth nothing.

## What it is not

It is not retrieval. The filter answers whether the memory contains something bearing on the cue, not
which item that is. Phases 31, 34 and 39 stand, and Phase 40 explains why: ranking N items needs N
comparisons, so no fixed-size state can do it.

## Interpretation boundary

Fifty-two questions means 0.08 is four of them. The intervals are resolved, and they are resolved on
fifty-two paired observations, not fifty-two thousand.

## The clutter objection, tested

The obvious threat was what scales. Each question's own turns stay in both memories, so everything
added comes from other conversations — semantically distant material, and a sum over cells might
simply absorb distant noise better than a top-k does. Filling instead with the turns *nearest* the
cue tests that directly.

Two versions of this test were wrong before one was right, and both errors showed up in the numbers
rather than in the reasoning. Taking the stand-in for the answering turn from the end of the fill made
it farther away as the fill grew, so load and difficulty moved together and every reader improved
with load. Making the stand-in the cue's nearest neighbour sent every win rate below 0.5, because it
asked whether the answer is the nearest turn rather than whether the memory holds it. The stand-in has
to be drawn at random and independently of the fill, so that only the clutter varies.

| Load | Distant clutter: top-256 / filter | Near clutter: top-256 / filter |
| ---: | ---: | ---: |
| 491 | 0.9861 / 0.9722 | 0.9861 / 0.9861 |
| 2,000 | 0.9872 / 0.9744 | 0.9006 / 0.9808 |
| 8,000 | 0.9455 / 0.9872 | 0.9038 / 0.9744 |
| 25,000 | 0.9038 / **0.9872** | 0.9038 / **0.9936** |

At 25,000 items the filter leads top-256 by +0.0833 [+0.0385, +0.1346] against distant clutter and by
+0.0897 [+0.0417, +0.1442] against near clutter; both resolved, and the near case is if anything
slightly better. **The objection does not hold.** The advantage is not an artefact of easy noise.

One nuance the table makes visible. Under near fill the list readers are load-invariant by
construction — the k nearest turns are already present at the smallest load, so adding farther ones
cannot change a top-k sum — and their values are identical from 2,000 onward. So against near clutter
the filter's advantage is a constant +0.09 rather than a growing one; the growth in the earlier table
belongs to the distant-clutter case.

# Phase 43: the winning mechanism is worthless at the job it was built for

Every number in this line, including Phase 42's, came from a paired comparison: two memories, one
cue, which scores higher. A memory that simply receives data and lets old material surface has no
such comparison. Items arrive, one number comes out, and a single global threshold decides whether
anything is emitted. The fraction of arrivals worth emitting is small.

The stream is the corpus in random order behind a sliding window. Positives and negatives are the
same texts: each question fires once at a point where the turn answering it is inside the window, and
many times at points where it is not. Base rate one percent.

An earlier version used ordinary turns as the negatives and every score inverted — AUC 0.18 — because
a conversational turn arriving beside its own neighbours is far more familiar than a short standalone
question. That measured text genre, not memory. Matched, the result is this:

| Window | Filter AUC | Max cosine AUC | Max cosine precision at best F1 |
| ---: | ---: | ---: | ---: |
| 200 | **0.5007** | 0.7093 | 0.2694 |
| 1,000 | **0.4895** | 0.6078 | 0.1001 |
| 4,000 | **0.4782** | 0.5277 | 0.0358 |
| 8,000 | **0.4798** | 0.5221 | 0.0378 |

**The filter is at chance at every window size.** The mechanism that beat every list reader at 25,000
items cannot tell, at any scale, whether the answering turn is in the memory. To catch all of them,
both readers need a threshold that admits about 97 false alarms per true positive.

## Why, and what it means for the design

The familiarity score is a sum over everything stored. Its absolute value is set by how many items
are in the window and what they are, and that variation swamps the contribution of one item among
even two hundred. A paired comparison holds all of it constant and isolates the one item, which is
exactly why Phase 42 read 0.97 and this reads 0.48. Both are correct measurements of different
quantities.

Maximum cosine survives at small windows because a maximum is robust to what else is present — the
right item can set a new one. It dies at large windows for the reason Phase 39 also found: with
enough stored material something is always nearly as close as the thing you want.

**A sum is the right statistic for a paired judgement and the wrong one for an absolute judgement.**
The accumulation that made Phase 38 and Phase 42 work is the same property that makes the score
incomparable across different memory contents. This is not a tuning problem; it is what the statistic
is.

## What this retracts and what it leaves

It does not retract Phase 42. Paired, at 25,000 items, against near and distant clutter, the fixed
size filter beats every list reader, and that stands as a statement about paired discrimination.

It does retract the practical reading of it. A memory that receives data and surfaces things
unprompted needs an absolute judgement, and on that the filter is worthless while a plain maximum is
merely bad. Everything positive this line has found was measured in a setting the intended system
does not have.

## The obvious next thing, and why it is not done here

The fix implied by the diagnosis is to remove the dependence on window content: score against a
running estimate of what familiarity looks like for this memory right now, rather than against a fixed
number. A z-score over recent arrivals costs nothing and is the standard move. Whether it recovers
anything is untested, and it should be tested before any of this is built on.

## Interpretation boundary

Fifty-two questions, 33 of which could be placed in the stream, one encoder, three seeds. The
positives are questions rather than ordinary turns, which is still not the intended cue — a real
system is triggered by incoming conversation, not by a query — and the corpus carries no annotation
of what *should* have surfaced for an ordinary turn. That label does not exist in any dataset here,
and without it the intended task cannot be scored at all, only approximated as it is above.

# Phase 44: the answer key was in the data the whole time

Every evaluation in this project has depended on a human annotation — which session holds the answer,
which question is unanswerable — and there are 52 of them. That is why Phase 43 could only
approximate the task it wanted to measure, and why Phase 33 could retract a whole endpoint on the
strength of 72 episodes.

Deep learning did not scale on annotations. It scaled on targets the data manufactures for itself,
one per position, and a memory has the same thing available: at time t what should surface is
whatever turns out to matter at t+1, and the stream says what that was. No labelling, a target at
every position, and continuous rather than a binary "should fire".

Using it forces one correction. Phase 43 shuffled the stream, which destroys the only structure this
objective runs on — in a random order nothing predicts anything. Turns are streamed here in their
true order.

At each scored position the memory holds every earlier turn, each reader surfaces one of them, and
the surfaced item then has to pick the real future out of a hundred candidate futures drawn from
elsewhere in the corpus. 2,080 positions, horizon five turns, three seeds:

| Reader | Top-1 | Top-10 | **Top-1, positions needing something old** |
| --- | ---: | ---: | ---: |
| Random | 0.0210 | 0.1152 | 0.0076 |
| Recency | 0.5580 | 0.7530 | 0.1606 |
| **Association (cosine to the current turn)** | 0.5564 | 0.7915 | **0.2145** |
| Oracle, the best item actually in memory | 0.6641 | 0.9149 | 0.3476 |

Over all positions association and recency are indistinguishable, 0.5564 against 0.5580, which is
what a conversation should look like: the near future is mostly the present topic and the last turn
already carries it.

The interesting positions separate themselves without any annotation. Where even the oracle's best
item sits far back, recency has nothing to offer and association does:

| "Old" means at least | Association minus recency | 95% interval | Headroom to the oracle |
| ---: | ---: | --- | ---: |
| 16 turns | +0.0563 | [+0.0330, +0.0811] | +0.1343 |
| 64 turns | +0.0634 | [+0.0336, +0.0933] | +0.1331 |
| 128 turns | +0.0772 | [+0.0415, +0.1128] | +0.1369 |

All resolved, and the advantage grows the further back the useful item is. The headroom is stable at
about 0.134 whichever cutoff is used — more than double the gain association already captures.

## Why this is different from everything before it

Phase 19 measured the headroom above the learned write gate at 0.0149 and concluded, correctly, that
further work on it was unjustified. Phase 31 found the store below its own baseline. Every positive
result since has either been a proxy (Phase 30), a paired comparison the intended system cannot make
(Phases 38, 42), or a mechanism at chance where it matters (Phase 43).

This is the first measurement in the project with both properties a learnable problem needs: a target
that exists at every position rather than at fifty-two of them, and a gap between the simple
heuristic and the ceiling — 0.2145 against 0.3476 — large enough to be worth closing.

It also says what the memory is for, in a way no annotation did. Association earns nothing on the
bulk of a conversation, where recency is already right. It earns its keep precisely on the positions
where the future returns to something old, those positions are a fifth of the stream, and they can be
found without anyone labelling them.

## Interpretation boundary

The target is a five-turn future centroid, which is a proxy for usefulness and not usefulness itself:
a memory that predicts what gets said next is not necessarily a memory that helps. The foils come
from the same corpus, so the hundred-way choice is easier than an open one. "Old" is defined by where
the oracle's pick sits, which is a property of this encoder's geometry rather than of the
conversation. And nothing is trained here — association is plain cosine, and whether a learned reader
captures any of the 0.134 is exactly the open question this makes askable.

# Phase 45: trained against the future, and out of data before out of ideas

Phase 44's target gives a gradient at every position, so the read can be trained rather than
hand-built. `PredictiveRead` scores memory bilinearly, `m^T (I + UV^T) q`, initialised at the
identity so it starts exactly at the cosine baseline; selection is a softmax over memory, making the
whole read differentiable; the loss is InfoNCE against the same hundred foils the evaluation uses.
Conversations are held out whole.

One implementation detail decided the first run. With the softmax temperature at one, cosines are
too narrow a range to select anything and the read returned the mean of memory — 0.0141 against a
cosine baseline of 0.2958. Starting the temperature at 0.02 makes the untrained model equal the
baseline it is supposed to begin at.

Five seeds, 3,000 steps, held out by conversation:

| Reader | Top-1 | Top-1, old positions | Per-seed spread, old |
| --- | ---: | ---: | ---: |
| Recency | 0.5719 | 0.1652 | 0.1248 |
| Hard pick, cosine | 0.5603 | 0.1873 | 0.1112 |
| Soft blend, untrained | 0.5722 | 0.2109 | 0.0880 |
| **Soft blend, trained** | 0.5872 | **0.2269** | 0.0918 |
| Best single item in memory | 0.6706 | 0.3402 | 0.1298 |

Decomposed: blending several memory items instead of taking one is worth +0.0236, and training on top
of that is worth +0.0159.

## The verdict is that the experiment cannot decide

**Both effects are smaller than the seed-to-seed spread.** With five seeds and a spread near 0.09 the
standard error is about 0.02, so training's +0.0159 is roughly one of them. Nothing here is
established. Reporting this as "training closed 26% of the gap to the oracle", which the first draft
of the script printed, would have been the same error as Phase 15's 72-episode gate and Phase 25's
mechanism table: a point estimate quoted past what the sample supports.

The cause is not subtle. There are 52 conversations because `prepare_longmemeval_turns.py` drops any
question whose gold answer cannot be found verbatim in an evidence turn, and that discarded 48 of the
100 sampled. Sixteen conversations in the held-out split is too few for a 0.02 effect.

The separation the decomposition draws is worth keeping regardless of power, because it is the
distinction this project has misread twice — Phase 29 mistook a hand-set constant for a dead
component, Phase 36 mistook a metric for a mechanism. Here the parameterisation change, blending
rather than picking, contributes more than the learning does, and a table reporting only "trained
against baseline" would have hidden that.

## Also: the oracle is not a ceiling for this read

Phase 44's "oracle" is the best *single* item in memory. A blended read is not restricted to one
item and can exceed it, which one short run did. The 0.134 headroom quoted in Phase 44 bounds hard
selection, not what a soft read can reach.

## Interpretation boundary

Fifty-two conversations, five seeds, one encoder, and an effect size around 0.02. The correct
statement is that the direction is consistent — trained above untrained above hard pick above
recency, in that order, at every cutoff tried — and that the sample cannot resolve any single step of
it. Re-running on the full 500 questions is the obvious next move and it is a data preparation job,
not a modelling one.

## Resolved, on 233 conversations

Loosening nothing and simply preparing all 500 questions yields 233 conversations and 115,007 turns,
four and a half times the data. Two things change, in opposite directions.

**The effects shrink.** Blending falls from +0.0236 to +0.0093 and training from +0.0159 to +0.0086,
which is what inflated small-sample estimates do. The per-seed spread falls with them, from about
0.09 to about 0.043.

**And they resolve**, because the comparison was never really between five seeds. Every reader saw
the same positions, so the difference is paired, and on 3,404 old positions:

| Comparison | Difference | 95% interval |
| --- | ---: | --- |
| Trained minus untrained blend | **+0.0085** | [+0.0006, +0.0165] |
| Blend minus hard pick | **+0.0094** | [+0.0026, +0.0162] |
| Hard pick minus recency | **+0.0282** | [+0.0176, +0.0391] |
| Trained minus hard pick | **+0.0179** | [+0.0091, +0.0267] |

All four resolved. Reading seed spreads rather than pairing the positions had hidden that, in the
same way the paired form in Phase 33 revealed what a comparison of means had hidden.

| Reader | Top-1 | Top-1, old positions |
| --- | ---: | ---: |
| Recency | 0.5904 | 0.1803 |
| Hard pick, cosine | 0.5849 | 0.2085 |
| Soft blend, untrained | 0.5961 | 0.2178 |
| **Soft blend, trained** | 0.6037 | **0.2264** |
| Best single item in memory | 0.6889 | 0.3697 |

**Training helps, and it is the smallest of the three steps.** Association over recency is worth
+0.028, blending over picking +0.009, and learning on top of both +0.009 — with the last interval
only just excluding zero. A report of "trained against baseline", +0.0179, would have been true and
would have credited learning with twice what it did.

0.143 of the distance to the best single item in memory is still unclosed, so the target has plenty
left in it; what is exhausted is this parameterisation, not the problem.

# Phase 46: more context helps, and telling it the age makes it lazy

Two ways to widen the read, both initialised to leave the model exactly where it was without them.
`context` lets the cue be a learned blend of the last few turns rather than only the current one,
starting as a near one-hot on the current turn. `age` gives the scorer an explicit handle on how far
back each memory item sits, starting at a coefficient of zero.

Five seeds, 3,000 steps, 233 conversations, 3,404 paired old positions:

| Read | Top-1, all | Top-1, old | Trained minus hard pick, old |
| --- | ---: | ---: | --- |
| Base, current turn only | 0.6037 | 0.2264 | +0.0179 [+0.0091, +0.0267] |
| **Context 8** | 0.6042 | **0.2341** | **+0.0256** [+0.0165, +0.0341] |
| Age term | 0.6121 | 0.2212 | +0.0126 [+0.0038, +0.0214] |
| Context 8 and age | 0.6123 | 0.2209 | +0.0123 [+0.0029, +0.0217] |
| Context 32 and age | 0.6141 | 0.2247 | +0.0162 [+0.0071, +0.0250] |
| Best single item | 0.6889 | 0.3697 | — |

**The age term splits the two metrics in opposite directions.** It raises overall top-1, 0.6037 to
0.6121, and lowers it on the positions that need an old item, 0.2264 to 0.2212. Adding context on top
does not rescue that: 0.2341 without age becomes 0.2209 with it.

The reading is straightforward. Recency is right on the bulk of a conversation, so a read handed an
explicit age handle learns to lean on it, which is the correct move for the average position and the
wrong one exactly where a memory is needed. The term does not add a capability, it offers a shortcut,
and the model takes it. Nothing about this was visible in the aggregate number, which improved.

Context without age is the best configuration found, 0.2341 on old positions against 0.2264 for the
base. That comparison is between separate runs rather than paired, and the two intervals overlap, so
it is suggestive rather than resolved; the age result is a consistent split across two independent
pairs of runs and is firmer.

## Interpretation boundary

The gap to the best single item in memory is 0.1356 after all of this, essentially unchanged from
Phase 45's 0.143. Widening the read by these two routes moved it by about 0.008, which says the
limitation is not the cue or the recency handle. What has not been tried is giving the read more than
one output — every configuration here surfaces one blended vector, and the target it is scored
against is a five-turn centroid that may simply not be reachable from any single read.

# Phase 47: the read is nowhere near its limit; the cue is

A single softmax is either sharp, returning one stored item, or flat, returning the mean of
everything. The target is a centroid of five turns, so several distinct items may be needed at once.
Heads give the read that, with the output still one vector so every baseline stays comparable.

| Heads, context 8 | Top-1, all | Top-1, old | Trained minus hard pick |
| ---: | ---: | ---: | --- |
| 1 | 0.6042 | 0.2341 | +0.0256 [+0.0165, +0.0341] |
| 2 | 0.6055 | 0.2321 | +0.0235 [+0.0147, +0.0326] |
| 4 | 0.6123 | 0.2364 | +0.0279 [+0.0188, +0.0370] |
| 8 | 0.6129 | 0.2358 | +0.0273 [+0.0179, +0.0364] |
| 16 | 0.6140 | **0.2391** | +0.0306 [+0.0217, +0.0397] |

Sixteen times the heads buys +0.005 on the positions that matter, and every interval overlaps every
other. Three routes to a wider read have now been tried and all three are worth about nothing there:
cue width +0.008, an age handle negative, output multiplicity +0.005.

## What the read can do when it is told what to look for

Phase 19 settled an equivalent question by fitting an unconstrained model to bound what was
extractable. The equivalent here is one flag: cue the same read with the target itself. That does not
produce a usable system — it needs the future — but it bounds what this read form can reach when the
information it lacks is handed to it.

| Condition, old positions | Top-1 |
| --- | ---: |
| Real cue, sixteen heads, context 8 | 0.2391 |
| Best single item in memory | 0.3697 |
| **Same read, cued by the future** | **0.6581** |

The read form is not the constraint. Given the right cue it reaches 0.658, nearly three times what it
manages with the real one and well past the single-item oracle — which also confirms that blending
beats picking when there is something to steer it with. Every point of the gap is the cue.

**This is Phase 19's finding in a new setting.** There, the write gate could not exceed about 0.54
because what to keep is not determined at write time. Here, the read cannot exceed about 0.24 because
which stored item will matter is not determined by the current turn. Same shape, opposite end of the
system, reached by a different route and a different target.

## What that leaves

The honest reading is that this line is now at the same wall from both sides, and the wall is the
representation rather than the mechanism. The current turn, encoded by BGE-small, does not carry
which old thing is about to become relevant. Three things could move it and none is a modelling
choice: a cue that spans more than turns (the conversation's state rather than its last utterances), an
encoder that was trained for this rather than for sentence similarity, or a target closer to
usefulness than a five-turn centroid.

## Interpretation boundary

The future-cued number is a ceiling for this read form and this target, not a claim about what any
system could do. It uses the answer to find the answer. Its only role is to separate "the mechanism
is too weak" from "the input does not say", and it says the second — which is the question three
phases of architecture work could not settle.

# Phase 48: more capacity on the same embedding makes it worse

Phase 47 located the limit in the cue, which points at the encoder. Fine-tuning BGE-small needs the
raw text, which these artifacts deliberately do not keep, and it needs hours. There is a cheaper test
that decides whether those hours are worth spending: the cue transform so far is low-rank and
*linear*, so adding a non-linear residual on both the cue and the memory side asks whether what
predicts the future is present in the frozen embedding but not linearly available. Both residuals end
in a zero-initialised layer, so the model still starts exactly at the cosine baseline.

Sixteen heads, context 8, 6,000 steps, five seeds:

| Non-linear residual | Top-1, all | Top-1, old | Trained minus hard pick |
| ---: | ---: | ---: | --- |
| none (linear) | 0.6150 | **0.2397** | **+0.0311** [+0.0214, +0.0408] |
| hidden 256 | 0.5775 | 0.2082 | -0.0003 [-0.0106, +0.0100] |
| hidden 1024 | 0.4881 | 0.1748 | **-0.0338** [-0.0449, -0.0217] |

Capacity makes it monotonically worse, and at hidden 1024 the trained read is *below* an untrained
hard pick by a resolved margin. This is overfitting, and it answers the question the test was built
to answer: there is nothing extra in the frozen embedding for more capacity to reach. What extra
capacity finds is the training conversations.

## What that says about fine-tuning the encoder

It does not say fine-tuning would fail. It says the obstacle is not representational capacity but
data, and fine-tuning adds far more capacity than these residuals did. 233 conversations produce
115,007 turns, which sounds like a large sample and is not one: positions inside a conversation share
its whole history, so the effective sample size is nearer 233 than 115,007. An encoder trained on
that will find the same thing hidden 1024 found.

This is the shape of nearly every wall in this project. 52 questions in Phase 45, 30 abstention
questions in Phase 38, 233 conversations here. The mechanisms have been cheap to test and the
measurements have been the expensive part, and what ran out each time was data rather than ideas.

## Interpretation boundary

One objective, one encoder, one corpus, and no attempt at the regularisation that might let a larger
model survive this sample — weight decay is at its default and nothing was tuned per configuration.
A better-regularised non-linear read could plausibly land between 0.2397 and 0.2082. What the result
rules out is the hope that capacity alone was the missing piece, which is what Phase 47 left open.

# Phase 49: the private corpus is smaller than it looked, and so was the sample behind Track A

Phase 48 ends at a data wall, and the obvious place to go is the private Claude corpus: real human
conversation, and the self-supervised target needs no labels so it applies to any of it. Counted, it
goes the wrong way.

| Corpus | Conversations | Turns |
| --- | ---: | ---: |
| LongMemEval, turn granularity | 233 | **115,007** |
| Local Claude logs | **50** | **1,467** |

Eighty times less, not more. The move does not work, and the reason it looked like it would is worth
recording, because it applies to this project's one surviving positive result.

## An episode count is not a sample size

Track A reported 775 training and 574 held-out episodes, which reads as a comfortable sample. Those
are episodes, drawn many per conversation and many conversations per project, and the held-out split
is **by project**. Re-running the same extraction on today's logs:

- 340 episodes
- from **16 projects**
- median 12 episodes per project, maximum 80
- split 314 train / 26 eval

So the independent unit is the project, and there are on the order of ten to twenty of them, not 574.
The standard deviations Track A reports — 0.5920 +/- 0.0099 for the embedding arm, 0.4056 +/- 0.0128
for the length arm — are spreads over five seeds on a fixed split. They describe initialisation
noise. They do not describe what happens if a different set of projects is held out, and no number in
Track A does.

This does not overturn Phase 16. Its margin is 0.1864 retention and 0.3477 top-1, which is far larger
than anything plausible from resampling a dozen projects, and the length control it rests on is a
within-sample comparison that project variation affects on both sides. What it does mean is that the
error bars are decorative, that the result rests on roughly a dozen independent conversations' worth
of behaviour, and that the phrase "574 held-out episodes" should not be quoted as a sample size.

## Where this leaves the data problem

Nothing local scales. The self-supervised target from Phase 44 is the one asset that transfers, since
it needs no annotation and applies to any corpus of long conversations. That makes the missing
ingredient a public multi-session dialogue corpus rather than a labelling effort — a download, not a
project. Whether any of the findings from Phase 44 onward survive at ten times the conversations is
the question every one of them now waits on.

# Phase 50: counting the wrong stream, and a transfer test that passes

This is a memory meant to improve a model's behaviour, and every count in Phase 49 measured human
conversation. The stream that matters is the model's own actions. Re-counting the same logs that way:

| Stream | Sessions | Events |
| --- | ---: | ---: |
| User turns only | 50 | 1,467 |
| All conversational turns | 228 | 6,730 |
| **Tool calls** | 219 | **15,125** |

The 249 MB dismissed in Phase 49 as "tool output rather than conversational text" was the data. That
dismissal applied a criterion inherited from the old task — the same mistake as the `require_answer`
filter removed minutes earlier, made twice in the same hour.

Three things keep it from being the answer to the data problem, though. It is still seven times less
than LongMemEval's 115,007. Three sessions hold 5,118 of the 15,125 calls, almost certainly this
project's own, so it is far more concentrated than 500 distinct personas. And the free label is thin:
441 errors, overwhelmingly in those same sessions, and `is_error` catches only hard failures while
the interesting case — an action that succeeded and should not have been taken — carries no label at
all.

So it is used for what small data is good for. Training on it would repeat Phase 48.

## The transfer test

`prepare_claude_tool_stream.py` encodes each action as one line naming the tool and what it acted on,
persisting embeddings only. 34 sessions, 12,495 actions, 298 failures. The read trained on
LongMemEval is then evaluated there without adaptation, on 10,513 paired positions:

| Reader | Top-1 | Top-1, old |
| --- | ---: | ---: |
| Recency | 0.3748 | 0.2946 |
| Hard pick, cosine | 0.3885 | 0.3346 |
| Soft blend, untrained | 0.3871 | 0.3323 |
| **Soft blend, trained on LongMemEval** | 0.3955 | **0.3407** |
| Best single item in memory | 0.7390 | **0.7075** |

| Comparison | Difference | 95% interval |
| --- | ---: | --- |
| Trained minus untrained blend | **+0.0084** | [+0.0039, +0.0128] |
| Blend minus hard pick | -0.0023 | [-0.0074, +0.0028] |
| Trained minus hard pick | **+0.0061** | [+0.0008, +0.0113] |

**It transfers.** A read trained on human conversations about hamsters and restaurants beats both an
untrained blend and a plain cosine pick on a stream of shell commands and file edits, by a resolved
margin. That is the evidence Phase 44's target was missing: predicting a five-turn future centroid is
a proxy, and a proxy that transfers across corpora this different is measuring something that is not
an artefact of either.

It also separates two things LongMemEval had confounded. There, blending and training both helped and
were hard to tell apart. Here blending alone buys nothing, -0.0023 and unresolved, and the entire
gain is the learned transform.

## The number that stands out

The best single item in memory reaches **0.7075** on the action stream against 0.3697 on
conversation. The right earlier action is far more identifiable from what follows than the right
earlier turn is — actions repeat, files recur, the same command comes back. And the trained read
captures 0.3407 of it.

So the headroom here is 0.367, nearly three times conversation's 0.133, on a stream where the
structure is more regular and the target is closer to a real outcome. Everything this project has
been measuring on conversation has a richer version of itself sitting in the logs it was generating
while running.

## Interpretation boundary

34 sessions with three dominating, one encoder, and an action rendered as its tool name plus one
field. The transfer margin is +0.0084, which is resolved and small; what it establishes is direction,
not size. The 0.7075 ceiling is measured the same way as everywhere else — it uses the future to pick
the item — so it bounds what a perfect reader of this stream could do and says nothing about whether
one is reachable.

# Phase 51: twice the data, and the prediction that came with it fails

Phase 48 found non-linear capacity making the read worse and concluded the obstacle was data rather
than representation. That conclusion carries a prediction: at more data the non-linear model should
close on the linear one. The `require_answer` filter was the only thing keeping the corpus at 233
conversations, and the self-supervised target never needed it, so removing it gives 470 conversations
and 231,595 turns — twice everything.

| Read | 233 conversations | 470 conversations |
| --- | ---: | ---: |
| Linear | 0.2397 | **0.2497** |
| Non-linear, hidden 256 | 0.2082 | 0.2145 |

| Read, 470 conversations | Trained minus hard pick |
| --- | --- |
| Linear | **+0.0346** [+0.0280, +0.0412], resolved, 7,202 positions |
| Non-linear, hidden 256 | -0.0007 [-0.0072, +0.0061], not resolved |

**Data helps the linear read.** 0.2397 to 0.2497, with the margin over an untrained hard pick growing
from +0.0311 to +0.0346 and its interval tightening. That is the strongest form of this result so far.

**It does not rescue the non-linear one.** Hidden 256 gains 0.006 and stays 0.035 below linear, still
unable to beat a hard pick. Doubling changed the ordering not at all.

So Phase 48's diagnosis was half right and the half it got wrong is the part it predicted. More data
does help; it does not make capacity usable. Either far more than twice is needed, or the
non-linearity hurts for a reason that is not sample size.

There is a candidate for the second, and it is untested. The residual transforms both the cue and the
memory before scoring, but the read still outputs the *original* memory vectors, because that is the
space the target lives in. A non-linear scoring space free to reshape similarity can therefore rank
highly an item whose untransformed embedding does not match the future at all — the space it selects
in and the space it is graded in come apart. The linear version cannot drift as far. If that is the
cause, the fix is to score and output in the same space rather than to collect more data, which is a
different project from the one Phase 48 pointed at.

## Interpretation boundary

One doubling. A factor of two is a weak test of a data hypothesis, and nothing here rules out a
non-linear read working at ten times or a hundred. What it does rule out is the specific expectation
Phase 48 set, which is worth recording because that phase was about to be used to justify fine-tuning
an encoder on the strength of it.

# Phase 52: it was the learning rate, and the tuned model loses what mattered

Phase 51 proposed a geometric reason for the non-linear read's failure — scoring space and output
space coming apart. Phase 48's own boundary section had already named the mundane alternative and it
was never tested: the residual carries sixteen times the parameters of the linear read, and every run
gave both the same learning rate and no decay.

The mundane alternative wins. 470 conversations, three seeds:

| Configuration | Top-1, old | Trained minus hard pick |
| --- | ---: | --- |
| Linear reference | 0.2542 | +0.0312 [+0.0231, +0.0391] |
| Hidden 256, lr 1e-3 (every earlier run) | 0.2276 | +0.0046, **not resolved** |
| Hidden 256, lr 3e-4 | 0.2429 | +0.0199 |
| **Hidden 256, lr 1e-4** | **0.2598** | **+0.0367** [+0.0289, +0.0448] |
| Hidden 256, lr 1e-4, decay 1e-1 | 0.2579 | +0.0349 |

Learning rate is the whole effect and it is monotone; weight decay does essentially nothing. "Capacity
makes it worse", the conclusion of Phase 48 and the premise of Phase 51's geometric story, was a
sixteen-times-larger model run at the smaller model's rate.

**This is the fourth time in this session a configuration difference has been reported as a finding.**
Phase 29 read hand-set constants as a dead component. Phase 36 read a metric containing its own write
as a mechanism ranking. Phase 45's first draft read a softmax temperature as a training effect. Now a
learning rate as a statement about capacity. The pattern is consistent enough to be a rule: before
believing that a component does not work, check that it was given the setting it needs.

## And the tuned model loses the property that mattered

Confirmed at five seeds on 7,202 paired positions, then carried to the action stream of Phase 50
without adaptation:

| Read | LongMemEval, old | Transfer to the action stream |
| --- | ---: | --- |
| Linear | 0.2466, +0.0314 [+0.0250, +0.0378] | **+0.0061** [+0.0008, +0.0113], resolved |
| Non-linear, tuned | **0.2512**, +0.0360 [+0.0297, +0.0425] | +0.0011 [-0.0044, +0.0065], **not resolved** |

In-domain the tuned non-linear read is ahead, though the two runs are not paired against each other
and their intervals overlap, so that ordering is suggestive rather than settled. Out of domain it
gives up exactly what made the linear read worth having. The extra capacity buys corpus-specific
structure.

So Phase 48's instinct was right about the nature of the problem and wrong about every measurement it
used to support it. The non-linear read does overfit — not in a way that shows up as a worse held-out
number on the corpus it trained on, which is what that phase looked for, but as a collapse the moment
the corpus changes. Only a transfer test could see it, and this project did not have one until
Phase 50.

The linear read stays, now for a measured reason rather than an assumed one.

## Interpretation boundary

The learning rate sweep is five points on one architecture at one corpus size; 1e-4 is simply the
best of what was tried and may not be the best available. The transfer corpus is 34 sessions with
three dominating, so "does not transfer" here means "does not transfer to this", and a second transfer
target would make the claim much stronger. Nothing about the in-domain ordering between linear and
tuned non-linear is resolved.

# Phase 53: the transfer is not three sessions

Phase 52's boundary named the obvious way its transfer result could be hollow: three sessions hold a
third of the action stream, and those three are this project's own. A transfer claim that only holds
with them in is a claim about one afternoon of work.

Dropping them, and then more, with the linear read trained on all 470 conversations:

| Transfer corpus | Trained minus hard pick | Positions |
| --- | --- | ---: |
| All 34 sessions | **+0.0116** [+0.0060, +0.0171] | 10,513 |
| Without the largest 3 | **+0.0082** [+0.0022, +0.0141] | 9,144 |
| Without the largest 6 | **+0.0093** [+0.0028, +0.0156] | 7,732 |
| Without the largest 10 | **+0.0084** [+0.0015, +0.0154] | 6,086 |

All four resolved. Removing the sessions that could have produced the effect does not remove the
effect; it shrinks it by about a quarter and leaves it clear of zero. The read trained on
conversations about restaurants and pets improves action selection on other people's shell sessions,
and the sessions this project generated are not what is carrying it.

## Training data improves transfer, not just fit

Phase 50 measured this transfer at +0.0061 with the read trained on 233 conversations for 3,000
steps. The same measurement on 470 conversations and 6,000 steps gives **+0.0116**, nearly double.
More training data does not only fit the training corpus better; it generalises further out of
domain. That is the opposite behaviour to the non-linear read of Phase 52, which gained in-domain and
lost the transfer entirely, and it is the clearest evidence so far that the linear read is learning
something about memory rather than about LongMemEval.

## Where this leaves the one working result

| Property | Status |
| --- | --- |
| Beats an untrained hard pick in-domain | +0.0314 [+0.0250, +0.0378], 7,202 positions |
| Transfers to a different corpus | +0.0116 [+0.0060, +0.0171], 10,513 positions |
| Survives removing the dominant transfer sessions | +0.0084 at ten dropped, still resolved |
| Improves with training data, in-domain and out | 0.2397 to 0.2497; +0.0061 to +0.0116 |
| Distance to the ceiling on the action stream | 0.3462 against 0.7075 |

It is small in every one of those rows. What it is not is fragile, and none of the other twenty
mechanisms measured in this project reached even the second row.

## Interpretation boundary

The transfer target is still one corpus of one person's sessions, and dropping large sessions removes
volume rather than adding independence — the remaining sessions come from the same user and the same
tools. A genuinely independent target, another user's logs or another agent's traces, is what would
turn this from "not an artefact of three sessions" into "transfers". The ceiling row is the one worth
staring at: 0.3462 against 0.7075 means that whatever is working here is capturing under half of what
is available on the stream this system is actually for.

# Phase 54: an independent target, and the transfer does not survive it

Phase 53 asked for another agent's traces. There are 1,118 Codex session files on this machine, a
different agent with different tool names and a different call format — `function_call` with JSON
argument strings rather than Claude's nested content blocks. Two hundred of them give 44,880 actions,
three and a half times the Claude stream.

| Transfer target | Trained minus hard pick | Positions |
| --- | --- | ---: |
| Claude action stream | **+0.0116** [+0.0060, +0.0171] | 10,513 |
| **Codex, linear read** | **-0.0230** [-0.0243, -0.0218] | 79,680 |
| Codex, without the largest 10 sessions | -0.0195 [-0.0207, -0.0183] | 73,769 |
| Codex, tuned non-linear read | -0.0258 [-0.0272, -0.0245] | 79,680 |

**All three negative, all resolved, on eight times the positions the positive result had.** On a
genuinely different agent the trained read is worse than doing nothing, and the tuned non-linear read
is worse still.

## What Phase 53 should have said

That phase concluded the linear read "learned something about memory rather than about LongMemEval",
on the strength of a transfer to the Claude action stream. That transfer was to a near domain: the
same user, the same tools, the same names — `Bash`, `Edit`, `Read` — so the action embeddings sit in
a space overlapping the one the read was trained in. Codex shares none of that.

The corrected statement is narrower and duller: the read learned something that survives a related
corpus and not an unrelated one. Every row of the summary table in Phase 53 stands as measured; the
sentence built on top of them does not.

## Why it might break this way

Codex streams are far more predictable than either corpus measured before. A hard cosine pick reaches
about 0.744 there against 0.335 on the Claude stream, and the ceiling is 0.8146 against 0.7075 — the
headroom is 0.09, not 0.36. Codex actions repeat heavily, so plain similarity is already close to the
best available, and a learned perturbation trained on conversational structure has almost nothing to
add and plenty to disturb. Where the baseline is near its ceiling, any transfer that is not exactly
right is a loss.

That is a testable story rather than a settled one, and the obvious check is whether the loss
concentrates on the positions where cosine was already correct.

## Interpretation boundary

Codex failure flags are read off output text rather than a structured field, which affects the
descriptive failure count and nothing in the table above. Both transfer corpora are still one
person's logs on one machine; a second *user* remains untested and would bear on a different question
than a second agent does. The negative result is large in confidence and small in size — two
percentage points — so what it rules out is the claim that this read transfers, not the possibility
that some read does.

# Phase 55: a memory is not a pre-learned system, and it gains before it knows anything

Every read in this project was fitted on one corpus, frozen, and carried elsewhere. Phase 54 measured
that carrying and found it fails on a different agent. The right reading of that is not that the read
is weak but that the question was wrong: a memory adapts while it runs, on the material it is
actually seeing, and asking whether a frozen transform generalises tests a property memory is not
supposed to have.

So `eval_online_memory.py` removes both training corpus and transfer. Each stream gets its own read,
starting at plain cosine, walking forward. Every position is scored **before** the update taken at it,
so nothing the read is graded on has touched its weights. The baseline is the same hard cosine pick
on the identical positions.

## It gains immediately, then destroys itself

Claude action stream, five seeds, no pre-training:

| Updates seen | Positions | Static | Online | Difference | 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| 0–25 | 4,250 | 0.4111 | 0.4252 | **+0.0141** | [+0.0059, +0.0219] |
| 25–50 | 4,015 | 0.4065 | 0.4276 | **+0.0212** | [+0.0100, +0.0329] |
| 50–100 | 6,345 | 0.3549 | 0.3556 | +0.0006 | not resolved |
| 100–200 | 10,370 | 0.4014 | 0.3324 | -0.0689 | [-0.0784, -0.0593] |
| 200–400 | 9,630 | 0.4207 | 0.2955 | **-0.1251** | [-0.1354, -0.1152] |

**A memory that knew nothing at the start of the stream is ahead within twenty-five updates**, and
the gain is resolved. That is the thing worth having, and it needs no corpus, no pre-training and
nothing carried in from anywhere.

It is also the shortest-lived result in this project. By a hundred updates it is gone, and by four
hundred the read is a fifth worse than doing nothing. Averaged over the whole stream the number is
-0.0512, which is what the first version of this measurement reported and which hides both halves of
what is actually happening.

## Anchoring does not fix it

The shift is `UV^T` initialised at zero, so weight decay pulls toward zero and zero is the identity —
decay is an anchor to plain cosine rather than generic shrinkage. It does almost nothing. At decay
1.0 the 200–400 bin moves from -0.1192 to -0.1080, and the early bins do not change in the fourth
decimal. The drift is not a magnitude the optimiser can be shrunk out of; single-position InfoNCE
gradients are simply noisy enough that continued updating walks the parameters somewhere useless.

## Where there is no headroom there is no early gain

Codex shows no positive bin at all, -0.0295 in the first twenty-five updates and worsening from
there. That fits Phase 54's account: a static pick already reaches 0.788 on the earliest Codex
positions, close to the 0.815 ceiling, so there is nothing for adaptation to add and the same drift
applies with none of the benefit.

## What this leaves

The mechanism is real and it is unstable. It appears in the regime the whole project was built for —
no annotation, no pre-training, learning on the stream it is in — and the obstacle is not whether
adaptation helps but that nothing stops it from continuing past the point where it did.

## Interpretation boundary

One optimiser, one rank, one temperature, and updates at every position. Whether the collapse is
specific to per-position AdamW steps, or to this objective, is untested; batching several positions,
replaying earlier ones, or stopping on a held-out signal are all standard answers that were not
tried. The early gain is measured on one corpus where headroom exists and is absent on the one where
it does not, so it is a statement about that regime rather than about streams in general.

# Phase 56: the collapse is mostly gradient noise, and mostly fixable

Phase 55 left one question: adaptation helps for fifty updates and then walks away, and nothing
stopped it. Two of the standard answers apply directly and both draw only on the past, so the causal
guarantee is unchanged — batching averages several positions before stepping, and replay adds earlier
ones so an update is not only about the newest thing.

Claude action stream, three seeds:

| Configuration | Whole stream | 100–200 updates | 200–400 updates |
| --- | ---: | ---: | ---: |
| Baseline, one position per update | -0.0607 | -0.0783 | -0.1405 |
| Batch 8 | -0.0199 | -0.0108 | -0.0715 |
| Batch 8, replay 8 | -0.0316 | -0.0198 | -0.1097 |
| Batch 16, replay 32 | -0.0185 | -0.0135 | -0.0673 |
| **Batch 16, replay 32, lr 3e-4** | **-0.0046** | **-0.0111** | **-0.0209** |

**Batching is the single largest fix**, cutting the whole-stream loss by two thirds on its own, which
identifies most of the collapse as gradient noise rather than anything structural. Replay on its own
makes things slightly worse — batch 8 with replay 8 is behind batch 8 alone — and only helps once the
batch is already large. Lowering the rate on top takes the whole stream to -0.0046.

The window where adaptation is useful extends with each fix. At one position per update it closes
after about fifty; at batch 16 the 100–200 bin is -0.0135 rather than -0.0783, so it closes nearer
two hundred.

**And the early gain survives every configuration.** The 25–50 bin is +0.0262, +0.0158, +0.0216,
+0.0203, +0.0203 across the five rows above, resolved in all of them. Whatever the stabilisers do to
the tail, they do not touch the thing worth having.

## What is left, and what the numbers say it is not

Lowering the rate further does nothing: 1e-4 gives -0.0044 against 3e-4's -0.0046, identical to the
third decimal. So the residual is not step size, and with Adam normalising by gradient magnitude the
cumulative-movement hypothesis cannot be cleanly separated from noise by tuning the rate at all.
Testing it needs a different optimiser or an explicit bound on how far the parameters may travel.

The honest state is close to break-even and still on the wrong side: a memory adapting on its own
stream is ahead by +0.020 through its first fifty updates, and behind by -0.0046 averaged over four
hundred. That is a sixty-fold improvement on where Phase 55 left it and it is not yet a system.

## Interpretation boundary

One corpus, three seeds, and a stabiliser sweep chosen by hand rather than searched. The per-bin
intervals are computed on the positions in each bin and the bins are not independent of one another,
since a read that drifted early is the same read being scored late. Nothing here was run on Codex,
where Phase 55 found no early gain to protect in the first place.

# Phase 57: the curve was comparing different streams, and the result is better than it looked

Phase 56 ended pointing at total distance travelled. Bounding it directly — rescaling the shift's
factors whenever their norms' product exceeds a budget — changes nothing at all: a budget of 0.03
gives the same numbers to the fourth decimal as no budget. The constraint never binds, so the read is
not travelling far, and the drift hypothesis is dead.

What was wrong is the curve itself. Streams have different lengths, so a short one contributes to the
early bins and not the late ones, and each bin holds a different set of streams. The static baseline
gives it away: 0.4027, 0.3898, 0.3575, 0.4016, 0.4266 across bins, moving around far more than a
fixed population should. The curve was confounding "more updates" with "different stream".

## Corrected, with the population fixed

Every stream required to reach the last bin, Claude action stream, five seeds:

| Stream length held | Streams | Whole stream | 25–50 updates |
| ---: | ---: | --- | --- |
| 100 positions | 24 | **+0.0066** [+0.0020, +0.0112] | **+0.0157** [+0.0067, +0.0247] |
| 200 positions | 17 | +0.0019 [-0.0022, +0.0060], not resolved | **+0.0254** [+0.0136, +0.0376] |
| 400 positions | 6 | -0.0268 [-0.0336, -0.0200] | -0.0022, not resolved |

**Over a hundred-action stream, a memory that started from nothing and adapted online beats a static
one, +0.0066, resolved.** That is the whole-stream number, not a favourable slice of it, and there is
no pre-training and nothing carried in from anywhere.

The 25–50 gain is real and survives the correction at +0.0157 and +0.0254. Its absence in the
400-position row is a power problem rather than a contradiction: only six streams reach that far, so
that bin holds 450 positions against 2,125, and its interval spans +/- 0.024.

The degradation is also real and now correctly sized. At 200 positions the 100–200 bin is -0.0066
rather than Phase 55's -0.0689; an order of magnitude of that apparent collapse was short streams
leaving the population, not the read getting worse.

## The state of it

| Stream length | Net effect of adapting online |
| --- | --- |
| 100 actions | positive, +0.0066, resolved |
| 200 actions | break-even, +0.0019, not resolved |
| 400 actions | negative, -0.0268, resolved |

A memory that learns on its own stream is ahead early, holds to about two hundred actions, and falls
behind after. That is a working mechanism with a known expiry, which is a more useful object than
either Phase 55's "gains then collapses" or Phase 56's "nearly break-even overall" — both of which
were reading a confounded curve.

## Interpretation boundary

Fixing the population costs streams: 24 at length 100, 6 at 400, from 34. The long-stream rows are
therefore thin, and the 400-position conclusion rests on six sessions of one user. What would settle
the expiry is more long streams, which the Codex corpus has — 200 sessions, many of them long — and
where Phase 55 found no early gain to begin with, so it tests the decay without the benefit.

# Phase 58: the expiry is partly the memory growing, and Codex never had anything to lose

Two questions were left open. Does the expiry replicate where there are enough long streams to
measure it, and is it about the number of updates or about the memory growing alongside them? Those
two grow together in every run so far, and separating them needs only a fixed window on the memory.

## Codex: negative everywhere, and the window changes nothing

200 sessions, population fixed at each length:

| Stream length | Memory grows | Memory held to 64 |
| ---: | ---: | ---: |
| 100 | -0.0206 | -0.0194 |
| 200 | -0.0199 | -0.0176 |
| 400 | -0.0155 | -0.0172 |
| 800 | -0.0136 | -0.0120 |

All resolved, all negative, and the window makes almost no difference. There is no expiry curve here
because there was never anything to expire: Phase 55 found no early gain on Codex, and Phase 54
explained why — a static pick already reaches 0.788 against a 0.815 ceiling, so adaptation has no
room and can only disturb. Longer streams are slightly *less* bad, which is the opposite of an expiry.

## Claude: bounding the memory bounds the decay

| Stream length | Memory grows | Memory held to 64 |
| ---: | ---: | ---: |
| 100 | +0.0066 [+0.0020, +0.0112] | **+0.0092** [+0.0047, +0.0138] |
| 400 | -0.0268 [-0.0336, -0.0200] | **-0.0084** [-0.0133, -0.0035] |

Capping the memory improves the short-stream gain slightly and **cuts the long-stream damage by a
factor of three**. So a real part of the decay is the memory growing under a read that was tuned when
it was small, not the updates accumulating on their own — which is what the Codex rows, taken alone,
would have suggested.

That matters because it is the one lever here that a system would pull anyway. A memory with a
bounded window is the ordinary engineering choice, and it happens to remove most of the failure mode.

## Where this leaves the online result

| Condition | Effect |
| --- | --- |
| Headroom present, short stream, bounded memory | **+0.0092**, resolved |
| Headroom present, long stream, bounded memory | -0.0084, resolved |
| Headroom present, long stream, unbounded memory | -0.0268, resolved |
| No headroom, any length or window | -0.012 to -0.021, resolved |

Adapting online helps only where a static reader has somewhere to improve to, and it helps most when
the memory it reads is bounded. Neither of those is a tuning detail; they are conditions on when the
mechanism applies at all, and they were invisible in every aggregate this line reported before
Phase 57 fixed the population.

## Interpretation boundary

Window 64 was the only size tried and was not searched; it is plausible that a different bound moves
both rows further. The Claude rows still rest on 24 and 6 streams. The headroom explanation ties four
results together and is consistent with all of them, but it has never been manipulated directly — no
experiment here varies headroom while holding the corpus fixed, which is what would turn it from a
pattern into a cause.

# Phase 59: the online memory was never adapting — Phases 55 to 58 retracted

Headroom can be manipulated within a corpus by changing the horizon: a one-action target is a
specific thing a single stored item can match, a twenty-action target is a broad average. Sweeping it
falsifies the headroom story outright.

| Corpus, horizon | Headroom | Online minus static |
| --- | ---: | ---: |
| Claude, 1 | 0.4472 | +0.0045 |
| Claude, 5 | 0.3218 | +0.0092 |
| Claude, 20 | 0.2499 | +0.0107 |
| Codex, 1 | 0.1248 | -0.0169 |
| Codex, 5 | 0.0767 | -0.0201 |
| Codex, 20 | 0.0480 | **+0.0253** |

The gain rises as headroom *falls*, and Codex turns strongly positive at its smallest headroom. What
actually tracks the effect is the horizon, in both corpora.

## And the reason is that nothing was adapting

The online read blends over memory; the static baseline takes one item. A longer horizon makes the
target an average of more actions, which favours a blend whether or not anything is learned. Phase 45
had already separated those two on LongMemEval and the control was not carried across when the
setting changed. Adding it back:

| Corpus, horizon | Blend minus static | **Online minus blend** |
| --- | ---: | --- |
| Claude, 1 | +0.0046 [+0.0004, +0.0088] | **+0.0000**, not resolved |
| Claude, 5 | +0.0089 [+0.0043, +0.0135] | +0.0003, not resolved |
| Claude, 20 | +0.0104 [+0.0057, +0.0152] | +0.0003, not resolved |
| Codex, 1 | -0.0142 [-0.0157, -0.0128] | -0.0027 [-0.0034, -0.0021] |
| Codex, 5 | -0.0209 [-0.0225, -0.0192] | +0.0008 [+0.0002, +0.0013] |
| Codex, 20 | +0.0253 [+0.0233, +0.0273] | +0.0001, not resolved |

**Online adaptation contributes nothing.** The largest effect it has anywhere is -0.0027, and it is
negative. Every number in Phases 55 through 58 was a softmax blend behaving differently from a hard
argmax, measured against a baseline that differed in parameterisation as well as in training.

## What that retracts

- Phase 55's "a memory gains before it knows anything, +0.0141 within twenty-five updates" — the gain
  is the blend, present at update zero, and the update count had nothing to do with it.
- Phase 56's batching and replay results — those changed how the read drifted, and the drift was
  moving a component that was contributing nothing.
- Phase 57's expiry curve — real as a description of the blend under a growing memory, not as
  anything about adaptation.
- Phase 58's memory-window finding — likewise. Bounding the window helps the *blend*, which is still
  a usable result but not the one that phase claimed.

What survives is the horizon result, and it is clean: a blended read beats a single-item pick in
proportion to how broad the target is, from +0.0046 at horizon 1 to +0.0104 at horizon 20 on Claude,
and from -0.0142 to +0.0253 on Codex. That is a statement about selection versus averaging, measured
on two corpora, with no learning involved anywhere.

## The fifth time

Phase 29 read hand-set constants as a dead component. Phase 36 read a metric containing its own write
as a mechanism ranking. Phase 45's first draft read a softmax temperature as a training effect. Phase
52 found a learning rate read as a statement about capacity. This is the fifth, and the worst, since
it ran for four phases.

The control that would have caught it existed in Phase 45 and was dropped when the experiment moved
from offline to online. The rule that keeps being relearned: when the setting changes, the baselines
have to move with it, and a baseline that differs from the treatment in two ways measures neither.

## Interpretation boundary

The frozen blend uses the same random initialisation as the adapting one, so the comparison is exact.
`online minus blend` being unresolved at five of six settings is a statement that the effect is below
about 0.001 there, not that it is exactly zero. Nothing here says online adaptation cannot work —
only that this implementation of it, at every setting tried across four phases, did nothing.

# Phase 60: it did nothing because it never moved, and where it moves it hurts

An effect of +0.0000 after thousands of gradient steps is not a small effect, it is a broken one.
Phase 56 had already reported that a travel budget of 0.03 never binds, which should have been the
clue: the read was not going anywhere. With a softmax temperature of 0.02 the blend is nearly an
argmax, so a small shift changes the scores without changing which item dominates, and the read is
frozen in output while its parameters drift.

Counting how often the adapted read actually attends somewhere the frozen one does not:

| Learning rate | Attention moved | Shift norm at stream end | Online minus blend |
| --- | ---: | ---: | --- |
| 3e-4, the rate used in Phases 56–59 | **0.0032** | 0.079 | +0.0006 [+0.0001, +0.0011] |
| 1e-2 | **0.3937** | 4.865 | **-0.0315** [-0.0394, -0.0238] |

**At the settings every online phase used, the read attended somewhere different on three positions
in a thousand.** That is why adaptation measured as nothing: it did not happen. Turn the rate up
until it does happen, on four positions in ten, and the effect is clearly negative and resolved.

## The complete account of the online line

- Where the read moves enough to matter, online adaptation selects worse items, -0.0315.
- Where it does not hurt, it is because it has not moved, 0.3% of positions.
- There is no setting tried in which it moves and helps.

This also explains Phase 56 backwards. Batching "fixed the collapse" by reducing effective movement —
it froze the read harder. The stabilisers were not protecting a mechanism, they were suppressing one
that only ever did damage.

So the honest conclusion is stronger than Phase 59's "contributes nothing": **this objective,
optimised online one stream at a time, actively picks worse memory items whenever it changes anything
at all.** Whether that is the objective, the single-stream sample size, or the parameterisation is not
separated here, and the three are testable independently.

## What is left standing

Only the horizon result, which involves no learning: a blended read beats a single-item pick in
proportion to how broad the target is, +0.0046 to +0.0104 on Claude and -0.0142 to +0.0253 on Codex
as the horizon goes from one action to twenty. That is a fact about averaging versus selecting,
measured on two corpora, and it is what this line of the project actually produced.

## Interpretation boundary

"Attention moved" counts changes in the top-scoring item, which understates movement in a blend where
the second and third weights also matter; a read could change its output meaningfully without the
argmax flipping. The two rates bracket the behaviour but nothing between them was tried, so it is
possible a rate exists where the read moves a little and helps a little. Given that both measured
points are negative or null, that would be a narrow window to go looking for.

# Phase 61: half the foils were the answer, and ties counted as wins

Training offline on the Codex action stream collapses the read — `trained minus blend` is **-0.3544**,
which is not a degradation, it is a destroyed model. Training on the Claude stream does nothing much,
+0.0127 and unresolved. A difference that large between two corpora of the same kind points at the
data rather than the method.

The objective asks the read to rank the true future above ninety-nine others drawn from the same
corpus. Counting how many of those are the same thing:

| Corpus | Foils above 0.99 cosine to the target | Positions with at least one |
| --- | ---: | ---: |
| **Codex action stream** | **49.87 of 99** | **0.765** |
| Claude action stream | 0.02 | 0.020 |
| LongMemEval turns | 0.00 | 0.003 |

**On Codex, half the foils are the target.** Repetitive tool use means the same five-action window
recurs constantly, so InfoNCE is demanding an impossible discrimination and its gradient can only be
destructive — which is exactly the -0.3544.

And the evaluation has the matching defect. A hit was counted as `(scored > scored[0]).sum() == 0`,
strict inequality, so **a foil scoring exactly equal to the target counts as a win**. A reader that
separates nothing at all reads as perfect wherever the duplicates are.

## What the defect was worth

Dropping foils within 0.9 cosine of the target:

| Corpus | All foils | Duplicates removed |
| --- | --- | --- |
| Codex, blend minus static | **-0.0200** [-0.0221, -0.0178] | **+0.0019** [+0.0009, +0.0029] |
| Claude, blend minus static | +0.0089 [+0.0043, +0.0135] | **+0.0262** [+0.0217, +0.0304] |

On Codex the sign flips. On Claude the effect triples. The absolute scores rise on both — 0.73 to
0.88 and 0.38 to 0.60 — because clean foils are easier than duplicate ones, so numbers are not
comparable across the two columns, only signs and orderings within them.

## What this invalidates

Every Codex number in this project was measured with a foil set that was half answers:

- **Phase 54's headline** — "transfer to a different agent fails, -0.0230" — is not supported. That
  measurement's negative sign is the same one that flips here.
- Phase 58's Codex rows, which concluded there was "never anything to expire", and the headroom
  explanation built on Codex's apparently high baseline of 0.79. That baseline was ties.
- Phase 59's Codex horizon rows and Phase 60's account of them.

LongMemEval is clean at 0.00 duplicates, so Phases 44 through 53 are unaffected. The Claude action
stream is nearly clean at 0.99 but has 11 of 99 foils above 0.9, so its numbers are understated
rather than wrong — the direction holds and the size was too small.

The surviving result therefore survives and grows: a blended read beats a single-item pick, by
+0.0262 on the Claude action stream once the foils are real, against the +0.0089 previously reported.

## The sixth

Phase 29's constants, Phase 36's metric, Phase 45's temperature, Phase 52's learning rate, Phase 59's
missing blend control, and now a corpus whose foils are copies of its answers. Five of the six were
found by asking why a number looked strange rather than by checking. This one came from a -0.3544
that was too large to be anything but a broken setup.

## Interpretation boundary

The 0.9 threshold is a judgement, not a derived quantity; 0.95 or 0.8 would give different absolute
numbers. Re-running every affected phase with clean foils is not done here — what is established is
that the Codex conclusions cannot stand as written and that the Claude direction survives with a
larger effect. The offline collapse of -0.3544 has not been re-measured with clean foils either, so
whether training on action streams works at all is now an open question rather than a settled
failure.

# Phase 62: with real foils, the transfer holds and training is worth more than reported

Phase 61 left the transfer question genuinely open. Re-running it with foils that are not copies of
the answer, five seeds:

| Setting | Trained minus blend | Blend minus hard | Trained minus hard |
| --- | --- | --- | --- |
| LongMemEval, in-domain | **+0.0214** [+0.0187, +0.0242] | +0.0082 [+0.0060, +0.0106] | **+0.0296** |
| Claude action stream, transfer | **+0.0121** [+0.0070, +0.0169] | +0.0238 [+0.0190, +0.0288] | **+0.0359** |
| Codex action stream, transfer | +0.0002, not resolved | +0.0007 [+0.0002, +0.0012] | **+0.0009** |

**Phase 54 is reversed.** Its headline was that transfer to a different agent fails at -0.0230. With
real foils the same measurement is +0.0009, resolved and positive. The negative sign was the blend
penalty that duplicate foils manufacture, and Phase 61 showed that penalty flipping on its own.

**And training is worth more than any phase reported.** The learned transform contributes +0.0214
in-domain and +0.0121 on the Claude transfer, both resolved, against the +0.0085 of Phase 45. That
increase is not the foil fix — LongMemEval was always clean — it is the configuration those later
phases established: context 8, sixteen heads, 470 conversations, 6,000 steps.

## The corrected state of the one working result

| Claim | Status |
| --- | --- |
| A self-supervised future target gives usable supervision with no annotation | 229,245 positions |
| A trained read beats an untrained hard pick in-domain | +0.0296, resolved |
| The learned transform, separated from the blend, contributes | +0.0214, resolved |
| It transfers to a near domain, another agent's tools under the same user | +0.0359, resolved |
| It transfers to a far domain, a different agent entirely | +0.0009, resolved but negligible |
| Distance to the best single item in memory | 0.3464 against 0.2320 in-domain |

The far-domain number is the honest one to quote as the limit: the read neither helps nor hurts on
Codex, and what little it does there is the blend rather than the learning. Whether that is domain
distance or the corpus's repetitiveness is not separated.

## What this session's corrections add up to

Six configuration or measurement defects were reported as findings and then caught: hand-set
constants (Phase 29), a metric containing its own write (36), a softmax temperature (45), a learning
rate (52), a missing blend control (59), and foils that were copies of the answer (61). Four of the
six inverted a conclusion. Two of them — the temperature and the foils — would have made a broken
system look like a working one rather than the reverse.

The pattern is not that the mechanisms were bad. Every effect measured in this project is between
0.001 and 0.04, and every defect above was worth more than that. **The measurements dominate the
mechanisms at this scale**, which is the most transferable thing here: an effect of 0.02 needs the
baseline to differ from the treatment in exactly one way, and that has to be checked rather than
assumed each time the setting changes.

## Interpretation boundary

Everything above is one encoder, one self-supervised target, and a proxy metric — ranking a future
centroid among ninety-nine foils — that has never been connected to anything a user would notice.
Phase 33 retired the only endpoint that tried. The effects are real, resolved, and small, and nothing
here establishes that a memory built this way would change an agent's behaviour.

# Phase 63: online adaptation does work, and Phase 60 tested two points

Phase 60 concluded that "there is no setting tried in which it moves and helps", which was accurate
about what had been tried and was then written as though it were a property of online adaptation. Two
learning rates were tried, 3e-4 and 1e-2, and they bracket the answer rather than containing it. The
online runs also trained and evaluated against the foils Phase 61 found to be half answers on one
corpus and partly duplicated on the other, so the gradient itself was polluted.

Both fixed, five seeds on the Claude action stream:

| Learning rate | Attention moved | Online minus blend, dirty foils | Online minus blend, clean foils |
| --- | ---: | --- | --- |
| 3e-4 | 0.0028 | +0.0003, not resolved | +0.0001, not resolved |
| **3e-3** | **0.0866** | +0.0013, not resolved | **+0.0040** [+0.0013, +0.0067] |
| 1e-2 | 0.3972 | -0.0295 | -0.0166 [-0.0228, -0.0105] |
| 3e-2 | 0.6559 | -0.1073 | -0.0917 [-0.1003, -0.0833] |

**There is a setting where the read moves and helps.** At 3e-3 it attends somewhere different on 8.7%
of positions and gains +0.0040, resolved. Below that it does not move; above it, it moves too much
and the damage grows monotonically. It is an inverted U, and Phase 60 sampled only its two tails.

The foil pollution mattered too: at the rate that works, the dirty-foil measurement reads +0.0013 and
unresolved. Both causes were needed to hide it.

## What this costs the earlier phases

Phase 60's retraction of Phases 55–58 stands in substance — those phases attributed to adaptation an
effect that was the blend, and the blend is still doing most of the work here, +0.0262 against
adaptation's +0.0040. What does not stand is the sentence that online adaptation actively harms
whenever it changes anything. It harms when it changes too much.

That also revises Phase 56 again. Batching and replay were described there as helping, then in Phase
60 as merely freezing a harmful component. With a working rate they are doing neither: the useful
regime is a movement rate, and the stabilisers are one way of reaching it.

## The honest size of it

| Component, Claude action stream, clean foils | Effect |
| --- | --- |
| Blending instead of picking one item | **+0.0262** [+0.0217, +0.0304] |
| Online adaptation at the rate that works | **+0.0040** [+0.0013, +0.0067] |
| Online adaptation at ten times that rate | -0.0917 |

Adaptation is real and is a sixth of the blend. A system built on this would get most of its value
from averaging several memories rather than selecting one, and a little more from learning on the
stream, and would be destroyed by learning slightly too fast.

## Interpretation boundary

Four rates on one corpus with one optimiser. The peak is somewhere near 3e-3 and has not been
located; the useful window's width is unknown and could be narrow enough to be impractical, since one
order of magnitude past it costs twenty times what it gains. Nothing here was run on Codex with clean
foils, and the movement measure still counts only changes in the top-scoring item.

# Phase 64: the window is a factor of five, and the peak is twice what was found

Mapping the rate finely, clean foils, five seeds:

| Claude, learning rate | Attention moved | Online minus blend |
| --- | ---: | --- |
| 1e-3 | 0.0128 | +0.0013 [+0.0003, +0.0022] |
| 2e-3 | 0.0417 | +0.0016, not resolved |
| 3e-3 | 0.0866 | +0.0040 [+0.0013, +0.0067] |
| **5e-3** | **0.1998** | **+0.0083** [+0.0037, +0.0127] |
| 7e-3 | 0.3037 | +0.0008, not resolved |
| 1e-2 | 0.3972 | -0.0166 [-0.0228, -0.0105] |

**The peak is at 5e-3 and is twice what Phase 63 reported**, +0.0083 against +0.0040, at about a fifth
of positions attending somewhere new. The useful window runs from roughly 1e-3 to 5e-3, a factor of
five. One further doubling to 7e-3 returns to zero and the next is clearly negative, so the fall is
much steeper than the rise.

| Codex, learning rate | Attention moved | Online minus blend |
| --- | ---: | --- |
| 1e-3 | 0.0029 | +0.0003 [+0.0001, +0.0005] |
| 3e-3 | 0.0123 | +0.0003, not resolved |
| 1e-2 | 0.0801 | -0.0049 [-0.0063, -0.0034] |

Codex peaks an order of magnitude lower and twenty times smaller, and is already negative where
Claude is still climbing.

## There is no universal knob

Attention movement looked like the corpus-independent quantity to tune on — it is the thing the rate
actually controls, and it is measurable without knowing the answer. It is not. Claude peaks near 20%
movement; Codex peaks below 1% and is harmed at 8%. The rate that works has to be found per corpus,
and the only signal for finding it is the objective itself.

That is a practical limit rather than a fatal one: the loss being optimised is available online, so a
system could search the rate on its own stream. But it means "adapt online" is not a setting that can
be shipped with a default.

## Where the online line ends

| Component, Claude action stream, clean foils | Effect |
| --- | --- |
| Blending instead of picking one item | **+0.0262** |
| Online adaptation at its best rate | **+0.0083** |
| Online adaptation one doubling past it | +0.0008 |
| Online adaptation one order of magnitude past it | -0.0166 |

Adaptation is a third of the blend at its peak rather than the sixth Phase 63 measured, real, and
fragile. Against Phase 60's version of this table, which had adaptation at zero or harmful
everywhere, the difference is entirely two learning rates and a foil set.

## Interpretation boundary

The peak is bracketed by 3e-3 and 7e-3 and not located more precisely; 5e-3 is the best of six points
rather than a fitted optimum, and choosing it from the same data that measures it inflates the
+0.0083 somewhat. One optimiser, one rank, one temperature, and two corpora whose optima differ by an
order of magnitude, which is the finding that most limits what can be claimed.

# Phase 65: the first connection to something that actually happened

Sixty-four phases of tuning have optimised one quantity: rank a future centroid among ninety-nine
foils. Phase 33 retired the only endpoint that tried to connect a memory to an outcome, and nothing
replaced it, so every gain since has been on a proxy with no established destination. Continuing to
improve it is not worth doing until it is known to point at anything.

The action stream carries a real outcome that nobody wrote for this purpose. A tool call either
returned an error or it did not, and Claude's logs record the flag directly. The question is whether
anything about the memory at position t carries information about whether the action at t+1 fails.

11,917 positions, 272 of them followed by a failure, base rate 0.0228:

| Signal available before the action is taken | AUC | 95% interval |
| --- | ---: | --- |
| Highest similarity to anything in memory | 0.4125 | [0.3774, 0.4458] |
| The blend's agreement with itself | 0.4184 | [0.3825, 0.4517] |
| **Novelty — the absence of any match** | **0.5787** | **[0.5453, 0.6142]** |

All three resolved, and the first two sit *below* chance, which is the same statement read the other
way: **an action that resembles what the memory already holds is more likely to succeed, and one that
does not is more likely to fail.**

This is the first time in this project that a quantity derived from memory has been shown to carry
information about something that actually happened to the agent. It is not large — 0.579 is a weak
predictor — but it is a real recorded outcome rather than a centroid, and the label was not
constructed for the experiment.

## What it does and does not license

It does not say that retrieval quality matters. Novelty is a property of the current action against
the memory as a whole, not of which item a reader surfaces, so the sixty-four phases of work on
*what to surface* are not validated by this. What is validated is weaker and more useful: the memory
state is not disconnected from outcomes, so there is a destination for this work to point at.

It also suggests the more promising target is the one nobody here was optimising. A memory that
answers "have I done something like this before, and did it go badly" is a different object from one
that answers "which stored item is most relevant", and only the first has shown any link to a real
consequence.

## Interpretation boundary

One corpus, 272 failures, all from one user's sessions and heavily concentrated in a few long ones.
`is_error` catches hard failures only — an action that succeeded and should not have been taken is
unlabelled, and that is most of what a memory would be for. Codex was not measured: its failure flags
are inferred from output text rather than recorded, and the bootstrap over ten thousand of them did
not finish. Novelty and failure could share a common cause — unfamiliar territory is both novel and
error-prone for reasons having nothing to do with memory — and nothing here separates those.

# Phase 66: remembering how it went, and the baseline that had to be beaten

Phase 65 pointed at a different object: not "which stored item is most relevant" but "have I done
something like this, and did it go badly". That is retrieval weighted by outcome, and it has one
baseline that decides whether it is an idea at all. **Failures might simply cluster in time**, in
which case the share of the last sixteen actions that failed carries everything and similarity adds
nothing.

11,917 positions, 272 failures, memory window 256, everything read strictly before the action being
predicted:

| Signal | AUC | 95% interval |
| --- | ---: | --- |
| Share of the last 16 actions that failed | 0.5571 | [0.5195, 0.5950] |
| Novelty (Phase 65) | 0.5797 | [0.5345, 0.6241] |
| Similarity-weighted average of past outcomes | 0.6012 | [0.5566, 0.6442] |
| **Both together** | **0.6105** | [0.5649, 0.6533] |

Paired, on shared resamples, which is the comparison those overlapping intervals cannot make:

| Comparison | Difference | 95% interval |
| --- | ---: | --- |
| Outcome retrieval minus clustering | +0.0442 | [+0.0000, +0.0868], marginal |
| Outcome retrieval minus novelty | +0.0215 | [-0.0346, +0.0808], not resolved |
| **Outcome retrieval plus novelty, minus novelty** | **+0.0308** | **[+0.0098, +0.0522]** |

**Remembering what went wrong adds something beyond noticing that things are going wrong**, though
the margin over the clustering baseline sits exactly on zero at its lower bound and should be called
marginal rather than established. What is resolved is that outcome retrieval and novelty are
complementary: adding the first to the second gains +0.0308, and neither subsumes the other.

The combination reaches 0.6105 on a real recorded outcome. That is a weak predictor, and it is the
strongest link to anything consequential that this project has produced in sixty-six phases.

## Why this is a different object from what came before

Everything from Phase 31 onward asked which stored item to surface, and Phase 65 showed that quantity
carries no outcome information at all — the similarity signals sit *below* chance. This asks what the
memory implies about what is about to happen, which needs the memory to store outcomes rather than
only content, and it works where the other did not.

That is worth saying plainly: the sixty phases of work on retrieval quality are not what produced
this. A memory holding one extra bit per item — how it went — outperforms all of it against a real
label.

## Interpretation boundary

The same 272 failures as Phase 65, with the same concentration and the same blindness to actions that
succeeded but should not have been taken. The clustering baseline uses a sixteen-action window chosen
without searching, and a better-tuned version of it could close the marginal +0.0442. Nothing is
trained here; these are fixed formulas over embeddings, so the comparison is between hand-built
signals and not between learned systems. And the causal question from Phase 65 stands untouched:
unfamiliar territory may be error-prone for reasons that have nothing to do with memory.

# Phase 67: the baseline was unsearched, and once searched the idea disappears

Phase 66 said its clustering baseline used an unsearched sixteen-action window and that a better one
could close the +0.0442. Searching it:

| Clustering baseline | AUC |
| --- | ---: |
| Last 16 actions (Phase 66's choice) | 0.5571 |
| Last 128 | 0.5818 |
| **Recency-decayed, half-life 128** | **0.5820** |

The baseline gains 0.025 from being tuned at all, and the comparison it was losing goes with it:

| Comparison | Difference | 95% interval |
| --- | ---: | --- |
| Outcome retrieval minus best clustering | +0.0192 | [-0.0185, +0.0544], **not resolved** |
| Outcome retrieval and novelty, minus best clustering | +0.0285 | [-0.0243, +0.0815], not resolved |

And the one result Phase 66 called resolved survives only in a form that empties it. Adding outcome
retrieval to novelty gains +0.0308 — but adding *clustering* to novelty gains +0.0294, and the two
combinations are indistinguishable:

| Comparison | Difference | 95% interval |
| --- | ---: | --- |
| Outcome retrieval and novelty, minus clustering and novelty | **+0.0014** | [-0.0159, +0.0190] |

**Similarity-weighted retrieval of past outcomes contributes nothing.** What works is novelty plus
any measure of how often things have been failing lately, and the similarity part of the memory is
not doing any of it. Phase 66's sentence — "remembering what went wrong adds something beyond
remembering that things are going wrong" — is false as stated.

## What is actually left

| Signal | AUC | Needs retrieval? |
| --- | ---: | --- |
| Novelty, one scalar summary of the memory | 0.5797 | no |
| Recency-decayed failure rate | 0.5820 | no |
| Both | 0.6091 | no |
| Both, plus similarity-weighted outcomes | 0.6105 | yes, and worth +0.0014 |

Two signals predict a real outcome at 0.609 combined, and **neither of them requires retrieving
anything.** Novelty is the mean similarity to the memory as a whole; the failure rate needs only a
counter. Sixty-seven phases of work on what to store and what to surface are not implicated in the
only result that touches a consequence.

## The seventh, and the same shape

Phase 29's constants, 36's metric, 45's temperature, 52's learning rate, 59's missing control, 61's
foils, and now a baseline left unsearched for one phase. Four of the seven were caught by tuning the
baseline rather than the treatment. The asymmetry is the lesson: effort went into the proposal every
time, and the comparison was decided by the thing nobody was tuning.

## Interpretation boundary

The clustering baseline was searched on the same data it is evaluated on, so its 0.5820 is optimistic
and the true gap to outcome retrieval is somewhat larger than the point estimates say — which makes
the negative conclusion safer, not weaker. Everything is still 272 failures on one corpus with no
training anywhere, so this rules out a hand-built similarity-weighted signal rather than the idea that
a learned one could work.

# Phase 68: at a real sample size the effect is four times larger, and it does not transfer

Every corpus used until now was one person's logs. Phase 49 established that the independent unit is
the project and there were on the order of a dozen, and Phase 51 tested the data hypothesis by
doubling 233 conversations to 470 and found it wanting. Doubling was too small a test.

`local-data/open-swe-v1.jsonl` had been on disk since the start of this project: 5,000 public agent
trajectories from **1,326 distinct repositories**, 416,541 tool calls, licence-filtered. Taking 1,200
streams of at least 64 actions gives 124,291 actions — ten times the Claude stream and from
independent sources rather than one user. Foils are clean at the exact-duplicate level, 0.04 above
0.99 against Codex's 49.87, though 22 of 99 sit above 0.9, so everything below uses the clean-foil
setting.

| Setting | Trained minus blend | Blend minus hard | Trained minus hard | Positions |
| --- | --- | --- | --- | ---: |
| **Trained on Open-SWE, evaluated on it** | **+0.1029** [+0.0981, +0.1075] | +0.0165 | **+0.1194** [+0.1145, +0.1241] | 52,414 |
| Trained on LongMemEval, evaluated on Open-SWE | **+0.0014** [+0.0007, +0.0022] | +0.0169 | +0.0184 | 174,587 |

**The in-domain effect is four times anything measured before** — +0.1194 against LongMemEval's best
of +0.0296 — and the composition inverts. In every earlier phase the blend did most of the work and
training added a little; here training is worth +0.1029 against the blend's +0.0165, six times more.

**And it does not transfer.** The same architecture trained on LongMemEval contributes +0.0014 on
Open-SWE, seventy-three times less than training on Open-SWE itself. The blend transfers fine, +0.0169,
because it is not learned.

## What this settles

- Phase 48 and 49 were right that data was the binding constraint, and Phase 51's retraction of that
  was premature: 233 to 470 conversations is not a test of a data hypothesis, 1,200 independent
  streams is.
- The transfer question, open since Phase 54 and reversed twice, now has an answer at 174,587
  positions: **the learned part is domain-specific.** Train on the stream you will run on.
- Every small number in Phases 44 through 67 — the +0.008s and +0.03s — was measured on a sample too
  small and too correlated to show what the method does. The method was never the limit.

## What it does not settle

The target is still the proxy: rank a five-action future centroid among ninety-nine foils. Phases 65
to 67 found that the signals predicting a *real* outcome need no retrieval at all, and nothing here
changes that. A large gain on the proxy and no gain on the outcome remain simultaneously true, and
which one matters is the question this project has never answered.

The ceiling is also still far away: 0.5720 against 0.8754 for the best single item in memory, so
seven tenths of what is available is still unclaimed even in-domain.

## Interpretation boundary

One encoder, five seeds, 1,200 of 5,000 available trajectories, and `resolved` is 1 on every row
because the downloader filtered to successes — so this corpus carries no task-level outcome contrast
and cannot address the Phase 65 question. Failure flags came out as zero for it, a bug in the
tool-result matching rather than a property of the data. Held-out is by stream within one dataset;
transfer *to* LongMemEval was not measured, so "domain-specific" is shown in one direction.

# Phase 69: at scale the outcome question answers itself, in the negative

The failure flags in Phase 68 came out zero because these traces carry no `tool_call_id` on the
result — a tool message holds only `content`, `reasoning_content`, `think` and `tool_calls`. The
transcript alternates strictly, so matching each result to the oldest waiting call recovers them:
26,071 failures in 124,291 actions.

That makes Phases 65 to 67 repeatable at ten times the scale and across 1,326 repositories instead of
one user. They do not repeat.

| Signal | Claude stream (272 failures) | **Open-SWE (26,071 failures)** |
| --- | ---: | ---: |
| Novelty | 0.5797, resolved | **0.5105, not resolved** |
| Best clustering baseline | 0.5820 | **0.6925** |
| Similarity-weighted outcome retrieval | 0.6012 | 0.6387 |

| Comparison, Open-SWE | Difference | 95% interval |
| --- | ---: | --- |
| Outcome retrieval minus best clustering | **-0.0538** | [-0.0877, -0.0202] |
| Retrieval and novelty, minus clustering and novelty | **-0.0392** | [-0.0725, -0.0077] |

**Retrieval is now resolved *worse* than a counter.** Phase 67 found it indistinguishable from
recency-decayed failure rate; at scale it is behind by 0.054, and adding novelty does not rescue it.

**And novelty does not replicate at all.** Phase 65's one link between memory and a real outcome —
that an unfamiliar action predicts the next one failing — reads 0.5105 and unresolved here. It was a
property of 272 failures in one person's sessions.

## What this answers

The question left open since Phase 33, and restated at the end of Phase 68, was whether the proxy
this project optimises has any connection to something consequential. Two facts now sit together, both
measured at proper scale:

- The trained read beats a hard pick on the proxy by **+0.1194** (Phase 68).
- On the only real outcome available, similarity-based retrieval is **-0.0538 behind a counter**.

Those are not in tension; they are the answer. **Getting better at surfacing the item that matches the
next five actions does not help predict whether the next action fails, and the machinery that does it
actively hurts there.** What predicts failure is how often things have lately been failing, which
needs no memory of content at all.

## Interpretation boundary

The failure label is keyword-derived — "error", "traceback", "no such file" and two others in the
first 2,000 characters of a result — and fires on 21% of actions against 2.3% on Claude's structured
`is_error`. Agents read source code that mentions errors, so this label is noisy and its absolute
AUCs are not comparable with Phase 67's. What it supports is the ordering, which noise attenuates
rather than inverts, and the ordering is resolved. A structured failure field for these traces, if one
exists in the full dataset rather than the downloaded subset, would settle the magnitudes.

# Phase 70: measuring behaviour, finally

Every phase of this project ranked something, and the requirement was always a memory that improves an
agent's behaviour. The two are not the same, and the gap has a structural reason: these logs record an
agent that had no memory, so nothing was ever surfaced and no observational number can show surfacing
helping. Worse, the proxy target rewards predicting what the agent actually did, mistakes included, so
optimising it builds a better model of that agent rather than a better adviser to it.

The fix is to intervene. Open-SWE trajectories are all `resolved`, so the action the agent took next
is known-good. A frozen `Qwen2.5-1.5B-Instruct` is asked how likely that action is given the
trajectory so far, with and without memory injected. Every condition injects four items, so length
cannot stand in for content.

250 trajectories:

| Condition | Action NLL | Gain over no memory | Win rate |
| --- | ---: | ---: | ---: |
| No memory | 1.4058 | — | — |
| Recency, the four preceding steps | 1.2548 | +0.1511 | 0.548 |
| Similarity, the four most like the current state | 1.3028 | +0.1030 | 0.668 |
| **Oracle, the four a reader with the future would pick** | **0.9193** | **+0.4865** | **0.828** |

**The instrument works.** The oracle beats no memory by +0.4865 at a win rate of 0.828, which is what
Phase 33 could not achieve and what retired the previous endpoint. The reason is the one Phase 40
identified: there, a memory item was a ten-thousand-character session averaged into one vector and the
answer was diluted away; here it is a single rendered action.

Paired, because recency leads on the mean while similarity leads on the win rate, and that
disagreement is exactly what retracted Phase 33:

| Comparison | Difference | 95% interval |
| --- | ---: | --- |
| Similarity over recency | -0.0481 | [-0.1254, +0.0195], **not resolved** |
| **Similarity over no memory** | **+0.1030** | [+0.0709, +0.1384] |
| **Oracle over similarity** | **+0.3835** | [+0.2713, +0.5104] |

## Three things, and the third is the point

**Memory changes behaviour for the better.** +0.1030, resolved, against having none. That is the
first behavioural result this project has produced, on a target nobody wrote for it.

**Choosing what to surface by similarity is no better than taking the most recent thing.** -0.0481 and
unresolved. Sixty-nine phases of work on retrieval quality do not beat a four-line baseline here.

**And choosing it well is worth nearly four times more.** The oracle is +0.3835 ahead of similarity,
resolved on a behavioural measure. So there *is* something to select — the selection problem is real,
it is large, and cosine does not solve it.

That last line is the one that justifies the project and has never before been demonstrated. Every
earlier headroom was on a proxy. This one is on what the model would do.

## Interpretation boundary

The oracle looks at the action it is being scored on, so it is a ceiling and not a method. Likelihood
of the known-good action is not the same as task success: `resolved` is 1 on every row here, so there
is no failed trajectory to contrast against and nothing rules out a memory that raises likelihood
while hurting outcomes. 250 trajectories, one generator, one prompt format, and four items per
condition chosen without searching. The ceiling is measured with a single-item-per-slot oracle over
the same stream, so it bounds selection from this memory rather than what any memory could offer.
