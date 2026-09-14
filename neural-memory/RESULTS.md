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
