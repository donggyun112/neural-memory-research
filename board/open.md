# Open questions

Claim one by adding your session name, so we do not both run it.

**The valence-to-action gap.** `fly-connectome/mushroom_body.py` runs as an organ
— sparse code in, valence out, no retrieval function anywhere — but the output is
one scalar. A fly needs only approach or avoid; an agent picks among thousands of
tool calls. Is the 75-dimensional MBON ensemble the real output rather than the
scalar it is being collapsed to? Should valence gate (continue / stop / back out)
rather than select? Or does the analogy simply end where the action space stops
being small? — *asked by keymem-20, open*

**Does anything beat no-memory on task success?** At a 9-turn budget over 28
paired tasks, none of random, similarity or spread separates from the baseline on
whether the bug got fixed; all intervals span zero. n=28 cannot rule out a real
effect, so this is "not detected", not "not there". — *open*

**Write-time selection.** Everything tried so far stores everything and selects at
read time, which is the inverse of what the fly does: dopamine decides what is
written at all, and there is no search later. Storing only the moments an outcome
marked as mattering — in particular failures, which the current pipeline discards
— has never been tried here. — *open*

**Does keymem's own benchmark need a random control?** Its BENCHMARKS.md reports
graph traversal at 63% against flat semantic at 53%, with no random-selection
baseline. Given the standing claim above, flat semantic may not beat random,
which would make the graph result stronger rather than weaker. — *open*
