# Failure cases — where the live routing subgraph is wrong or incomplete

Failures are cases where ablating extracted edges does **not** beat size-matched random (or hurts the metric). Also: qualitative gaps vs known circuits.

## Causal failures (drop_S ≤ 0 or drop_S ≤ drop_R)

Same protocol: `attn_x_vnorm`, k=15, B=4, ban_bos.

### GPT-2 Small — induction
**0 / 23 failures.** All examples: drop_S > drop_R.

### GPT-2 Small — IOI
**0 / 23 failures.** All examples: drop_S > drop_R.

### GPT-2 Medium — induction
**2 / 23 failures** (`results/runs/induction_gpt2-medium_20260907_113405`):

| Example | drop_S | drop_R | Mode |
| --- | ---: | ---: | --- |
| wind / blew | −1.181 | −0.039 | ablation *helps* target (wrong edges) |
| door / opened | −1.418 | +0.000 | ablation *helps* target (wrong edges) |

### Pythia-410M — induction
**3 / 23 failures** (`results/runs/induction_pythia-410m_20260907_114107`):

| Example | drop_S | drop_R | Mode |
| --- | ---: | ---: | --- |
| dog / ran | −0.294 | +0.050 | wrong edges |
| girl / smiled | −0.422 | −0.011 | wrong edges |
| door / opened | +0.027 | +0.056 | weak / incomplete (below random) |

**Rate:** 5 causal fails out of 92 induction+IOI example-runs across three models (~5%). Concentrated on Medium/Pythia induction, not Small.

## Incomplete structure (even when causal drop looks good)

From GPT-2 Small induction head heatmap (`results/figures/induction_head_heatmap.png`):

| Known head | Times in extracted S ( /23 ) | Note |
| --- | ---: | --- |
| L4H11 (prev-token) | 23 | recovered |
| L5H5 (induction) | 23 | recovered |
| L5H8 (induction) | 22 | recovered |
| L7H9 | 5 | sparse |
| L7H3 | 6 | sparse |
| L6H9 | 3 | mostly missed |
| L6H10 | 0 | missed |

So the live graph often recovers *part* of the textbook circuit and mixes in other high-score edges. Causal verify can still pass because the extracted set overlaps the true path — not because every known head is present.

## Failure modes (interpretation)

1. **Attention sinks / mass ≠ causation** — early runs without `ban_bos` picked BOS edges; ablating them helped or did nothing. Mitigated by BOS ban + attn×‖v‖, but residual junk edges remain.
2. **Wrong high-score routes** — Medium `wind/blew`, `door/opened`: extracted edges anti-correlated with the target; model may use different routes or the top-k walk latches onto distractors.
3. **Incomplete multi-hop circuits** — fixed small `B` and flat top-k can miss deeper IOI/induction composition; `|S|` also scales with prompt length (IOI ~195 edges vs induction ~60–75).
4. **Single-head oracle ≠ edge set** — cost bench: best single-head knockout often weaker than ablating the full live edge set; conversely, a live set can look causal while still omitting named heads.

## What we are *not* claiming

- Not that every extracted edge is necessary or monosemantic.
- Not full ACDC / attribution-graph completeness.
- Not that failure rate is zero outside GPT-2 Small.

Honest paper line: **the method usually finds a causally useful routing subgraph quickly; it is not a complete circuit enumerator, and ~5% of scale/transfer examples still fail the random-control test.**
