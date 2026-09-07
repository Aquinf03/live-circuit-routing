# Thesis

**Claim.** On known transformer circuits, a small subgraph extracted from the live attention routing graph of a single forward pass recovers the same causal structure as heavier offline methods: ablating those edges hurts the target behavior more than size-matched random edges, at cost close to one forward pass.

**Falsifier.** If, on induction and IOI (or equivalent known circuits), the live routing subgraph does not beat matched random controls under ablation, or only works after adding the same heavy attribution machinery we claim to avoid, the claim is false.

## Them vs us

- **QK / OV framework + induction / IOI papers** map specific heads by hand. We reuse those as ground truth; we do not rediscover them manually.
- **ACDC / activation patching** find circuits by many interventions. We extract a candidate from the attention routing graph first, then verify with a small number of ablations.
- **Information Flow Routes** build a subgraph via attribution over the full computation graph. We stay attention-routing-first: edges are attention routes, not a full attributed residual/MLP graph.
- **Attention rollout / flow (Abnar & Zuidema)** compose attentions across layers for token importance. We keep per-edge \((\ell,h)\) identity and a short walk, then verify by ablation rather than rolling out.
- **Norm-based attention (Kobayashi et al.)** analyzes \(\|\alpha f(x)\|\) because weight alone understates the write. Our default score \(A\cdot\|V\|\) is a cheap edge-level cousin used for ranking, not a full norm-based analysis.
- **Attribution graphs + QK feature attributions** need replacement models / sparse features and often freeze attention. We do not train SAEs/CLTs; we read the live attention graph as the circuit sketch.
- **Sparse feature circuits** explain *what* features mean. We only claim *where* information was routed; feature semantics are out of scope.

## Method definitions

**Nodes.** One node per token position in the prompt (sequence index `i = 0 … T-1`). Nodes are positions, not vocabulary ids: the same word at two places is two nodes. We do not use SAE/feature nodes in v1. The answer/query position (usually the last token) is the root we extract toward.

**Edges.** A directed edge `(ℓ, h, j → i)` means: at layer `ℓ`, head `h`, query position `i` attended to key position `j` (information routed from `j` to `i`). Use **all layers and all heads**; keep `(ℓ, h)` identity on every edge. Do **not** mean/max-pool heads for the main method (induction/IOI need head identity). Mean/max across heads is only an ablation variant.

**Edge score.** Main score = **attn × ‖value‖** `A[ℓ, h, i, j] · ‖V[ℓ, h, j]‖` (one forward; closer to “how much was written”). Report **raw attention** as a secondary score. Do **not** use gradient / IFR-style attribution as the main score (that collapses into the methods we claim to avoid).

**Subgraph extract rule.** Root = answer position `t*`. Keep the **top-k** incoming edges to `t*` by score (primary). Recurse backward up to **B** hops: for each newly included key position, again keep its top-k incoming edges. Drop edges that cannot reach `t*` under this expansion. Threshold `τ` is the ablation alternative to top-k (same extract, different cut). Default starting point: `k` small (e.g. 5–20), `B` small (e.g. 2–4).

**Verification.** Main check = **edge ablation**: zero (or mean-ablate) the attention weights on extracted edges only; measure drop on the task metric (induction accuracy / IOI logit diff). Control = **size-matched random edges**. Secondary = **head knockout** on heads that dominate the subgraph (coarser). Skip path patching as required for v1 (optional later if edge ablation is ambiguous).

**Pseudocode.**

```
function live_circuit(model, tokens, t*, k, B):
  A = attention_from_forward(model, tokens)          # all layers, heads
  score[ℓ,h,i,j] = A[ℓ,h,i,j] * ||V[ℓ,h,j]||         # attn×‖v‖ (main)
  S = {}                                             # selected edges
  frontier = {t*}
  for hop in 1..B:
    next = {}
    for i in frontier:
      E = top_k_incoming(score, query=i, k=k)        # edges (ℓ,h,j→i)
      S = S ∪ E
      next = next ∪ {j for each edge j→i in E}
    frontier = next - already_expanded
  metric_full = task_metric(model, tokens)
  metric_ablate = task_metric(ablate_edges(model, tokens, S))
  metric_rand = task_metric(ablate_edges(model, tokens, random_edges(|S|)))
  return S, metric_full, metric_ablate, metric_rand
```

## Baselines

**Random edge subgraph (matched size).** Control for ablation. Let `N = |S|`. Sample `N` edges uniformly from valid causal attention edges (`j ≤ i`, all `ℓ, h`). Ablate them the same way as `S`. Repeat **R = 20** draws; report mean ± std metric drop. Success = extracted `S` hurts clearly more than random.

**Attention-mass only.** Same raw-attn score as main, but **flat**: take the global top-`N` edges into `t*` (or global top-`N`), **no B-hop expansion**. Tests whether the walk/extract rule matters vs “biggest attentions near the answer.” (attn×‖v‖ remains a score variant, not this baseline.)

**Heavier baseline.** **Activation patching** on the same prompts: patch candidate edges/heads, keep those that move the task metric, build a patching-selected set `S_patch`. Compare (1) causal drop of `S` vs `S_patch`, (2) overlap with known induction/IOI heads. Skip full IFR implementation for v1 unless it comes almost free; cite IFR as related, don’t reimplement as required.

**Optional diagnostic (not a claim).** Aquin-style **head entropy** and **sink-token** labels on figures only — helps readers see dead/sink heads. Not scored as a baseline win; not part of the main theorem.

