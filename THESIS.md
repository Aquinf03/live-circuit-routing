# Thesis

**Claim.** On known transformer circuits, a small subgraph extracted from the live attention routing graph of a single forward pass recovers the same causal structure as heavier offline methods: ablating those edges hurts the target behavior more than size-matched random edges, at cost close to one forward pass.

**Falsifier.** If, on induction and IOI (or equivalent known circuits), the live routing subgraph does not beat matched random controls under ablation, or only works after adding the same heavy attribution machinery we claim to avoid, the claim is false.

## Them vs us

- **QK / OV framework + induction / IOI papers** map specific heads by hand. We reuse those as ground truth; we do not rediscover them manually.
- **ACDC / activation patching** find circuits by many interventions. We extract a candidate from the attention routing graph first, then verify with a small number of ablations.
- **Information Flow Routes** build a subgraph via attribution over the full computation graph. We stay attention-routing-first: edges are attention routes, not a full attributed residual/MLP graph.
- **Attribution graphs + QK feature attributions** need replacement models / sparse features and often freeze attention. We do not train SAEs/CLTs; we read the live attention graph as the circuit sketch.
- **Sparse feature circuits** explain *what* features mean. We only claim *where* information was routed; feature semantics are out of scope.

