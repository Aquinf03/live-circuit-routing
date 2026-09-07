# Thesis

**Claim.** On known transformer circuits, a small subgraph extracted from the live attention routing graph of a single forward pass recovers the same causal structure as heavier offline methods: ablating those edges hurts the target behavior more than size-matched random edges, at cost close to one forward pass.

**Falsifier.** If, on induction and IOI (or equivalent known circuits), the live routing subgraph does not beat matched random controls under ablation, or only works after adding the same heavy attribution machinery we claim to avoid, the claim is false.
