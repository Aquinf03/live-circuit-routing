# Attention as a Routing Graph as Live Circuit Extraction

## What this is

Transformers move information with attention. Each head decides, for every token, where to look. That pattern is a **routing graph**: tokens are nodes, attention weights are edges.

Most circuit work finds important paths **after** the fact (patching, SAEs, attribution graphs). This paper asks whether we can treat the attention graph itself as a **live circuit**: extract the small subgraph that matters for a prediction **during** a forward pass, cheaply enough to use while debugging or watching a model.

## One-line claim

Attention already draws the circuit. We extract the live routing subgraph and show it recovers the same causal story as heavier offline methods on known tasks.

## Why it matters

- Offline circuit finding is slow and tool-heavy.
- Attention maps are free on every forward pass.
- If the routing subgraph is enough, interpretability can sit inside train / eval / watch loops, not only in separate research runs.

## What we are not claiming

- Not "attention heatmaps explain everything."
- Not a replacement for SAEs or feature circuits when you need feature meaning.
- Not that every head is a clean named circuit.

We claim a usable middle layer: **routing topology → candidate circuit → cheap causal check**.

## Method (simple)

1. Run the model on a prompt.
2. Build a graph from attention (token × token, per head / layer).
3. Keep only strong / high-attribution edges (threshold or top-k).
4. Read out a small subgraph for the answer token.
5. Verify with ablation / patching: breaking those edges should hurt the behavior; breaking random edges should not.

## Success looks like

On known circuits (induction, IOI, simple fact recall):

- Live routing subgraph overlaps with known important heads/edges.
- Ablating the extracted edges hurts the target metric more than matched random controls.
- Cost stays close to one forward pass (+ light post-process), not a full attribution campaign.

## Related ideas (nearby, not the same)

- Attention pattern viz / head entropy (diagnostics).
- Information flow routes / attribution graphs (powerful, usually heavier / more frozen).
- Feature circuits with SAEs / transcoders (feature meaning, not live routing alone).

Our wedge is **live + routing-first + cheap to verify**.
