# Cost note — live extract vs head-patching

Model: `gpt2-small` · device: `cpu` · n=5 induction prompts  
Live settings: score=`attn_x_vnorm`, k=15, B=4  
Source: `results/runs/cost_gpt2-small_20260907_115125/cost.json`

## Per-example means

| Method | Forwards / example | Mean wall-clock |
| --- | ---: | ---: |
| Live extract (graph + top-k) | **1** | **0.056s** |
| Live + verify (S + 10 random ablations) | 12 | 0.481s |
| Head-patch sweep (each of 144 heads) | 145 | 5.939s |

**Forward ratio** head-patch / live extract = **145×**  
**Wall-clock speedup** ≈ **106×**

Live extract is one forward plus cheap post-processing. Head-patching needs a separate forward per head (plus a clean run). Full edge-ACDC is even more expensive; this head sweep is a lower bound on patching cost.

Also: live edge ablation often hurts the target more than the best single-head knockout on these prompts (multi-edge circuit vs one head), so the cheap method is not just faster — the verify step still carries causal signal.
