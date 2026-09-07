# Ablation — all heads/layers vs selected layer bands

Model: `gpt2-small` · score=`attn_x_vnorm` · k=15 · B=4 · n=10
Edges kept only if their layer is in the band; head identity still preserved.
Source: `/Users/ashm/work/live/results/runs/ablation_layers_gpt2-small_20260907_121951/layer_ablation.json`

| Band | Layers | mean |S| | drop_S | drop_R | gap | fails |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | all | 67.5 | +4.047 | +0.120 | **+3.927** | 0/10 |
| early | 0,1,2,3 | 75.0 | +3.831 | +0.044 | **+3.788** | 0/10 |
| mid | 4,5,6,7 | 75.0 | +1.851 | +0.201 | **+1.650** | 0/10 |
| late | 8,9,10,11 | 70.5 | -0.207 | -0.016 | **-0.191** | 6/10 |
| induction_band | 4,5,6,7 | 75.0 | +1.851 | +0.201 | **+1.650** | 0/10 |

**Findings:** **all** is best (gap +3.93). **early (0–3)** nearly matches all (+3.79). **mid / induction_band (4–7)** still beats random (+1.65) but weaker. **late (8–11)** fails (gap negative, 6/10 fails).

So the live method should keep **all layers** for v1; restricting to the textbook induction band alone under-extracts causal mass on this prompt set, and late-only routing is not enough.
