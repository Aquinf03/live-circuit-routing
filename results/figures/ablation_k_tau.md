# Ablation — top-k / threshold τ sensitivity

Model: `gpt2-small` · score=`attn_x_vnorm` · B=4 · n=10
Source: `/Users/ashm/work/live/results/runs/sensitivity_k_tau_gpt2-small_20260907_121149/sensitivity.json`

## Top-k

| k | mean \|S\| | mean drop_S | mean drop_R | gap | fails |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 17.0 | +0.074 | +0.037 | **+0.037** | 4/10 |
| 10 | 40.0 | +2.198 | +0.053 | **+2.145** | 1/10 |
| 15 | 67.5 | +4.047 | +0.120 | **+3.927** | 0/10 |
| 20 | 98.0 | +4.990 | +0.141 | **+4.849** | 0/10 |
| 30 | 150.0 | +6.363 | +0.214 | **+6.150** | 0/10 |

## Threshold τ

| τ | mean \|S\| | mean drop_S | mean drop_R | gap | fails |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 548.6 | +9.392 | +1.194 | **+8.198** | 0/10 |
| 1.00 | 222.9 | +8.559 | +0.356 | **+8.203** | 0/10 |
| 2.00 | 68.7 | +3.182 | +0.145 | **+3.037** | 0/10 |
| 5.00 | 4.9 | -0.360 | +0.013 | **-0.373** | 8/10 |

Success: gap stays clearly > 0 for **k ≥ 10** and moderate τ (0.5–2).  
Default **k=15** is in the stable regime (0 fails on this slice).  
**k=5** under-extracts; **τ=5** over-prunes (gap flips negative).
