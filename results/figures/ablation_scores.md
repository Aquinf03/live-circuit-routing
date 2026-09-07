# Ablation — edge score variants

Model: `gpt2-small` · k=15 · B=4 · ban_bos · n=10
Source: `/Users/ashm/work/live/results/runs/ablation_scores_gpt2-small_20260907_122234/score_ablation.json`

| Score | mean |S| | drop_S | drop_R | gap | fails |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw` | 75.0 | +3.359 ± 1.852 | +0.145 ± 0.150 | **+3.214** | 0/10 |
| `attn_x_vnorm` | 67.5 | +4.047 ± 1.939 | +0.120 ± 0.142 | **+3.927** | 0/10 |

- `raw` = attention mass (routing-first)
- `attn_x_vnorm` = A × ‖value‖ at key (default in main results)

Both beat random with 0 fails. **`attn_x_vnorm` wins** (gap +3.93 vs +3.21) with a slightly smaller subgraph — keep it as the paper default.
