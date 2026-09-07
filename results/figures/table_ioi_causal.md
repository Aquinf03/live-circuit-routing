# Task 2 — IOI causal verification (GPT-2 Small)

Source run: `results/runs/ioi_gpt2-small_20260907_114655`  
Protocol: same live extract as induction (`attn_x_vnorm`, k=15, B=4, ban_bos)  
Metric: **logit(IO) − logit(S)**; drop = full − ablated

## Summary

| Task | Model | n | mean \|S\| | mean drop_S | mean drop_R | gap (S − R) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Induction | GPT-2 Small | 23 | 68.5 | +3.655 | +0.106 | **+3.548** |
| **IOI** | GPT-2 Small | 23 | 195.0 | **+5.270 ± 1.923** | +0.098 ± 0.270 | **+5.171** |

Not one-trick: same method transfers to IOI with an even larger gap.

## Per-example

| # | A → B (IO) | ld_full | drop_S | drop_R |
| ---: | --- | ---: | ---: | ---: |
| 1 | John → Mary | +3.172 | +3.566 | −0.074 |
| 2 | Mary → John | +2.479 | +1.435 | −0.001 |
| 3 | Alice → Bob | +3.096 | +3.546 | −0.033 |
| 4 | Bob → Alice | +3.891 | +3.872 | +0.206 |
| 5 | Tom → Sarah | +5.349 | +8.648 | +0.067 |
| 6 | Sarah → Tom | +5.460 | +4.414 | +0.212 |
| 7 | James → Emily | +5.553 | +6.573 | −0.076 |
| 8 | Emily → James | +5.588 | +7.169 | +0.621 |
| 9 | David → Lisa | +5.374 | +4.495 | +0.219 |
| 10 | Lisa → David | +3.019 | +4.237 | +0.129 |
| 11 | Paul → Anna | +5.343 | +5.480 | −0.565 |
| 12 | Anna → Paul | +3.647 | +4.068 | −0.048 |
| 13 | Mark → Laura | +5.499 | +6.236 | +0.013 |
| 14 | Laura → Mark | +3.876 | +4.726 | −0.307 |
| 15 | Chris → Emma | +4.818 | +5.655 | +0.614 |
| 16 | Emma → Chris | +3.527 | +4.487 | −0.020 |
| 17 | Sam → Kate | +4.494 | +6.144 | +0.143 |
| 18 | Kate → Sam | +4.475 | +2.187 | −0.015 |
| 19 | Dan → Ruth | +5.319 | +6.835 | −0.104 |
| 20 | Ruth → Dan | +7.292 | +4.887 | +0.578 |
| 21 | Joe → Amy | +6.760 | +9.188 | +0.229 |
| 22 | Amy → Joe | +5.267 | +4.707 | +0.119 |
| 23 | Tim → Nina | +6.204 | +8.646 | +0.355 |

All 23 examples: drop_S > drop_R.

Raw: `results/runs/ioi_gpt2-small_20260907_114655/metrics.csv`
