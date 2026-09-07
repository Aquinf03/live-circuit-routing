# Task 1 — Causal verification vs random control (GPT-2 Small)

Source run: `results/runs/induction_20260907_105902`  
Protocol: induction `The A B. The A → B`, score=`attn_x_vnorm`, k=15, B=4, ban_bos=True, n_random=10, seed=0

## Summary

| Model | n | mean \|S\| | mean drop_S ↓ | mean drop_R ↓ | gap (S − R) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 Small | 23 | 68.5 | **+3.655 ± 1.778** | +0.106 ± 0.118 | **+3.548** |

drop = logit(target) before ablation − after ablation. Higher = extracted edges were more causal.  
Success criterion: drop_S ≫ drop_R.

## Per-example

| # | A / B | \|S\| | drop_S | drop_R (mean±std) |
| ---: | --- | ---: | ---: | ---: |
| 1 | cat / sat | 60 | +7.232 | +0.001 ± 0.356 |
| 2 | dog / ran | 60 | +5.601 | +0.112 ± 0.080 |
| 3 | bird / flew | 75 | +2.559 | +0.006 ± 0.184 |
| 4 | boy / jumped | 60 | +3.409 | +0.029 ± 0.156 |
| 5 | girl / smiled | 75 | +3.618 | −0.086 ± 0.550 |
| 6 | man / walked | 60 | +3.357 | +0.026 ± 0.640 |
| 7 | woman / talked | 60 | +7.347 | +0.014 ± 0.127 |
| 8 | car / stopped | 75 | +0.996 | +0.232 ± 0.335 |
| 9 | train / moved | 75 | +3.052 | +0.262 ± 0.398 |
| 10 | king / ruled | 75 | +3.295 | −0.061 ± 0.663 |
| 11 | queen / spoke | 45 | +2.353 | +0.105 ± 0.173 |
| 12 | teacher / asked | 60 | +4.370 | +0.151 ± 0.374 |
| 13 | student / answered | 75 | +2.363 | +0.203 ± 0.565 |
| 14 | doctor / helped | 60 | +3.172 | +0.362 ± 0.342 |
| 15 | player / scored | 75 | +2.631 | −0.017 ± 0.271 |
| 16 | artist / painted | 75 | +6.155 | +0.046 ± 0.224 |
| 17 | writer / wrote | 75 | +2.448 | +0.142 ± 0.144 |
| 18 | chef / cooked | 75 | +6.854 | +0.254 ± 0.413 |
| 19 | farmer / worked | 60 | +4.234 | +0.163 ± 0.124 |
| 20 | river / flowed | 75 | +2.625 | +0.067 ± 0.225 |
| 21 | fire / burned | 75 | +2.252 | +0.222 ± 0.501 |
| 22 | wind / blew | 75 | +3.086 | +0.238 ± 0.307 |
| 23 | door / opened | 75 | +1.047 | −0.027 ± 0.151 |

Raw rows: `results/runs/induction_20260907_105902/metrics.csv`
