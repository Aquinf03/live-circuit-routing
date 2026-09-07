# Pythia-410M — Causal verification vs random control

Source run: `results/runs/induction_pythia-410m_20260907_114107`  
Protocol: same as Small/Medium — induction, score=`attn_x_vnorm`, k=15, B=4, ban_bos=True, n_random=10  
Model: `EleutherAI/pythia-410m`

## Summary across models

| Model | n | mean \|S\| | mean drop_S | mean drop_R | gap (S − R) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 Small | 23 | 68.5 | **+3.655 ± 1.778** | +0.106 ± 0.118 | **+3.548** |
| GPT-2 Medium | 23 | 60.7 | **+1.853 ± 1.920** | +0.004 ± 0.072 | **+1.849** |
| Pythia-410M | 23 | 59.3 | **+1.812 ± 1.334** | +0.031 ± 0.115 | **+1.781** |

Pythia reproduces the effect: extracted subgraph ≫ random, similar gap to Medium.

## Per-example (Pythia-410M)

| # | A / B | \|S\| | drop_S | drop_R (mean±std) |
| ---: | --- | ---: | ---: | ---: |
| 1 | cat / sat | 60 | +0.229 | −0.085 ± 0.240 |
| 2 | dog / ran | 60 | −0.294 | +0.050 ± 0.212 |
| 3 | bird / flew | 60 | +2.095 | +0.029 ± 0.142 |
| 4 | boy / jumped | 60 | +1.799 | −0.141 ± 0.496 |
| 5 | girl / smiled | 60 | −0.422 | −0.011 ± 0.094 |
| 6 | man / walked | 60 | +1.393 | +0.054 ± 0.288 |
| 7 | woman / talked | 60 | +1.398 | −0.038 ± 0.297 |
| 8 | car / stopped | 60 | +0.080 | −0.036 ± 0.366 |
| 9 | train / moved | 60 | +2.720 | +0.037 ± 0.204 |
| 10 | king / ruled | 60 | +4.380 | +0.467 ± 0.552 |
| 11 | queen / spoke | 60 | +1.694 | +0.010 ± 0.138 |
| 12 | teacher / asked | 60 | +3.260 | −0.040 ± 0.156 |
| 13 | student / answered | 60 | +2.774 | −0.047 ± 0.217 |
| 14 | doctor / helped | 60 | +3.074 | +0.139 ± 0.455 |
| 15 | player / scored | 60 | +0.543 | −0.030 ± 0.212 |
| 16 | artist / painted | 60 | +3.991 | +0.050 ± 0.534 |
| 17 | writer / wrote | 60 | +2.569 | +0.020 ± 0.241 |
| 18 | chef / cooked | 45 | +2.163 | −0.077 ± 0.318 |
| 19 | farmer / worked | 60 | +2.876 | +0.107 ± 0.198 |
| 20 | river / flowed | 60 | +0.730 | +0.084 ± 0.298 |
| 21 | fire / burned | 60 | +1.793 | −0.002 ± 0.219 |
| 22 | wind / blew | 60 | +2.812 | +0.127 ± 0.297 |
| 23 | door / opened | 60 | +0.027 | +0.056 ± 0.240 |

Failures / weak: `dog/ran`, `girl/smiled` (negative); `door/opened` (below random).

Raw: `results/runs/induction_pythia-410m_20260907_114107/metrics.csv`
