# GPT-2 Medium — Causal verification vs random control

Source run: `results/runs/induction_gpt2-medium_20260907_113405`  
Protocol: same as Small — induction `The A B. The A → B`, score=`attn_x_vnorm`, k=15, B=4, ban_bos=True, n_random=10

## Summary vs GPT-2 Small

| Model | n | mean \|S\| | mean drop_S | mean drop_R | gap (S − R) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 Small | 23 | 68.5 | **+3.655 ± 1.778** | +0.106 ± 0.118 | **+3.548** |
| GPT-2 Medium | 23 | 60.7 | **+1.853 ± 1.920** | +0.004 ± 0.072 | **+1.849** |

Medium reproduces the sign of the effect: extracted subgraph ≫ random. Absolute drop is smaller than Small (expected; deeper/wider model, same k/B budget).

## Per-example (Medium)

| # | A / B | \|S\| | drop_S | drop_R (mean±std) |
| ---: | --- | ---: | ---: | ---: |
| 1 | cat / sat | 60 | +1.112 | −0.065 ± 0.139 |
| 2 | dog / ran | 60 | +0.360 | −0.011 ± 0.107 |
| 3 | bird / flew | 60 | +1.749 | +0.026 ± 0.067 |
| 4 | boy / jumped | 60 | +0.724 | +0.026 ± 0.074 |
| 5 | girl / smiled | 60 | +1.581 | −0.060 ± 0.162 |
| 6 | man / walked | 60 | +4.360 | +0.104 ± 0.131 |
| 7 | woman / talked | 60 | +3.540 | +0.101 ± 0.165 |
| 8 | car / stopped | 60 | +0.335 | −0.009 ± 0.091 |
| 9 | train / moved | 60 | +0.353 | −0.125 ± 0.281 |
| 10 | king / ruled | 60 | +3.309 | −0.010 ± 0.075 |
| 11 | queen / spoke | 60 | +2.592 | −0.000 ± 0.075 |
| 12 | teacher / asked | 60 | +0.759 | −0.013 ± 0.064 |
| 13 | student / answered | 60 | +3.153 | +0.093 ± 0.120 |
| 14 | doctor / helped | 60 | +0.119 | −0.021 ± 0.094 |
| 15 | player / scored | 60 | +3.708 | +0.050 ± 0.082 |
| 16 | artist / painted | 60 | +5.327 | −0.140 ± 0.399 |
| 17 | writer / wrote | 60 | +4.668 | +0.171 ± 0.218 |
| 18 | chef / cooked | 60 | +5.182 | −0.079 ± 0.236 |
| 19 | farmer / worked | 60 | +1.074 | +0.017 ± 0.099 |
| 20 | river / flowed | 60 | +0.232 | +0.006 ± 0.170 |
| 21 | fire / burned | 75 | +0.984 | +0.074 ± 0.133 |
| 22 | wind / blew | 60 | −1.181 | −0.039 ± 0.063 |
| 23 | door / opened | 60 | −1.418 | +0.000 ± 0.053 |

Failure cases to note: `wind/blew`, `door/opened` (negative drop_S).

Raw: `results/runs/induction_gpt2-medium_20260907_113405/metrics.csv`
