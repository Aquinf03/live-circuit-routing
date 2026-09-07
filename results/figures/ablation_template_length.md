# Ablation — prompt template / length sensitivity

Model: `gpt2-small` · score=`attn_x_vnorm` · k=15 · B=4 · n≤10
Source: `/Users/ashm/work/live/results/runs/ablation_template_length_gpt2-small_20260907_123953/template_length.json`

## Templates

| Template | mean seq len | mean |S| | drop_S | drop_R | gap | fails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `the_dot` | 7.0 | 67.5 | +4.047 | +0.120 | **+3.927** | 0/10 |
| `bare` | 4.0 | 30.0 | +2.949 | +0.311 | **+2.637** | 1/10 |
| `then` | 7.0 | 75.0 | +5.480 | +0.069 | **+5.411** | 0/10 |
| `yesterday` | 10.0 | 111.0 | +2.313 | +0.123 | **+2.190** | 0/10 |

## Length (filler sentences before `the_dot` pattern)

| Fillers | mean seq len | mean |S| | drop_S | drop_R | gap | fails |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7.0 | 67.5 | +4.047 | +0.120 | **+3.927** | 0/10 |
| 1 | 12.0 | 144.0 | +2.588 | +0.014 | **+2.574** | 1/10 |
| 2 | 17.0 | 207.0 | +3.189 | +0.092 | **+3.097** | 0/10 |
| 4 | 27.0 | 228.0 | +4.065 | +0.013 | **+4.052** | 0/10 |

Success: gap stays > 0 across templates and moderate lengths. Main template `the_dot` should not be a one-template artifact.
