# scripts/

- `helper/` — shared library (load, graph, extract, ablate, save, plot, table, run_induction)
- `test-gpt2-small/` — GPT-2 Small induction protocol
- `test-gpt2-medium/` — GPT-2 Medium reproduction
- (later) `test-pythia-410m/`

## Run (you)

```bash
source .venv/bin/activate

# small (already done for Task 1)
python scripts/test-gpt2-small/run_induction.py

# medium reproduction (same defaults: attn_x_vnorm, k=15, B=4)
python scripts/test-gpt2-medium/run_induction.py

# optional smoke
python scripts/test-gpt2-medium/run_induction.py --limit 5 --n-random 5
```

Results land in `results/runs/induction_gpt2-medium_<timestamp>/`.
