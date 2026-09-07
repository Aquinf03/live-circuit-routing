# scripts/

- `helper/` — shared library (load, graph, extract, ablate, save, plot, table, run_induction)
- `test-gpt2-small/` — GPT-2 Small induction protocol
- `test-gpt2-medium/` — GPT-2 Medium reproduction
- `test-pythia-410m/` — Pythia-410M reproduction

## Run (you)

```bash
source .venv/bin/activate

# small (already done for Task 1)
python scripts/test-gpt2-small/run_induction.py

# medium reproduction (same defaults: attn_x_vnorm, k=15, B=4)
python scripts/test-gpt2-medium/run_induction.py

# pythia-410m reproduction
python scripts/test-pythia-410m/run_induction.py --limit 5 --n-random 5   # smoke
python scripts/test-pythia-410m/run_induction.py                         # full

# Task 2: IOI on gpt2-small (same extract protocol)
python scripts/test-gpt2-small/run_ioi.py --limit 5 --n-random 5
python scripts/test-gpt2-small/run_ioi.py

# optional smoke
python scripts/test-gpt2-medium/run_induction.py --limit 5 --n-random 5
```

Results land in `results/runs/induction_<model>_<timestamp>/`.
