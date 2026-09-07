"""GPT-2 Small: raw vs attn×‖v‖ score ablation.

You run:
  python scripts/test-gpt2-small/run_ablation_scores.py
  python scripts/test-gpt2-small/run_ablation_scores.py --limit 10 --n-random 5
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

from ablation_scores import run_score_ablation  # noqa: E402

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--tag" not in argv:
        argv = ["--tag", "ablation_scores_gpt2-small", *argv]
    run_score_ablation(argv, default_model="gpt2-small")
