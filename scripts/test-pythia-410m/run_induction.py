"""Pythia-410M induction protocol (reproduce GPT-2 Small/Medium result).

Same defaults: score=attn_x_vnorm, k=15, B=4, ban_bos.

You run:
  python scripts/test-pythia-410m/run_induction.py
  python scripts/test-pythia-410m/run_induction.py --limit 5 --n-random 5
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

from run_induction import run_induction_eval  # noqa: E402

# TransformerLens / HF id for Pythia 410M
DEFAULT_MODEL = "EleutherAI/pythia-410m"

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--tag" not in argv:
        argv = ["--tag", "induction_pythia-410m", *argv]
    run_induction_eval(argv, default_model=DEFAULT_MODEL)
