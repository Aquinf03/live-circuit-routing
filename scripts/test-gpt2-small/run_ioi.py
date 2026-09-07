"""GPT-2 Small IOI protocol (Task 2).

Same extract settings as induction: attn_x_vnorm, k=15, B=4, ban_bos.
Metric: logit(IO) - logit(S).

You run:
  python scripts/test-gpt2-small/run_ioi.py --limit 5 --n-random 5
  python scripts/test-gpt2-small/run_ioi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

from run_ioi import run_ioi_eval  # noqa: E402

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--tag" not in argv:
        argv = ["--tag", "ioi_gpt2-small", *argv]
    run_ioi_eval(argv, default_model="gpt2-small")
