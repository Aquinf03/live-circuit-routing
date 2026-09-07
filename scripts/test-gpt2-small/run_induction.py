"""GPT-2 Small induction protocol.

You run:
  python scripts/test-gpt2-small/run_induction.py
  python scripts/test-gpt2-small/run_induction.py --limit 5 --n-random 5
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

from run_induction import run_induction_eval  # noqa: E402

if __name__ == "__main__":
    run_induction_eval(sys.argv[1:], default_model="gpt2-small")
