"""GPT-2 Small: top-k / τ sensitivity.

You run:
  python scripts/test-gpt2-small/run_sensitivity_k_tau.py
  python scripts/test-gpt2-small/run_sensitivity_k_tau.py --limit 8 --n-random 5
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

from sweep_k_tau import run_sensitivity  # noqa: E402

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--tag" not in argv:
        argv = ["--tag", "sensitivity_k_tau_gpt2-small", *argv]
    run_sensitivity(argv, default_model="gpt2-small")
