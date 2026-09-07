"""GPT-2 Medium: IOI eval (n=100 by default)."""

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
        argv = ["--tag", "ioi_gpt2-medium", *argv]
    run_ioi_eval(argv, default_model="gpt2-medium")
