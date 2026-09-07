"""GPT-2 Small cost bench: live extract vs head-patching.

You run:
  python scripts/test-gpt2-small/run_cost.py
  python scripts/test-gpt2-small/run_cost.py --limit 3
  python scripts/test-gpt2-small/run_cost.py --skip-head-patch   # faster smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

from bench_cost import run_cost_bench  # noqa: E402

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--tag" not in argv:
        argv = ["--tag", "cost_gpt2-small", *argv]
    run_cost_bench(argv, default_model="gpt2-small")
