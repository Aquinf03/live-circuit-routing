"""GPT-2 Small: all layers vs early/mid/late/induction-band.

You run:
  python scripts/test-gpt2-small/run_ablation_layers.py
  python scripts/test-gpt2-small/run_ablation_layers.py --limit 10 --n-random 5
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

from ablation_layers import run_layer_ablation  # noqa: E402

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--tag" not in argv:
        argv = ["--tag", "ablation_layers_gpt2-small", *argv]
    run_layer_ablation(argv, default_model="gpt2-small")
