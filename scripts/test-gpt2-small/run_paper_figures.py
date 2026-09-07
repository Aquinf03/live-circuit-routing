"""Write paper/figures/fig{1..4}.png

You run:
  python scripts/helper/fig_paper.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

from fig_paper import main  # noqa: E402

if __name__ == "__main__":
    main()
