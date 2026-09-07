"""Repo paths for scripts/helper and scripts/test-*."""

from __future__ import annotations

import sys
from pathlib import Path

HELPER_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = HELPER_DIR.parent
ROOT = SCRIPTS_DIR.parent
RESULTS_RUNS = ROOT / "results" / "runs"
RESULTS_FIGURES = ROOT / "results" / "figures"


def ensure_helper_imports() -> None:
    if str(HELPER_DIR) not in sys.path:
        sys.path.insert(0, str(HELPER_DIR))
