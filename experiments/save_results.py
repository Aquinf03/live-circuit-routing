"""Save experiment metrics and subgraphs to disk."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from routing_graph import Edge

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results"


def new_run_dir(tag: str = "induction") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{tag}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def edge_to_dict(e: Edge) -> dict[str, Any]:
    return {
        "layer": e.layer,
        "head": e.head,
        "key": e.key,
        "query": e.query,
        "score": e.score,
    }


def save_run(
    run_dir: Path,
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    subgraphs: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "subgraphs.json").write_text(json.dumps(subgraphs, indent=2) + "\n")

    if rows:
        fieldnames = list(rows[0].keys())
        with (run_dir / "metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
