"""Build causal-vs-random summary table from a saved induction run.

Usage (you run this):
  python experiments/make_table.py --run data/results/induction_20260907_105902
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parent
ROOT = _EXPERIMENTS.parent
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))


def latest_run() -> Path | None:
    root = ROOT / "data" / "results"
    runs = sorted(root.glob("induction_*"), key=lambda p: p.name) if root.exists() else []
    return runs[-1] if runs else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default="")
    p.add_argument("--out-dir", type=str, default=str(ROOT / "figures"))
    args = p.parse_args()

    run_dir = Path(args.run) if args.run else latest_run()
    if run_dir is None or not (run_dir / "metrics.csv").exists():
        raise SystemExit("no run found; pass --run data/results/induction_...")

    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    with (run_dir / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV summary (paper table one-liner)
    csv_path = out_dir / "table_causal_vs_random.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "n",
                "mean_abs_S",
                "mean_drop_S",
                "std_drop_S",
                "mean_drop_R",
                "std_drop_R",
                "gap_S_minus_R",
                "score",
                "k",
                "B",
                "ban_bos",
                "n_random",
                "seed",
                "run",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "model": config.get("model", ""),
                "n": summary["n_examples"],
                "mean_abs_S": f"{summary['mean_n_edges']:.3f}",
                "mean_drop_S": f"{summary['mean_drop_S']:.4f}",
                "std_drop_S": f"{summary['std_drop_S']:.4f}",
                "mean_drop_R": f"{summary['mean_drop_R']:.4f}",
                "std_drop_R": f"{summary['std_drop_R']:.4f}",
                "gap_S_minus_R": f"{summary['gap_S_minus_R']:.4f}",
                "score": config.get("score", ""),
                "k": config.get("k", ""),
                "B": config.get("B", ""),
                "ban_bos": config.get("ban_bos", ""),
                "n_random": config.get("n_random", ""),
                "seed": config.get("seed", ""),
                "run": run_dir.name,
            }
        )

    # Markdown table
    md_path = out_dir / "table_causal_vs_random.md"
    lines = [
        "# Task 1 — Causal verification vs random control (GPT-2 Small)",
        "",
        f"Source run: `{run_dir}`",
        f"Protocol: induction, score=`{config.get('score')}`, k={config.get('k')}, "
        f"B={config.get('B')}, ban_bos={config.get('ban_bos')}, "
        f"n_random={config.get('n_random')}, seed={config.get('seed')}",
        "",
        "## Summary",
        "",
        "| Model | n | mean |S| | mean drop_S ↓ | mean drop_R ↓ | gap (S − R) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| {config.get('model')} | {summary['n_examples']} | "
        f"{summary['mean_n_edges']:.1f} | "
        f"**{summary['mean_drop_S']:+.3f} ± {summary['std_drop_S']:.3f}** | "
        f"{summary['mean_drop_R']:+.3f} ± {summary['std_drop_R']:.3f} | "
        f"**{summary['gap_S_minus_R']:+.3f}** |",
        "",
        "drop = logit(target) before − after ablation. Want drop_S ≫ drop_R.",
        "",
        "## Per-example",
        "",
        "| # | A / B | |S| | drop_S | drop_R (mean±std) |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {int(r['idx'])+1} | {r['a']} / {r['b']} | {r['n_edges_S']} | "
            f"{float(r['drop_S']):+.3f} | "
            f"{float(r['drop_random_mean']):+.3f} ± {float(r['drop_random_std']):.3f} |"
        )
    lines.append("")
    lines.append(f"Raw rows: `{run_dir / 'metrics.csv'}`")
    lines.append("")
    md_path.write_text("\n".join(lines))

    n_pos = sum(1 for r in rows if float(r["drop_S"]) > float(r["drop_random_mean"]))
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(
        f"drop_S > drop_R on {n_pos}/{len(rows)} examples; "
        f"gap={summary['gap_S_minus_R']:+.3f}"
    )


if __name__ == "__main__":
    main()
