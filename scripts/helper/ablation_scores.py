"""Ablation: edge score variants (raw vs attn×‖v‖)."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Sequence

from ablate import measure_drop
from extract_subgraph import extract_subgraph
from induction_data import build_induction_set
from load_model import attention_from_forward, get_device, load_model
from paths import RESULTS_FIGURES, RESULTS_RUNS, ensure_helper_imports
from routing_graph import build_routing_graph
from save_results import new_run_dir

ensure_helper_imports()

SCORE_MODES = ("raw", "attn_x_vnorm")


def build_arg_parser(default_model: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Edge score variant ablation")
    p.add_argument("--model", type=str, default=default_model)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scores",
        type=str,
        default="raw,attn_x_vnorm",
        help="comma list from {raw,attn_x_vnorm}",
    )
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--no-save", action="store_true")
    return p


def run_score_ablation(argv: Sequence[str] | None, *, default_model: str) -> None:
    args = build_arg_parser(default_model).parse_args(argv)
    device = "cpu" if get_device() == "mps" else get_device()
    model = load_model(name=args.model, device=device)
    examples = build_induction_set(model)
    if args.limit > 0:
        examples = examples[: args.limit]

    modes = [m.strip() for m in args.scores.split(",") if m.strip()]
    for m in modes:
        if m not in SCORE_MODES:
            raise SystemExit(f"unknown score mode {m!r}; choose from {SCORE_MODES}")

    print(
        f"score-ablation model={args.model} device={device} "
        f"k={args.k} B={args.B} n_examples={len(examples)} modes={modes}"
    )

    rows = []
    for mode in modes:
        drops_S, drops_R, n_edges = [], [], []
        for i, ex in enumerate(examples):
            tokens, _, A, Vn = attention_from_forward(model, ex.prefix)
            target_id = int(model.to_tokens(ex.target, prepend_bos=False)[0, 0].item())
            edges = build_routing_graph(
                A, Vn, mode=mode, ban_bos=True, exclude_self=True
            )
            t_star = int(tokens.shape[1] - 1)
            S = extract_subgraph(edges, t_star, k=args.k, B=args.B)
            result = measure_drop(
                model,
                tokens,
                target_id,
                S,
                all_edges=edges,
                n_random=args.n_random,
                seed=args.seed + 1000 * i,
            )
            drops_S.append(result["drop_S"])
            drops_R.append(result["drop_random_mean"])
            n_edges.append(result["n_edges"])

        mean_S = statistics.mean(drops_S)
        mean_R = statistics.mean(drops_R)
        row = {
            "score": mode,
            "mean_n_edges": statistics.mean(n_edges),
            "mean_drop_S": mean_S,
            "std_drop_S": statistics.pstdev(drops_S) if len(drops_S) > 1 else 0.0,
            "mean_drop_R": mean_R,
            "std_drop_R": statistics.pstdev(drops_R) if len(drops_R) > 1 else 0.0,
            "gap": mean_S - mean_R,
            "n_fail": sum(1 for s, r in zip(drops_S, drops_R) if s <= r),
            "n_examples": len(examples),
        }
        rows.append(row)
        print(
            f"{mode:14s} |S|={row['mean_n_edges']:.1f} "
            f"drop_S={row['mean_drop_S']:+.3f} drop_R={row['mean_drop_R']:+.3f} "
            f"gap={row['gap']:+.3f} fails={row['n_fail']}/{row['n_examples']}"
        )

    if not args.no_save:
        tag = args.tag or f"ablation_scores_{args.model.replace('/', '-')}"
        run_dir = new_run_dir(tag)
        payload = {
            "config": {
                "model": args.model,
                "device": device,
                "k": args.k,
                "B": args.B,
                "n_random": args.n_random,
                "limit": args.limit,
                "scores": modes,
            },
            "rows": rows,
        }
        (run_dir / "score_ablation.json").write_text(json.dumps(payload, indent=2) + "\n")

        RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
        md = RESULTS_FIGURES / "ablation_scores.md"
        lines = [
            "# Ablation — edge score variants",
            "",
            f"Model: `{args.model}` · k={args.k} · B={args.B} · ban_bos · n={len(examples)}",
            f"Source: `{run_dir / 'score_ablation.json'}`",
            "",
            "| Score | mean |S| | drop_S | drop_R | gap | fails |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in rows:
            lines.append(
                f"| `{r['score']}` | {r['mean_n_edges']:.1f} | "
                f"{r['mean_drop_S']:+.3f} ± {r['std_drop_S']:.3f} | "
                f"{r['mean_drop_R']:+.3f} ± {r['std_drop_R']:.3f} | "
                f"**{r['gap']:+.3f}** | {r['n_fail']}/{r['n_examples']} |"
            )
        lines += [
            "",
            "- `raw` = attention mass (routing-first)",
            "- `attn_x_vnorm` = A × ‖value‖ at key (default in main results)",
            "",
            "Both should beat random; prefer the one with larger stable gap for the main paper setting.",
            "",
        ]
        md.write_text("\n".join(lines))
        print(f"saved → {run_dir / 'score_ablation.json'}")
        print(f"wrote → {md}")
        print(f"(runs root: {RESULTS_RUNS})")
