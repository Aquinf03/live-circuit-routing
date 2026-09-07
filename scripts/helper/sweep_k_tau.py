"""Sweep top-k and threshold τ; report mean drop_S / drop_R / gap."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Sequence

from ablate import measure_drop
from extract_subgraph import extract_subgraph, extract_subgraph_threshold
from induction_data import build_induction_set
from load_model import attention_from_forward, get_device, load_model
from paths import RESULTS_FIGURES, RESULTS_RUNS, ensure_helper_imports
from routing_graph import build_routing_graph
from save_results import new_run_dir

ensure_helper_imports()


def build_arg_parser(default_model: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="k / τ sensitivity sweep")
    p.add_argument("--model", type=str, default=default_model)
    p.add_argument("--score", choices=("raw", "attn_x_vnorm"), default="attn_x_vnorm")
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--n-random", type=int, default=5, help="random controls per example")
    p.add_argument("--limit", type=int, default=10, help="examples (0=all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ks", type=str, default="5,10,15,20,30", help="comma list of k")
    p.add_argument(
        "--taus",
        type=str,
        default="0.5,1.0,2.0,5.0",
        help="comma list of score thresholds (attn_x_vnorm; use smaller for --score raw)",
    )
    p.add_argument("--skip-tau", action="store_true")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--no-save", action="store_true")
    return p


def _mean_gap(model, examples, *, score, B, n_random, seed, extract_fn):
    drops_S, drops_R, n_edges = [], [], []
    for i, ex in enumerate(examples):
        tokens, _, A, Vn = attention_from_forward(model, ex.prefix)
        target_id = int(model.to_tokens(ex.target, prepend_bos=False)[0, 0].item())
        edges = build_routing_graph(
            A, Vn, mode=score, ban_bos=True, exclude_self=True
        )
        t_star = int(tokens.shape[1] - 1)
        S = extract_fn(edges, t_star)
        if not S:
            drops_S.append(0.0)
            drops_R.append(0.0)
            n_edges.append(0)
            continue
        result = measure_drop(
            model,
            tokens,
            target_id,
            S,
            all_edges=edges,
            n_random=n_random,
            seed=seed + 1000 * i,
        )
        drops_S.append(result["drop_S"])
        drops_R.append(result["drop_random_mean"])
        n_edges.append(result["n_edges"])
    mean_S = statistics.mean(drops_S)
    mean_R = statistics.mean(drops_R)
    return {
        "mean_n_edges": statistics.mean(n_edges),
        "mean_drop_S": mean_S,
        "mean_drop_R": mean_R,
        "gap": mean_S - mean_R,
        "n_examples": len(examples),
        "n_fail": sum(1 for s, r in zip(drops_S, drops_R) if s <= r),
    }


def run_sensitivity(argv: Sequence[str] | None, *, default_model: str) -> None:
    args = build_arg_parser(default_model).parse_args(argv)
    device = "cpu" if get_device() == "mps" else get_device()
    model = load_model(name=args.model, device=device)
    examples = build_induction_set(model)
    if args.limit > 0:
        examples = examples[: args.limit]

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    taus = [float(x) for x in args.taus.split(",") if x.strip()]

    print(
        f"sensitivity model={args.model} device={device} score={args.score} "
        f"B={args.B} n_examples={len(examples)} ks={ks} taus={taus if not args.skip_tau else 'skipped'}"
    )

    k_rows = []
    for k in ks:
        stats = _mean_gap(
            model,
            examples,
            score=args.score,
            B=args.B,
            n_random=args.n_random,
            seed=args.seed,
            extract_fn=lambda edges, t, k=k: extract_subgraph(
                edges, t, k=k, B=args.B
            ),
        )
        row = {"cut": "k", "value": k, **stats}
        k_rows.append(row)
        print(
            f"k={k:3d} |S|={stats['mean_n_edges']:.1f} "
            f"drop_S={stats['mean_drop_S']:+.3f} drop_R={stats['mean_drop_R']:+.3f} "
            f"gap={stats['gap']:+.3f} fails={stats['n_fail']}/{stats['n_examples']}"
        )

    tau_rows = []
    if not args.skip_tau:
        for tau in taus:
            stats = _mean_gap(
                model,
                examples,
                score=args.score,
                B=args.B,
                n_random=args.n_random,
                seed=args.seed,
                extract_fn=lambda edges, t, tau=tau: extract_subgraph_threshold(
                    edges, t, tau=tau, B=args.B
                ),
            )
            row = {"cut": "tau", "value": tau, **stats}
            tau_rows.append(row)
            print(
                f"τ={tau:.2f} |S|={stats['mean_n_edges']:.1f} "
                f"drop_S={stats['mean_drop_S']:+.3f} drop_R={stats['mean_drop_R']:+.3f} "
                f"gap={stats['gap']:+.3f} fails={stats['n_fail']}/{stats['n_examples']}"
            )

    if not args.no_save:
        tag = args.tag or f"sensitivity_k_tau_{args.model.replace('/', '-')}"
        run_dir = new_run_dir(tag)
        payload = {
            "config": {
                "model": args.model,
                "device": device,
                "score": args.score,
                "B": args.B,
                "n_random": args.n_random,
                "limit": args.limit,
                "ks": ks,
                "taus": taus if not args.skip_tau else [],
            },
            "k_rows": k_rows,
            "tau_rows": tau_rows,
        }
        (run_dir / "sensitivity.json").write_text(json.dumps(payload, indent=2) + "\n")

        RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
        md = RESULTS_FIGURES / "ablation_k_tau.md"
        lines = [
            "# Ablation — top-k / threshold τ sensitivity",
            "",
            f"Model: `{args.model}` · score=`{args.score}` · B={args.B} · n={len(examples)}",
            f"Source: `{run_dir / 'sensitivity.json'}`",
            "",
            "## Top-k",
            "",
            "| k | mean |S| | mean drop_S | mean drop_R | gap | fails |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in k_rows:
            lines.append(
                f"| {int(r['value'])} | {r['mean_n_edges']:.1f} | "
                f"{r['mean_drop_S']:+.3f} | {r['mean_drop_R']:+.3f} | "
                f"**{r['gap']:+.3f}** | {r['n_fail']}/{r['n_examples']} |"
            )
        if tau_rows:
            lines += [
                "",
                "## Threshold τ",
                "",
                "| τ | mean |S| | mean drop_S | mean drop_R | gap | fails |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for r in tau_rows:
                lines.append(
                    f"| {float(r['value']):.2f} | {r['mean_n_edges']:.1f} | "
                    f"{r['mean_drop_S']:+.3f} | {r['mean_drop_R']:+.3f} | "
                    f"**{r['gap']:+.3f}** | {r['n_fail']}/{r['n_examples']} |"
                )
        lines += [
            "",
            "Success: gap stays clearly > 0 across reasonable k / τ. "
            "Default paper setting k=15 should not be a brittle spike.",
            "",
        ]
        md.write_text("\n".join(lines))
        print(f"saved → {run_dir / 'sensitivity.json'}")
        print(f"wrote → {md}")
        print(f"(runs root: {RESULTS_RUNS})")
