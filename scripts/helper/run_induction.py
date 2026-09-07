"""Shared induction eval runner (used by test-gpt2-small / test-gpt2-medium / …)."""

from __future__ import annotations

import argparse
import statistics
from typing import Sequence

from ablate import measure_drop
from extract_subgraph import extract_subgraph
from induction_data import build_induction_set
from load_model import attention_from_forward, get_device, load_model
from paths import RESULTS_RUNS, ensure_helper_imports
from routing_graph import build_routing_graph
from save_results import edge_to_dict, new_run_dir, save_run
from stats_gap import format_gap_stats, paired_gap_stats

ensure_helper_imports()


def build_arg_parser(default_model: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Live routing-graph induction eval ({default_model})")
    p.add_argument("--model", type=str, default=default_model)
    p.add_argument("--k", type=int, default=15, help="top-k incoming edges per hop")
    p.add_argument("--B", type=int, default=4, help="backward hops from t*")
    p.add_argument(
        "--score",
        choices=("raw", "attn_x_vnorm"),
        default="attn_x_vnorm",
        help="edge score mode",
    )
    p.add_argument("--n-random", type=int, default=10, help="random controls per example")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n",
        type=int,
        default=100,
        help="number of clean induction examples (0 = use full filtered pool)",
    )
    p.add_argument("--limit", type=int, default=0, help="alias for --n if >0 (deprecated)")
    p.add_argument("--no-ban-bos", action="store_true", help="keep BOS sink edges")
    p.add_argument("--no-save", action="store_true", help="skip writing results/runs")
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="results folder tag prefix (default: induction_<model>)",
    )
    return p


def run_induction_eval(argv: Sequence[str] | None, *, default_model: str) -> None:
    args = build_arg_parser(default_model).parse_args(argv)
    device = "cpu" if get_device() == "mps" else get_device()
    model = load_model(name=args.model, device=device)
    n = args.limit if args.limit > 0 else args.n
    n_arg = None if n == 0 else n
    examples = build_induction_set(model, n=n_arg)

    ban_bos = not args.no_ban_bos
    drops_S: list[float] = []
    drops_R: list[float] = []
    n_edges: list[int] = []
    rows: list[dict] = []
    subgraphs: list[dict] = []

    print(
        f"model={args.model} device={device} score={args.score} k={args.k} B={args.B} "
        f"ban_bos={ban_bos} n_examples={len(examples)} n_random={args.n_random}"
    )

    for i, ex in enumerate(examples):
        tokens, _, A, Vn = attention_from_forward(model, ex.prefix)
        target_id = int(model.to_tokens(ex.target, prepend_bos=False)[0, 0].item())
        edges = build_routing_graph(
            A,
            Vn,
            mode=args.score,
            ban_bos=ban_bos,
            exclude_self=True,
        )
        t_star = int(tokens.shape[1] - 1)
        S = extract_subgraph(edges, t_star, k=args.k, B=args.B, exclude_self=True)
        ex_seed = args.seed + 1000 * i
        result = measure_drop(
            model,
            tokens,
            target_id,
            S,
            all_edges=edges,
            n_random=args.n_random,
            seed=ex_seed,
        )
        drops_S.append(result["drop_S"])
        drops_R.append(result["drop_random_mean"])
        n_edges.append(result["n_edges"])
        str_tokens = model.to_str_tokens(tokens[0])
        rows.append(
            {
                "idx": i,
                "a": ex.a,
                "b": ex.b,
                "prefix": ex.prefix,
                "target": ex.target,
                "target_id": target_id,
                "t_star": t_star,
                "n_edges_S": result["n_edges"],
                "logit_full": result["logit_full"],
                "logit_ablate_S": result["logit_ablate_S"],
                "drop_S": result["drop_S"],
                "drop_random_mean": result["drop_random_mean"],
                "drop_random_std": result["drop_random_std"],
                "seed": ex_seed,
            }
        )
        subgraphs.append(
            {
                "idx": i,
                "a": ex.a,
                "b": ex.b,
                "tokens": str_tokens,
                "t_star": t_star,
                "edges": [edge_to_dict(e) for e in S],
            }
        )
        print(
            f"[{i+1:02d}/{len(examples)}] {ex.a}/{ex.b} "
            f"|S|={result['n_edges']} drop_S={result['drop_S']:+.3f} "
            f"drop_R={result['drop_random_mean']:+.3f}±{result['drop_random_std']:.3f}"
        )

    mean_S = statistics.mean(drops_S)
    mean_R = statistics.mean(drops_R)
    std_S = statistics.pstdev(drops_S) if len(drops_S) > 1 else 0.0
    std_R = statistics.pstdev(drops_R) if len(drops_R) > 1 else 0.0
    gap = mean_S - mean_R
    gap_stats = paired_gap_stats(drops_S, drops_R, seed=args.seed)
    summary = {
        "mean_n_edges": statistics.mean(n_edges),
        "mean_drop_S": mean_S,
        "std_drop_S": std_S,
        "mean_drop_R": mean_R,
        "std_drop_R": std_R,
        "gap_S_minus_R": gap,
        "n_examples": len(examples),
        **{f"paired_{k}": v for k, v in gap_stats.items()},
    }

    print("---")
    print(f"mean |S|={summary['mean_n_edges']:.1f}")
    print(f"mean drop_S={mean_S:+.4f} ± {std_S:.4f}")
    print(f"mean drop_R={mean_R:+.4f} ± {std_R:.4f}")
    print(f"gap (S - R)={gap:+.4f}  (want clearly > 0)")
    print(format_gap_stats(gap_stats))

    if not args.no_save:
        tag = args.tag or f"induction_{args.model.replace('/', '-')}"
        run_dir = new_run_dir(tag)
        config = {
            "model": args.model,
            "device": device,
            "score": args.score,
            "k": args.k,
            "B": args.B,
            "ban_bos": ban_bos,
            "exclude_self": True,
            "n_random": args.n_random,
            "seed": args.seed,
            "n": n,
            "limit": args.limit,
            "n_examples": len(examples),
        }
        save_run(
            run_dir,
            config=config,
            rows=rows,
            subgraphs=subgraphs,
            summary=summary,
        )
        print(f"saved → {run_dir}")
        print(f"(runs root: {RESULTS_RUNS})")
