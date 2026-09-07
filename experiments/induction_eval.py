"""Induction eval: extract live subgraph, ablate, average drop vs random."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parent
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from ablate import measure_drop
from extract_subgraph import extract_subgraph
from induction_data import build_induction_set
from load_model import attention_from_forward, get_device, load_model
from routing_graph import build_routing_graph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live routing-graph induction eval")
    p.add_argument("--k", type=int, default=10, help="top-k incoming edges per hop")
    p.add_argument("--B", type=int, default=3, help="backward hops from t*")
    p.add_argument(
        "--score",
        choices=("raw", "attn_x_vnorm"),
        default="attn_x_vnorm",
        help="edge score mode",
    )
    p.add_argument("--n-random", type=int, default=10, help="random controls per example")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="use first N examples (0=all)")
    p.add_argument("--no-ban-bos", action="store_true", help="keep BOS sink edges")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cpu" if get_device() == "mps" else get_device()
    model = load_model(device=device)
    examples = build_induction_set(model)
    if args.limit > 0:
        examples = examples[: args.limit]

    ban_bos = not args.no_ban_bos
    drops_S: list[float] = []
    drops_R: list[float] = []
    n_edges: list[int] = []

    print(
        f"device={device} score={args.score} k={args.k} B={args.B} "
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
        # random pool should match extract constraints
        t_star = int(tokens.shape[1] - 1)
        S = extract_subgraph(edges, t_star, k=args.k, B=args.B, exclude_self=True)
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
        print(
            f"[{i+1:02d}/{len(examples)}] {ex.a}/{ex.b} "
            f"|S|={result['n_edges']} drop_S={result['drop_S']:+.3f} "
            f"drop_R={result['drop_random_mean']:+.3f}±{result['drop_random_std']:.3f}"
        )

    mean_S = statistics.mean(drops_S)
    mean_R = statistics.mean(drops_R)
    std_S = statistics.pstdev(drops_S) if len(drops_S) > 1 else 0.0
    std_R = statistics.pstdev(drops_R) if len(drops_R) > 1 else 0.0
    print("---")
    print(f"mean |S|={statistics.mean(n_edges):.1f}")
    print(f"mean drop_S={mean_S:+.4f} ± {std_S:.4f}")
    print(f"mean drop_R={mean_R:+.4f} ± {std_R:.4f}")
    print(f"gap (S - R)={mean_S - mean_R:+.4f}  (want clearly > 0)")


if __name__ == "__main__":
    main()
