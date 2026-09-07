"""Ablation: extract using all layers vs selected layer bands."""

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
from routing_graph import Edge, build_routing_graph
from save_results import new_run_dir

ensure_helper_imports()

# Named bands for GPT-2 Small (12 layers). For other models, bands are computed by thirds.
GPT2_SMALL_BANDS = {
    "all": None,  # no filter
    "early": list(range(0, 4)),
    "mid": list(range(4, 8)),
    "late": list(range(8, 12)),
    "induction_band": list(range(4, 8)),  # prev-token + induction neighborhood
}


def filter_edges_by_layers(edges: list[Edge], layers: list[int] | None) -> list[Edge]:
    if layers is None:
        return edges
    allow = set(layers)
    return [e for e in edges if e.layer in allow]


def bands_for_model(n_layers: int, model_name: str) -> dict[str, list[int] | None]:
    if "gpt2-small" in model_name.lower() or n_layers == 12:
        return GPT2_SMALL_BANDS
    a, b = n_layers // 3, 2 * n_layers // 3
    return {
        "all": None,
        "early": list(range(0, a)),
        "mid": list(range(a, b)),
        "late": list(range(b, n_layers)),
    }


def build_arg_parser(default_model: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Layer-band ablation")
    p.add_argument("--model", type=str, default=default_model)
    p.add_argument("--score", choices=("raw", "attn_x_vnorm"), default="attn_x_vnorm")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--no-save", action="store_true")
    return p


def run_layer_ablation(argv: Sequence[str] | None, *, default_model: str) -> None:
    args = build_arg_parser(default_model).parse_args(argv)
    device = "cpu" if get_device() == "mps" else get_device()
    model = load_model(name=args.model, device=device)
    examples = build_induction_set(model)
    if args.limit > 0:
        examples = examples[: args.limit]

    n_layers = int(model.cfg.n_layers)
    bands = bands_for_model(n_layers, args.model)

    print(
        f"layer-ablation model={args.model} device={device} score={args.score} "
        f"k={args.k} B={args.B} n_examples={len(examples)} bands={list(bands)}"
    )

    rows = []
    for name, layers in bands.items():
        drops_S, drops_R, n_edges = [], [], []
        for i, ex in enumerate(examples):
            tokens, _, A, Vn = attention_from_forward(model, ex.prefix)
            target_id = int(model.to_tokens(ex.target, prepend_bos=False)[0, 0].item())
            edges = build_routing_graph(
                A, Vn, mode=args.score, ban_bos=True, exclude_self=True
            )
            edges = filter_edges_by_layers(edges, layers)
            t_star = int(tokens.shape[1] - 1)
            S = extract_subgraph(edges, t_star, k=args.k, B=args.B)
            if not S:
                drops_S.append(0.0)
                drops_R.append(0.0)
                n_edges.append(0)
                continue
            # Random controls must come from the same layer-restricted pool
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
            "band": name,
            "layers": layers if layers is not None else list(range(n_layers)),
            "mean_n_edges": statistics.mean(n_edges),
            "mean_drop_S": mean_S,
            "mean_drop_R": mean_R,
            "gap": mean_S - mean_R,
            "n_fail": sum(1 for s, r in zip(drops_S, drops_R) if s <= r),
            "n_examples": len(examples),
        }
        rows.append(row)
        print(
            f"{name:16s} layers={row['layers']} |S|={row['mean_n_edges']:.1f} "
            f"drop_S={row['mean_drop_S']:+.3f} drop_R={row['mean_drop_R']:+.3f} "
            f"gap={row['gap']:+.3f} fails={row['n_fail']}/{row['n_examples']}"
        )

    if not args.no_save:
        tag = args.tag or f"ablation_layers_{args.model.replace('/', '-')}"
        run_dir = new_run_dir(tag)
        payload = {
            "config": {
                "model": args.model,
                "device": device,
                "score": args.score,
                "k": args.k,
                "B": args.B,
                "n_random": args.n_random,
                "limit": args.limit,
                "n_layers": n_layers,
            },
            "rows": rows,
        }
        (run_dir / "layer_ablation.json").write_text(json.dumps(payload, indent=2) + "\n")

        RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
        md = RESULTS_FIGURES / "ablation_layers.md"
        lines = [
            "# Ablation — all heads/layers vs selected layer bands",
            "",
            f"Model: `{args.model}` · score=`{args.score}` · k={args.k} · B={args.B} · n={len(examples)}",
            "Edges kept only if their layer is in the band; head identity still preserved.",
            f"Source: `{run_dir / 'layer_ablation.json'}`",
            "",
            "| Band | Layers | mean |S| | drop_S | drop_R | gap | fails |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in rows:
            layer_str = (
                "all"
                if r["band"] == "all"
                else ",".join(str(x) for x in r["layers"])
            )
            lines.append(
                f"| {r['band']} | {layer_str} | {r['mean_n_edges']:.1f} | "
                f"{r['mean_drop_S']:+.3f} | {r['mean_drop_R']:+.3f} | "
                f"**{r['gap']:+.3f}** | {r['n_fail']}/{r['n_examples']} |"
            )
        lines += [
            "",
            "Expect: **all** and **induction_band/mid** strongest on GPT-2 Small induction; "
            "early/late alone weaker. If all ≫ every band, the circuit is spread across depth.",
            "",
        ]
        md.write_text("\n".join(lines))
        print(f"saved → {run_dir / 'layer_ablation.json'}")
        print(f"wrote → {md}")
        print(f"(runs root: {RESULTS_RUNS})")
