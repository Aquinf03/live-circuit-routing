"""Qualitative figures: does the live subgraph recover known induction structure?"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Writable matplotlib cache inside the repo (sandbox / CI friendly).
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parent
ROOT = _EXPERIMENTS.parent
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from extract_subgraph import extract_subgraph
from load_model import attention_from_forward, get_device, load_model
from routing_graph import build_routing_graph

# Commonly cited GPT-2 Small heads for induction (Olsson et al. / TL community).
# Format: (layer, head)
GPT2_SMALL_PREV_TOKEN = {(4, 11)}
GPT2_SMALL_INDUCTION = {
    (5, 5),
    (5, 8),
    (5, 9),
    (6, 9),
    (6, 10),
    (7, 3),
    (7, 9),
}


def _head_kind(layer: int, head: int) -> str:
    if (layer, head) in GPT2_SMALL_PREV_TOKEN:
        return "prev_token"
    if (layer, head) in GPT2_SMALL_INDUCTION:
        return "induction"
    return "other"


def _color(kind: str) -> str:
    return {
        "prev_token": "#c45c26",
        "induction": "#1f6f8b",
        "other": "#9a9a9a",
    }[kind]


def plot_example_graph(
    tokens: list[str],
    edges: list[dict],
    t_star: int,
    out_path: Path,
    title: str,
) -> None:
    """Arc diagram: token positions as nodes, edges as arcs colored by head type."""
    n = len(tokens)
    # Aggregate parallel edges (same key→query) for readability; keep top score label
    by_pair: dict[tuple[int, int], list[dict]] = {}
    for e in edges:
        by_pair.setdefault((e["key"], e["query"]), []).append(e)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(n)
    ys = np.zeros(n)
    ax.scatter(xs, ys, s=120, c="#222222", zorder=3)

    for i, tok in enumerate(tokens):
        label = tok.replace("Ġ", " ").replace("▁", " ")
        if label == "<|endoftext|>":
            label = "BOS"
        ax.text(i, -0.18, label, ha="center", va="top", fontsize=9)
        if i == t_star:
            ax.scatter([i], [0], s=220, facecolors="none", edgecolors="#111111", linewidths=2, zorder=4)

    max_score = max((e["score"] for e in edges), default=1.0) or 1.0
    for (key, query), elist in sorted(by_pair.items()):
        if key == query:
            continue
        elist = sorted(elist, key=lambda e: e["score"], reverse=True)
        top = elist[0]
        kind = _head_kind(top["layer"], top["head"])
        mid = 0.5 * (key + query)
        height = 0.15 + 0.55 * abs(query - key) / max(n - 1, 1)
        rad = 0.25 + 0.15 * abs(query - key) / max(n - 1, 1)
        ax.annotate(
            "",
            xy=(query, 0),
            xytext=(key, 0),
            arrowprops=dict(
                arrowstyle="->",
                color=_color(kind),
                lw=0.8 + 2.0 * (top["score"] / max_score),
                connectionstyle=f"arc3,rad={rad if query > key else -rad}",
                alpha=0.85,
            ),
            zorder=2,
        )
        if kind != "other":
            ax.text(
                mid,
                height if query > key else -height,
                f"L{top['layer']}H{top['head']}",
                ha="center",
                va="center",
                fontsize=7,
                color=_color(kind),
            )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-1.0, 1.35)
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)
    legend = [
        mpatches.Patch(color=_color("prev_token"), label="prev-token (known)"),
        mpatches.Patch(color=_color("induction"), label="induction (known)"),
        mpatches.Patch(color=_color("other"), label="other"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_head_heatmap(
    head_counts: np.ndarray,
    out_path: Path,
    title: str,
) -> dict:
    """n_layers x n_heads counts; mark known induction / prev-token cells."""
    n_layers, n_heads = head_counts.shape
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(head_counts, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(title, fontsize=11)

    for layer, head in GPT2_SMALL_PREV_TOKEN:
        if layer < n_layers and head < n_heads:
            ax.add_patch(
                plt.Rectangle(
                    (head - 0.5, layer - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=_color("prev_token"),
                    linewidth=2,
                )
            )
    for layer, head in GPT2_SMALL_INDUCTION:
        if layer < n_layers and head < n_heads:
            ax.add_patch(
                plt.Rectangle(
                    (head - 0.5, layer - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=_color("induction"),
                    linewidth=1.5,
                )
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="times in extracted S")
    legend = [
        mpatches.Patch(edgecolor=_color("prev_token"), facecolor="none", label="prev-token"),
        mpatches.Patch(edgecolor=_color("induction"), facecolor="none", label="induction"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # recovery stats
    known = GPT2_SMALL_PREV_TOKEN | GPT2_SMALL_INDUCTION
    recovered = {
        f"L{l}H{h}": int(head_counts[l, h])
        for (l, h) in sorted(known)
        if l < n_layers and h < n_heads and head_counts[l, h] > 0
    }
    return recovered


def heads_from_subgraphs(subgraphs: list[dict], n_layers: int, n_heads: int) -> np.ndarray:
    counts = np.zeros((n_layers, n_heads), dtype=np.int32)
    for g in subgraphs:
        seen = set()
        for e in g["edges"]:
            key = (e["layer"], e["head"])
            if key in seen:
                continue
            seen.add(key)
            if e["layer"] < n_layers and e["head"] < n_heads:
                counts[e["layer"], e["head"]] += 1
    return counts


def load_latest_run() -> Path | None:
    root = ROOT / "data" / "results"
    if not root.exists():
        return None
    runs = sorted(root.glob("induction_*"), key=lambda p: p.name)
    return runs[-1] if runs else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default="", help="path to data/results/induction_*")
    p.add_argument("--example-idx", type=int, default=0, help="which subgraph to draw")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--score", choices=("raw", "attn_x_vnorm"), default="attn_x_vnorm")
    p.add_argument("--fresh", action="store_true", help="ignore saved run; extract fresh")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run) if args.run else load_latest_run()
    subgraphs = None
    config = {}
    if run_dir and not args.fresh and (run_dir / "subgraphs.json").exists():
        subgraphs = json.loads((run_dir / "subgraphs.json").read_text())
        if (run_dir / "config.json").exists():
            config = json.loads((run_dir / "config.json").read_text())
        print(f"loaded run {run_dir}")
    else:
        device = "cpu" if get_device() == "mps" else get_device()
        model = load_model(device=device)
        # build a small set for heatmap + one example graph
        from induction_data import build_induction_set

        examples = build_induction_set(model)
        subgraphs = []
        for ex in examples:
            tokens, _, A, Vn = attention_from_forward(model, ex.prefix)
            edges = build_routing_graph(
                A, Vn, mode=args.score, ban_bos=True, exclude_self=True
            )
            t_star = int(tokens.shape[1] - 1)
            S = extract_subgraph(edges, t_star, k=args.k, B=args.B)
            subgraphs.append(
                {
                    "a": ex.a,
                    "b": ex.b,
                    "tokens": model.to_str_tokens(tokens[0]),
                    "t_star": t_star,
                    "edges": [
                        {
                            "layer": e.layer,
                            "head": e.head,
                            "key": e.key,
                            "query": e.query,
                            "score": e.score,
                        }
                        for e in S
                    ],
                }
            )
        config = {"k": args.k, "B": args.B, "score": args.score}

    g = subgraphs[min(args.example_idx, len(subgraphs) - 1)]
    # infer grid size
    max_l = max((e["layer"] for sg in subgraphs for e in sg["edges"]), default=11)
    max_h = max((e["head"] for sg in subgraphs for e in sg["edges"]), default=11)
    n_layers, n_heads = max_l + 1, max_h + 1

    title_ex = (
        f"Extracted routing subgraph — {g.get('a','?')}/{g.get('b','?')} "
        f"(k={config.get('k','?')}, B={config.get('B','?')}, {config.get('score','?')})"
    )
    out_graph = fig_dir / "induction_subgraph_example.png"
    plot_example_graph(g["tokens"], g["edges"], g["t_star"], out_graph, title_ex)

    counts = heads_from_subgraphs(subgraphs, n_layers, n_heads)
    out_heat = fig_dir / "induction_head_heatmap.png"
    recovered = plot_head_heatmap(
        counts,
        out_heat,
        title=f"Heads appearing in extracted S across {len(subgraphs)} induction prompts",
    )

    meta = {
        "example": {"a": g.get("a"), "b": g.get("b"), "n_edges": len(g["edges"])},
        "recovered_known_heads": recovered,
        "n_examples": len(subgraphs),
        "figures": [str(out_graph), str(out_heat)],
        "config": config,
        "run": str(run_dir) if run_dir else None,
    }
    (fig_dir / "induction_structure_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_graph}")
    print(f"wrote {out_heat}")
    print(f"recovered known heads: {recovered}")


if __name__ == "__main__":
    main()
