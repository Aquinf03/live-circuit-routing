"""Extract a live circuit subgraph from the routing graph."""

from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parent
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from load_model import attention_from_forward, load_model
from routing_graph import Edge, build_routing_graph


def extract_subgraph(
    edges: list[Edge],
    t_star: int,
    *,
    k: int = 10,
    B: int = 3,
    exclude_self: bool = True,
) -> list[Edge]:
    """Top-k incoming walk from answer position `t_star` for `B` hops.

    Matches THESIS.md: keep top-k into the frontier, recurse backward,
    drop edges that never enter the walk.
    """
    if k <= 0 or B <= 0:
        return []

    # incoming[query] -> edges ending at that query
    incoming: dict[int, list[Edge]] = {}
    for e in edges:
        if exclude_self and e.key == e.query:
            continue
        incoming.setdefault(e.query, []).append(e)

    for q, lst in incoming.items():
        lst.sort(key=lambda e: e.score, reverse=True)

    selected: dict[tuple[int, int, int, int], Edge] = {}
    frontier = {t_star}
    expanded: set[int] = set()

    for _ in range(B):
        nxt: set[int] = set()
        for i in frontier:
            if i in expanded:
                continue
            expanded.add(i)
            for e in incoming.get(i, [])[:k]:
                key = (e.layer, e.head, e.key, e.query)
                selected[key] = e
                nxt.add(e.key)
        frontier = nxt - expanded
        if not frontier:
            break

    return list(selected.values())


def extract_attention_mass_only(
    edges: list[Edge],
    t_star: int,
    *,
    n: int,
    exclude_self: bool = True,
) -> list[Edge]:
    """Flat baseline: top-n edges into `t_star` only (no B-hop walk)."""
    cand = [
        e
        for e in edges
        if e.query == t_star and not (exclude_self and e.key == e.query)
    ]
    cand.sort(key=lambda e: e.score, reverse=True)
    return cand[:n]


if __name__ == "__main__":
    model = load_model()
    text = "The cat sat. The cat"
    tokens, _, A = attention_from_forward(model, text)
    edges = build_routing_graph(A, batch=0)
    t_star = int(tokens.shape[1] - 1)
    S = extract_subgraph(edges, t_star, k=10, B=3)
    flat = extract_attention_mass_only(edges, t_star, n=len(S))

    strs = model.to_str_tokens(tokens[0])
    print(f"tokens={strs}")
    print(f"t*={t_star} ({strs[t_star]!r})")
    print(f"full_edges={len(edges)}  subgraph={len(S)}  mass_only={len(flat)}")
    print("subgraph top by score:")
    for e in sorted(S, key=lambda e: e.score, reverse=True)[:10]:
        print(
            f"  L{e.layer}H{e.head} {e.key}→{e.query} "
            f"({strs[e.key]!r}→{strs[e.query]!r}) score={e.score:.4f}"
        )
