"""Build the attention routing graph from one forward pass."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

_EXPERIMENTS = Path(__file__).resolve().parent
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from load_model import attention_from_forward, load_model


@dataclass(frozen=True, slots=True)
class Edge:
    """Directed routing edge: key j → query i at (layer, head)."""

    layer: int
    head: int
    key: int  # j
    query: int  # i
    score: float

    def as_tuple(self) -> tuple[int, int, int, int, float]:
        return self.layer, self.head, self.key, self.query, self.score


def build_routing_graph(
    A: torch.Tensor,
    *,
    batch: int = 0,
    min_score: float = 0.0,
    exclude_self: bool = False,
) -> list[Edge]:
    """Turn attention `A` into a list of edges.

    Args:
        A: [batch, n_layers, n_heads, query, key] from `attention_from_forward`.
        batch: which batch row to use.
        min_score: drop edges with score <= this (0 keeps all causal mass).
        exclude_self: if True, drop j == i edges.

    Returns:
        Edges `(ℓ, h, j → i)` with score = raw attention A[b, ℓ, h, i, j].
    """
    if A.ndim != 5:
        raise ValueError(f"expected A [B,L,H,Q,K], got shape {tuple(A.shape)}")

    Ab = A[batch].detach().float().cpu()  # [L, H, Q, K]
    n_layers, n_heads, n_q, n_k = Ab.shape
    edges: list[Edge] = []

    for layer in range(n_layers):
        for head in range(n_heads):
            for query in range(n_q):
                # causal: key <= query (TransformerLens patterns are already masked)
                for key in range(n_k):
                    if key > query:
                        continue
                    if exclude_self and key == query:
                        continue
                    score = float(Ab[layer, head, query, key])
                    if score <= min_score:
                        continue
                    edges.append(
                        Edge(
                            layer=layer,
                            head=head,
                            key=key,
                            query=query,
                            score=score,
                        )
                    )
    return edges


def routing_graph_summary(edges: list[Edge]) -> dict:
    return {
        "n_edges": len(edges),
        "n_layers": (max((e.layer for e in edges), default=-1) + 1),
        "n_heads": (max((e.head for e in edges), default=-1) + 1),
        "score_min": min((e.score for e in edges), default=0.0),
        "score_max": max((e.score for e in edges), default=0.0),
        "top5": [asdict(e) for e in sorted(edges, key=lambda e: e.score, reverse=True)[:5]],
    }


if __name__ == "__main__":
    model = load_model()
    tokens, _, A = attention_from_forward(model, "The cat sat. The cat")
    edges = build_routing_graph(A, batch=0)
    summary = routing_graph_summary(edges)
    print(f"tokens={model.to_str_tokens(tokens[0])}")
    print(f"A={tuple(A.shape)} → {summary['n_edges']} edges")
    print(f"score range [{summary['score_min']:.4f}, {summary['score_max']:.4f}]")
    print("top5:")
    for e in summary["top5"]:
        print(f"  L{e['layer']}H{e['head']} {e['key']}→{e['query']} score={e['score']:.4f}")
