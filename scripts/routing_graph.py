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


def edge_scores(
    A: torch.Tensor,
    Vn: torch.Tensor | None = None,
    *,
    mode: str = "raw",
) -> torch.Tensor:
    """Score tensor [B, L, H, Q, K].

    mode:
      - raw: attention mass
      - attn_x_vnorm: A * ‖v‖ at the key
    """
    if mode == "raw":
        return A
    if mode == "attn_x_vnorm":
        if Vn is None:
            raise ValueError("attn_x_vnorm requires Vn")
        # A: [B,L,H,Q,K], Vn: [B,L,H,K] → broadcast over Q
        return A * Vn[:, :, :, None, :]
    raise ValueError(f"unknown score mode: {mode!r}")


def build_routing_graph(
    A: torch.Tensor,
    Vn: torch.Tensor | None = None,
    *,
    batch: int = 0,
    mode: str = "raw",
    min_score: float = 0.0,
    exclude_self: bool = False,
    ban_bos: bool = True,
    bos_pos: int = 0,
) -> list[Edge]:
    """Turn attention into edges `(ℓ, h, j → i)`.

    ban_bos: drop edges with key == bos_pos (attention sinks into BOS).
    """
    scores = edge_scores(A, Vn, mode=mode)
    if scores.ndim != 5:
        raise ValueError(f"expected scores [B,L,H,Q,K], got {tuple(scores.shape)}")

    Sb = scores[batch].detach().float().cpu()  # [L, H, Q, K]
    n_layers, n_heads, n_q, n_k = Sb.shape
    edges: list[Edge] = []

    for layer in range(n_layers):
        for head in range(n_heads):
            for query in range(n_q):
                for key in range(min(n_k, query + 1)):
                    if exclude_self and key == query:
                        continue
                    if ban_bos and key == bos_pos:
                        continue
                    score = float(Sb[layer, head, query, key])
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
    tokens, _, A, Vn = attention_from_forward(model, "The cat sat. The cat")
    for mode in ("raw", "attn_x_vnorm"):
        edges = build_routing_graph(A, Vn, mode=mode, ban_bos=True, exclude_self=True)
        summary = routing_graph_summary(edges)
        print(f"mode={mode} edges={summary['n_edges']} top={summary['top5'][:3]}")
