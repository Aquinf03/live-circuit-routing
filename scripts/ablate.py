"""Ablate routing edges and measure task-metric drop."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

_EXPERIMENTS = Path(__file__).resolve().parent
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from extract_subgraph import extract_subgraph
from load_model import attention_from_forward, get_device, load_model
from routing_graph import Edge, build_routing_graph
from transformer_lens import HookedTransformer


def sample_random_edges(
    edges: list[Edge],
    n: int,
    *,
    exclude_self: bool = True,
    ban_bos: bool = True,
    bos_pos: int = 0,
    seed: int = 0,
) -> list[Edge]:
    """Size-matched random control from valid causal edges."""
    pool = []
    for e in edges:
        if exclude_self and e.key == e.query:
            continue
        if ban_bos and e.key == bos_pos:
            continue
        pool.append(e)
    n = min(n, len(pool))
    rng = random.Random(seed)
    return rng.sample(pool, n)


def _ablation_hooks(
    edges: list[Edge],
    *,
    renormalize: bool = True,
):
    """Zero selected pattern entries; optionally renormalize over keys."""
    by_layer: dict[int, list[Edge]] = {}
    for e in edges:
        by_layer.setdefault(e.layer, []).append(e)

    hooks = []
    for layer, layer_edges in by_layer.items():
        name = f"blocks.{layer}.attn.hook_pattern"

        def make_hook(layer_edges: list[Edge]):
            def hook_fn(pattern: torch.Tensor, hook):
                # pattern: [batch, head, query, key]
                out = pattern.clone()
                for e in layer_edges:
                    out[:, e.head, e.query, e.key] = 0.0
                if renormalize:
                    denom = out.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    out = out / denom
                return out

            return hook_fn

        hooks.append((name, make_hook(layer_edges)))
    return hooks


@torch.no_grad()
def next_token_logit(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_id: int,
    *,
    ablate: list[Edge] | None = None,
) -> float:
    """Logit of `target_id` at the last position (optionally with edge ablation)."""
    if ablate:
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=_ablation_hooks(ablate),
        )
    else:
        logits = model(tokens)
    return float(logits[0, -1, target_id].item())


@torch.no_grad()
def measure_drop(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_id: int,
    S: list[Edge],
    *,
    all_edges: list[Edge],
    n_random: int = 20,
    seed: int = 0,
) -> dict:
    """Full vs extracted ablation vs random ablations (matched |S|)."""
    full = next_token_logit(model, tokens, target_id)
    ablated = next_token_logit(model, tokens, target_id, ablate=S)
    random_drops = []
    for r in range(n_random):
        R = sample_random_edges(all_edges, len(S), seed=seed + r)
        rand_logit = next_token_logit(model, tokens, target_id, ablate=R)
        random_drops.append(full - rand_logit)

    drop_S = full - ablated
    rand_mean = sum(random_drops) / max(len(random_drops), 1)
    rand_std = (
        (sum((d - rand_mean) ** 2 for d in random_drops) / max(len(random_drops), 1))
        ** 0.5
    )
    return {
        "logit_full": full,
        "logit_ablate_S": ablated,
        "drop_S": drop_S,
        "drop_random_mean": rand_mean,
        "drop_random_std": rand_std,
        "n_edges": len(S),
        "n_random": n_random,
    }


if __name__ == "__main__":
    # Prefer CPU for metric numbers (MPS can be silently wrong on TL).
    device = "cpu" if get_device() == "mps" else get_device()
    model = load_model(device=device)

    prefix = "The cat sat. The cat"
    target_str = " sat"
    target_id = int(model.to_tokens(target_str, prepend_bos=False)[0, 0].item())

    tokens, _, A, Vn = attention_from_forward(model, prefix)
    edges = build_routing_graph(
        A, Vn, mode="attn_x_vnorm", ban_bos=True, exclude_self=True
    )
    t_star = int(tokens.shape[1] - 1)
    S = extract_subgraph(edges, t_star, k=10, B=3)

    result = measure_drop(
        model, tokens, target_id, S, all_edges=edges, n_random=10, seed=0
    )
    print(f"device={device}")
    print(f"prefix={prefix!r} target={target_str!r} (id={target_id})")
    print(f"tokens={model.to_str_tokens(tokens[0])}")
    print(f"|S|={result['n_edges']}")
    print(
        f"logit_full={result['logit_full']:.4f}  "
        f"logit_ablate_S={result['logit_ablate_S']:.4f}  "
        f"drop_S={result['drop_S']:.4f}"
    )
    print(
        f"drop_random={result['drop_random_mean']:.4f} "
        f"± {result['drop_random_std']:.4f} (n={result['n_random']})"
    )
