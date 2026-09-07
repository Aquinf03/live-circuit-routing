"""Wall-clock / forward-count: live extract vs head-patching baseline."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from ablate import next_token_logit, sample_random_edges
from extract_subgraph import extract_subgraph
from induction_data import build_induction_set
from load_model import attention_from_forward, get_device, load_model
from paths import RESULTS_FIGURES, RESULTS_RUNS, ensure_helper_imports
from routing_graph import build_routing_graph
from save_results import new_run_dir

ensure_helper_imports()


@dataclass
class TimingRow:
    example: str
    live_extract_s: float
    live_forwards: int
    verify_s: float
    verify_forwards: int
    head_patch_s: float
    head_patch_forwards: int
    n_heads_total: int
    n_edges_S: int
    drop_S: float
    drop_head_oracle: float


@torch.no_grad()
def _ablate_whole_heads(
    model,
    tokens: torch.Tensor,
    target_id: int,
    heads: list[tuple[int, int]],
) -> float:
    """Zero all attention from listed (layer, head) pairs; return target logit."""
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    hooks = []
    for layer, head_list in by_layer.items():
        name = f"blocks.{layer}.attn.hook_pattern"

        def make_hook(head_list: list[int]):
            def hook_fn(pattern: torch.Tensor, hook):
                out = pattern.clone()
                for h in head_list:
                    out[:, h, :, :] = 0.0
                denom = out.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                return out / denom

            return hook_fn

        hooks.append((name, make_hook(head_list)))

    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return float(logits[0, -1, target_id].item())


def _time_live_extract(model, prefix: str, *, score: str, k: int, B: int, ban_bos: bool):
    t0 = time.perf_counter()
    tokens, _, A, Vn = attention_from_forward(model, prefix)
    edges = build_routing_graph(
        A, Vn, mode=score, ban_bos=ban_bos, exclude_self=True
    )
    t_star = int(tokens.shape[1] - 1)
    S = extract_subgraph(edges, t_star, k=k, B=B, exclude_self=True)
    elapsed = time.perf_counter() - t0
    return tokens, edges, S, elapsed, 1  # 1 forward


def _time_verify(model, tokens, target_id, S, edges, n_random: int, seed: int):
    """1 full + 1 S-ablate + n_random random ablations."""
    t0 = time.perf_counter()
    full = next_token_logit(model, tokens, target_id)
    ablated = next_token_logit(model, tokens, target_id, ablate=S)
    drops_r = []
    for r in range(n_random):
        R = sample_random_edges(edges, len(S), seed=seed + r)
        drops_r.append(full - next_token_logit(model, tokens, target_id, ablate=R))
    elapsed = time.perf_counter() - t0
    forwards = 1 + 1 + n_random
    drop_S = full - ablated
    drop_R = sum(drops_r) / max(len(drops_r), 1)
    return elapsed, forwards, drop_S, drop_R


def _time_head_patch(model, tokens, target_id, n_layers: int, n_heads: int):
    """Ablate each head alone; keep best single-head drop (oracle upper bound cost)."""
    t0 = time.perf_counter()
    full = next_token_logit(model, tokens, target_id)
    best_drop = float("-inf")
    for l in range(n_layers):
        for h in range(n_heads):
            logit = _ablate_whole_heads(model, tokens, target_id, [(l, h)])
            best_drop = max(best_drop, full - logit)
    elapsed = time.perf_counter() - t0
    forwards = 1 + n_layers * n_heads
    return elapsed, forwards, best_drop


def build_arg_parser(default_model: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cost bench: live vs head-patching")
    p.add_argument("--model", type=str, default=default_model)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--score", choices=("raw", "attn_x_vnorm"), default="attn_x_vnorm")
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--limit", type=int, default=5, help="examples to time (default 5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ban-bos", action="store_true")
    p.add_argument("--skip-head-patch", action="store_true", help="only time live path")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--no-save", action="store_true")
    return p


def run_cost_bench(argv: Sequence[str] | None, *, default_model: str) -> None:
    args = build_arg_parser(default_model).parse_args(argv)
    device = "cpu" if get_device() == "mps" else get_device()
    model = load_model(name=args.model, device=device)
    examples = build_induction_set(model)
    if args.limit > 0:
        examples = examples[: args.limit]
    ban_bos = not args.no_ban_bos
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)

    print(
        f"cost-bench model={args.model} device={device} score={args.score} "
        f"k={args.k} B={args.B} n_examples={len(examples)} "
        f"heads={n_layers}x{n_heads}={n_layers * n_heads}"
    )

    rows: list[TimingRow] = []
    for i, ex in enumerate(examples):
        target_id = int(model.to_tokens(ex.target, prepend_bos=False)[0, 0].item())
        tokens, edges, S, live_s, live_fwd = _time_live_extract(
            model,
            ex.prefix,
            score=args.score,
            k=args.k,
            B=args.B,
            ban_bos=ban_bos,
        )
        verify_s, verify_fwd, drop_S, _ = _time_verify(
            model,
            tokens,
            target_id,
            S,
            edges,
            n_random=args.n_random,
            seed=args.seed + 1000 * i,
        )
        if args.skip_head_patch:
            head_s, head_fwd, drop_head = float("nan"), 0, float("nan")
        else:
            head_s, head_fwd, drop_head = _time_head_patch(
                model, tokens, target_id, n_layers, n_heads
            )

        row = TimingRow(
            example=f"{ex.a}/{ex.b}",
            live_extract_s=live_s,
            live_forwards=live_fwd,
            verify_s=verify_s,
            verify_forwards=verify_fwd,
            head_patch_s=head_s,
            head_patch_forwards=head_fwd,
            n_heads_total=n_layers * n_heads,
            n_edges_S=len(S),
            drop_S=drop_S,
            drop_head_oracle=drop_head,
        )
        rows.append(row)
        print(
            f"[{i+1:02d}/{len(examples)}] {row.example} "
            f"live={row.live_extract_s:.3f}s/{row.live_forwards}fwd "
            f"verify={row.verify_s:.3f}s/{row.verify_forwards}fwd "
            f"headpatch={row.head_patch_s:.3f}s/{row.head_patch_forwards}fwd "
            f"|S|={row.n_edges_S}"
        )

    def mean(xs):
        xs = [x for x in xs if x == x]  # drop nan
        return statistics.mean(xs) if xs else float("nan")

    summary = {
        "model": args.model,
        "device": device,
        "score": args.score,
        "k": args.k,
        "B": args.B,
        "n_examples": len(rows),
        "n_heads_total": n_layers * n_heads,
        "mean_live_extract_s": mean([r.live_extract_s for r in rows]),
        "mean_verify_s": mean([r.verify_s for r in rows]),
        "mean_head_patch_s": mean([r.head_patch_s for r in rows]),
        "live_forwards_per_example": 1,
        "verify_forwards_per_example": 2 + args.n_random,
        "head_patch_forwards_per_example": 1 + n_layers * n_heads,
        "speedup_extract_vs_head_patch": (
            mean([r.head_patch_s for r in rows]) / mean([r.live_extract_s for r in rows])
            if mean([r.live_extract_s for r in rows]) > 0
            else float("nan")
        ),
        "forward_ratio_head_patch_vs_live": float(1 + n_layers * n_heads),
    }

    print("---")
    print(f"mean live extract: {summary['mean_live_extract_s']:.3f}s  (1 forward)")
    print(
        f"mean verify (S + {args.n_random} random): {summary['mean_verify_s']:.3f}s  "
        f"({summary['verify_forwards_per_example']} forwards)"
    )
    if not args.skip_head_patch:
        print(
            f"mean head-patch sweep: {summary['mean_head_patch_s']:.3f}s  "
            f"({summary['head_patch_forwards_per_example']} forwards)"
        )
        print(
            f"wall-clock speedup (head-patch / live extract): "
            f"{summary['speedup_extract_vs_head_patch']:.1f}x"
        )
        print(
            f"forward ratio (head-patch / live): "
            f"{summary['forward_ratio_head_patch_vs_live']:.0f}x"
        )

    if not args.no_save:
        tag = args.tag or f"cost_{args.model.replace('/', '-')}"
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
                "ban_bos": ban_bos,
                "skip_head_patch": args.skip_head_patch,
            },
            "summary": summary,
            "rows": [asdict(r) for r in rows],
        }
        (run_dir / "cost.json").write_text(json.dumps(payload, indent=2) + "\n")

        # Paper-facing markdown note
        RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
        md = RESULTS_FIGURES / "cost_note.md"
        md.write_text(
            "\n".join(
                [
                    "# Cost note — live extract vs head-patching",
                    "",
                    f"Model: `{args.model}` · device: `{device}` · n={len(rows)} induction prompts",
                    f"Live settings: score=`{args.score}`, k={args.k}, B={args.B}",
                    "",
                    "## Per-example means",
                    "",
                    "| Method | Forwards / example | Mean wall-clock |",
                    "| --- | ---: | ---: |",
                    f"| Live extract (graph + top-k) | **1** | **{summary['mean_live_extract_s']:.3f}s** |",
                    f"| Live + verify (S + {args.n_random} random ablations) | {summary['verify_forwards_per_example']} | {summary['mean_verify_s']:.3f}s |",
                    (
                        f"| Head-patch sweep (each of {n_layers * n_heads} heads) | {summary['head_patch_forwards_per_example']} | {summary['mean_head_patch_s']:.3f}s |"
                        if not args.skip_head_patch
                        else "| Head-patch sweep | skipped | — |"
                    ),
                    "",
                    (
                        f"**Forward ratio** head-patch / live extract = **{summary['forward_ratio_head_patch_vs_live']:.0f}×**  \n"
                        f"**Wall-clock speedup** ≈ **{summary['speedup_extract_vs_head_patch']:.1f}×**"
                        if not args.skip_head_patch
                        else ""
                    ),
                    "",
                    "Live extract is one forward plus cheap post-processing. "
                    "Head-patching needs a separate forward per head (plus a clean run). "
                    "Full edge-ACDC is even more expensive; this head sweep is a lower bound on patching cost.",
                    "",
                    f"Raw: `{run_dir / 'cost.json'}`",
                    "",
                ]
            )
        )
        print(f"saved → {run_dir / 'cost.json'}")
        print(f"wrote → {md}")
        print(f"(runs root: {RESULTS_RUNS})")
