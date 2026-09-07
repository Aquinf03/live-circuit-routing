"""Ablation: induction template + prompt-length sensitivity."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Sequence

from ablate import measure_drop
from extract_subgraph import extract_subgraph
from induction_templates import TEMPLATES, build_length_set, build_template_set
from load_model import attention_from_forward, get_device, load_model
from paths import RESULTS_FIGURES, RESULTS_RUNS, ensure_helper_imports
from routing_graph import build_routing_graph
from save_results import new_run_dir

ensure_helper_imports()


def build_arg_parser(default_model: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Template / length sensitivity")
    p.add_argument("--model", type=str, default=default_model)
    p.add_argument("--score", choices=("raw", "attn_x_vnorm"), default="attn_x_vnorm")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--templates",
        type=str,
        default="the_dot,bare,then,yesterday",
        help="comma list of template names",
    )
    p.add_argument(
        "--fillers",
        type=str,
        default="0,1,2,4",
        help="comma list of filler sentence counts for length sweep (the_dot base)",
    )
    p.add_argument("--skip-length", action="store_true")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--no-save", action="store_true")
    return p


def _eval_examples(model, examples, *, score, k, B, n_random, seed):
    drops_S, drops_R, n_edges, lengths = [], [], [], []
    for i, ex in enumerate(examples):
        tokens, _, A, Vn = attention_from_forward(model, ex.prefix)
        target_id = int(model.to_tokens(ex.target, prepend_bos=False)[0, 0].item())
        edges = build_routing_graph(
            A, Vn, mode=score, ban_bos=True, exclude_self=True
        )
        t_star = int(tokens.shape[1] - 1)
        S = extract_subgraph(edges, t_star, k=k, B=B)
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
        lengths.append(int(tokens.shape[1]))
    mean_S = statistics.mean(drops_S)
    mean_R = statistics.mean(drops_R)
    return {
        "mean_n_edges": statistics.mean(n_edges),
        "mean_seq_len": statistics.mean(lengths),
        "mean_drop_S": mean_S,
        "mean_drop_R": mean_R,
        "gap": mean_S - mean_R,
        "n_fail": sum(1 for s, r in zip(drops_S, drops_R) if s <= r),
        "n_examples": len(examples),
    }


def run_template_length(argv: Sequence[str] | None, *, default_model: str) -> None:
    args = build_arg_parser(default_model).parse_args(argv)
    device = "cpu" if get_device() == "mps" else get_device()
    model = load_model(name=args.model, device=device)

    template_names = [t.strip() for t in args.templates.split(",") if t.strip()]
    unknown = [t for t in template_names if t not in TEMPLATES]
    if unknown:
        raise SystemExit(f"unknown template(s) {unknown}; known={sorted(TEMPLATES)}")

    print(
        f"template/length model={args.model} device={device} score={args.score} "
        f"k={args.k} B={args.B} templates={template_names}"
    )

    tmpl_rows = []
    for name in template_names:
        examples = build_template_set(model, name)
        if args.limit > 0:
            examples = examples[: args.limit]
        stats = _eval_examples(
            model,
            examples,
            score=args.score,
            k=args.k,
            B=args.B,
            n_random=args.n_random,
            seed=args.seed,
        )
        row = {"kind": "template", "name": name, **stats}
        tmpl_rows.append(row)
        print(
            f"template={name:12s} len={stats['mean_seq_len']:.1f} |S|={stats['mean_n_edges']:.1f} "
            f"drop_S={stats['mean_drop_S']:+.3f} drop_R={stats['mean_drop_R']:+.3f} "
            f"gap={stats['gap']:+.3f} fails={stats['n_fail']}/{stats['n_examples']}"
        )

    len_rows = []
    if not args.skip_length:
        fillers = [int(x) for x in args.fillers.split(",") if x.strip()]
        for n_fill in fillers:
            examples = build_length_set(model, n_filler_sentences=n_fill)
            if args.limit > 0:
                examples = examples[: args.limit]
            stats = _eval_examples(
                model,
                examples,
                score=args.score,
                k=args.k,
                B=args.B,
                n_random=args.n_random,
                seed=args.seed,
            )
            row = {"kind": "length", "name": f"filler_{n_fill}", "n_filler": n_fill, **stats}
            len_rows.append(row)
            print(
                f"length filler={n_fill} len={stats['mean_seq_len']:.1f} |S|={stats['mean_n_edges']:.1f} "
                f"drop_S={stats['mean_drop_S']:+.3f} drop_R={stats['mean_drop_R']:+.3f} "
                f"gap={stats['gap']:+.3f} fails={stats['n_fail']}/{stats['n_examples']}"
            )

    if not args.no_save:
        tag = args.tag or f"ablation_template_length_{args.model.replace('/', '-')}"
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
                "templates": template_names,
            },
            "template_rows": tmpl_rows,
            "length_rows": len_rows,
        }
        (run_dir / "template_length.json").write_text(json.dumps(payload, indent=2) + "\n")

        RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
        md = RESULTS_FIGURES / "ablation_template_length.md"
        lines = [
            "# Ablation — prompt template / length sensitivity",
            "",
            f"Model: `{args.model}` · score=`{args.score}` · k={args.k} · B={args.B} · n≤{args.limit}",
            f"Source: `{run_dir / 'template_length.json'}`",
            "",
            "## Templates",
            "",
            "| Template | mean seq len | mean |S| | drop_S | drop_R | gap | fails |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in tmpl_rows:
            lines.append(
                f"| `{r['name']}` | {r['mean_seq_len']:.1f} | {r['mean_n_edges']:.1f} | "
                f"{r['mean_drop_S']:+.3f} | {r['mean_drop_R']:+.3f} | "
                f"**{r['gap']:+.3f}** | {r['n_fail']}/{r['n_examples']} |"
            )
        if len_rows:
            lines += [
                "",
                "## Length (filler sentences before `the_dot` pattern)",
                "",
                "| Fillers | mean seq len | mean |S| | drop_S | drop_R | gap | fails |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for r in len_rows:
                lines.append(
                    f"| {r['n_filler']} | {r['mean_seq_len']:.1f} | {r['mean_n_edges']:.1f} | "
                    f"{r['mean_drop_S']:+.3f} | {r['mean_drop_R']:+.3f} | "
                    f"**{r['gap']:+.3f}** | {r['n_fail']}/{r['n_examples']} |"
                )
        lines += [
            "",
            "Success: gap stays > 0 across templates and moderate lengths. "
            "Main template `the_dot` should not be a one-template artifact.",
            "",
        ]
        md.write_text("\n".join(lines))
        print(f"saved → {run_dir / 'template_length.json'}")
        print(f"wrote → {md}")
        print(f"(runs root: {RESULTS_RUNS})")
