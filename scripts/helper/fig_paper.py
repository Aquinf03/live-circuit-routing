"""Generate the four paper figures into paper/figures/.

Fig 1 — VisualTorch flow of the live-extract pipeline (stand-in modules).
Fig 2 — Extracted routing subgraph arc diagram (from a saved induction run).
Fig 3 — Causal drop_S vs drop_R across model/task cells.
Fig 4 — Cost: wall-clock / forwards vs head-patching.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HELPER = Path(__file__).resolve().parent
_ROOT = _HELPER.parents[1]
if str(_HELPER) not in sys.path:
    sys.path.insert(0, str(_HELPER))

os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import visualtorch
from PIL import Image, ImageDraw, ImageFont

from plot_structure import plot_example_graph

OUT_DIR = _ROOT / "paper" / "figures"
RUN_SMALL = _ROOT / "results" / "runs" / "induction_20260907_105902"


class SingleForward(nn.Linear):
    """Stage 1: one model forward → A, V."""


class RoutingGraph(nn.Linear):
    """Stage 2: attention routes (ℓ,h,j→i)."""


class TopkWalk(nn.Linear):
    """Stage 3: top-k walk from t* → S."""


class VerifyAblate(nn.Linear):
    """Stage 4: ablate S vs matched random."""


class LiveExtractPipeline(nn.Module):
    """Named stages so VisualTorch labels the paper pipeline."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.single_forward = SingleForward(d, d)
        self.routing_graph = RoutingGraph(d, d)
        self.topk_walk = TopkWalk(d, d)
        self.verify_vs_random = VerifyAblate(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.single_forward(x)
        x = self.routing_graph(x)
        x = self.topk_walk(x)
        return self.verify_vs_random(x)


def _annotate_pipeline(img: Image.Image) -> Image.Image:
    """Add a short title strip so the VisualTorch panel reads as Fig 1."""
    pad = 40
    w, h = img.size
    canvas = Image.new("RGB", (w, h + pad), "white")
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((16, 10), "Live circuit extraction pipeline", fill=(30, 30, 30), font=font)
    return canvas


def fig1_pipeline(out: Path) -> None:
    model = LiveExtractPipeline()
    model.eval()
    img = visualtorch.render(
        model,
        input_shape=(1, 32),
        style="graph",
        background_fill="white",
        padding=24,
        legend=True,
        show_dimension=False,
        show_neurons=True,
        node_size=40,
        layer_spacing=220,
    )
    _annotate_pipeline(img).save(out)
    print(f"wrote {out}")


def fig2_subgraph(out: Path, run_dir: Path = RUN_SMALL) -> None:
    data = json.loads((run_dir / "subgraphs.json").read_text())
    g = data[0]  # cat / sat
    title = (
        f"Extracted routing subgraph — {g['a']}/{g['b']} "
        f"(GPT-2 Small, attn×‖v‖, k=15, B=4)"
    )
    plot_example_graph(g["tokens"], g["edges"], g["t_star"], out, title)
    print(f"wrote {out}")


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def fig3_causal(out: Path) -> None:
    # Means from paper tables (torch tensors for the plotting path).
    labels = [
        "Ind.\nSmall",
        "Ind.\nMedium",
        "Ind.\nPythia",
        "IOI\nSmall",
    ]
    drop_s = torch.tensor([3.655, 1.853, 1.812, 5.270])
    drop_r = torch.tensor([0.106, 0.004, 0.031, 0.098])
    x = torch.arange(len(labels), dtype=torch.float32)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.36
    ax.bar(
        (x - w / 2).numpy(),
        drop_s.numpy(),
        width=w,
        color="#1f6f8b",
        label=r"$\Delta_S$ (extracted)",
    )
    ax.bar(
        (x + w / 2).numpy(),
        drop_r.numpy(),
        width=w,
        color="#b8b8b8",
        label=r"$\Delta_R$ (random)",
    )
    ax.set_xticks(x.numpy())
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean logit drop")
    ax.set_title("Causal verification: extracted vs size-matched random")
    ax.legend(frameon=False, fontsize=9)
    ax.axhline(0.0, color="#444444", lw=0.6)
    _style_axes(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig4_cost(out: Path) -> None:
    methods = ["Live extract", "Live + verify", "Head-patch"]
    forwards = torch.tensor([1.0, 12.0, 145.0])
    wall = torch.tensor([0.056, 0.481, 5.939])  # seconds

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    colors = ["#2a9d8f", "#e9c46a", "#c45c26"]

    axes[0].bar(methods, forwards.numpy(), color=colors)
    axes[0].set_ylabel("Forwards / example")
    axes[0].set_title("Compute")
    axes[0].set_yscale("log")
    _style_axes(axes[0])
    for i, v in enumerate(forwards.tolist()):
        axes[0].text(i, v * 1.15, f"{int(v)}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(methods, wall.numpy(), color=colors)
    axes[1].set_ylabel("Wall-clock (s, CPU)")
    axes[1].set_title("Latency")
    _style_axes(axes[1])
    for i, v in enumerate(wall.tolist()):
        axes[1].text(i, v + 0.12, f"{v:.2f}s", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Cost vs head-patching (GPT-2 Small induction, n=5)", fontsize=11, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1_pipeline(OUT_DIR / "fig1_pipeline.png")
    fig2_subgraph(OUT_DIR / "fig2_subgraph.png")
    fig3_causal(OUT_DIR / "fig3_causal.png")
    fig4_cost(OUT_DIR / "fig4_cost.png")
    print(f"all figures → {OUT_DIR}")


if __name__ == "__main__":
    main()
