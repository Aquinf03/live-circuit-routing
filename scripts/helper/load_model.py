"""Load models via TransformerLens and read attention (+ values)."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


DEFAULT_MODEL = "gpt2-small"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str = DEFAULT_MODEL, device: str | None = None) -> HookedTransformer:
    device = device or get_device()
    return HookedTransformer.from_pretrained(name, device=device)


def attention_from_forward(
    model: HookedTransformer,
    text: str | list[str],
    *,
    tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward pass → attention + value norms.

    Returns:
        tokens: [batch, T]
        logits: [batch, T, d_vocab]
        A: [batch, n_layers, n_heads, query, key]
        Vn: [batch, n_layers, n_heads, key]  = ‖value‖ at each key position
    """
    if tokens is None:
        tokens = model.to_tokens(text)
    logits, cache = model.run_with_cache(tokens)
    patterns = [
        cache[f"blocks.{l}.attn.hook_pattern"] for l in range(model.cfg.n_layers)
    ]
    values = [cache[f"blocks.{l}.attn.hook_v"] for l in range(model.cfg.n_layers)]
    # pattern: [B, H, Q, K] → [B, L, H, Q, K]
    A = torch.stack(patterns, dim=1)
    # value: [B, pos, H, d_head] → stack [B, L, pos, H, d] → norm → [B, L, H, pos]
    V = torch.stack(values, dim=1)
    Vn = V.norm(dim=-1).permute(0, 1, 3, 2).contiguous()
    return tokens, logits, A, Vn


if __name__ == "__main__":
    model = load_model()
    tokens, logits, A, Vn = attention_from_forward(model, "The cat sat. The cat")
    print(f"model={DEFAULT_MODEL} device={model.cfg.device}")
    print(f"tokens={tuple(tokens.shape)} A={tuple(A.shape)} Vn={tuple(Vn.shape)}")
    print(model.to_str_tokens(tokens[0]))
