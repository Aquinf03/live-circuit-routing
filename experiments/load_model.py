"""Load GPT-2 Small (v1) via TransformerLens and read attention patterns."""

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward pass → attention routing tensor.

    Returns:
        tokens: [batch, T]
        logits: [batch, T, d_vocab]
        A: [batch, n_layers, n_heads, query, key]  (edge weight for (ℓ,h,j→i) is A[..., ℓ, h, i, j])
    """
    tokens = model.to_tokens(text)
    logits, cache = model.run_with_cache(tokens)
    patterns = [
        cache[f"blocks.{l}.attn.hook_pattern"] for l in range(model.cfg.n_layers)
    ]
    # each: [batch, head, query, key] → stack to [batch, layer, head, query, key]
    A = torch.stack(patterns, dim=1)
    return tokens, logits, A


if __name__ == "__main__":
    model = load_model()
    tokens, logits, A = attention_from_forward(model, "The cat sat. The cat")
    print(f"model={DEFAULT_MODEL} device={model.cfg.device}")
    print(f"tokens={tuple(tokens.shape)} A={tuple(A.shape)}  # [B, L, H, Q, K]")
    print(model.to_str_tokens(tokens[0]))
