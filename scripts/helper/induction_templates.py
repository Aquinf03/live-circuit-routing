"""Induction template / length variants for sensitivity checks."""

from __future__ import annotations

from transformer_lens import HookedTransformer

from induction_data import InductionExample, _PAIRS, _is_single_token


def _build_pair_examples(model: HookedTransformer, fmt) -> list[InductionExample]:
    out: list[InductionExample] = []
    for a, b in _PAIRS:
        a_tok = f" {a}"
        b_tok = f" {b}"
        if not (_is_single_token(model, a_tok) and _is_single_token(model, b_tok)):
            continue
        prefix = fmt(a_tok, b_tok)
        out.append(InductionExample(prefix=prefix, target=b_tok, a=a, b=b))
    if len(out) < 8:
        raise RuntimeError(f"too few clean pairs for template: {len(out)}")
    return out


# Each fmt(a_tok, b_tok) builds the prefix ending at the second A (predict B).
TEMPLATES = {
    "the_dot": lambda a, b: f"The{a}{b}. The{a}",
    "bare": lambda a, b: f"{a}{b}{a}",
    "then": lambda a, b: f"Then{a}{b}. Then{a}",
    "yesterday": lambda a, b: f"Yesterday the{a}{b} quickly. Today the{a}",
}


def build_template_set(model: HookedTransformer, name: str) -> list[InductionExample]:
    if name not in TEMPLATES:
        raise KeyError(f"unknown template {name!r}; choose from {sorted(TEMPLATES)}")
    return _build_pair_examples(model, TEMPLATES[name])


def build_length_set(
    model: HookedTransformer,
    *,
    n_filler_sentences: int,
) -> list[InductionExample]:
    """Same the_dot pattern, preceded by n short filler sentences (length control)."""
    filler = " The sky is blue." * n_filler_sentences

    def fmt(a, b):
        return f"{filler} The{a}{b}. The{a}".lstrip()

    return _build_pair_examples(model, fmt)
