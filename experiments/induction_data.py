"""Clean induction prompts: A B … A → B."""

from __future__ import annotations

from dataclasses import dataclass

from transformer_lens import HookedTransformer


@dataclass(frozen=True, slots=True)
class InductionExample:
    prefix: str
    target: str  # expected next token string (usually with leading space)
    a: str
    b: str


# Word pairs that are typically single GPT-2 tokens (with leading space).
_PAIRS: list[tuple[str, str]] = [
    ("cat", "sat"),
    ("dog", "ran"),
    ("bird", "flew"),
    ("fish", "swam"),
    ("boy", "jumped"),
    ("girl", "smiled"),
    ("man", "walked"),
    ("woman", "talked"),
    ("car", "stopped"),
    ("train", "moved"),
    ("king", "ruled"),
    ("queen", "spoke"),
    ("teacher", "asked"),
    ("student", "answered"),
    ("doctor", "helped"),
    ("player", "scored"),
    ("artist", "painted"),
    ("writer", "wrote"),
    ("chef", "cooked"),
    ("farmer", "worked"),
    ("river", "flowed"),
    ("fire", "burned"),
    ("wind", "blew"),
    ("door", "opened"),
]


def _is_single_token(model: HookedTransformer, piece: str) -> bool:
    ids = model.to_tokens(piece, prepend_bos=False)[0]
    return int(ids.shape[0]) == 1


def build_induction_set(model: HookedTransformer) -> list[InductionExample]:
    """Build `The {A} {B}. The {A}` → `{B}` examples that tokenize cleanly."""
    out: list[InductionExample] = []
    for a, b in _PAIRS:
        a_tok = f" {a}"
        b_tok = f" {b}"
        if not (_is_single_token(model, a_tok) and _is_single_token(model, b_tok)):
            continue
        prefix = f"The{a_tok}{b_tok}. The{a_tok}"
        out.append(InductionExample(prefix=prefix, target=b_tok, a=a, b=b))
    if len(out) < 8:
        raise RuntimeError(f"too few clean induction pairs: {len(out)}")
    return out
