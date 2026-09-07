"""IOI prompts for Task 2 (Indirect Object Identification).

Template (Wang et al. style):
  When {A} and {B} went to the store, {A} gave a drink to
  → {B}   (indirect object, not the repeated subject {A})
"""

from __future__ import annotations

from dataclasses import dataclass

from transformer_lens import HookedTransformer


@dataclass(frozen=True, slots=True)
class IOIExample:
    prefix: str
    io: str  # correct next token string (usually leading space)
    s: str  # incorrect subject name token string
    a: str
    b: str


# Single-token-friendly GPT-2 first names (with leading space when used as tokens).
_NAME_PAIRS: list[tuple[str, str]] = [
    ("John", "Mary"),
    ("Mary", "John"),
    ("Alice", "Bob"),
    ("Bob", "Alice"),
    ("Tom", "Sarah"),
    ("Sarah", "Tom"),
    ("James", "Emily"),
    ("Emily", "James"),
    ("David", "Lisa"),
    ("Lisa", "David"),
    ("Paul", "Anna"),
    ("Anna", "Paul"),
    ("Mark", "Laura"),
    ("Laura", "Mark"),
    ("Chris", "Emma"),
    ("Emma", "Chris"),
    ("Sam", "Kate"),
    ("Kate", "Sam"),
    ("Dan", "Ruth"),
    ("Ruth", "Dan"),
    ("Joe", "Amy"),
    ("Amy", "Joe"),
    ("Tim", "Nina"),
]


def _is_single_token(model: HookedTransformer, piece: str) -> bool:
    ids = model.to_tokens(piece, prepend_bos=False)[0]
    return int(ids.shape[0]) == 1


def build_ioi_set(model: HookedTransformer) -> list[IOIExample]:
    out: list[IOIExample] = []
    for a, b in _NAME_PAIRS:
        a_tok = f" {a}"
        b_tok = f" {b}"
        if not (_is_single_token(model, a_tok) and _is_single_token(model, b_tok)):
            continue
        # A is subject (repeated); B is indirect object (correct completion)
        prefix = f"When{a_tok} and{b_tok} went to the store,{a_tok} gave a drink to"
        out.append(IOIExample(prefix=prefix, io=b_tok, s=a_tok, a=a, b=b))
    if len(out) < 8:
        raise RuntimeError(f"too few clean IOI pairs: {len(out)}")
    return out
