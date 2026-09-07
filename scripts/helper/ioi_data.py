"""IOI prompts for Task 2 (Indirect Object Identification).

Large name × template pool so we can run n≈100 clean examples.
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
    template: str = "store"


# First names that are often single tokens with a leading space.
_NAMES: list[str] = [
    "John",
    "Mary",
    "Alice",
    "Bob",
    "Tom",
    "Sarah",
    "James",
    "Emily",
    "David",
    "Lisa",
    "Paul",
    "Anna",
    "Mark",
    "Laura",
    "Chris",
    "Emma",
    "Sam",
    "Kate",
    "Dan",
    "Ruth",
    "Joe",
    "Amy",
    "Tim",
    "Nina",
    "Ryan",
    "Grace",
    "Luke",
    "Olivia",
    "Jack",
    "Sophie",
    "Ben",
    "Claire",
    "Adam",
    "Helen",
    "Eric",
    "Julia",
    "Kevin",
    "Rachel",
    "Brian",
    "Megan",
    "Jason",
    "Lauren",
    "Matt",
    "Hannah",
    "Steve",
    "Chloe",
    "Andrew",
    "Natalie",
    "Peter",
    "Victoria",
]

# (name, place/object filler) → When A and B went to the {place}, A gave a {obj} to
_TEMPLATES: list[tuple[str, str, str]] = [
    ("store", "store", "drink"),
    ("park", "park", "ball"),
    ("cafe", "cafe", "coffee"),
    ("school", "school", "book"),
    ("office", "office", "pen"),
    ("station", "station", "ticket"),
    ("library", "library", "note"),
    ("market", "market", "bag"),
]


def _is_single_token(model: HookedTransformer, piece: str) -> bool:
    ids = model.to_tokens(piece, prepend_bos=False)[0]
    return int(ids.shape[0]) == 1


def build_ioi_set(
    model: HookedTransformer,
    *,
    n: int | None = None,
) -> list[IOIExample]:
    """Build IOI examples that tokenize cleanly; optionally cap at ``n``."""
    out: list[IOIExample] = []
    clean_names = [name for name in _NAMES if _is_single_token(model, f" {name}")]
    if len(clean_names) < 4:
        raise RuntimeError(f"too few single-token names: {len(clean_names)}")

    for tmpl_name, place, obj in _TEMPLATES:
        for i, a in enumerate(clean_names):
            for b in clean_names[i + 1 :]:
                for a_name, b_name in ((a, b), (b, a)):
                    a_tok = f" {a_name}"
                    b_tok = f" {b_name}"
                    prefix = (
                        f"When{a_tok} and{b_tok} went to the {place},"
                        f"{a_tok} gave a {obj} to"
                    )
                    out.append(
                        IOIExample(
                            prefix=prefix,
                            io=b_tok,
                            s=a_tok,
                            a=a_name,
                            b=b_name,
                            template=tmpl_name,
                        )
                    )
                    if n is not None and len(out) >= n:
                        return out

    if len(out) < 8:
        raise RuntimeError(f"too few clean IOI pairs: {len(out)}")
    if n is not None and len(out) < n:
        raise RuntimeError(f"only {len(out)} IOI examples; need n={n}")
    return out
