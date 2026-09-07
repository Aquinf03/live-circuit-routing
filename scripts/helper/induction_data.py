"""Clean induction prompts: A B … A → B (large pool for n≈100 evals)."""

from __future__ import annotations

from dataclasses import dataclass

from transformer_lens import HookedTransformer


@dataclass(frozen=True, slots=True)
class InductionExample:
    prefix: str
    target: str  # expected next token string (usually with leading space)
    a: str
    b: str


# Prefer words that are usually a single GPT-2 / GPT-NeoX token with a leading space.
# Pool is oversized so each model can filter to ≥100 clean pairs.
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
    ("horse", "galloped"),
    ("mouse", "hid"),
    ("wolf", "howled"),
    ("bear", "slept"),
    ("fox", "waited"),
    ("duck", "swam"),
    ("goat", "climbed"),
    ("sheep", "grazed"),
    ("cow", "stood"),
    ("pig", "ate"),
    ("hen", "laid"),
    ("eagle", "soared"),
    ("shark", "hunted"),
    ("whale", "dived"),
    ("frog", "leaped"),
    ("snake", "slithered"),
    ("baby", "cried"),
    ("child", "laughed"),
    ("nurse", "cared"),
    ("pilot", "flew"),
    ("driver", "turned"),
    ("guard", "watched"),
    ("judge", "ruled"),
    ("lawyer", "argued"),
    ("soldier", "marched"),
    ("sailor", "sailed"),
    ("singer", "sang"),
    ("dancer", "danced"),
    ("actor", "acted"),
    ("poet", "read"),
    ("baker", "baked"),
    ("butcher", "cut"),
    ("tailor", "sewed"),
    ("miner", "dug"),
    ("hunter", "tracked"),
    ("ranger", "patrolled"),
    ("coach", "yelled"),
    ("referee", "whistled"),
    ("captain", "ordered"),
    ("mayor", "spoke"),
    ("priest", "prayed"),
    ("monk", "meditated"),
    ("robot", "moved"),
    ("engine", "roared"),
    ("plane", "landed"),
    ("ship", "sailed"),
    ("boat", "drifted"),
    ("bike", "rolled"),
    ("truck", "halted"),
    ("bus", "arrived"),
    ("bridge", "stood"),
    ("tower", "rose"),
    ("castle", "stood"),
    ("garden", "bloomed"),
    ("forest", "whispered"),
    ("mountain", "loomed"),
    ("valley", "echoed"),
    ("ocean", "crashed"),
    ("lake", "froze"),
    ("storm", "raged"),
    ("cloud", "drifted"),
    ("moon", "rose"),
    ("sun", "set"),
    ("star", "shone"),
    ("rock", "fell"),
    ("tree", "swayed"),
    ("flower", "bloomed"),
    ("leaf", "fell"),
    ("seed", "grew"),
    ("wave", "crashed"),
    ("flame", "flickered"),
    ("smoke", "rose"),
    ("shadow", "moved"),
    ("light", "faded"),
    ("bell", "rang"),
    ("clock", "ticked"),
    ("phone", "rang"),
    ("camera", "clicked"),
    ("window", "shattered"),
    ("gate", "closed"),
    ("wall", "cracked"),
    ("road", "ended"),
    ("path", "turned"),
    ("crowd", "cheered"),
    ("team", "won"),
    ("class", "listened"),
    ("band", "played"),
    ("orchestra", "played"),
    ("audience", "clapped"),
    ("hero", "arrived"),
    ("villain", "escaped"),
    ("ghost", "vanished"),
    ("dragon", "roared"),
    ("wizard", "cast"),
    ("knight", "charged"),
    ("prince", "bowed"),
    ("princess", "smiled"),
    ("giant", "stomped"),
    ("alien", "landed"),
    ("machine", "hummed"),
    ("computer", "crashed"),
    ("server", "failed"),
    ("signal", "faded"),
    ("message", "arrived"),
    ("letter", "arrived"),
    ("package", "arrived"),
    ("guest", "entered"),
    ("host", "greeted"),
    ("client", "paid"),
    ("customer", "left"),
    ("patient", "rested"),
    ("victim", "escaped"),
    ("witness", "spoke"),
    ("suspect", "fled"),
    ("officer", "arrested"),
    ("agent", "reported"),
    ("boss", "decided"),
    ("worker", "finished"),
    ("intern", "learned"),
    ("mentor", "guided"),
    ("rival", "competed"),
    ("partner", "agreed"),
    ("friend", "helped"),
    ("neighbor", "waved"),
    ("stranger", "appeared"),
    ("traveler", "rested"),
    ("explorer", "discovered"),
    ("scientist", "measured"),
    ("engineer", "built"),
    ("architect", "designed"),
    ("designer", "sketched"),
    ("photographer", "shot"),
    ("journalist", "wrote"),
    ("editor", "reviewed"),
    ("publisher", "released"),
    ("investor", "bought"),
    ("trader", "sold"),
    ("banker", "approved"),
    ("accountant", "checked"),
    ("manager", "approved"),
    ("director", "signed"),
    ("founder", "launched"),
    ("leader", "spoke"),
]


def _is_single_token(model: HookedTransformer, piece: str) -> bool:
    ids = model.to_tokens(piece, prepend_bos=False)[0]
    return int(ids.shape[0]) == 1


def build_induction_set(
    model: HookedTransformer,
    *,
    n: int | None = None,
) -> list[InductionExample]:
    """Build `The {A} {B}. The {A}` → `{B}` examples that tokenize cleanly.

    If ``n`` is set, return the first ``n`` clean examples (stable order).
    """
    out: list[InductionExample] = []
    for a, b in _PAIRS:
        a_tok = f" {a}"
        b_tok = f" {b}"
        if not (_is_single_token(model, a_tok) and _is_single_token(model, b_tok)):
            continue
        prefix = f"The{a_tok}{b_tok}. The{a_tok}"
        out.append(InductionExample(prefix=prefix, target=b_tok, a=a, b=b))
        if n is not None and len(out) >= n:
            break
    if len(out) < 8:
        raise RuntimeError(f"too few clean induction pairs: {len(out)}")
    if n is not None and len(out) < n:
        raise RuntimeError(
            f"only {len(out)} clean induction pairs after filter; need n={n}. "
            "Expand _PAIRS or lower --n."
        )
    return out
