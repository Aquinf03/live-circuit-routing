"""Paired causal-gap statistics: bootstrap CI + sign test (+ Wilcoxon if scipy)."""

from __future__ import annotations

import math
import random
from typing import Sequence


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _binom_sf_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided binomial test via exact sum (n small enough for n≈100)."""
    if n == 0:
        return 1.0
    # P(X=i)
    probs = []
    # compute recursively
    log_c = 0.0  # log C(n,0)
    for i in range(n + 1):
        # C(n,i) * p^i * (1-p)^(n-i)
        if i > 0:
            log_c += math.log(n - i + 1) - math.log(i)
        probs.append(math.exp(log_c + i * math.log(p) + (n - i) * math.log(1 - p)))
    observed = probs[k]
    # two-sided: sum probs <= observed (+ tiny tol)
    total = sum(pr for pr in probs if pr <= observed + 1e-15)
    return min(1.0, total)


def paired_gap_stats(
    drops_S: Sequence[float],
    drops_R: Sequence[float],
    *,
    seed: int = 0,
    n_boot: int = 10_000,
) -> dict:
    """Stats on per-example gaps g_i = drop_S_i - drop_R_i."""
    if len(drops_S) != len(drops_R):
        raise ValueError("drops_S and drops_R length mismatch")
    gaps = [float(s) - float(r) for s, r in zip(drops_S, drops_R)]
    n = len(gaps)
    mean_gap = _mean(gaps)
    std_gap = _std(gaps)
    se_gap = std_gap / math.sqrt(n) if n else 0.0

    n_pos = sum(1 for g in gaps if g > 0)
    n_neg = sum(1 for g in gaps if g < 0)
    n_zero = n - n_pos - n_neg
    n_fail = sum(
        1 for s, r in zip(drops_S, drops_R) if float(s) <= float(r)
    )  # drop_S <= drop_R

    # Sign test on non-ties: H0 P(g>0)=1/2
    n_eff = n_pos + n_neg
    sign_p = _binom_sf_two_sided(n_pos, n_eff) if n_eff else 1.0

    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [gaps[rng.randrange(n)] for _ in range(n)]
        boots.append(_mean(sample))
    boots.sort()
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]

    wilcoxon_p = None
    try:
        from scipy.stats import wilcoxon

        # zero_method="wilcox" drops zeros
        if n_eff >= 1:
            res = wilcoxon(
                drops_S,
                drops_R,
                zero_method="wilcox",
                alternative="greater",
                method="auto",
            )
            wilcoxon_p = float(res.pvalue)
    except Exception:
        wilcoxon_p = None

    return {
        "n": n,
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "se_gap": se_gap,
        "gap_ci95_low": lo,
        "gap_ci95_high": hi,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "n_fail": n_fail,
        "sign_test_p": sign_p,
        "wilcoxon_greater_p": wilcoxon_p,
    }


def format_gap_stats(stats: dict) -> str:
    lines = [
        f"paired gap mean={stats['mean_gap']:+.4f} ± {stats['std_gap']:.4f} "
        f"(SE {stats['se_gap']:.4f})",
        f"95% bootstrap CI [{stats['gap_ci95_low']:+.4f}, {stats['gap_ci95_high']:+.4f}]",
        f"sign test (g>0): {stats['n_pos']}/{stats['n_pos']+stats['n_neg']} "
        f"non-ties, p={stats['sign_test_p']:.4g}; fails(drop_S≤drop_R)={stats['n_fail']}/{stats['n']}",
    ]
    if stats.get("wilcoxon_greater_p") is not None:
        lines.append(
            f"Wilcoxon signed-rank (S>R): p={stats['wilcoxon_greater_p']:.4g}"
        )
    return "\n".join(lines)
