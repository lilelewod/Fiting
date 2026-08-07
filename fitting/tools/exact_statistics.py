"""Small-sample paired statistics with explicit zero/tie handling."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import rankdata


def exact_wilcoxon_signed_rank(differences):
    """Exact two-sided signed-rank permutation test after dropping exact zeros.

    Average ranks handle tied absolute differences, and a dynamic program
    enumerates the full conditional sign distribution without Monte Carlo.
    """
    values = np.asarray(differences, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Signed-rank differences must all be finite")
    total_pairs = int(values.size)
    if total_pairs < 2:
        return None
    zero_pairs = int(np.sum(values == 0.0))
    values = values[values != 0.0]
    if values.size == 0:
        return {
            "statistic": 0.0,
            "exact_two_sided_p": 1.0,
            "nonzero_pairs": 0,
            "zero_pairs": zero_pairs,
            "method": "degenerate exact conditional signed-rank test",
        }

    ranks_twice = np.rint(2.0 * rankdata(np.abs(values), method="average")).astype(int)
    total = int(np.sum(ranks_twice))
    observed_positive = int(np.sum(ranks_twice[values > 0.0]))
    observed_statistic = min(observed_positive, total - observed_positive)

    subset_counts = {0: 1}
    for rank in ranks_twice:
        updated = defaultdict(int)
        for subtotal, count in subset_counts.items():
            updated[subtotal] += count
            updated[subtotal + int(rank)] += count
        subset_counts = dict(updated)
    extreme = sum(
        count
        for positive_sum, count in subset_counts.items()
        if min(positive_sum, total - positive_sum) <= observed_statistic
    )
    assignments = 1 << int(values.size)
    return {
        "statistic": float(observed_statistic / 2.0),
        "exact_two_sided_p": float(extreme / assignments),
        "nonzero_pairs": int(values.size),
        "zero_pairs": zero_pairs,
        "method": "exact conditional sign permutation of Wilcoxon average ranks",
    }
