import numpy as np
from scipy.stats import wilcoxon

from tools.exact_statistics import exact_wilcoxon_signed_rank


def test_exact_signed_rank_matches_scipy_without_zeros_or_ties():
    differences = np.asarray([0.2, -0.7, 1.1, 0.4, -1.8, 2.3])
    expected = wilcoxon(differences, alternative="two-sided", method="exact")
    actual = exact_wilcoxon_signed_rank(differences)

    assert actual["statistic"] == expected.statistic
    assert actual["exact_two_sided_p"] == expected.pvalue
    assert actual["nonzero_pairs"] == 6
    assert actual["zero_pairs"] == 0


def test_exact_signed_rank_handles_zeros_without_asymptotic_fallback():
    actual = exact_wilcoxon_signed_rank([0.0, 1.0, 2.0])

    assert actual["statistic"] == 0.0
    assert actual["exact_two_sided_p"] == 0.5
    assert actual["nonzero_pairs"] == 2
    assert actual["zero_pairs"] == 1


def test_exact_signed_rank_handles_tied_average_ranks():
    actual = exact_wilcoxon_signed_rank([1.0, 1.0, -1.0])

    assert actual["statistic"] == 2.0
    assert actual["exact_two_sided_p"] == 1.0


def test_exact_signed_rank_reports_degenerate_all_zero_pairs():
    actual = exact_wilcoxon_signed_rank([0.0, 0.0, 0.0, 0.0, 0.0])

    assert actual["statistic"] == 0.0
    assert actual["exact_two_sided_p"] == 1.0
    assert actual["nonzero_pairs"] == 0
    assert actual["zero_pairs"] == 5


def test_exact_signed_rank_requires_at_least_two_pairs():
    assert exact_wilcoxon_signed_rank([1.0]) is None
