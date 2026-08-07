import numpy as np

from tools.summarize_v3_superquadric_robustness import (
    describe,
    summarize_diagnostics,
    summarize_strata,
    threshold_sensitivity,
)


def test_describe_retains_extrema_that_can_fall_outside_the_iqr():
    summary = describe([0.02920, 0.02926, 0.02933, 0.02937, 0.04668])

    assert np.isclose(summary["minimum"], 0.02920)
    assert np.isclose(summary["median"], 0.02933)
    assert np.isclose(summary["maximum"], 0.04668)
    assert summary["maximum"] > summary["q3"]


def _row(case, value, success):
    return {
        "case": case,
        "gt_chamfer": value,
        "success": success,
    }


def test_stratified_summary_preserves_independent_cases_and_run_denominators():
    case_strata = {
        f"case_{index:03d}": {
            "shape": ("smooth", "mixed", "boxy")[index % 3],
            "aspect": ("balanced", "anisotropic", "extreme")[index // 3],
        }
        for index in range(9)
    }
    pso_rows = []
    case_medians = {}
    ems_rows = []
    for index, case in enumerate(case_strata):
        value = 0.01 * (index + 1)
        case_medians[case] = value
        successes = 5 if case_strata[case]["shape"] == "smooth" else 0
        pso_rows.extend(_row(case, value, int(repeat < successes)) for repeat in range(5))
        ems_rows.append(_row(case, value / 2.0, 1))

    result = summarize_strata(
        case_strata,
        "shape",
        ("smooth", "mixed", "boxy"),
        pso_rows,
        ems_rows,
        case_medians,
    )

    assert result["smooth"]["cases"] == ["case_000", "case_003", "case_006"]
    assert result["smooth"]["guided_pso_runs"]["runs"] == 15
    assert result["smooth"]["guided_pso_runs"]["successes"] == 15
    assert result["mixed"]["guided_pso_runs"]["runs"] == 15
    assert result["mixed"]["guided_pso_runs"]["successes"] == 0
    assert result["boxy"]["ems_cases"]["cases"] == 3
    assert result["boxy"]["ems_cases"]["successes"] == 3
    assert result["smooth"]["guided_pso_case_medians"]["count"] == 3
    assert np.isclose(result["smooth"]["guided_pso_case_medians"]["median"], 0.04)


def test_stratified_summary_marks_incomplete_cells_without_inventing_runs():
    case_strata = {
        "case_000": {"shape": "smooth", "aspect": "balanced"},
        "case_001": {"shape": "smooth", "aspect": "anisotropic"},
        "case_002": {"shape": "smooth", "aspect": "extreme"},
    }
    pso_rows = [_row("case_000", 0.02, 1) for _ in range(5)]
    pso_rows += [_row("case_001", 0.04, 0) for _ in range(3)]
    result = summarize_strata(
        case_strata,
        "shape",
        ("smooth",),
        pso_rows,
        [],
        {"case_000": 0.02, "case_001": 0.04},
    )

    cell = result["smooth"]
    assert cell["guided_pso_runs"]["runs"] == 8
    assert cell["guided_pso_runs"]["successes"] == 5
    assert cell["guided_pso_case_medians"]["count"] == 2
    assert cell["ems_cases"]["cases"] == 0
    assert cell["ems_cases"]["chamfer"] is None


def test_symmetry_aware_diagnostic_summary_counts_axis_roles_and_permutations():
    rows = [
        {
            "center_error_normalized": 0.01 + index * 0.001,
            "axis_frame_error_deg_any_permutation": 1.0 + index,
            "scale_relative_mae_at_best_frame": 0.02 + index * 0.001,
            "shape_mae": 0.1 + index * 0.01,
            "best_axis_permutation": "102" if index < 4 else "210",
            "z_role_preserved": int(index < 4),
        }
        for index in range(5)
    ]

    summary = summarize_diagnostics(rows)

    assert summary["runs"] == 5
    assert summary["z_role_preserved"] == 4
    assert summary["dominant_axis_permutation"] == "102"
    assert summary["axis_permutation_counts"] == {"102": 4, "210": 1}
    assert summary["shape_mae"]["count"] == 5
    assert np.isclose(summary["shape_mae"]["median"], 0.12)


def test_threshold_sensitivity_keeps_the_preregistered_threshold_centered():
    rows = [
        {"gt_chamfer": value}
        for value in (0.039, 0.040, 0.049, 0.050, 0.059, 0.060, 0.061)
    ]

    counts = threshold_sensitivity(rows, 0.05)

    assert counts == {"0.040": 2, "0.050": 4, "0.060": 6}
