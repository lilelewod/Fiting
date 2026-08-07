import json
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree

from core.estimator.mm_estimator import MeanMeasureEstimator
from models.rule import Token
from models.surface.superquadric_rule import SuperquadricRule
from tools.superquadric_evaluation import geometric_metrics, sample_trait, trait_from_mapping
from tools.superquadric_initialization import (
    adaptive_density_support,
    density_support,
    guided_population,
    parameter_hypotheses,
)
from tools.tool import json_default


def _implicit_value(local_points, scale, shape):
    x = np.abs(local_points[:, 0] / scale[0]) ** (2.0 / shape[0])
    y = np.abs(local_points[:, 1] / scale[1]) ** (2.0 / shape[0])
    z = np.abs(local_points[:, 2] / scale[2]) ** (2.0 / shape[1])
    return (x + y) ** (shape[0] / shape[1]) + z


def test_unit_sphere_sampling_and_area():
    points, weights, area = SuperquadricRule._spherical_product(
        1.0, 1.0, 1.0, 1.0, 1.0, n_eta=96, n_omega=96, return_weights=True
    )

    np.testing.assert_allclose(np.sum(points * points, axis=1), 1.0, atol=3e-6)
    np.testing.assert_allclose(weights.sum(), area, rtol=1e-6)
    assert abs(area - 4.0 * np.pi) / (4.0 * np.pi) < 1e-3
    assert np.all(weights > 0.0)


def test_area_weight_ablation_only_disables_token_weights():
    class FakeEstimator:
        dimension = 3

        def __init__(self, enabled):
            self.cfg = {
                "model": {
                    "sample_eta": 16,
                    "sample_omega": 12,
                    "use_area_weights": enabled,
                }
            }
            self.token = None

        def get_data(self):
            return np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)

        def add_token(self, token):
            self.token = token

    tokens = []
    for enabled in (False, True):
        estimator = FakeEstimator(enabled)
        rule = SuperquadricRule(estimator)
        rule.parse(action=np.zeros(11, dtype=np.float32))
        rule.generate()
        tokens.append(estimator.token)

    uniform_token, weighted_token = tokens
    assert uniform_token.weights is None
    assert weighted_token.weights is not None
    assert uniform_token.points.shape == weighted_token.points.shape
    np.testing.assert_allclose(uniform_token.points, weighted_token.points)
    np.testing.assert_allclose(uniform_token.measure, weighted_token.measure)


def test_parameter_mapping_rotation_and_implicit_surface():
    class FakeEstimator:
        cfg = {"model": {"sample_eta": 48, "sample_omega": 40}}
        dimension = 3

        @staticmethod
        def get_data():
            return np.array([[-2.0, -1.5, -1.0], [2.0, 1.5, 1.0]], dtype=np.float32)

    rule = SuperquadricRule(FakeEstimator())
    rule._init_bounds()
    center = np.array([0.3, -0.2, 0.4], dtype=np.float32)
    scale = np.array([1.2, 0.8, 0.55], dtype=np.float32)
    shape = np.array([0.65, 1.35], dtype=np.float32)
    angles = np.array([0.25, -0.4, 0.7], dtype=np.float32)
    physical = np.concatenate((center, scale, shape, angles))
    action = 2.0 * (physical - rule.lb) / (rule.ub - rule.lb) - 1.0

    trait = rule.parse(action=action)
    points, weights, area = rule.sample_with_weights()
    local = (points - trait.center) @ trait.rot_matrix

    np.testing.assert_allclose(trait.center, center, atol=2e-6)
    np.testing.assert_allclose(trait.scale, scale, atol=2e-6)
    np.testing.assert_allclose(trait.shape, shape, atol=2e-6)
    np.testing.assert_allclose(
        trait.rot_matrix, Rotation.from_euler("xyz", angles).as_matrix(), atol=2e-6
    )
    np.testing.assert_allclose(
        _implicit_value(local, trait.scale, trait.shape), 1.0, atol=2e-5
    )
    assert points.shape == (48 * 40, 3)
    assert weights.shape == (48 * 40,)
    assert area > 0.0


def test_surface_uniform_sampling_is_reproducible_and_area_balanced():
    trait = type("UniformSamplingTrait", (), {})()
    trait.center = np.array([0.2, -0.3, 0.4], dtype=np.float32)
    trait.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    trait.shape = np.array([1.0, 1.0], dtype=np.float32)
    trait.rot_matrix = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()

    first = SuperquadricRule.sample_surface_uniform(
        trait, 12000, seed=17, n_eta=128, n_omega=96
    )
    second = SuperquadricRule.sample_surface_uniform(
        trait, 12000, seed=17, n_eta=128, n_omega=96
    )
    third = SuperquadricRule.sample_surface_uniform(
        trait, 12000, seed=18, n_eta=128, n_omega=96
    )

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)
    assert first.shape == (12000, 3)
    assert np.all(np.isfinite(first))

    # Undo the pose. Equal-area sphere samples have a uniform z marginal, so
    # equally sized z bins should contain approximately equal point counts.
    local = (first - trait.center) @ trait.rot_matrix
    radii = np.linalg.norm(local, axis=1)
    np.testing.assert_allclose(radii, 1.0, atol=8e-4)
    counts, _ = np.histogram(local[:, 2], bins=np.linspace(-1.0, 1.0, 9))
    assert counts.max() / counts.min() < 1.18


def test_area_uniform_external_metrics_are_reproducible():
    mapping = {
        "center": [0.1, -0.2, 0.3],
        "scale": [1.0, 0.8, 0.6],
        "shape": [1.0, 1.0],
        "rotation": [0.2, -0.3, 0.4],
    }
    truth = trait_from_mapping(mapping)
    reference = sample_trait(truth, 8000, seed=101, grid_resolution=128)
    model = sample_trait(truth, 8000, seed=102, grid_resolution=128)
    first = geometric_metrics(reference, model, threshold=0.06)
    second = geometric_metrics(reference, model, threshold=0.06)

    assert first == second
    assert first["gt_chamfer"] < 0.04
    assert first["gt_fscore"] > 0.99

    shifted_mapping = dict(mapping)
    shifted_mapping["center"] = [0.35, -0.2, 0.3]
    shifted = sample_trait(
        trait_from_mapping(shifted_mapping), 8000, seed=102, grid_resolution=128
    )
    shifted_metrics = geometric_metrics(reference, shifted, threshold=0.06)
    assert shifted_metrics["gt_chamfer"] > first["gt_chamfer"] * 2.0
    assert shifted_metrics["gt_fscore"] < first["gt_fscore"]


def _make_estimator(data, estimator_overrides=None):
    def load_data(_estimator):
        return data.copy()

    cfg = {
        "model": {"type": "superquadric", "sample_eta": 64, "sample_omega": 64},
        "estimator": {
            "data_resolution": 0.05,
            "model_resolution": 0.02,
            "regularization_factor": 0.5,
            "incremental_coverage": True,
            "coverage_distance_factor": 3.0,
            "coverage_power": 1.0,
            "outlier_distance_factor": 3.0,
            "rule_class": SuperquadricRule,
            "load_data_fn": load_data,
        },
    }
    if estimator_overrides:
        cfg["estimator"].update(estimator_overrides)
    return MeanMeasureEstimator(cfg)


def test_mm_estimator_optional_density_support_is_label_free_and_recorded():
    rng = np.random.default_rng(91)
    surface = np.column_stack(
        (rng.uniform(-1.0, 1.0, 300), rng.normal(0.0, 0.01, 300), rng.normal(0.0, 0.01, 300))
    ).astype(np.float32)
    outliers = rng.uniform(-5.0, 5.0, size=(100, 3)).astype(np.float32)
    data = np.vstack((surface, outliers))
    estimator = _make_estimator(
        data,
        {"density_support_fraction": 0.75, "density_support_neighbors": 8},
    )

    assert estimator.raw_data.shape == (400, 3)
    assert estimator.data.shape == (300, 3)
    assert estimator.density_support_info == {
        "raw_points": 400,
        "support_points": 300,
        "support_fraction": 0.75,
        "actual_support_fraction": 0.75,
        "support_neighbors": 8,
        "support_mode": "fixed",
    }
    # The sparse gross outliers should not survive this deterministic density cut.
    assert np.max(np.linalg.norm(estimator.data, axis=1)) < 2.0


def test_adaptive_density_support_recovers_unknown_dense_fraction():
    rng = np.random.default_rng(92)
    dense = rng.normal(0.0, 0.08, size=(300, 3))
    sparse = rng.uniform(-5.0, 5.0, size=(200, 3))
    points = np.vstack((dense, sparse))
    support = adaptive_density_support(points, neighbors=12)

    assert 280 <= len(support) <= 320
    assert np.mean(np.linalg.norm(support, axis=1) < 0.5) > 0.97


def _score_surface(estimator, points, weights, area):
    token = Token(3)
    token.points = points.astype(np.float32)
    token.weights = weights.astype(np.float32)
    token.measure = float(area)
    token.action = np.zeros(11, dtype=np.float32)
    token.trait = None
    estimator.reset()
    estimator.add_token(token)
    return estimator.score, estimator.coverage_ratio


def test_area_weighted_mm_and_coverage_favor_true_surface():
    rng = np.random.default_rng(7)
    dense_points, dense_weights, _ = SuperquadricRule._spherical_product(
        1.4, 0.9, 0.6, 0.7, 1.25, n_eta=160, n_omega=120, return_weights=True
    )
    probability = dense_weights / dense_weights.sum()
    data = dense_points[rng.choice(len(dense_points), 3000, p=probability)]
    data = data + rng.normal(0.0, 0.01, data.shape)
    estimator = _make_estimator(data.astype(np.float32))

    true_points, true_weights, true_area = SuperquadricRule._spherical_product(
        1.4, 0.9, 0.6, 0.7, 1.25, n_eta=64, n_omega=64, return_weights=True
    )
    true_score, true_coverage = _score_surface(
        estimator, true_points, true_weights, true_area
    )

    bad_points = true_points * np.array([0.55, 0.55, 0.55], dtype=np.float32)
    bad_weights = true_weights * (0.55 ** 2)
    bad_score, bad_coverage = _score_surface(
        estimator, bad_points, bad_weights, true_area * (0.55 ** 2)
    )

    assert true_coverage > 0.95
    assert bad_coverage < 0.25
    assert true_score > bad_score * 5.0


def test_fitted_trait_is_json_reproducible():
    # Use the real class name convention consumed by json_default.
    trait = type("SyntheticTrait", (), {})()
    trait.center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    trait.shape = np.array([0.5, 1.5], dtype=np.float32)
    restored = json.loads(json.dumps(trait, default=json_default))
    assert restored == {"center": [1.0, 2.0, 3.0], "shape": [0.5, 1.5]}


def test_geometry_guided_hypotheses_recover_clean_cylinder_axes_without_trait_input():
    mapping = {
        "center": [0.1, -0.2, 0.3],
        "scale": [0.55, 0.55, 1.4],
        "shape": [1.0, 0.2],
        "rotation": [0.35, -0.25, 0.15],
    }
    trait = trait_from_mapping(mapping)
    points = sample_trait(trait, 12000, seed=91, grid_resolution=192)
    hypotheses = parameter_hypotheses(points)
    cylinder_hypotheses = [item for item in hypotheses if np.allclose(item[6:8], [1.0, 0.25])]

    true_axis = trait.rot_matrix[:, 2]
    best = max(
        cylinder_hypotheses,
        key=lambda item: abs(np.dot(Rotation.from_euler("xyz", item[8:11]).as_matrix()[:, 2], true_axis)),
    )
    recovered_rotation = Rotation.from_euler("xyz", best[8:11]).as_matrix()
    assert abs(np.dot(recovered_rotation[:, 2], true_axis)) > 0.995
    np.testing.assert_allclose(best[:3], mapping["center"], atol=0.03)
    np.testing.assert_allclose(np.sort(best[3:5]), [0.55, 0.55], atol=0.06)
    np.testing.assert_allclose(best[5], 1.4, atol=0.08)


def test_guided_population_is_bounded_reproducible_and_mixed():
    mapping = {
        "center": [0.0, 0.0, 0.0],
        "scale": [0.8, 0.6, 0.5],
        "shape": [0.2, 0.2],
        "rotation": [-0.3, 0.2, 0.45],
    }
    points = sample_trait(trait_from_mapping(mapping), 6000, seed=93, grid_resolution=160)

    class FakeEstimator:
        cfg = {"model": {}}

        @staticmethod
        def get_data():
            return points

    rule = SuperquadricRule(FakeEstimator())
    rule._init_bounds()
    first, first_info = guided_population(
        points, rule.lb, rule.ub, 16, np.random.default_rng(17)
    )
    second, second_info = guided_population(
        points, rule.lb, rule.ub, 16, np.random.default_rng(17)
    )

    np.testing.assert_array_equal(first, second)
    assert first_info == second_info == {
        "guided_count": 12,
        "random_count": 4,
        "exact_anchor_count": 9,
        "hypothesis_count": 9,
        "support_fraction": 1.0,
        "support_neighbors": 8,
    }
    assert first.shape == (16, 11)
    assert np.all(first >= -1.0) and np.all(first <= 1.0)
    # The exact box anchors use no fitted trait or ground-truth metadata.
    assert np.any(np.all(np.isclose(first[:9, 6:8], 2.0 * (0.25 - rule.lb[6:8]) / (rule.ub[6:8] - rule.lb[6:8]) - 1.0), axis=1))


def test_density_support_rejects_sparse_gross_outliers_without_labels():
    mapping = {
        "center": [0.0, 0.0, 0.0],
        "scale": [0.55, 0.55, 1.4],
        "shape": [1.0, 0.2],
        "rotation": [0.35, -0.25, 0.15],
    }
    trait = trait_from_mapping(mapping)
    inliers = sample_trait(trait, 4000, seed=111, grid_resolution=192)
    rng = np.random.default_rng(112)
    outliers = rng.uniform([-2.5, -2.5, -2.5], [2.5, 2.5, 2.5], size=(1000, 3))
    contaminated = np.vstack((inliers, outliers))
    support = density_support(contaminated, support_fraction=0.75, neighbors=8)

    assert support.shape == (3750, 3)
    # Dense surface points should dominate without using the known labels.
    nearest_inlier = KDTree(inliers).query(support, k=1)[0].ravel()
    assert np.mean(nearest_inlier < 1e-8) > 0.99

    hypotheses = parameter_hypotheses(contaminated, support_fraction=0.75)
    cylinder_hypotheses = [item for item in hypotheses if np.allclose(item[6:8], [1.0, 0.25])]
    true_axis = trait.rot_matrix[:, 2]
    alignment = max(
        abs(np.dot(Rotation.from_euler("xyz", item[8:11]).as_matrix()[:, 2], true_axis))
        for item in cylinder_hypotheses
    )
    assert alignment > 0.99


def test_ems_shape_conversion_uses_project_storage_order():
    from tools.external_parameter_conventions import ems_shape_to_project

    # EMS stores [meridional epsilon1, azimuthal epsilon2], while this
    # project's existing trait files store [azimuthal, meridional].
    np.testing.assert_array_equal(ems_shape_to_project([0.2, 1.0]), [1.0, 0.2])


def test_randomized_benchmark_traits_are_stratified_and_reproducible():
    from tools.prepare_randomized_superquadric_benchmark import randomized_trait

    first = [randomized_trait(index, 20260721) for index in range(9)]
    second = [randomized_trait(index, 20260721) for index in range(9)]
    assert first == second
    assert {item[1]["shape"] for item in first} == {"smooth", "mixed", "boxy"}
    assert {item[1]["aspect"] for item in first} == {"balanced", "anisotropic", "extreme"}
    for payload, strata, seeds in first:
        assert np.isclose(max(payload["scale"]), 1.2)
        assert min(payload["scale"]) >= 0.15 * 1.2 - 1e-12
        assert np.all((np.asarray(strata["ems_shape_epsilon1_meridional_epsilon2_azimuthal"]) >= 0.15))
        assert np.all((np.asarray(strata["ems_shape_epsilon1_meridional_epsilon2_azimuthal"]) <= 1.0))
        assert len(set(seeds.values())) == len(seeds)
