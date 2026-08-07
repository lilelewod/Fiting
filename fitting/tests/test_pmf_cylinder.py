import numpy as np

from tools.audit_pmf_cylinder_experiment import audit_status

from models.surface.pmf_cylinder_rule import PMFCylinderTrait, sample_partial_cylinder


def test_partial_cylinder_sampling_and_measure():
    trait = PMFCylinderTrait()
    trait.radius = 2.0
    trait.height = 6.0
    trait.start_angle = -2.4
    trait.angular_span = 4.8
    points = sample_partial_cylinder(trait, sample_angle=64, sample_height=32)
    assert points.shape == (2048, 3)
    radial = np.linalg.norm(points[:, :2], axis=1)
    assert np.allclose(radial, trait.radius, atol=1e-6)
    assert points[:, 2].min() > 0.0
    assert points[:, 2].max() < trait.height
    assert np.isclose(trait.radius * trait.angular_span * trait.height, 57.6)


def test_audit_status_requires_complete_and_error_free_matrix():
    assert audit_status([], set()) == "PASS"
    assert audit_status([], {("clean", "pso", 1)}) == "INCOMPLETE"
    assert audit_status(["metric mismatch"], set()) == "FAIL"
    assert audit_status(["metric mismatch"], {("clean", "pso", 1)}) == "FAIL"
