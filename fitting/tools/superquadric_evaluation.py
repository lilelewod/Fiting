"""Area-consistent sampling and metrics for superquadric evaluation."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree

from models.surface.superquadric_rule import SuperquadricRule, SuperquadricTrait


def trait_from_mapping(mapping):
    """Build a validated ``SuperquadricTrait`` from JSON-compatible values."""
    if "trait" in mapping:
        mapping = mapping["trait"]
    trait = SuperquadricTrait()
    trait.center = np.asarray(mapping["center"], dtype=np.float64).reshape(3)
    trait.scale = np.asarray(mapping["scale"], dtype=np.float64).reshape(3)
    trait.shape = np.asarray(mapping["shape"], dtype=np.float64).reshape(2)
    trait.rotation = np.asarray(mapping.get("rotation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    matrix_value = mapping.get("rotation_matrix", mapping.get("rot_matrix"))
    if matrix_value is not None:
        trait.rot_matrix = np.asarray(matrix_value, dtype=np.float64).reshape(3, 3)
    else:
        trait.rot_matrix = Rotation.from_euler("xyz", trait.rotation).as_matrix()
    return trait


def load_trait(path):
    with open(Path(path), encoding="utf-8") as stream:
        return trait_from_mapping(json.load(stream))


def sample_trait(trait, count, seed, grid_resolution=256):
    return SuperquadricRule.sample_surface_uniform(
        trait,
        count=count,
        seed=seed,
        n_eta=grid_resolution,
        n_omega=grid_resolution,
    )


def geometric_metrics(reference, model, threshold):
    """Symmetric point-to-surface proxy on independently area-uniform clouds."""
    reference = np.asarray(reference, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    if reference.ndim != 2 or model.ndim != 2 or reference.shape[1:] != (3,) or model.shape[1:] != (3,):
        raise ValueError("reference and model must both have shape (N, 3)")
    if len(reference) == 0 or len(model) == 0:
        raise ValueError("reference and model must be non-empty")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    d2m = KDTree(model).query(reference, k=1)[0].ravel()
    m2d = KDTree(reference).query(model, k=1)[0].ravel()
    precision = float(np.mean(m2d < threshold))
    recall = float(np.mean(d2m < threshold))
    return {
        "gt_chamfer": float(np.mean(d2m) + np.mean(m2d)),
        "gt_d2m": float(np.mean(d2m)),
        "gt_m2d": float(np.mean(m2d)),
        "gt_fscore": float(2.0 * precision * recall / (precision + recall + 1e-8)),
    }
