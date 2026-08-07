"""Geometry-guided, ground-truth-free initialization for superquadric search."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


DEFAULT_EXPONENT_ANCHORS = (
    (1.0, 1.0),   # ellipsoid-like
    (1.0, 0.25),  # cylinder-like
    (0.25, 0.25), # box-like
)


def density_support(
    points: np.ndarray,
    support_fraction: float = 1.0,
    neighbors: int = 8,
) -> np.ndarray:
    """Select a label-free high-density support for robust pose estimation."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError("points must have shape (N, 3) with N >= 4")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    if not 0.0 < support_fraction <= 1.0:
        raise ValueError("support_fraction must lie in (0, 1]")
    if support_fraction == 1.0:
        return points
    k = min(max(2, int(neighbors)), points.shape[0] - 1)
    distances = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(points).kneighbors(
        points, return_distance=True
    )[0][:, -1]
    keep_count = max(4, int(np.floor(points.shape[0] * support_fraction)))
    indices = np.argpartition(distances, keep_count - 1)[:keep_count]
    return points[np.sort(indices)]


def adaptive_density_support(points: np.ndarray, neighbors: int = 8) -> np.ndarray:
    """Split log k-NN distances into dense/sparse modes without labels.

    The component with the smaller median k-neighbor distance is retained.
    This adapts the retained fraction to the observed density separation while
    keeping the rule deterministic through a fixed clustering seed.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError("points must have shape (N, 3) with N >= 4")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    k = min(max(2, int(neighbors)), points.shape[0] - 1)
    distances = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(points).kneighbors(
        points, return_distance=True
    )[0][:, -1]
    labels = KMeans(n_clusters=2, random_state=0, n_init=20).fit_predict(
        np.log(np.maximum(distances, np.finfo(float).eps)).reshape(-1, 1)
    )
    medians = np.asarray([np.median(distances[labels == label]) for label in (0, 1)])
    dense_label = int(np.argmin(medians))
    indices = np.flatnonzero(labels == dense_label)
    if indices.size < 4:
        raise RuntimeError("adaptive density support selected fewer than four points")
    return points[indices]


def _canonical_principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)

    center = np.median(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigenvalues, axes = np.linalg.eigh(covariance)
    axes = axes[:, np.argsort(eigenvalues)[::-1]]

    # Resolve the arbitrary eigenvector signs deterministically. Closed
    # superquadrics are sign-symmetric, so this changes only representation.
    for column in range(3):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return center, axes


def parameter_hypotheses(
    points: np.ndarray,
    extent_quantile: float = 0.005,
    exponent_anchors=DEFAULT_EXPONENT_ANCHORS,
    support_fraction: float = 1.0,
    support_neighbors: int = 8,
) -> list[np.ndarray]:
    """Build pose/scale/exponent hypotheses using only the observed cloud.

    Three PCA-axis assignments are considered because the superquadric's
    second exponent has a distinguished local z-axis role. Swapping local x
    and y is geometrically redundant for the shared first exponent.
    """
    if not 0.0 <= extent_quantile < 0.25:
        raise ValueError("extent_quantile must lie in [0, 0.25)")
    points = np.asarray(points, dtype=np.float64)
    support = density_support(points, support_fraction, support_neighbors)
    base_center, principal_axes = _canonical_principal_axes(support)
    hypotheses = []
    for z_index in range(3):
        remaining = [index for index in range(3) if index != z_index]
        rotation = principal_axes[:, [remaining[0], remaining[1], z_index]].copy()
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 1] *= -1.0

        local = (support - base_center) @ rotation
        low = np.quantile(local, extent_quantile, axis=0)
        high = np.quantile(local, 1.0 - extent_quantile, axis=0)
        local_midpoint = 0.5 * (low + high)
        center = base_center + local_midpoint @ rotation.T
        scale = np.maximum(0.5 * (high - low), np.finfo(np.float32).eps)
        euler = Rotation.from_matrix(rotation).as_euler("xyz")

        for anchor in exponent_anchors:
            shape = np.asarray(anchor, dtype=np.float64)
            if shape.shape != (2,) or np.any(shape <= 0.0):
                raise ValueError("every exponent anchor must contain two positive values")
            hypotheses.append(np.concatenate([center, scale, shape, euler]))
    return hypotheses


def parameters_to_action(parameters: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if parameters.shape[-1] != lower.size or lower.shape != upper.shape:
        raise ValueError("parameter and bound dimensions do not match")
    span = np.maximum(upper - lower, np.finfo(np.float64).eps)
    return np.clip(2.0 * (parameters - lower) / span - 1.0, -1.0, 1.0)


def guided_population(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    population_size: int,
    rng,
    guided_fraction: float = 0.75,
    jitter: float = 0.04,
    extent_quantile: float = 0.005,
    support_fraction: float = 1.0,
    support_neighbors: int = 8,
) -> tuple[np.ndarray, dict]:
    """Return a mixed guided/random normalized population in [-1, 1]."""
    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if not 0.0 < guided_fraction <= 1.0:
        raise ValueError("guided_fraction must lie in (0, 1]")
    if jitter < 0.0:
        raise ValueError("jitter must be nonnegative")

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    population = rng.uniform(-1.0, 1.0, size=(population_size, lower.size))
    anchors = np.asarray(
        [parameters_to_action(item, lower, upper) for item in parameter_hypotheses(
            points,
            extent_quantile=extent_quantile,
            support_fraction=support_fraction,
            support_neighbors=support_neighbors,
        )],
        dtype=np.float64,
    )
    guided_count = min(population_size, max(1, int(round(population_size * guided_fraction))))
    exact_count = min(guided_count, anchors.shape[0])
    population[:exact_count] = anchors[:exact_count]
    for index in range(exact_count, guided_count):
        anchor = anchors[(index - exact_count) % anchors.shape[0]]
        population[index] = np.clip(anchor + rng.normal(0.0, jitter, size=lower.size), -1.0, 1.0)

    info = {
        "guided_count": guided_count,
        "random_count": population_size - guided_count,
        "exact_anchor_count": exact_count,
        "hypothesis_count": int(anchors.shape[0]),
        "support_fraction": float(support_fraction),
        "support_neighbors": int(support_neighbors),
    }
    return population.astype(np.float32), info
