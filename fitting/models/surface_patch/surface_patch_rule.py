import math

import numpy as np
from easydict import EasyDict

from ..rule import ModelRule
from tools.tool import rescale


def _normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def _build_tangent_frame(normal, psi):
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(normal, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    axis_u = _normalize(np.cross(ref, normal))
    axis_v = _normalize(np.cross(normal, axis_u))

    cos_psi = math.cos(psi)
    sin_psi = math.sin(psi)
    rotated_u = cos_psi * axis_u + sin_psi * axis_v
    rotated_v = -sin_psi * axis_u + cos_psi * axis_v
    return _normalize(rotated_u), _normalize(rotated_v)


class SurfacePatchTrait(EasyDict):
    def __init__(self):
        super().__init__()
        self.center_local = np.zeros(3, dtype=np.float32)
        self.theta = 0.0
        self.phi = 0.0
        self.psi = 0.0
        self.extent_u = 0.0
        self.extent_v = 0.0
        self.curvature_uu = 0.0
        self.curvature_vv = 0.0
        self.curvature_uv = 0.0


class SurfacePatchRule(ModelRule):
    def __init__(self, estimator=None):
        super().__init__(estimator)
        self.lower_bound = None
        self.upper_bound = None
        self.action = None
        self.trait = None

        self.data_mean = None
        self.pca_basis = None
        self.local_min = None
        self.local_max = None
        self.max_span = None

        self._prepare_reference_frame()
        self.set_trait_range()

    def _prepare_reference_frame(self):
        data = np.asarray(self.estimator.raw_data, dtype=np.float32)
        self.data_mean = data.mean(axis=0)
        centered = data - self.data_mean
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        self.pca_basis = eigenvectors[:, order]
        local_points = centered @ self.pca_basis
        self.local_min = local_points.min(axis=0)
        self.local_max = local_points.max(axis=0)
        self.max_span = float(np.linalg.norm(self.local_max - self.local_min))

    def set_trait_range(self):
        min_extent = max(2.0 * self.estimator.data_resolution, 1e-3)
        max_extent = max(float(np.max(self.local_max - self.local_min)), min_extent * 2.0)
        beyond = np.maximum((self.local_max - self.local_min) / 10.0, self.estimator.data_resolution)
        curvature_limit = 1.5 / max(self.max_span, self.estimator.data_resolution)

        self.lower_bound = np.asarray(
            [
                self.local_min[0] - beyond[0],
                self.local_min[1] - beyond[1],
                self.local_min[2] - beyond[2],
                0.0,
                0.0,
                0.0,
                min_extent,
                min_extent,
                -curvature_limit,
                -curvature_limit,
                -curvature_limit,
            ],
            dtype=np.float32,
        )
        self.upper_bound = np.asarray(
            [
                self.local_max[0] + beyond[0],
                self.local_max[1] + beyond[1],
                self.local_max[2] + beyond[2],
                math.pi,
                2.0 * math.pi,
                2.0 * math.pi,
                max_extent,
                max_extent,
                curvature_limit,
                curvature_limit,
                curvature_limit,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def measure(trait):
        return float(trait.extent_u * trait.extent_v)

    def compute_top_dividing_level(self):
        resolution = max(self.estimator.resolution, 1e-4)
        level_u = math.floor(math.log2(1.0 + self.trait.extent_u / resolution))
        level_v = math.floor(math.log2(1.0 + self.trait.extent_v / resolution))
        self.top_level = np.asarray([max(2, level_u), max(2, level_v)], dtype=np.int64)

    def get_num_variables(self):
        return int(self.lower_bound.size)

    def parse(self, **kwargs):
        action = kwargs['action']
        assert action.size == self.get_num_variables()
        trait_flat = rescale(action, self.lower_bound, self.upper_bound)

        trait = SurfacePatchTrait()
        trait.center_local = np.asarray(trait_flat[0:3], dtype=np.float32)
        trait.theta = float(trait_flat[3])
        trait.phi = float(trait_flat[4])
        trait.psi = float(trait_flat[5])
        trait.extent_u = float(trait_flat[6])
        trait.extent_v = float(trait_flat[7])
        trait.curvature_uu = float(trait_flat[8])
        trait.curvature_vv = float(trait_flat[9])
        trait.curvature_uv = float(trait_flat[10])

        self.action = action
        self.trait = trait
        self.compute_top_dividing_level()
        return trait

    def _sample_local_surface(self):
        level_u, level_v = self.compute_current_dividing_level()
        nu = min(24, max(6, int(2 ** int(level_u)) + 1))
        nv = min(24, max(6, int(2 ** int(level_v)) + 1))

        u = np.linspace(-0.5 * self.trait.extent_u, 0.5 * self.trait.extent_u, nu, dtype=np.float32)
        v = np.linspace(-0.5 * self.trait.extent_v, 0.5 * self.trait.extent_v, nv, dtype=np.float32)
        uu, vv = np.meshgrid(u, v, indexing='xy')
        ww = (
            self.trait.curvature_uu * (uu ** 2)
            + self.trait.curvature_vv * (vv ** 2)
            + self.trait.curvature_uv * uu * vv
        )
        return uu.reshape(-1), vv.reshape(-1), ww.reshape(-1)

    def generate(self):
        uu, vv, ww = self._sample_local_surface()

        normal_local = np.array(
            [
                math.sin(self.trait.theta) * math.cos(self.trait.phi),
                math.sin(self.trait.theta) * math.sin(self.trait.phi),
                math.cos(self.trait.theta),
            ],
            dtype=np.float32,
        )
        normal_local = _normalize(normal_local)
        axis_u_local, axis_v_local = _build_tangent_frame(normal_local, self.trait.psi)

        local_points = (
            self.trait.center_local[None, :]
            + uu[:, None] * axis_u_local[None, :]
            + vv[:, None] * axis_v_local[None, :]
            + ww[:, None] * normal_local[None, :]
        )
        global_points = self.data_mean[None, :] + local_points @ self.pca_basis.T

        self.estimator.add_model(
            new_measure=self.measure(self.trait),
            new_model=np.ascontiguousarray(global_points, dtype=np.float32),
        )
        return global_points
