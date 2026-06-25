"""Sphere/Hemisphere rule for geometric model fitting.

Action space (4D): center_x, center_y, center_z, radius
"""

import numpy as np

from tools.tool import rescale


class SphereTrait:
    def __init__(self):
        self.center = None
        self.radius = None


class SphereRule:
    """Sphere/Hemisphere: 4D action → hemisphere point cloud.

    ``parse(action)`` maps [-1,1]⁴ to [center(3), radius(1)].
    ``generate()`` samples hemisphere surface (z >= center_z).
    """

    def __init__(self, estimator=None):
        self.estimator = estimator
        self.trait = SphereTrait()
        self.action = None
        self._initialized = False

    def _init_bounds(self):
        if self._initialized:
            return
        pts = self.estimator.get_data()
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        extent = hi - lo
        padding = 0.2 * extent

        self.lb = np.zeros(4, dtype=np.float32)
        self.ub = np.zeros(4, dtype=np.float32)
        self.lb[0:3] = lo - padding
        self.ub[0:3] = hi + padding
        self.lb[3] = 0.1 * max(extent)
        self.ub[3] = 1.5 * max(extent)
        self._initialized = True

    def get_num_variables(self):
        return 4

    def parse(self, **kwargs):
        self._init_bounds()
        action = kwargs["action"]
        if action.size != 4:
            raise ValueError(f"SphereRule expects 4 variables, got {action.size}")
        flat = rescale(action, self.lb, self.ub).astype(float)

        trait = SphereTrait()
        trait.center = flat[0:3]
        trait.radius = float(flat[3])

        self.trait = trait
        self.action = action
        return trait

    @staticmethod
    def measure(trait):
        """Hemisphere surface area: 2πr²"""
        return 2.0 * np.pi * (trait.radius ** 2)

    def sample(self):
        """Deterministic hemisphere surface sampling (Fibonacci sphere)."""
        r = self.trait.radius
        c = self.trait.center
        n = 3000

        # Fibonacci sphere — deterministic, uniform area distribution
        i = np.arange(n)
        phi = np.arccos(1 - 2 * (i + 0.5) / n)  # polar angle [0, π]
        # Keep only top hemisphere: phi <= π/2 → cos(phi) >= 0
        mask = phi <= np.pi / 2
        phi = phi[mask]
        i_kept = i[mask]
        theta = np.pi * (1 + np.sqrt(5)) * i_kept  # golden angle

        pts = np.zeros((len(phi), 3), dtype=np.float32)
        pts[:, 0] = r * np.sin(phi) * np.cos(theta)
        pts[:, 1] = r * np.sin(phi) * np.sin(theta)
        pts[:, 2] = r * np.cos(phi)
        pts += c
        return pts

    def generate(self):
        from models.rule import Token

        cloud = self.sample()
        token = Token(self.estimator.dimension)
        token.points = cloud
        token.trait = self.trait
        token.measure = self.measure(self.trait)
        token.action = self.action
        self.estimator.add_token(token)
        return cloud
