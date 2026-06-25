"""Superquadric rule for geometric model fitting.

Action space (11D): center(3) + scale(3) + shape(2) + rotation(3)

Superquadric parametric surface:
  r(η, ω) = [a1 * C(η,ε1) * C(ω,ε2),
              a2 * S(η,ε1) * C(ω,ε2),
              a3 * S(ω,ε2)]
  where C(θ,ε)=sgn(cos θ)·|cos θ|^ε, S(θ,ε)=sgn(sin θ)·|sin θ|^ε
  η∈[-π,π], ω∈[-π/2,π/2], ε1,ε2>0

Reference: Barr, "Superquadrics and Angle-Preserving Transformations", CG&A 1981
"""

import numpy as np
from scipy.spatial.transform import Rotation


class SuperquadricTrait:
    def __init__(self):
        self.center = None       # (3,)  np.float32
        self.scale = None        # (3,)  np.float32 — a1, a2, a3
        self.shape = None        # (2,)  np.float32 — ε1, ε2
        self.rotation = None     # (3,)  np.float32 — Euler angles (rx, ry, rz)
        self.rot_matrix = None   # (3,3) np.float32


class SuperquadricRule:
    """Superquadric: 11D action → point cloud.

    ``parse(action)`` maps [-1,1]¹¹ to [center(3), scale(3), shape(2), rotation(3)].
    ``generate()`` samples the parametric surface.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator
        self.trait = SuperquadricTrait()
        self.action = None
        self._initialized = False

    def _init_bounds(self):
        if self._initialized:
            return
        pts = self.estimator.get_data()
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        extent = hi - lo
        data_scale = float(np.linalg.norm(extent))

        # --- center ---
        padding = 0.2 * extent
        self.lb = np.zeros(11, dtype=np.float32)
        self.ub = np.zeros(11, dtype=np.float32)
        self.lb[0:3] = lo - padding
        self.ub[0:3] = hi + padding

        # --- scale ---
        self.lb[3:6] = 0.02 * data_scale
        self.ub[3:6] = 1.5 * data_scale

        # --- shape (ε1, ε2) ---
        self.lb[6:8] = 0.1
        self.ub[6:8] = 2.5

        # --- rotation (Euler angles) ---
        self.lb[8:11] = -np.pi
        self.ub[8:11] = np.pi

        self._initialized = True

    def get_num_variables(self):
        return 11

    def parse(self, **kwargs):
        self._init_bounds()
        action = kwargs["action"]
        if action.size != 11:
            raise ValueError(f"SuperquadricRule expects 11 variables, got {action.size}")
        flat = self._rescale(action).astype(float)

        trait = SuperquadricTrait()
        trait.center = flat[0:3]
        trait.scale = flat[3:6]
        trait.shape = flat[6:8]
        euler = flat[8:11]
        trait.rotation = euler
        trait.rot_matrix = Rotation.from_euler('xyz', euler).as_matrix().astype(np.float32)

        self.trait = trait
        self.action = action
        return trait

    def _rescale(self, action):
        return self.lb + (self.ub - self.lb) * (np.clip(action, -1.0, 1.0) + 1.0) / 2.0

    @staticmethod
    def _spherical_product(a1, a2, a3, e1, e2, n_eta=80, n_omega=80):
        """Generate superquadric surface points via spherical product.

        Returns points (N, 3) and approximate measure.
        """
        eta = np.linspace(-np.pi, np.pi, n_eta, dtype=np.float32)
        omega = np.linspace(-np.pi / 2, np.pi / 2, n_omega, dtype=np.float32)
        eta_g, omega_g = np.meshgrid(eta, omega, indexing='ij')  # (n_eta, n_omega)

        cos_e = np.cos(eta_g)
        sin_e = np.sin(eta_g)
        cos_o = np.cos(omega_g)
        sin_o = np.sin(omega_g)

        def _spow(val, exp):
            return np.sign(val) * (np.abs(val) ** exp)

        c1 = _spow(cos_e, e1) * _spow(cos_o, e2)
        c2 = _spow(sin_e, e1) * _spow(cos_o, e2)
        c3 = _spow(sin_o, e2)

        x = a1 * c1
        y = a2 * c2
        z = a3 * c3

        pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Approximate measure: sum triangle areas
        p00 = np.stack([x[:-1, :-1], y[:-1, :-1], z[:-1, :-1]], axis=-1)
        p10 = np.stack([x[1:, :-1], y[1:, :-1], z[1:, :-1]], axis=-1)
        p01 = np.stack([x[:-1, 1:], y[:-1, 1:], z[:-1, 1:]], axis=-1)
        p11 = np.stack([x[1:, 1:], y[1:, 1:], z[1:, 1:]], axis=-1)

        area1 = 0.5 * np.linalg.norm(np.cross(p10 - p00, p01 - p00, axis=-1), axis=-1).sum()
        area2 = 0.5 * np.linalg.norm(np.cross(p11 - p10, p01 - p10, axis=-1), axis=-1).sum()
        measure = float(area1 + area2)

        return pts, measure

    @staticmethod
    def measure(trait):
        """Approximate surface area via spherical product sampling."""
        pts, m = SuperquadricRule._spherical_product(
            trait.scale[0], trait.scale[1], trait.scale[2],
            trait.shape[0], trait.shape[1],
            n_eta=60, n_omega=60,
        )
        return m

    def sample(self):
        """Generate superquadric surface points."""
        t = self.trait
        pts, _ = self._spherical_product(
            t.scale[0], t.scale[1], t.scale[2],
            t.shape[0], t.shape[1],
            n_eta=64, n_omega=64,
        )
        pts = pts @ t.rot_matrix.T + t.center
        return pts.astype(np.float32)

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
