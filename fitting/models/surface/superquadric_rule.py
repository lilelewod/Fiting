"""
Superquadric rule for geometric model fitting.

Action space 11D:
    center(3) + scale(3) + shape(2) + rotation(3)

This rule generates a closed superquadric surface.

Parameterization used here:
    x = a1 * C(eta, e1) * C(omega, e2)
    y = a2 * S(eta, e1) * C(omega, e2)
    z = a3 * S(omega, e2)

where:
    C(t, e) = sign(cos(t)) * |cos(t)|^e
    S(t, e) = sign(sin(t)) * |sin(t)|^e

    eta   ∈ [-pi, pi)
    omega ∈ (-pi/2, pi/2)

Meaning:
    a1, a2, a3 : half-size / scale parameters
    e1         : controls shape in x-y section
    e2         : controls vertical profile, cylinder-like when small

Reference:
    Barr, "Superquadrics and Angle-Preserving Transformations", IEEE CG&A, 1981
"""

import numpy as np
from scipy.spatial.transform import Rotation


class SuperquadricTrait:
    def __init__(self):
        self.center = None       # (3,)
        self.scale = None        # (3,)  a1, a2, a3
        self.shape = None        # (2,)  e1, e2
        self.rotation = None     # (3,)  Euler angles rx, ry, rz
        self.rot_matrix = None   # (3,3)


class SuperquadricRule:
    """
    Superquadric: 11D action -> point cloud.

    parse(action):
        maps action in [-1, 1]^11 to actual geometric parameters.

    sample():
        returns sampled closed superquadric points.

    sample_with_weights():
        returns sampled points and local surface-area weights.
        This is useful for improving MM/PMF estimator.
    """

    def __init__(
        self,
        estimator=None,
        n_eta=None,
        n_omega=None,
        shape_min=0.10,
        shape_max=2.50,
        pole_eps=1e-4,
    ):
        self.estimator = estimator
        self.trait = SuperquadricTrait()
        self.action = None

        model_cfg = {} if estimator is None else estimator.cfg.get("model", {})
        self.n_eta = int(model_cfg.get("sample_eta", 96) if n_eta is None else n_eta)
        self.n_omega = int(model_cfg.get("sample_omega", 96) if n_omega is None else n_omega)
        # Ablation switch: keep the same angular samples and geometric surface
        # measure, but optionally replace local-area quadrature weights with a
        # uniform mean in the model-to-data distance.
        self.use_area_weights = bool(model_cfg.get("use_area_weights", True))
        if self.n_eta < 8 or self.n_omega < 5:
            raise ValueError("superquadric sampling requires sample_eta >= 8 and sample_omega >= 5")
        self.shape_min = float(shape_min)
        self.shape_max = float(shape_max)
        self.pole_eps = float(pole_eps)

        self.lb = None
        self.ub = None
        self._initialized = False

    # ------------------------------------------------------------
    # Basic interface
    # ------------------------------------------------------------
    def get_num_variables(self):
        return 11

    def _init_bounds(self):
        if self._initialized:
            return

        if self.estimator is None:
            raise ValueError("SuperquadricRule requires estimator to initialize bounds.")

        pts = self.estimator.get_data()
        pts = np.asarray(pts, dtype=np.float32)

        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        extent = hi - lo
        diag = float(np.linalg.norm(extent))

        if diag <= 1e-8:
            diag = 1.0

        self.lb = np.zeros(11, dtype=np.float32)
        self.ub = np.zeros(11, dtype=np.float32)

        # center bounds
        padding = 0.2 * extent
        self.lb[0:3] = lo - padding
        self.ub[0:3] = hi + padding

        # scale bounds: a1, a2, a3 are half-size parameters
        self.lb[3:6] = 0.02 * diag
        self.ub[3:6] = 1.20 * diag

        # shape bounds: e1, e2
        self.lb[6:8] = self.shape_min
        self.ub[6:8] = self.shape_max

        # Euler rotation bounds
        self.lb[8:11] = -np.pi
        self.ub[8:11] = np.pi

        self._initialized = True

    def _rescale(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)
        return self.lb + (self.ub - self.lb) * (action + 1.0) / 2.0

    def parse(self, **kwargs):
        self._init_bounds()

        action = np.asarray(kwargs["action"], dtype=np.float32).reshape(-1)

        if action.size != 11:
            raise ValueError(
                f"SuperquadricRule expects 11 variables, got {action.size}"
            )

        flat = self._rescale(action).astype(np.float32)

        trait = SuperquadricTrait()
        trait.center = flat[0:3].astype(np.float32)
        trait.scale = flat[3:6].astype(np.float32)
        trait.shape = flat[6:8].astype(np.float32)

        euler = flat[8:11].astype(np.float32)
        trait.rotation = euler
        trait.rot_matrix = Rotation.from_euler("xyz", euler).as_matrix().astype(np.float32)

        self.trait = trait
        self.action = action.astype(np.float32)

        return trait

    # ------------------------------------------------------------
    # Superquadric geometry
    # ------------------------------------------------------------
    @staticmethod
    def _signed_power(v, e):
        """
        Stable signed power:
            sign(v) * |v|^e
        """
        v = np.asarray(v, dtype=np.float32)
        return np.sign(v) * (np.abs(v) ** e)

    @classmethod
    def _local_surface_grid(
        cls,
        a1,
        a2,
        a3,
        e1,
        e2,
        n_eta=96,
        n_omega=96,
        pole_eps=1e-4,
    ):
        """
        Generate local superquadric grid before rotation and translation.

        Returns:
            grid: (n_eta, n_omega, 3)
        """

        # eta is periodic. endpoint=False avoids duplicated seam.
        eta = np.linspace(
            -np.pi,
            np.pi,
            n_eta,
            endpoint=False,
            dtype=np.float32,
        )

        # Avoid exact poles to prevent many duplicated points at top/bottom.
        omega = np.linspace(
            -np.pi / 2.0 + pole_eps,
            np.pi / 2.0 - pole_eps,
            n_omega,
            endpoint=True,
            dtype=np.float32,
        )

        eta_g, omega_g = np.meshgrid(eta, omega, indexing="ij")

        c_eta = cls._signed_power(np.cos(eta_g), e1)
        s_eta = cls._signed_power(np.sin(eta_g), e1)
        c_omega = cls._signed_power(np.cos(omega_g), e2)
        s_omega = cls._signed_power(np.sin(omega_g), e2)

        x = a1 * c_eta * c_omega
        y = a2 * s_eta * c_omega
        z = a3 * s_omega

        grid = np.stack([x, y, z], axis=-1).astype(np.float32)
        return grid

    @staticmethod
    def _triangle_area(p0, p1, p2):
        return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0, axis=-1), axis=-1)

    @classmethod
    def _vertex_area_weights(cls, grid):
        """
        Approximate local area weight for each vertex.

        The eta direction is periodic, so the last eta row connects to the first.
        The omega direction is not periodic.

        Returns:
            weights: (n_eta, n_omega)
            total_area: float
        """

        n_eta, n_omega, _ = grid.shape
        weights = np.zeros((n_eta, n_omega), dtype=np.float32)

        # Vectorized periodic quad mesh. Each quad is split into two triangles,
        # and one third of each triangle's area is assigned to every vertex.
        next_eta = np.roll(grid, -1, axis=0)
        p00 = grid[:, :-1]
        p10 = next_eta[:, :-1]
        p01 = grid[:, 1:]
        p11 = next_eta[:, 1:]
        share1 = cls._triangle_area(p00, p10, p01) / 3.0
        share2 = cls._triangle_area(p10, p11, p01) / 3.0

        weights[:, :-1] += share1
        weights[:, :-1] += np.roll(share1, 1, axis=0)
        weights[:, 1:] += share1
        weights[:, :-1] += np.roll(share2, 1, axis=0)
        weights[:, 1:] += np.roll(share2, 1, axis=0)
        weights[:, 1:] += share2

        total_area = float(weights.sum())
        return weights, total_area

    @classmethod
    def _surface_triangles(cls, grid):
        """Convert a periodic superquadric parameter grid to surface triangles."""
        grid = np.asarray(grid, dtype=np.float64)
        if grid.ndim != 3 or grid.shape[2] != 3:
            raise ValueError("grid must have shape (n_eta, n_omega, 3)")
        if grid.shape[0] < 3 or grid.shape[1] < 2:
            raise ValueError("grid is too small to form a closed surface mesh")

        next_eta = np.roll(grid, -1, axis=0)
        p00 = grid[:, :-1]
        p10 = next_eta[:, :-1]
        p01 = grid[:, 1:]
        p11 = next_eta[:, 1:]
        triangles = np.concatenate(
            (
                np.stack((p00, p10, p01), axis=-2),
                np.stack((p10, p11, p01), axis=-2),
            ),
            axis=0,
        )
        return triangles.reshape(-1, 3, 3)

    @classmethod
    def sample_surface_uniform(
        cls,
        trait,
        count,
        seed=0,
        n_eta=256,
        n_omega=256,
        pole_eps=1e-4,
    ):
        """Sample a posed superquadric approximately uniformly by surface area.

        A high-resolution deterministic parameter grid is triangulated first.
        Triangles are selected in proportion to their physical area, followed
        by uniform barycentric sampling inside the selected triangles.  This is
        intended for dataset generation, visualization and external metrics;
        optimization should keep using ``sample_with_weights``.
        """
        count = int(count)
        n_eta = int(n_eta)
        n_omega = int(n_omega)
        if count <= 0:
            raise ValueError("count must be positive")
        if n_eta < 8 or n_omega < 5:
            raise ValueError("uniform surface sampling requires n_eta >= 8 and n_omega >= 5")

        scale = np.asarray(trait.scale, dtype=np.float64).reshape(3)
        shape = np.asarray(trait.shape, dtype=np.float64).reshape(2)
        center = np.asarray(trait.center, dtype=np.float64).reshape(3)
        rotation = np.asarray(trait.rot_matrix, dtype=np.float64).reshape(3, 3)
        if not all(np.all(np.isfinite(value)) for value in (scale, shape, center, rotation)):
            raise ValueError("trait contains non-finite values")
        if np.any(scale <= 0.0) or np.any(shape <= 0.0):
            raise ValueError("trait scale and shape parameters must be positive")

        # Evaluation/data generation uses float64 independently of the float32
        # optimization grid so the standalone dataset generator and fitting
        # project produce byte-identical samples under the same protocol.
        eta = np.linspace(-np.pi, np.pi, n_eta, endpoint=False, dtype=np.float64)
        omega = np.linspace(
            -np.pi / 2.0 + pole_eps,
            np.pi / 2.0 - pole_eps,
            n_omega,
            endpoint=True,
            dtype=np.float64,
        )
        eta_g, omega_g = np.meshgrid(eta, omega, indexing="ij")
        signed_power = lambda values, exponent: np.sign(values) * np.abs(values) ** exponent
        c_eta = signed_power(np.cos(eta_g), shape[0])
        s_eta = signed_power(np.sin(eta_g), shape[0])
        c_omega = signed_power(np.cos(omega_g), shape[1])
        s_omega = signed_power(np.sin(omega_g), shape[1])
        local_grid = np.stack(
            (
                scale[0] * c_eta * c_omega,
                scale[1] * s_eta * c_omega,
                scale[2] * s_omega,
            ),
            axis=-1,
        )
        posed_grid = np.asarray(local_grid, dtype=np.float64) @ rotation.T + center
        triangles = cls._surface_triangles(posed_grid)
        areas = cls._triangle_area(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        valid = np.isfinite(areas) & (areas > np.finfo(np.float64).eps)
        if not np.any(valid):
            raise ValueError("superquadric mesh contains no positive-area triangles")
        triangles = triangles[valid]
        areas = np.asarray(areas[valid], dtype=np.float64)
        probabilities = areas / areas.sum()

        rng = np.random.default_rng(seed)
        chosen = rng.choice(triangles.shape[0], size=count, replace=True, p=probabilities)
        selected = triangles[chosen]
        root_u = np.sqrt(rng.random(count))
        v = rng.random(count)
        points = (
            (1.0 - root_u)[:, None] * selected[:, 0]
            + (root_u * (1.0 - v))[:, None] * selected[:, 1]
            + (root_u * v)[:, None] * selected[:, 2]
        )
        return points.astype(np.float32)

    @classmethod
    def _spherical_product(
        cls,
        a1,
        a2,
        a3,
        e1,
        e2,
        n_eta=96,
        n_omega=96,
        pole_eps=1e-4,
        return_weights=False,
    ):
        """
        Generate local superquadric surface points.

        Returns:
            if return_weights=False:
                points, measure

            if return_weights=True:
                points, weights, measure
        """

        grid = cls._local_surface_grid(
            a1=a1,
            a2=a2,
            a3=a3,
            e1=e1,
            e2=e2,
            n_eta=n_eta,
            n_omega=n_omega,
            pole_eps=pole_eps,
        )

        area_weights, measure = cls._vertex_area_weights(grid)

        pts = grid.reshape(-1, 3).astype(np.float32)
        weights = area_weights.reshape(-1).astype(np.float32)

        # Avoid zero weights caused by numerical degeneracy
        weights = np.maximum(weights, 1e-12).astype(np.float32)

        if return_weights:
            return pts, weights, measure

        return pts, measure

    # ------------------------------------------------------------
    # Public sampling and measure
    # ------------------------------------------------------------
    @staticmethod
    def measure(trait, n_eta=96, n_omega=96, pole_eps=1e-4):
        """
        Approximate surface area of the superquadric.
        Rotation and translation do not change area.
        """
        _, m = SuperquadricRule._spherical_product(
            trait.scale[0],
            trait.scale[1],
            trait.scale[2],
            trait.shape[0],
            trait.shape[1],
            n_eta=n_eta,
            n_omega=n_omega,
            pole_eps=pole_eps,
            return_weights=False,
        )
        return float(m)

    def sample_with_weights(self):
        """
        Generate transformed superquadric points and local area weights.

        Returns:
            pts:     (N, 3)
            weights: (N,)
            measure: float
        """

        t = self.trait

        pts, weights, measure = self._spherical_product(
            t.scale[0],
            t.scale[1],
            t.scale[2],
            t.shape[0],
            t.shape[1],
            n_eta=self.n_eta,
            n_omega=self.n_omega,
            pole_eps=self.pole_eps,
            return_weights=True,
        )

        pts = pts @ t.rot_matrix.T + t.center
        pts = pts.astype(np.float32)

        return pts, weights.astype(np.float32), float(measure)

    def sample(self):
        """
        Generate transformed superquadric surface points.
        """
        pts, _, _ = self.sample_with_weights()
        return pts.astype(np.float32)

    def generate(self):
        """
        Generate token and add it to estimator.

        If your current MM estimator does not use token.weights,
        this field will simply be ignored.
        Later, when you modify MM, you can use token.weights for
        area-weighted model-to-data distance.
        """

        from models.rule import Token

        cloud, weights, measure = self.sample_with_weights()

        token = Token(self.estimator.dimension)
        token.points = cloud
        token.trait = self.trait
        token.measure = measure
        token.action = self.action

        # Optional field for improved MM estimator
        token.weights = weights if self.use_area_weights else None

        self.estimator.add_token(token)

        return cloud
