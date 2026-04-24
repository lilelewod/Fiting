from pathlib import Path
import math
import sys

import numpy as np
from easydict import EasyDict

from ..rule import ModelRule, Token
from tools.tool import rescale


def _open_uniform_knot_vector(num_ctrl_pts, degree):
    n = num_ctrl_pts - 1
    interior_count = n - degree
    knots = [0.0] * (degree + 1)
    if interior_count > 0:
        step = 1.0 / float(interior_count + 1)
        knots.extend(step * i for i in range(1, interior_count + 1))
    knots.extend([1.0] * (degree + 1))
    return np.asarray(knots, dtype=np.float32)


def _basis_functions(u_values, num_ctrl_pts, degree, knots):
    basis = np.zeros((u_values.size, num_ctrl_pts), dtype=np.float32)
    last_knot = knots[-1]

    for row, u in enumerate(u_values):
        if np.isclose(u, knots[0]):
            basis[row, 0] = 1.0
            continue
        if np.isclose(u, last_knot):
            basis[row, -1] = 1.0
            continue

        for i in range(num_ctrl_pts):
            if knots[i] <= u < knots[i + 1]:
                basis[row, i] = 1.0

        for k in range(1, degree + 1):
            next_basis = np.zeros(num_ctrl_pts, dtype=np.float32)
            for i in range(num_ctrl_pts):
                left_denom = knots[i + k] - knots[i]
                if left_denom > 0:
                    next_basis[i] += ((u - knots[i]) / left_denom) * basis[row, i]

                right_index = i + 1
                if right_index < num_ctrl_pts:
                    right_denom = knots[i + k + 1] - knots[right_index]
                    if right_denom > 0:
                        next_basis[i] += (
                            (knots[i + k + 1] - u) / right_denom
                        ) * basis[row, right_index]
            basis[row] = next_basis

    return basis


def _try_import_nurbsdiff():
    candidate_roots = []
    current_root = Path(__file__).resolve().parents[3]
    candidate_roots.append(current_root / "code" / "NURBSDiff")
    candidate_roots.append(current_root.parent / "NURBSDiff")

    for root in candidate_roots:
        package_root = root / "NURBSDiff"
        if package_root.exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            try:
                from NURBSDiff.surf_eval import SurfEval

                return SurfEval
            except Exception:
                continue
    return None


class NURBSSurfaceTrait(EasyDict):
    def __init__(self):
        EasyDict.__init__(self)
        self.control_points = None
        self.weights = None


class NURBSSurfaceRule(ModelRule):
    name = "nurbs_surface"

    def __init__(self, estimator=None):
        ModelRule.__init__(self, estimator)
        self.cfg = estimator.cfg.get("model", {})
        self.num_ctrl_u = int(self.cfg.get("num_ctrl_u", 4))
        self.num_ctrl_v = int(self.cfg.get("num_ctrl_v", 4))
        self.degree_u = int(self.cfg.get("degree_u", 3))
        self.degree_v = int(self.cfg.get("degree_v", 3))
        if self.degree_u >= self.num_ctrl_u or self.degree_v >= self.num_ctrl_v:
            raise ValueError("NURBS degree must be smaller than the number of control points.")

        self.final_samples_u = int(self.cfg.get("sample_u", max(17, self.num_ctrl_u * 4 + 1)))
        self.final_samples_v = int(self.cfg.get("sample_v", max(17, self.num_ctrl_v * 4 + 1)))
        self.weight_lb = float(self.cfg.get("weight_lb", 0.2))
        self.weight_ub = float(self.cfg.get("weight_ub", 3.0))

        self.trait = None
        self.action = None
        self.lb = None
        self.ub = None
        self.knot_u = _open_uniform_knot_vector(self.num_ctrl_u, self.degree_u)
        self.knot_v = _open_uniform_knot_vector(self.num_ctrl_v, self.degree_v)
        self._surf_eval_cls = _try_import_nurbsdiff()
        self._surf_eval = None
        self.set_trait_range()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Torch modules/tensors inside the evaluator are rebuilt lazily after unpickling.
        state["_surf_eval"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._surf_eval = None

    def set_trait_range(self):
        num_ctrl = self.num_ctrl_u * self.num_ctrl_v
        dims = self.estimator.dimension
        xyz_dims = num_ctrl * dims
        num_vars = xyz_dims + num_ctrl

        self.lb = np.empty(num_vars, dtype=np.float32)
        self.ub = np.empty(num_vars, dtype=np.float32)

        data_min = np.asarray(self.estimator.min_point, dtype=np.float32)
        data_max = np.asarray(self.estimator.max_point, dtype=np.float32)
        extent = np.maximum(data_max - data_min, self.estimator.resolution)
        padding = 0.25 * np.maximum(extent, self.estimator.resolution)

        xyz_lb = data_min - padding
        xyz_ub = data_max + padding

        for index in range(num_ctrl):
            start = index * dims
            end = start + dims
            self.lb[start:end] = xyz_lb
            self.ub[start:end] = xyz_ub

        self.lb[xyz_dims:] = self.weight_lb
        self.ub[xyz_dims:] = self.weight_ub

    @staticmethod
    def _top_level_from_samples(sample_count):
        return max(0, int(math.ceil(math.log2(max(2, sample_count - 1)))) - 1)

    def compute_top_dividing_level(self):
        self.top_level = np.asarray(
            [
                self._top_level_from_samples(self.final_samples_u),
                self._top_level_from_samples(self.final_samples_v),
            ],
            dtype=np.int64,
        )

    @staticmethod
    def _measure(points):
        if points.shape[0] < 4:
            return float(points.shape[0])

        area = 0.0
        p00 = points[:-1, :-1]
        p10 = points[1:, :-1]
        p01 = points[:-1, 1:]
        p11 = points[1:, 1:]
        area += np.linalg.norm(np.cross(p10 - p00, p01 - p00), axis=-1).sum() * 0.5
        area += np.linalg.norm(np.cross(p11 - p10, p01 - p10), axis=-1).sum() * 0.5
        return float(area)

    def _sample_counts(self):
        level = self.compute_current_dividing_level().astype(np.int64)
        sample_u = min(self.final_samples_u, max(3, 2 ** (int(level[0]) + 1) + 1))
        sample_v = min(self.final_samples_v, max(3, 2 ** (int(level[1]) + 1) + 1))
        return sample_u, sample_v

    def _sample_with_numpy(self, sample_u, sample_v):
        u_values = np.linspace(0.0, 1.0, sample_u, dtype=np.float32)
        v_values = np.linspace(0.0, 1.0, sample_v, dtype=np.float32)
        basis_u = _basis_functions(u_values, self.num_ctrl_u, self.degree_u, self.knot_u)
        basis_v = _basis_functions(v_values, self.num_ctrl_v, self.degree_v, self.knot_v)

        weighted_ctrl = self.trait.control_points * self.trait.weights[..., None]
        numerators = np.einsum("ui,vj,ijd->uvd", basis_u, basis_v, weighted_ctrl)
        denominators = np.einsum("ui,vj,ij->uv", basis_u, basis_v, self.trait.weights)
        denominators = np.clip(denominators[..., None], 1e-8, None)
        return numerators / denominators

    def _get_surf_eval(self, sample_u, sample_v):
        if self._surf_eval_cls is None:
            return None

        cache_key = (sample_u, sample_v)
        if self._surf_eval is None or getattr(self._surf_eval, "_cache_key", None) != cache_key:
            import torch

            self._surf_eval = self._surf_eval_cls(
                m=self.num_ctrl_u - 1,
                n=self.num_ctrl_v - 1,
                dimension=self.estimator.dimension,
                p=self.degree_u,
                q=self.degree_v,
                knot_u=self.knot_u,
                knot_v=self.knot_v,
                out_dim_u=sample_u,
                out_dim_v=sample_v,
                device="cpu",
            )
            self._surf_eval._cache_key = cache_key
        return self._surf_eval

    def _sample_with_nurbsdiff(self, sample_u, sample_v):
        surf_eval = self._get_surf_eval(sample_u, sample_v)
        if surf_eval is None:
            return None

        import torch

        ctrl_pts = np.concatenate(
            [self.trait.control_points * self.trait.weights[..., None], self.trait.weights[..., None]],
            axis=-1,
        )
        ctrl_pts = torch.as_tensor(ctrl_pts[None, ...], dtype=torch.float32)
        surface = surf_eval(ctrl_pts).detach().cpu().numpy()[0]
        return surface

    def sample(self):
        sample_u, sample_v = self._sample_counts()
        surface = self._sample_with_nurbsdiff(sample_u, sample_v)
        if surface is None:
            surface = self._sample_with_numpy(sample_u, sample_v)
        return surface.reshape(-1, self.estimator.dimension), surface

    def generate(self):
        points_flat, points_grid = self.sample()
        token = Token(self.estimator.dimension)
        token.points = points_flat
        token.trait = self.trait
        token.measure = self._measure(points_grid)
        token.action = self.action
        self.estimator.add_token(token)
        return points_flat

    def get_num_variables(self):
        return int(self.lb.size)

    def parse(self, **kwargs):
        action = kwargs["action"]
        if action.size != self.get_num_variables():
            raise ValueError(
                f"Expected {self.get_num_variables()} variables for NURBS surface, got {action.size}."
            )

        trait_flat = rescale(action, self.lb, self.ub)
        dims = self.estimator.dimension
        num_ctrl = self.num_ctrl_u * self.num_ctrl_v
        xyz_size = num_ctrl * dims

        trait = NURBSSurfaceTrait()
        trait.control_points = trait_flat[:xyz_size].reshape(self.num_ctrl_u, self.num_ctrl_v, dims)
        trait.weights = trait_flat[xyz_size:].reshape(self.num_ctrl_u, self.num_ctrl_v)

        self.trait = trait
        self.action = action
        self.compute_top_dividing_level()
        return trait
