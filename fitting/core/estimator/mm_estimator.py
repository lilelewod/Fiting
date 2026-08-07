from copy import deepcopy
import time

import numpy as np
from sklearn.neighbors import KDTree

from models.rule import Token
from tools.geometry import compute_resolution
from tools.superquadric_initialization import adaptive_density_support, density_support

try:
    import point_cloud_utils as pcu
except ModuleNotFoundError:
    pcu = None

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None

try:
    import faiss
except ModuleNotFoundError:
    faiss = None

try:
    import torch
except ModuleNotFoundError:
    torch = None


class MeanMeasureEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dimension = None
        self.raw_data = None
        self.data = None
        self.num_data_points = None
        self.min_point = None
        self.max_point = None
        self.data_kDTree = None
        self.data_resolution = None
        self.model_resolution = None
        self.resolution = None
        self.load_data()

        self.rule = None
        self.set_rule()

        estimator_cfg = cfg["estimator"]
        self.regularization_factor = estimator_cfg.get("regularization_factor", 0.5)

        self.current_dividing_level = -1
        self.instance_index = 0

        self.model = np.empty((0, self.dimension), dtype=np.float32)
        self.labels = np.empty(0, dtype=np.int64)
        self.sum_errors = 0.0
        self.supporters = np.empty(0, dtype=np.int64)
        self.nearest_points = np.empty(0, dtype=np.int64)
        self.num_points = 0

        self.base_sum_errors = 0.0
        self.base_supporters = np.empty(0, dtype=np.int64)
        self.base_num_points = 0
        self.new_supporters = np.empty(0, dtype=np.int64)
        self.overlap_ratio = 0.0
        self.outlier_ratio = 0.0
        self.bbox_violation_ratio = 0.0
        self.coverage_ratio = 1.0
        self.control_smoothness = 0.0

        self.measure = 0.0
        self.single_model_error = None
        self.score = None
        self.score_mm = 0.0

        # FAISS GPU 加速
        requested_backend = str(estimator_cfg.get("nearest_neighbor_backend", "legacy"))
        if requested_backend not in {"legacy", "sklearn", "faiss", "torch_cuda"}:
            raise ValueError(
                "nearest_neighbor_backend must be legacy, sklearn, faiss, or torch_cuda"
            )
        if requested_backend == "torch_cuda":
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError("torch_cuda nearest-neighbor backend requires CUDA PyTorch")
            self._nearest_neighbor_backend = "torch_cuda"
        elif requested_backend == "faiss":
            if faiss is None:
                raise RuntimeError("faiss nearest-neighbor backend requested but FAISS is unavailable")
            self._nearest_neighbor_backend = "faiss"
        elif requested_backend == "sklearn":
            self._nearest_neighbor_backend = "sklearn"
        else:
            self._nearest_neighbor_backend = (
                "faiss" if faiss is not None and estimator_cfg.get("use_faiss", False) else "sklearn"
            )
        self._use_faiss = self._nearest_neighbor_backend == "faiss"
        self._faiss_index = None
        self._torch_data = None
        self._torch_device = None
        self._torch_cached_data_to_model_errors = None
        self._last_model_to_data_time = 0.0
        self.token = None
        self.model_color = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in {
                "_torch_data",
                "_torch_device",
                "_torch_cached_data_to_model_errors",
                "_faiss_index",
            }:
                setattr(result, k, None)
            elif k == "rule":
                setattr(result, k, deepcopy(v, memo))
                if result.rule is not None:
                    result.rule.estimator = result
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    # ---- data loading ----
    def load_data(self):
        load_data_fn = self.cfg["estimator"]["load_data_fn"]
        data = load_data_fn(self)
        self.raw_data = data.copy()
        estimator_cfg = self.cfg["estimator"]
        support_mode = str(estimator_cfg.get("density_support_mode", "fixed"))
        support_fraction = float(estimator_cfg.get("density_support_fraction", 1.0))
        support_neighbors = int(estimator_cfg.get("density_support_neighbors", 8))
        if support_mode not in {"fixed", "adaptive"}:
            raise ValueError("density_support_mode must be 'fixed' or 'adaptive'")
        if not 0.0 < support_fraction <= 1.0:
            raise ValueError("density_support_fraction must lie in (0, 1]")
        if support_neighbors < 2:
            raise ValueError("density_support_neighbors must be at least 2")
        if support_mode == "adaptive":
            data = adaptive_density_support(data, support_neighbors)
        elif support_fraction < 1.0:
            data = density_support(data, support_fraction, support_neighbors)
        self.density_support_info = {
            "raw_points": int(self.raw_data.shape[0]),
            "support_points": int(data.shape[0]),
            "support_fraction": support_fraction,
            "actual_support_fraction": float(data.shape[0] / self.raw_data.shape[0]),
            "support_neighbors": support_neighbors,
            "support_mode": support_mode,
        }
        self.dimension = data.shape[1]
        if self.data_resolution is None:
            self.preprocess(data)
        else:
            self.create_kdtree(data)

    def preprocess(self, data, synthetic=False):
        assert data.shape[0] > 1
        cfg = self.cfg["estimator"]

        if synthetic:
            self.data_resolution = cfg["synthetic_data_resolution"]
            self.data = data
        elif "voxel_size_for_down_sampling" in cfg:
            if pcu is None:
                raise ModuleNotFoundError(
                    "point_cloud_utils is required for voxel down-sampling."
                )
            self.data_resolution = cfg["voxel_size_for_down_sampling"]
            self.data = pcu.downsample_point_cloud_on_voxel_grid(
                self.data_resolution, data
            )
        elif "data_resolution" in cfg:
            self.data_resolution = cfg["data_resolution"]
            self.data = data
        else:
            self.data_resolution, self.data = compute_resolution(data.copy())

        self.min_point = self.data.min(0)
        self.max_point = self.data.max(0)
        self.data_kDTree = KDTree(self.data)
        if hasattr(self, "_torch_data"):
            self._torch_data = None

        self.model_resolution = cfg.get("model_resolution", 0.45 * self.data_resolution)
        assert self.model_resolution < 0.5 * self.data_resolution
        self.num_data_points = self.data.shape[0]
        self.resolution = self.model_resolution

    def create_kdtree(self, data):
        assert data.shape[0] > 1
        self.data = data
        self.dimension = data.shape[1]
        self.num_data_points = data.shape[0]
        self.data_kDTree = KDTree(data)
        if hasattr(self, "_torch_data"):
            self._torch_data = None
        self.min_point = np.min(data, axis=0)
        self.max_point = np.max(data, axis=0)

    # ---- rule ----
    def set_rule(self):
        rule_class = self.cfg["estimator"]["rule_class"]
        print(f"rule is {rule_class.__name__}")
        assert self.raw_data is not None
        self.rule = rule_class(estimator=self)

    def num_variables(self):
        assert self.rule is not None
        return self.rule.get_num_variables()

    def parse(self, **kwargs):
        return self.rule.parse(**kwargs)

    def generate(self, current_dividing_level=-1):
        self.current_dividing_level = current_dividing_level
        assert self.rule.trait is not None
        self.rule.generate()

    # ---- state management ----
    def reset(self):
        self.sum_errors = deepcopy(self.base_sum_errors)
        self.supporters = deepcopy(self.base_supporters)
        self.nearest_points = deepcopy(self.base_supporters)
        self.num_points = deepcopy(self.base_num_points)
        self.measure = 0.0
        self.model = np.empty((0, self.dimension), dtype=np.float32)
        self.labels = np.empty(0, dtype=np.int64)
        self.token = None
        self.model_color = None
        self.control_smoothness = 0.0
        self.coverage_ratio = 1.0
        self._torch_cached_data_to_model_errors = None

    def update(self, supporters, sum_errors, num_points):
        self.base_sum_errors = deepcopy(sum_errors)
        self.base_supporters = deepcopy(supporters)
        self.base_num_points = int(num_points)

    # ---- accessors ----
    def get_model(self):
        return deepcopy(self.model)

    def get_token(self):
        return deepcopy(self.token)

    def get_data(self):
        return self.raw_data if self.raw_data is not None else self.data

    def get_score(self):
        return deepcopy(self.score)

    def get_single_model_error(self):
        return deepcopy(self.single_model_error)

    def set_resolution(self, resolution):
        self.resolution = resolution

    # ---- scoring (Mean Measure, Zhang et al. PR 2019) ----
    def estimate(self):
        """MM = |M|^λ / (d(M,D) / δ)

        与张老师 estimator.py 完全一致。
        """
        if self.data_kDTree is None or self.num_points == 0:
            self.score_mm = 0.0
            self.score = 0.0
            return 0.0

        error = self.sum_errors / float(self.num_points)
        if np.isclose(error, 0.0):
            error = np.finfo(np.float32).eps

        normalized_error = error / self.data_resolution
        self.score_mm = (self.measure ** self.regularization_factor) / normalized_error

        # MM is model-to-data directed. For a single closed primitive this can
        # otherwise reward a surface that matches only part of the observation.
        # The optional coverage factor adds the complementary data-to-model term.
        score = self.score_mm
        cfg = self.cfg["estimator"]
        if cfg.get("incremental_coverage", False):
            coverage_power = float(cfg.get("coverage_power", 1.0))
            score *= max(float(self.coverage_ratio), np.finfo(np.float32).eps) ** coverage_power

        penalty = 1.0
        penalty += float(cfg.get("outlier_penalty_factor", 0.0)) * self.outlier_ratio
        penalty += (
            float(cfg.get("bbox_penalty_factor", 0.0))
            + float(cfg.get("mm_bbox_penalty_factor", 0.0))
        ) * self.bbox_violation_ratio
        penalty += float(cfg.get("overlap_penalty_factor", 0.0)) * self.overlap_ratio
        penalty += float(cfg.get("control_smoothness_penalty_factor", 0.0)) * self.control_smoothness
        self.score = score / penalty
        return self.score

    # ---- model-to-data error ----
    def _build_faiss_index(self):
        if self._faiss_index is not None:
            return
        data = np.ascontiguousarray(self.data, dtype=np.float32)
        self._faiss_index = faiss.IndexFlatL2(data.shape[1])
        self._faiss_index.add(data)

    def _torch_cuda_bidirectional_nearest(self, points):
        """Return CPU-exact distances to neighbors selected on CUDA."""
        if self._torch_device is None:
            configured = str(self.cfg.get("device", "cuda:0"))
            self._torch_device = torch.device(configured if "cuda" in configured else "cuda:0")
        if self._torch_data is None:
            self._torch_data = torch.as_tensor(
                np.ascontiguousarray(self.data, dtype=np.float32), device=self._torch_device
            )
        model = torch.as_tensor(
            np.ascontiguousarray(points, dtype=np.float32), device=self._torch_device
        )
        distances = torch.cdist(model, self._torch_data, compute_mode="use_mm_for_euclid_dist")
        model_indexes = distances.argmin(dim=1).cpu().numpy().astype(np.int64, copy=False)
        data_indexes = distances.argmin(dim=0).cpu().numpy().astype(np.int64, copy=False)
        del distances, model

        points64 = np.asarray(points, dtype=np.float64)
        data64 = np.asarray(self.data, dtype=np.float64)
        model_errors = np.linalg.norm(points64 - data64[model_indexes], axis=1)
        data_errors = np.linalg.norm(data64 - points64[data_indexes], axis=1)
        self._torch_cached_data_to_model_errors = data_errors
        return model_errors, model_indexes

    def compute_model_to_data_error(self, points, sample_weights=None):
        if self.data_kDTree is None:
            print("no data")
            return np.inf, np.empty(0, dtype=np.int64)

        t0 = time.perf_counter()
        if self._nearest_neighbor_backend == "torch_cuda" and points.shape[0] >= 32:
            errors, indexes = self._torch_cuda_bidirectional_nearest(points)
            indexes = indexes[:, None]
        elif self._use_faiss and points.shape[0] >= 32:
            self._build_faiss_index()
            points_f32 = np.ascontiguousarray(points, dtype=np.float32)
            distances_sq, idx = self._faiss_index.search(points_f32, 1)
            errors = np.sqrt(np.maximum(distances_sq[:, 0], 0.0))
            indexes = np.asarray(idx, dtype=np.int64)
        else:
            errors, indexes = self.data_kDTree.query(points)
        errors = np.asarray(errors).reshape(-1)

        if sample_weights is None:
            normalized_weights = np.ones(points.shape[0], dtype=np.float64)
        else:
            normalized_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
            if normalized_weights.shape[0] != points.shape[0]:
                raise ValueError("token.weights must have one value per sampled point")
            if not np.all(np.isfinite(normalized_weights)) or np.any(normalized_weights < 0.0):
                raise ValueError("token.weights must be finite and non-negative")
            weight_sum = float(normalized_weights.sum())
            if weight_sum <= 0.0:
                raise ValueError("token.weights must have a positive sum")
            # Preserve the existing sum_errors / num_points accumulator while
            # making that quotient an area-weighted mean distance.
            normalized_weights *= points.shape[0] / weight_sum
        sum_errors = float(np.dot(errors, normalized_weights))
        self._last_model_to_data_time = time.perf_counter() - t0
        new_supporters = indexes[:, 0]
        outlier_distance_factor = float(
            self.cfg["estimator"].get("outlier_distance_factor", 0.0)
        )
        if outlier_distance_factor > 0.0:
            max_distance = outlier_distance_factor * float(self.data_resolution)
            self.outlier_ratio = float(
                np.dot((errors > max_distance).astype(np.float64), normalized_weights)
                / normalized_weights.sum()
            )
        else:
            self.outlier_ratio = 0.0

        bbox_margin_factor = float(self.cfg["estimator"].get("bbox_margin_factor", 0.0))
        if bbox_margin_factor > 0.0:
            margin = bbox_margin_factor * float(self.data_resolution)
            below = points < (self.min_point - margin)
            above = points > (self.max_point + margin)
            self.bbox_violation_ratio = float(
                np.dot(np.any(below | above, axis=1).astype(np.float64), normalized_weights)
                / normalized_weights.sum()
            )
        else:
            self.bbox_violation_ratio = 0.0

        if self.base_supporters.size > 0:
            base_supporter_set = np.unique(self.base_supporters)
            self.new_supporters = np.setdiff1d(
                np.unique(new_supporters), base_supporter_set, assume_unique=False
            )
            self.overlap_ratio = float(
                np.isin(new_supporters, base_supporter_set).sum()
            ) / float(new_supporters.size)
        else:
            self.new_supporters = np.unique(new_supporters)
            self.overlap_ratio = 0.0
        self.supporters = np.unique(np.concatenate((self.supporters, new_supporters)))
        self.nearest_points = deepcopy(self.supporters)
        return sum_errors, new_supporters

    def compute_data_coverage(self, points):
        """Fraction of observed points explained by the candidate surface."""
        factor = float(self.cfg["estimator"].get("coverage_distance_factor", 5.0))
        threshold = max(factor * float(self.data_resolution), np.finfo(np.float32).eps)
        if (
            self._nearest_neighbor_backend == "torch_cuda"
            and self._torch_cached_data_to_model_errors is not None
        ):
            distances = self._torch_cached_data_to_model_errors
            self._torch_cached_data_to_model_errors = None
        else:
            distances = KDTree(points).query(self.data, k=1, return_distance=True)[0].reshape(-1)
        return float(np.mean(distances <= threshold))

    def _coerce_points(self, points):
        if o3d is not None:
            if isinstance(points, o3d.core.Tensor):
                points = points.cpu().numpy()
            elif isinstance(points, o3d.geometry.PointCloud):
                points = np.asarray(points.points)

        points = np.asarray(points)
        if self.dimension == 2 and points.shape[1] == 3:
            points = points[:, :2]
        return points

    def add_token(self, token):
        points = getattr(token, "points", None)
        if points is None:
            raise AttributeError("token must provide a `points` attribute.")

        points = self._coerce_points(points)
        if points.shape[0] == 0:
            raise ValueError("the new model instance has no points")

        if points.shape[0] < 5 and self.current_dividing_level != 0:
            self.score_mm = -1
            self.score = -1
            self.single_model_error = float("inf")
            return

        sample_weights = getattr(token, "weights", None)
        sum_errors, supporters = self.compute_model_to_data_error(points, sample_weights)
        self.single_model_error = sum_errors / float(points.shape[0])

        token.supporters = supporters
        token.sum_errors = sum_errors
        self.control_smoothness = self.compute_control_smoothness(token)
        if self.cfg["estimator"].get("incremental_coverage", False):
            self.coverage_ratio = self.compute_data_coverage(points)

        self.sum_errors += sum_errors
        self.num_points += points.shape[0]
        self.measure += getattr(token, "measure", points.shape[0])
        self.model = np.vstack((self.model, points))
        new_labels = np.full(points.shape[0], self.instance_index)
        self.labels = np.concatenate((self.labels, new_labels))
        self.token = token

        self.estimate()

    def add_model(self, **kwargs):
        points = self._coerce_points(kwargs["new_model"])
        token = Token(self.dimension)
        token.points = points
        token.measure = kwargs.get("new_measure", float(points.shape[0]))
        token.trait = getattr(self.rule, "trait", None)
        token.action = getattr(self.rule, "action", None)
        self.add_token(token)

        if "model_color" in kwargs:
            new_color = kwargs["model_color"]
            if self.model_color is None:
                self.model_color = deepcopy(new_color)
            else:
                self.model_color = np.vstack((self.model_color, new_color))

    def compute_control_smoothness(self, token):
        trait = getattr(token, "trait", None)
        control_points = getattr(trait, "control_points", None)
        if control_points is None:
            return 0.0

        control_points = np.asarray(control_points, dtype=np.float32)
        if control_points.ndim != 3 or control_points.shape[0] < 3 or control_points.shape[1] < 3:
            return 0.0

        second_u = control_points[2:, :, :] - 2.0 * control_points[1:-1, :, :] + control_points[:-2, :, :]
        second_v = control_points[:, 2:, :] - 2.0 * control_points[:, 1:-1, :] + control_points[:, :-2, :]
        roughness = np.mean(np.linalg.norm(second_u, axis=-1))
        roughness += np.mean(np.linalg.norm(second_v, axis=-1))

        extent = np.linalg.norm(self.max_point - self.min_point)
        scale = max(float(extent), float(self.data_resolution), np.finfo(np.float32).eps)
        return float(roughness / scale)
