from copy import deepcopy

import numpy as np
from sklearn.neighbors import KDTree

from models.rule import Token
from tools.geometry import compute_resolution

try:
    import point_cloud_utils as pcu
except ModuleNotFoundError:
    pcu = None

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None


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
        self.control_smoothness = 0.0

        self.measure = 0.0
        self.single_model_error = None
        self.score = None
        self.score_mm = 0.0
        self.token = None
        self.model_color = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "rule":
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

    # ---- scoring (mean-measure) ----
    def estimate(self):
        if self.data_kDTree is None or self.num_points == 0:
            print("no data or no model")
            self.score = 0
            return 0

        error = self.sum_errors / float(self.num_points)
        if np.isclose(error, 0.0):
            print(
                "the model-to-data error is too close to zero; clamping it for numerical stability."
            )
            error = np.finfo(np.float32).eps

        factor = self.regularization_factor
        normalized_error = error / self.data_resolution
        self.score_mm = (self.measure ** factor) / normalized_error

        overlap_penalty = float(self.cfg["estimator"].get("overlap_penalty_factor", 0.0))
        if overlap_penalty > 0.0 and self.overlap_ratio > 0.0:
            penalty = max(0.0, 1.0 - self.overlap_ratio) ** overlap_penalty
            self.score_mm *= penalty

        outlier_penalty = float(self.cfg["estimator"].get("outlier_penalty_factor", 0.0))
        if outlier_penalty > 0.0 and self.outlier_ratio > 0.0:
            penalty = max(0.0, 1.0 - self.outlier_ratio) ** outlier_penalty
            self.score_mm *= penalty

        bbox_penalty = float(self.cfg["estimator"].get("bbox_penalty_factor", 0.0))
        if bbox_penalty > 0.0 and self.bbox_violation_ratio > 0.0:
            penalty = max(0.0, 1.0 - self.bbox_violation_ratio) ** bbox_penalty
            self.score_mm *= penalty

        smoothness_penalty = float(
            self.cfg["estimator"].get("control_smoothness_penalty_factor", 0.0)
        )
        if smoothness_penalty > 0.0 and self.control_smoothness > 0.0:
            penalty = 1.0 / (1.0 + smoothness_penalty * self.control_smoothness)
            self.score_mm *= penalty

        self.score = self.score_mm
        return self.score

    # ---- model-to-data error ----
    def compute_model_to_data_error(self, points):
        if self.data_kDTree is None:
            print("no data")
            return np.inf, np.empty(0, dtype=np.int64)

        errors, indexes = self.data_kDTree.query(points)
        sum_errors = float(np.sum(errors))
        new_supporters = indexes[:, 0]
        outlier_distance_factor = float(
            self.cfg["estimator"].get("outlier_distance_factor", 0.0)
        )
        if outlier_distance_factor > 0.0:
            max_distance = outlier_distance_factor * float(self.data_resolution)
            self.outlier_ratio = float(np.mean(errors > max_distance))
        else:
            self.outlier_ratio = 0.0

        bbox_margin_factor = float(self.cfg["estimator"].get("bbox_margin_factor", 0.0))
        if bbox_margin_factor > 0.0:
            margin = bbox_margin_factor * float(self.data_resolution)
            below = points < (self.min_point - margin)
            above = points > (self.max_point + margin)
            self.bbox_violation_ratio = float(np.mean(np.any(below | above, axis=1)))
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

        sum_errors, supporters = self.compute_model_to_data_error(points)
        self.single_model_error = sum_errors / float(points.shape[0])

        token.supporters = supporters
        token.sum_errors = sum_errors
        self.control_smoothness = self.compute_control_smoothness(token)

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
