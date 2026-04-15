from copy import deepcopy
from typing import Any, Union

import numpy as np
from sklearn.neighbors import KDTree

from tools.geometry import compute_resolution

try:
    import point_cloud_utils as pcu
except ModuleNotFoundError:
    pcu = None

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None

try:
    from pytorch3d.loss.chamfer import (
        _apply_batch_reduction,
        _chamfer_distance_single_direction,
        _handle_pointcloud_input,
        _validate_chamfer_reduction_inputs,
    )
except ModuleNotFoundError:
    _validate_chamfer_reduction_inputs = None
    _handle_pointcloud_input = None
    _chamfer_distance_single_direction = None
    _apply_batch_reduction = None


def npre(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    abs_cosine: bool = True,
):
    if _validate_chamfer_reduction_inputs is None:
        raise ModuleNotFoundError(
            "pytorch3d is required to use fitting.core.estimator.npre()."
        )

    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if norm not in (1, 2):
        raise ValueError("Support for 1 or 2 norm.")

    if point_reduction == "max" and (x_normals is not None or y_normals is not None):
        raise ValueError('Normals must be None if point_reduction is "max"')

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_normals,
        y_normals,
        weights,
        point_reduction,
        norm,
        abs_cosine,
    )

    return _apply_batch_reduction(cham_x, cham_norm_x, weights, batch_reduction)


class NPREEstimator:
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]

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

        self.color = None
        self.model_color = None

        estimator_cfg = cfg["estimator"]
        self.regularization_factor = estimator_cfg.get("regularization_factor", 0.5)
        self.estimator_type = estimator_cfg.get(
            "estimator_type", estimator_cfg.get("type", "npre")
        ).lower()

        self.current_dividing_level = -1
        self.instance_index = 0

        self.model = np.empty((0, self.dimension), dtype=np.float32)
        self.labels = np.empty(0, dtype=np.int64)
        self.sum_errors = 0.0
        self.nearest_points = np.empty(0, dtype=np.int64)
        self.supporters = np.empty(0, dtype=np.int64)
        self.num_points = 0

        self.initial_model = np.empty((0, self.dimension), dtype=np.float32)
        self.initial_labels = np.empty(0, dtype=np.int64)
        self.initial_sum_errors = 0.0
        self.initial_nearest_points = np.empty(0, dtype=np.int64)
        self.initial_num_points = 0

        self.base_sum_errors = 0.0
        self.base_supporters = np.empty(0, dtype=np.int64)
        self.base_num_points = 0

        self.single_model_error = None
        self.score = None
        self.measure = 0.0
        self.score_npre = 0.0
        self.score_mm = 0.0
        self.best_models = [None]
        self.token = None

    def reset(self):
        self.model = deepcopy(self.initial_model)
        self.labels = deepcopy(self.initial_labels)
        self.sum_errors = deepcopy(self.base_sum_errors)
        self.nearest_points = deepcopy(self.base_supporters)
        self.supporters = deepcopy(self.base_supporters)
        self.num_points = deepcopy(self.base_num_points)
        self.measure = 0.0
        self.token = None

    def update(self, *args):
        if len(args) == 5:
            model, sum_errors, nearest_points, labels, instance_index = args
            self.initial_model = deepcopy(model)
            self.initial_labels = deepcopy(labels)
            self.initial_sum_errors = deepcopy(sum_errors)
            self.initial_nearest_points = deepcopy(nearest_points)
            self.initial_num_points = int(model.shape[0])
            self.base_sum_errors = deepcopy(sum_errors)
            self.base_supporters = deepcopy(nearest_points)
            self.base_num_points = self.initial_num_points
            self.instance_index = deepcopy(instance_index)
            if model.shape[0] > 0:
                self.best_models.append(None)
            return

        if len(args) == 3:
            supporters, sum_errors, num_points = args
            self.base_sum_errors = deepcopy(sum_errors)
            self.base_supporters = deepcopy(supporters)
            self.base_num_points = int(num_points)
            return

        raise TypeError(
            "update() expects either 5 args "
            "(model, sum_errors, nearest_points, labels, instance_index) "
            "or 3 args (supporters, sum_errors, num_points)."
        )

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
        if current_dividing_level != -1:
            self.current_dividing_level = current_dividing_level
        assert self.rule.trait is not None
        self.rule.generate()

    def estimate(self):
        if self.data_kDTree is None or self.num_points == 0:
            print("no data or no model")
            self.score = 0
            return 0

        error = self.sum_errors / float(self.num_points)
        if np.isclose(error, 0.0):
            print(
                "the model-to-data error is too close to zero; "
                "clamping it for numerical stability."
            )
            error = np.finfo(np.float32).eps

        factor = self.regularization_factor
        normalized_error = error / self.data_resolution
        normalized_regularizer = float(self.supporters.size) / float(
            self.num_data_points
        )
        self.score_npre = (normalized_regularizer**factor) / normalized_error
        self.score_mm = (self.measure**factor) / normalized_error

        if self.estimator_type == "npre":
            self.score = self.score_npre
        elif self.estimator_type in {"mm", "mean measure"}:
            self.score = self.score_mm
        else:
            raise ValueError(f"Unknown estimator_type: {self.estimator_type}")
        return self.score

    def npre(self):
        self.estimate()
        return self.score_npre

    def compute_model_to_data_error(self, model):
        if self.data_kDTree is None:
            print("no data")
            return np.inf, np.empty(0, dtype=np.int64)

        errors, indexes = self.data_kDTree.query(model)
        sum_errors = float(np.sum(errors))
        new_supporters = indexes[:, 0]
        self.supporters = np.unique(
            np.concatenate((self.supporters, new_supporters))
        )
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

    def add_token(self, token: Any):
        points = getattr(token, "points", None)
        if points is None:
            raise AttributeError("token must provide a `points` attribute.")

        points = self._coerce_points(points)
        if points.shape[0] == 0:
            raise ValueError("the new model instance has no points")

        if points.shape[0] < 5 and self.current_dividing_level != 0:
            self.score_npre = -1
            self.score_mm = -1
            self.score = -1
            self.single_model_error = float("inf")
            return

        self.best_models[-1] = points

        sum_errors, supporters = self.compute_model_to_data_error(points)
        self.single_model_error = sum_errors / float(points.shape[0])

        token.supporters = supporters
        token.sum_errors = sum_errors

        self.sum_errors += sum_errors
        self.num_points += points.shape[0]
        self.measure += getattr(token, "measure", points.shape[0])
        self.model = np.vstack((self.model, points))
        new_labels = np.full(points.shape[0], self.instance_index)
        self.labels = np.concatenate((self.labels, new_labels))
        self.token = token

        self.estimate()

    def add_model(self, **kwargs):
        new_model = self._coerce_points(kwargs["new_model"])
        if new_model.shape[0] == 0:
            raise ValueError("the new model has no points")

        if new_model.shape[0] < 5 and self.current_dividing_level != 0:
            self.score_npre = -1
            self.score_mm = -1
            self.score = -1
            self.single_model_error = float("inf")
            return

        self.best_models[-1] = new_model
        self.measure += kwargs.get("new_measure", new_model.shape[0])

        sum_errors, _ = self.compute_model_to_data_error(new_model)
        self.single_model_error = sum_errors / float(new_model.shape[0])

        self.sum_errors += sum_errors
        self.num_points += new_model.shape[0]
        self.model = np.vstack((self.model, new_model))
        new_labels = np.full(new_model.shape[0], self.instance_index)
        self.labels = np.concatenate((self.labels, new_labels))

        if "model_color" in kwargs:
            new_color = kwargs["model_color"]
            if self.model_color is None:
                self.model_color = deepcopy(new_color)
            else:
                self.model_color = np.vstack((self.model_color, new_color))

        self.estimate()
