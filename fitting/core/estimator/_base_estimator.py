from copy import deepcopy
from typing import Union

import numpy as np
import torch
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


def _validate_reduction_inputs(batch_reduction, point_reduction):
    valid = {"mean", "sum", None}
    point_valid = {"mean", "sum", "max", None}
    if batch_reduction not in valid:
        raise ValueError(
            f"batch_reduction must be one of {valid}, got {batch_reduction!r}"
        )
    if point_reduction not in point_valid:
        raise ValueError(
            f"point_reduction must be one of {point_valid}, got {point_reduction!r}"
        )


def _handle_pointcloud_input(points, lengths=None, normals=None):
    if normals is not None:
        raise ValueError("Normals are not supported by this local npre implementation.")

    if not torch.is_tensor(points):
        points = torch.as_tensor(points, dtype=torch.float32)

    if points.ndim == 2:
        points = points.unsqueeze(0)
    elif points.ndim != 3:
        raise ValueError(
            f"Expected points to have shape (N, D) or (B, N, D), got {tuple(points.shape)}"
        )

    if lengths is None:
        lengths = torch.full(
            (points.shape[0],),
            points.shape[1],
            dtype=torch.int64,
            device=points.device,
        )
    elif not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths, dtype=torch.int64, device=points.device)

    return points, lengths, None


def _reduce_point_losses(losses, lengths, point_reduction):
    reduced = []
    for batch_idx in range(losses.shape[0]):
        valid = losses[batch_idx, : lengths[batch_idx]]
        if point_reduction in (None, "mean"):
            reduced.append(valid.mean())
        elif point_reduction == "sum":
            reduced.append(valid.sum())
        elif point_reduction == "max":
            reduced.append(valid.max())
        else:
            raise ValueError(f"Unsupported point_reduction: {point_reduction}")
    return torch.stack(reduced)


def _apply_batch_reduction(losses, weights, batch_reduction):
    if weights is not None and not torch.is_tensor(weights):
        weights = torch.as_tensor(weights, dtype=losses.dtype, device=losses.device)

    if weights is not None:
        losses = losses * weights

    if batch_reduction is None:
        return losses
    if batch_reduction == "sum":
        return losses.sum()
    if batch_reduction == "mean":
        if weights is not None:
            weight_sum = weights.sum().clamp_min(torch.finfo(losses.dtype).eps)
            return losses.sum() / weight_sum
        return losses.mean()
    raise ValueError(f"Unsupported batch_reduction: {batch_reduction}")


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
    del abs_cosine

    _validate_reduction_inputs(batch_reduction, point_reduction)

    if norm not in (1, 2):
        raise ValueError("Support for 1 or 2 norm.")

    if point_reduction == "max" and (x_normals is not None or y_normals is not None):
        raise ValueError('Normals must be None if point_reduction is "max"')

    x, x_lengths, _ = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, _ = _handle_pointcloud_input(y, y_lengths, y_normals)

    if x.shape[0] != y.shape[0]:
        if x.shape[0] == 1:
            x = x.expand(y.shape[0], -1, -1)
            x_lengths = x_lengths.expand(y.shape[0])
        elif y.shape[0] == 1:
            y = y.expand(x.shape[0], -1, -1)
            y_lengths = y_lengths.expand(x.shape[0])
        else:
            raise ValueError(
                "Batch sizes of x and y must match or one of them must be 1."
            )

    pairwise = torch.cdist(x, y, p=norm)
    losses = []
    for batch_idx in range(pairwise.shape[0]):
        dist = pairwise[batch_idx, : x_lengths[batch_idx], : y_lengths[batch_idx]]
        min_dist = dist.min(dim=1).values
        if point_reduction is None:
            losses.append(min_dist)
        else:
            losses.append(
                _reduce_point_losses(
                    min_dist.unsqueeze(0),
                    x_lengths[batch_idx : batch_idx + 1],
                    point_reduction,
                )[0]
            )

    if point_reduction is None:
        return losses, None

    losses = torch.stack(losses)
    return _apply_batch_reduction(losses, weights, batch_reduction), None


class BaseEstimator:
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
        self.estimator_type = estimator_cfg.get(
            "estimator_type", estimator_cfg.get("type", "npre")
        ).lower()

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

        self.measure = 0.0
        self.single_model_error = None
        self.score = None
        self.score_npre = 0.0
        self.score_mm = 0.0
        self.token = None
        self.model_color = None

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

    def update(self, supporters, sum_errors, num_points):
        self.base_sum_errors = deepcopy(sum_errors)
        self.base_supporters = deepcopy(supporters)
        self.base_num_points = int(num_points)

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
                "the model-to-data error is too close to zero; clamping it for numerical stability."
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

    def compute_model_to_data_error(self, points):
        if self.data_kDTree is None:
            print("no data")
            return np.inf, np.empty(0, dtype=np.int64)

        errors, indexes = self.data_kDTree.query(points)
        sum_errors = float(np.sum(errors))
        new_supporters = indexes[:, 0]
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
            self.score_npre = -1
            self.score_mm = -1
            self.score = -1
            self.single_model_error = float("inf")
            return

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
