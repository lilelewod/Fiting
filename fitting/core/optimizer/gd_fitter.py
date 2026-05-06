from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from core.record import Record, SubRecord
from models.surface.nurbs_surface_rule import _basis_functions
from tools.tool import get_seeds, init_device, set_seed


def _inverse_rescale(value, lb, ub):
    """将原始参数值反向映射到 [-1, 1] 的归一化动作空间"""
    denom = np.maximum(ub - lb, np.finfo(np.float32).eps)
    action = 2.0 * (value - lb) / denom - 1.0
    return np.clip(action, -1.0, 1.0).astype(np.float32)


class Fitter:
    """
    基于梯度下降的 NURBS 曲面拟合器。

    loss =  data_to_model_weight * data_to_model      (数据→模型 Chamfer 距离)
          + model_to_data_weight * model_to_data      (模型→数据 Chamfer 距离)
          - coverage_weight     * soft_coverage       (软覆盖率奖励，sigmoid)
          + smoothness_weight   * smoothness          (控制网格二阶光滑度)
          + bbox_weight         * bbox_penalty        (包围盒越界 relu 惩罚)
          + weight_reg_weight   * weight_reg          (NURBS 权重 L2 正则化)
          + overlap_weight      * overlap_penalty     (多实例间重叠惩罚)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        seeds = self.cfg.get("seeds", None)
        if seeds is None:
            seeds = get_seeds(1)
            self.cfg["seeds"] = seeds
            self.cfg["raw_seeds"] = None
        set_seed(seeds[-1])

        self.device = init_device(cfg["device"])
        cfg["raw_device"] = deepcopy(cfg["device"])
        cfg["device"] = self.device

        self.estimator = self.cfg["estimator"]["estimator_class"](self.cfg)
        self.rule = self.estimator.rule

        data_cloud = self.estimator.get_data()
        self.record = Record(cfg, dimension=data_cloud.shape[1])
        self.record.data_cloud = data_cloud

        fitter_cfg = cfg["fitter"]
        model_cfg = cfg["model"]

        # 训练控制
        self.max_steps = int(fitter_cfg.get("max_episode", 20000))
        self.lr = float(fitter_cfg.get("gd_lr", 1e-2))
        self.lr_min_factor = float(fitter_cfg.get("gd_lr_min_factor", 0.3))
        self.eval_interval = int(fitter_cfg.get("gd_eval_interval", 100))
        self.data_batch_size = int(fitter_cfg.get("gd_data_batch_size", 4096))  # 0=全量

        # loss 权重
        self.data_to_model_weight = float(fitter_cfg.get("gd_data_to_model_weight", 1.0))
        self.model_to_data_weight = float(fitter_cfg.get("gd_model_to_data_weight", 1.0))
        self.coverage_weight = float(fitter_cfg.get("gd_coverage_weight", 0.5))  # 0=关闭覆盖率奖励
        self.coverage_threshold_factor = float(fitter_cfg.get("gd_coverage_threshold_factor", 2.5))
        self.coverage_temperature_factor = float(fitter_cfg.get("gd_coverage_temperature_factor", 0.5))
        self.smoothness_weight = float(fitter_cfg.get("gd_smoothness_weight", 0.05))
        self.bbox_weight = float(fitter_cfg.get("gd_bbox_weight", 0.2))
        self.weight_reg_weight = float(fitter_cfg.get("gd_weight_reg_weight", 0.01))
        self.overlap_weight = float(fitter_cfg.get("gd_overlap_weight", 0.05))  # num_instances>1 时生效

        # NURBS 结构参数
        self.num_ctrl_u = int(model_cfg["num_ctrl_u"])
        self.num_ctrl_v = int(model_cfg["num_ctrl_v"])
        self.degree_u = int(model_cfg["degree_u"])
        self.degree_v = int(model_cfg["degree_v"])
        self.sample_u = int(model_cfg["sample_u"])
        self.sample_v = int(model_cfg["sample_v"])
        self.dimension = int(self.estimator.dimension)

        # 预计算 B 样条基函数
        basis_u = _basis_functions(
            np.linspace(0.0, 1.0, self.sample_u, dtype=np.float32),
            self.num_ctrl_u, self.degree_u, self.rule.knot_u,
        )
        basis_v = _basis_functions(
            np.linspace(0.0, 1.0, self.sample_v, dtype=np.float32),
            self.num_ctrl_v, self.degree_v, self.rule.knot_v,
        )
        self.basis_u = torch.as_tensor(basis_u, dtype=torch.float32, device=self.device)
        self.basis_v = torch.as_tensor(basis_v, dtype=torch.float32, device=self.device)

        # 控制点上下界，用于投影约束
        self.ctrl_lb = torch.as_tensor(
            self.rule.lb[: self.num_ctrl_u * self.num_ctrl_v * self.dimension].reshape(
                self.num_ctrl_u, self.num_ctrl_v, self.dimension),
            dtype=torch.float32, device=self.device,
        )
        self.ctrl_ub = torch.as_tensor(
            self.rule.ub[: self.num_ctrl_u * self.num_ctrl_v * self.dimension].reshape(
                self.num_ctrl_u, self.num_ctrl_v, self.dimension),
            dtype=torch.float32, device=self.device,
        )
        self.weight_lb = float(model_cfg.get("weight_lb", 0.2))
        self.weight_ub = float(model_cfg.get("weight_ub", 3.0))

        self.data_min = torch.as_tensor(self.estimator.min_point, dtype=torch.float32, device=self.device)
        self.data_max = torch.as_tensor(self.estimator.max_point, dtype=torch.float32, device=self.device)
        self.data_resolution = float(self.estimator.data_resolution)

    def _sample_surface(self, control_points, weights):
        """NURBS 曲面采样：分子/分母分别做基函数组合，得到齐次坐标归一化后的曲面点云"""
        weighted_ctrl = control_points * weights[..., None]
        numerators = torch.einsum("ui,vj,ijd->uvd", self.basis_u, self.basis_v, weighted_ctrl)
        denominators = torch.einsum("ui,vj,ij->uv", self.basis_u, self.basis_v, weights)
        denominators = denominators.clamp_min(1e-8).unsqueeze(-1)
        return (numerators / denominators).reshape(-1, self.dimension)

    def _initial_control_grid(self, target_points):
        """SVD 主方向初始化控制网格，沿点云前两个主方向展开控制点"""
        points = np.asarray(target_points, dtype=np.float32)
        center = points.mean(axis=0)
        centered = points - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_u = vh[0]
        axis_v = vh[1] if vh.shape[0] > 1 else np.roll(axis_u, 1)

        coord_u = centered @ axis_u
        coord_v = centered @ axis_v
        u_values = np.linspace(coord_u.min(), coord_u.max(), self.num_ctrl_u, dtype=np.float32)
        v_values = np.linspace(coord_v.min(), coord_v.max(), self.num_ctrl_v, dtype=np.float32)

        grid = np.zeros((self.num_ctrl_u, self.num_ctrl_v, self.dimension), dtype=np.float32)
        for i, u in enumerate(u_values):
            for j, v in enumerate(v_values):
                grid[i, j] = center + u * axis_u + v * axis_v

        lb = self.ctrl_lb.detach().cpu().numpy()
        ub = self.ctrl_ub.detach().cpu().numpy()
        return np.clip(grid, lb, ub)

    def _target_points_for_instance(self):
        """多实例拟合时，排除已被之前实例覆盖的数据点"""
        data = self.estimator.get_data()
        base_supporters = np.asarray(self.estimator.base_supporters, dtype=np.int64)
        if base_supporters.size == 0:
            return data

        mask = np.ones(data.shape[0], dtype=bool)
        mask[np.unique(base_supporters)] = False
        remaining = data[mask]
        if remaining.shape[0] < max(64, self.num_ctrl_u * self.num_ctrl_v):
            return data
        return remaining

    def _evaluate_candidate(self, control_points, weights):
        """用 NPRE 评分评估当前曲面质量"""
        trait_flat = np.concatenate([control_points.reshape(-1), weights.reshape(-1)]).astype(np.float32)
        action = _inverse_rescale(trait_flat, self.rule.lb, self.rule.ub)

        self.estimator.reset()
        self.estimator.current_dividing_level = -1
        self.estimator.parse(action=action)
        self.estimator.generate(current_dividing_level=-1)
        return float(self.estimator.get_score())

    def optimize_instance(self):
        """对单张 NURBS 曲面执行梯度下降优化"""
        target_points_np = self._target_points_for_instance()
        target_points = torch.as_tensor(target_points_np, dtype=torch.float32, device=self.device)
        use_full_batch = self.data_batch_size <= 0 or target_points.shape[0] <= self.data_batch_size

        base_cloud_np = self.record.base_cloud
        base_cloud = None
        if base_cloud_np is not None and len(base_cloud_np) > 0:
            base_cloud = torch.as_tensor(base_cloud_np, dtype=torch.float32, device=self.device)

        # 控制点：SVD 初始化；权重：logit 映射到无约束空间训练
        init_ctrl = self._initial_control_grid(target_points_np)
        control_points = torch.nn.Parameter(
            torch.as_tensor(init_ctrl, dtype=torch.float32, device=self.device))
        init_weights = torch.ones((self.num_ctrl_u, self.num_ctrl_v), dtype=torch.float32, device=self.device)
        weights_raw = torch.nn.Parameter(
            torch.logit(((init_weights - self.weight_lb) / (self.weight_ub - self.weight_lb)).clamp(1e-4, 1 - 1e-4)))

        optimizer = torch.optim.Adam([control_points, weights_raw], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.max_steps, 1), eta_min=self.lr * self.lr_min_factor)
        sub_record = SubRecord(self.cfg, env_id=0)
        sub_record.data_cloud = self.record.data_cloud
        best_score = float("-inf")

        coverage_threshold = self.coverage_threshold_factor * self.data_resolution
        coverage_temperature = max(self.coverage_temperature_factor * self.data_resolution,
                                   np.finfo(np.float32).eps)

        for step in range(1, self.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)

            if use_full_batch:
                data_batch = target_points
            else:
                perm = torch.randperm(target_points.shape[0], device=self.device)[: self.data_batch_size]
                data_batch = target_points[perm]

            # 投影 + 前向采样
            control = torch.max(torch.min(control_points, self.ctrl_ub), self.ctrl_lb)
            weights = self.weight_lb + (self.weight_ub - self.weight_lb) * torch.sigmoid(weights_raw)
            model_points = self._sample_surface(control, weights)

            # 双向 Chamfer 距离
            pairwise = torch.cdist(data_batch.unsqueeze(0), model_points.unsqueeze(0), p=2).squeeze(0)
            data_to_model_min = pairwise.min(dim=1).values  # 每个数据点到曲面的最近距离
            model_to_data_min = pairwise.min(dim=0).values  # 每个曲面点到数据的最近距离
            data_to_model = data_to_model_min.mean()
            model_to_data = model_to_data_min.mean()

            # 软覆盖率奖励（sigmoid）：距离 < 阈值 → 视为被覆盖
            soft_coverage = torch.sigmoid(
                (coverage_threshold - data_to_model_min) / coverage_temperature).mean()

            # 光滑度：控制网格 u/v 方向二阶差分
            second_u = control[2:, :, :] - 2.0 * control[1:-1, :, :] + control[:-2, :, :]
            second_v = control[:, 2:, :] - 2.0 * control[:, 1:-1, :] + control[:, :-2, :]
            smoothness = (second_u.norm(dim=-1).mean() + second_v.norm(dim=-1).mean()) / max(
                float(torch.linalg.norm(self.data_max - self.data_min).item()),
                self.data_resolution, np.finfo(np.float32).eps)

            # 包围盒越界：relu 只惩罚超出 min/max 的曲面点
            bbox_penalty = (torch.relu(self.data_min - model_points) +
                            torch.relu(model_points - self.data_max)).sum(dim=1).mean()

            # NURBS 权重 L2 正则化：鼓励权重接近 1.0
            weight_reg = ((weights - 1.0) ** 2).mean()

            # 多实例重叠惩罚：与之前实例点云距离 < 2*resolution 视为重叠
            overlap_penalty = torch.tensor(0.0, device=self.device)
            if base_cloud is not None and base_cloud.numel() > 0:
                overlap_dist = torch.cdist(model_points.unsqueeze(0), base_cloud.unsqueeze(0), p=2).squeeze(0)
                overlap_penalty = torch.relu(2.0 * self.data_resolution - overlap_dist.min(dim=1).values).mean()

            loss = (self.data_to_model_weight * data_to_model +
                    self.model_to_data_weight * model_to_data -
                    self.coverage_weight * soft_coverage +
                    self.smoothness_weight * smoothness +
                    self.bbox_weight * bbox_penalty +
                    self.weight_reg_weight * weight_reg +
                    self.overlap_weight * overlap_penalty)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % self.eval_interval == 0 or step == 1 or step == self.max_steps:
                with torch.no_grad():
                    control_eval = torch.max(torch.min(control_points, self.ctrl_ub), self.ctrl_lb).detach().cpu().numpy()
                    weights_eval = (self.weight_lb + (self.weight_ub - self.weight_lb) *
                                    torch.sigmoid(weights_raw)).detach().cpu().numpy()
                score = self._evaluate_candidate(control_eval, weights_eval)
                sub_record.update(score, self.estimator)
                self.record.update(sub_record, 1)
                best_score = max(best_score, score)
                print(f"GD Step: {step}/{self.max_steps}, Loss: {loss.item():.6f}, Score: {score:.6f}",
                      end="\r", flush=True)

        return best_score

    def fit(self):
        """顺序拟合多个 NURBS 曲面实例"""
        for i in range(self.cfg["fitter"]["num_instances"]):
            self.record.token_index = i
            self.record.best_score = 0.0
            self.record.best_sub_record = -1
            self.record.base_cloud = None
            self.record.base_color = None

            self.estimator.instance_index = i
            self.estimator.reset()
            if i > 0:
                supporters, sum_errors, num_points = self.record.get_base()
                self.estimator.update(supporters, sum_errors, num_points)

            print(f"Fitting for the model instance {i} begins")
            best_score = self.optimize_instance()
            print(f"\nFitting for the model instance {i} finished. Best Score: {best_score}\n")

        print("The GD Multi-Instance fitting is finished.")

    def close(self):
        self.record.close()
