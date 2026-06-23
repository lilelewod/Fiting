from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from core.record import Record, SubRecord
from models.surface.nurbs_surface_rule import _basis_functions
from tools.tool import get_seeds, init_device, set_seed
from models.character.pybpl import splines as char_splines


def _inverse_rescale(value, lb, ub):
    """将原始参数值反向映射到 [-1, 1] 的归一化动作空间"""
    denom = np.maximum(ub - lb, np.finfo(np.float32).eps)
    action = 2.0 * (value - lb) / denom - 1.0
    return np.clip(action, -1.0, 1.0).astype(np.float32)


class Fitter:
    """
    基于梯度下降的 NURBS 曲面 / 圆柱拟合器。

    loss = soft_model_to_data_error / |M|^λ + smoothness

    SVD/PCA 初始化（与 Zhang et al. 的随机初始化对比，是我们的贡献点）。
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
        model_cfg = cfg.get("model", {})

        # ── 训练控制 ──
        self.max_steps = int(fitter_cfg.get("max_episode", 20000))
        self.lr = float(fitter_cfg.get("gd_lr", 1e-2))
        self.lr_min_factor = float(fitter_cfg.get("gd_lr_min_factor", 0.3))
        self.eval_interval = int(fitter_cfg.get("gd_eval_interval", 100))
        self.data_batch_size = int(fitter_cfg.get("gd_data_batch_size", 4096))
        self.patience = int(fitter_cfg.get("gd_patience", 0))

        # ── 初始化 ──
        self.init_mode = str(fitter_cfg.get("gd_init", "svd"))

        # ── loss 权重 ──
        self.smoothness_weight = float(fitter_cfg.get("gd_smoothness_weight", 0.05))

        # ── 模型类型 ──
        self._model_type = str(model_cfg.get('type', 'nurbs_surface')).lower()
        # 自动检测字符模型（从 rule 类名）
        if 'CharacterRule' in type(self.rule).__name__:
            self._model_type = 'character'
        self.dimension = int(self.estimator.dimension)

        # ── 字符模型 ──
        if self._model_type == 'character':
            self._load_character_pivot()

        if self._model_type in ('nurbs_surface', 'surface'):
            self.num_ctrl_u = int(model_cfg["num_ctrl_u"])
            self.num_ctrl_v = int(model_cfg["num_ctrl_v"])
            self.degree_u = int(model_cfg["degree_u"])
            self.degree_v = int(model_cfg["degree_v"])
            self.sample_u = int(model_cfg["sample_u"])
            self.sample_v = int(model_cfg["sample_v"])

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

    # ══════════════════════════════════════════════════════════════════
    #  NURBS 曲面
    # ══════════════════════════════════════════════════════════════════

    def _sample_surface(self, control_points, weights):
        weighted_ctrl = control_points * weights[..., None]
        numerators = torch.einsum("ui,vj,ijd->uvd", self.basis_u, self.basis_v, weighted_ctrl)
        denominators = torch.einsum("ui,vj,ij->uv", self.basis_u, self.basis_v, weights)
        denominators = denominators.clamp_min(1e-8).unsqueeze(-1)
        return (numerators / denominators).reshape(-1, self.dimension)

    def _compute_measure(self, control_points, weights):
        """可微 NURBS 曲面面积"""
        weighted_ctrl = control_points * weights[..., None]
        numerators = torch.einsum("ui,vj,ijd->uvd", self.basis_u, self.basis_v, weighted_ctrl)
        denominators = torch.einsum("ui,vj,ij->uv", self.basis_u, self.basis_v, weights)
        denominators = denominators.clamp_min(1e-8).unsqueeze(-1)
        grid = numerators / denominators

        p00, p10 = grid[:-1, :-1], grid[1:, :-1]
        p01, p11 = grid[:-1, 1:], grid[1:, 1:]

        area1 = 0.5 * torch.linalg.norm(torch.cross(p10 - p00, p01 - p00, dim=-1), dim=-1)
        area2 = 0.5 * torch.linalg.norm(torch.cross(p11 - p10, p01 - p10, dim=-1), dim=-1)
        return area1.sum() + area2.sum()

    def _initial_control_grid(self, target_points):
        """SVD/PCA 初始化控制网格"""
        init_mode = self.init_mode
        points = np.asarray(target_points, dtype=np.float32)
        lb = self.ctrl_lb.detach().cpu().numpy()
        ub = self.ctrl_ub.detach().cpu().numpy()

        if init_mode == 'random':
            rng = np.random.default_rng(self.cfg.get('seeds', [42])[0])
            return rng.uniform(lb, ub).astype(np.float32)

        # SVD
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

        return np.clip(grid, lb, ub)

    def _evaluate_candidate(self, control_points, weights):
        trait_flat = np.concatenate([control_points.reshape(-1), weights.reshape(-1)]).astype(np.float32)
        action = _inverse_rescale(trait_flat, self.rule.lb, self.rule.ub)
        self.estimator.reset()
        self.estimator.current_dividing_level = -1
        self.estimator.parse(action=action)
        self.estimator.generate(current_dividing_level=-1)
        return float(self.estimator.get_score())

    # ══════════════════════════════════════════════════════════════════
    #  Soft MM loss
    # ══════════════════════════════════════════════════════════════════

    def _soft_mm_loss(self, model_points, model_measure, data_batch, iteration):
        """Soft MM 代理 loss: soft_model→data_error / |M|^λ

        τ 线性退火: 3×res → 0.5×res
        """
        eps = 1e-8
        alpha = self.cfg['estimator'].get('regularization_factor', 1.2)
        progress = min(iteration / max(1, self.max_steps - 1), 1.0)

        tau_start = self.data_resolution * 3.0
        tau_end = self.data_resolution * 0.5
        tau = tau_start * (1.0 - progress) + tau_end * progress

        diff = model_points.unsqueeze(1) - data_batch.unsqueeze(0)
        dist = torch.sqrt((diff ** 2).sum(-1) + eps)

        logits = -dist / tau
        soft_assign = torch.softmax(logits, dim=1)
        soft_error = (soft_assign * dist).sum(dim=1).mean()

        safe_measure = model_measure.clamp(min=eps)
        loss = soft_error / (safe_measure ** alpha + eps)
        return loss

    # ══════════════════════════════════════════════════════════════════
    #  圆柱
    # ══════════════════════════════════════════════════════════════════

    def _cylinder_forward(self, action):
        """圆柱可微前向：tanh(action) → (points, measure)"""
        lo = torch.as_tensor(self.rule.lb, dtype=torch.float32, device=self.device)
        hi = torch.as_tensor(self.rule.ub, dtype=torch.float32, device=self.device)
        p = lo + (hi - lo) * (action + 1.0) / 2.0

        x0, y0, z0 = p[0], p[1], p[2]
        az, el = p[3], p[4]
        r = torch.clamp(p[5], min=0.01)
        h = torch.clamp(p[6], min=0.01)

        axis = torch.stack([torch.cos(az) * torch.cos(el),
                            torch.sin(az) * torch.cos(el),
                            torch.sin(el)])
        axis = axis / (axis.norm() + 1e-8)

        ref = torch.tensor([0., 0., 1.], device=self.device)
        if torch.abs(axis[2]) > 0.9:
            ref = torch.tensor([1., 0., 0.], device=self.device)
        u_dir = torch.linalg.cross(axis, ref)
        u_dir = u_dir / (u_dir.norm() + 1e-8)
        v_dir = torch.linalg.cross(axis, u_dir)

        su, sv = 80, 40
        u = torch.linspace(0, 2 * torch.pi, su, device=self.device)
        v = torch.linspace(0, h, sv, device=self.device)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        uu, vv = uu.unsqueeze(-1), vv.unsqueeze(-1)

        base = torch.stack([x0, y0, z0])
        radius_vec = r * (torch.cos(uu) * u_dir + torch.sin(uu) * v_dir)
        axis_vec = vv * axis
        points = (base + radius_vec + axis_vec).reshape(-1, 3)

        measure = 2.0 * torch.pi * r * h
        return points, measure

    def _init_cylinder_action(self, pts):
        """SVD/PCA 初始化圆柱参数，返回 raw action（供 tanh 前向）"""
        lo = np.percentile(pts, 5, axis=0)
        hi = np.percentile(pts, 95, axis=0)
        inlier = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts_clean = pts[inlier] if inlier.sum() > 100 else pts

        center = pts_clean.mean(axis=0)
        _, _, vh = np.linalg.svd(pts_clean - center, full_matrices=False)
        axis_dir = vh[0]

        az = np.arctan2(axis_dir[1], axis_dir[0])
        el = np.arcsin(np.clip(axis_dir[2], -1, 1))

        radial_dist = np.linalg.norm(
            (pts_clean - center) - np.outer((pts_clean - center) @ axis_dir, axis_dir), axis=1)
        r_est = float(np.mean(radial_dist))

        proj = pts @ axis_dir
        h_est = float(np.percentile(proj, 95) - np.percentile(proj, 5))
        h_est = np.clip(h_est, 0.1, self.rule.ub[6])

        base_pt = center - axis_dir * (h_est / 2.0)

        desired = np.array([base_pt[0], base_pt[1], base_pt[2],
                            az, el, r_est, h_est], dtype=np.float32)
        lb, ub = self.rule.lb, self.rule.ub
        action_clipped = np.clip(2.0 * (desired - lb) / np.maximum(ub - lb, 1e-8) - 1.0, -0.999, 0.999)
        return np.arctanh(action_clipped).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════
    #  优化循环
    # ══════════════════════════════════════════════════════════════════

    def _target_points_for_instance(self):
        data = self.estimator.get_data()
        base_supporters = np.asarray(self.estimator.base_supporters, dtype=np.int64)
        if base_supporters.size == 0:
            return data
        mask = np.ones(data.shape[0], dtype=bool)
        mask[np.unique(base_supporters)] = False
        return data[mask]

    def optimize_instance(self):
        if self._model_type == 'cylinder':
            return self._optimize_cylinder()
        elif self._model_type == 'character':
            return self._optimize_character()
        return self._optimize_nurbs()

    def _optimize_nurbs(self):
        target_points_np = self._target_points_for_instance()
        target_points = torch.as_tensor(target_points_np, dtype=torch.float32, device=self.device)
        use_full_batch = self.data_batch_size <= 0 or target_points.shape[0] <= self.data_batch_size

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
        best_ctrl = None
        best_weights = None
        steps_without_improvement = 0

        for step in range(1, self.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)

            if use_full_batch:
                data_batch = target_points
            else:
                perm = torch.randperm(target_points.shape[0], device=self.device)[:self.data_batch_size]
                data_batch = target_points[perm]

            control = torch.max(torch.min(control_points, self.ctrl_ub), self.ctrl_lb)
            weights = self.weight_lb + (self.weight_ub - self.weight_lb) * torch.sigmoid(weights_raw)
            model_points = self._sample_surface(control, weights)
            raw_measure = self._compute_measure(control, weights)

            loss = self._soft_mm_loss(model_points, raw_measure, data_batch, step)

            second_u = control[2:, :, :] - 2.0 * control[1:-1, :, :] + control[:-2, :, :]
            second_v = control[:, 2:, :] - 2.0 * control[:, 1:-1, :] + control[:, :-2, :]
            smoothness = (second_u.norm(dim=-1).mean() + second_v.norm(dim=-1).mean()) / max(
                float(torch.linalg.norm(self.data_max - self.data_min).item()),
                self.data_resolution, np.finfo(np.float32).eps)
            loss = loss + self.smoothness_weight * smoothness

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

                if score > best_score + 1e-8:
                    best_score = score
                    best_ctrl = control_eval.copy()
                    best_weights = weights_eval.copy()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                print(f"GD Step: {step}/{self.max_steps}, Loss: {loss.item():.6f}, Score: {score:.6f}",
                      end="\r", flush=True)

                if self.patience > 0 and steps_without_improvement >= self.patience:
                    print(f"\nEarly stop at step {step}, best={best_score:.4f}")
                    break

        if best_ctrl is not None and best_weights is not None:
            score = self._evaluate_candidate(best_ctrl, best_weights)
            if score > best_score:
                best_score = score
        return best_score

    def _optimize_cylinder(self):
        target_points_np = self._target_points_for_instance()
        target_points = torch.as_tensor(target_points_np, dtype=torch.float32, device=self.device)
        use_full_batch = self.data_batch_size <= 0 or target_points.shape[0] <= self.data_batch_size

        if self.init_mode == 'svd':
            init_action = self._init_cylinder_action(np.asarray(target_points_np, dtype=np.float32))
        else:
            init_action = np.random.default_rng(42).uniform(-1, 1, 7).astype(np.float32)

        action = torch.nn.Parameter(torch.as_tensor(init_action, dtype=torch.float32, device=self.device))
        optimizer = torch.optim.Adam([action], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.max_steps, 1), eta_min=self.lr * self.lr_min_factor)

        sub_record = SubRecord(self.cfg, env_id=0)
        sub_record.data_cloud = self.record.data_cloud
        best_score = float("-inf")
        best_action = None
        steps_without_improvement = 0

        for step in range(1, self.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)

            if use_full_batch:
                data_batch = target_points
            else:
                perm = torch.randperm(target_points.shape[0], device=self.device)[:self.data_batch_size]
                data_batch = target_points[perm]

            model_points, model_measure = self._cylinder_forward(torch.tanh(action))
            loss = self._soft_mm_loss(model_points, model_measure, data_batch, step)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % self.eval_interval == 0 or step == 1 or step == self.max_steps:
                with torch.no_grad():
                    act_np = np.clip(action.detach().cpu().numpy(), -1.0, 1.0)
                self.estimator.reset()
                self.estimator.current_dividing_level = -1
                self.estimator.parse(action=act_np)
                self.estimator.generate(current_dividing_level=-1)
                score = float(self.estimator.get_score())
                sub_record.update(score, self.estimator)
                self.record.update(sub_record, 1)

                if score > best_score + 1e-8:
                    best_score = score
                    best_action = act_np.copy()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                print(f"GD Step: {step}/{self.max_steps}, Loss: {loss.item():.6f}, Score: {score:.6f}",
                      end="\r", flush=True)

                if self.patience > 0 and steps_without_improvement >= self.patience:
                    print(f"\nEarly stop at step {step}, best={best_score:.4f}")
                    break

        if best_action is not None:
            self.estimator.reset()
            self.estimator.current_dividing_level = -1
            self.estimator.parse(action=best_action)
            self.estimator.generate(current_dividing_level=-1)
            score = float(self.estimator.get_score())
            if score > best_score:
                best_score = score
        return best_score

    # ══════════════════════════════════════════════════════════════════
    #  字符模型 (笔画轨迹, ~150D)
    # ══════════════════════════════════════════════════════════════════

    def _load_character_pivot(self):
        """加载字符 pivot token，解析笔画结构和参数维度"""
        import scipy.io as sio
        rule_cfg = self.cfg.get('rule', self.cfg['estimator'].get('rule_cfg', {}))
        token_file = rule_cfg['token_file']
        token = sio.loadmat(token_file, squeeze_me=True)
        positions = token['positions_token']
        invscales = token['invscales_token']
        controls = token['shapes_token']
        relation = token['relation']

        self.char_num_vars = 4  # ink, global_angle, global_scale(2)
        num_strokes = invscales.size - 1
        self.char_strokes = []

        for i in range(num_strokes):
            stroke = {
                'relation': relation[i],
                'position': positions[i],
            }
            if isinstance(stroke['relation'], str) and stroke['relation'] == 'independent':
                self.char_num_vars += 2
            elif isinstance(stroke['relation'], np.ndarray) and stroke['relation'][0] == 'mid':
                self.char_num_vars += 1
            # start/end: no extra vars

            if isinstance(invscales[i], float):
                stroke['nsub'] = 1
                stroke['pivot_scale'] = np.array([invscales[i]], dtype=np.float32)
                stroke['pivot_control'] = np.expand_dims(controls[i], axis=-1).astype(np.float32)
            else:
                stroke['nsub'] = len(invscales[i])
                stroke['pivot_scale'] = np.array(invscales[i], dtype=np.float32)
                stroke['pivot_control'] = controls[i].astype(np.float32)

            self.char_num_vars += 12 * stroke['nsub']  # angle(1) + scale(1) + control(10) per sub-stroke
            self.char_strokes.append(stroke)

        self.char_num_strokes = num_strokes

    def _character_forward(self, action):
        """字符可微前向：action → motors → (points, measure)。纯 PyTorch。"""
        rule_cfg = self.cfg.get('rule', self.cfg['estimator'].get('rule_cfg', {}))
        max_global_rot = float(rule_cfg.get('max_global_rotation', 0.5236))
        max_global_scale = float(rule_cfg.get('max_global_scale', 1.5))
        max_global_trans = float(rule_cfg.get('max_global_translation', 20))
        max_local_trans = float(rule_cfg.get('max_local_translation', 5))
        max_local_rot = float(rule_cfg.get('max_local_rotation', 0.349))
        max_local_scale = float(rule_cfg.get('max_local_scale', 1.5))
        max_control = float(rule_cfg.get('max_control', 5))

        device = action.device
        count = 0
        _ink = action[count]; count += 1
        _global_angle = action[count] * max_global_rot; count += 1
        _global_scales = action[count:count+2] * max_global_scale; count += 2

        all_points = []
        all_lengths = []
        global_translation = torch.zeros(2, device=device)
        neval = 50

        for i, stk in enumerate(self.char_strokes):
            relation = stk['relation']
            pivot_pos = torch.tensor(stk['position'], dtype=torch.float32, device=device)
            nsub = stk['nsub']
            pivot_ctrl = torch.tensor(stk['pivot_control'], dtype=torch.float32, device=device)
            pivot_scl = torch.tensor(stk['pivot_scale'], dtype=torch.float32, device=device)

            # position
            if isinstance(relation, str) and relation == 'independent':
                if i == 0:
                    global_translation = action[count:count+2] * max_global_trans
                    position = pivot_pos + global_translation
                else:
                    local_translation = action[count:count+2] * max_local_trans
                    position = pivot_pos + local_translation + global_translation
                count += 2
            elif isinstance(relation, np.ndarray) and relation[0] == 'mid':
                count += 1  # skip eval_spot
                position = pivot_pos + global_translation
            else:  # start/end
                position = pivot_pos + global_translation

            # angle, scale, control
            angle = action[count:count+nsub] * max_local_rot; count += nsub
            raw_scale = action[count:count+nsub] * max_local_scale; count += nsub
            scale = torch.clamp(pivot_scl * raw_scale, 0.01, 10.0)
            ctrl_delta = action[count:count+10*nsub].reshape(pivot_ctrl.shape) * max_control
            control = pivot_ctrl + ctrl_delta; count += 10 * nsub

            # ── 纯 PyTorch motor generation ──
            motor = torch.zeros(nsub, neval, 2, device=device)
            previous_pos = position
            for s in range(nsub):
                shapes_scaled = scale[s] * control[:, :, s]
                traj = char_splines.get_stk_from_bspline(shapes_scaled, neval)
                offset = traj[0] - previous_pos
                traj = traj - offset

                # Pure torch 2D rotation
                cos_a = torch.cos(angle[s])
                sin_a = torch.sin(angle[s])
                rot = torch.stack([cos_a, -sin_a, sin_a, cos_a]).reshape(2, 2)
                traj = traj @ rot.T

                motor[s] = traj
                previous_pos = traj[-1]

            pts = motor.reshape(-1, 2)
            all_points.append(pts)

            for s in range(nsub):
                seg_len = torch.norm(motor[s, 1:] - motor[s, :-1], dim=1).sum()
                all_lengths.append(seg_len)

        model_points = torch.cat(all_points, dim=0)
        measure = torch.stack(all_lengths).sum()
        return model_points, measure

    def _init_character_action(self):
        """初始化字符 action：从 pivot token 的参数出发，映射回 [-1,1]"""
        rule_cfg = self.cfg.get('rule', self.cfg['estimator'].get('rule_cfg', {}))
        max_global_rot = float(rule_cfg.get('max_global_rotation', 0.5236))
        max_global_scale = float(rule_cfg.get('max_global_scale', 1.5))
        max_global_trans = float(rule_cfg.get('max_global_translation', 20))
        max_local_trans = float(rule_cfg.get('max_local_translation', 5))
        max_local_rot = float(rule_cfg.get('max_local_rotation', 0.349))
        max_local_scale = float(rule_cfg.get('max_local_scale', 1.5))
        max_control = float(rule_cfg.get('max_control', 5))
        max_eval_spot = float(rule_cfg.get('max_eval_spot', 1))

        action = np.zeros(self.char_num_vars, dtype=np.float32)
        count = 0
        action[count] = 0.0; count += 1                      # ink (不优化)
        action[count] = 0.0; count += 1                      # global_angle
        action[count:count+2] = 0.5 / max_global_scale; count += 2  # global_scales

        for i, stk in enumerate(self.char_strokes):
            relation = stk['relation']
            nsub = stk['nsub']

            if isinstance(relation, str) and relation == 'independent':
                action[count:count+2] = 0.0; count += 2
            elif isinstance(relation, np.ndarray) and relation[0] == 'mid':
                action[count] = 0.0; count += 1

            action[count:count+nsub] = 0.0; count += nsub       # angle
            action[count:count+nsub] = 1.0 / max_local_scale; count += nsub  # scale
            action[count:count+10*nsub] = 0.0; count += 10 * nsub  # control delta

        return action.astype(np.float32)

    def _optimize_character(self):
        """GD 优化字符模型（笔画轨迹，~150D）"""
        target_points_np = self._target_points_for_instance()
        target_points = torch.as_tensor(target_points_np, dtype=torch.float32, device=self.device)
        use_full_batch = self.data_batch_size <= 0 or target_points.shape[0] <= self.data_batch_size

        init_action = self._init_character_action()
        action = torch.nn.Parameter(torch.as_tensor(init_action, dtype=torch.float32, device=self.device))

        optimizer = torch.optim.Adam([action], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.max_steps, 1), eta_min=self.lr * self.lr_min_factor)

        sub_record = SubRecord(self.cfg, env_id=0)
        sub_record.data_cloud = self.record.data_cloud
        best_score = float("-inf")
        best_action = None
        steps_without_improvement = 0

        for step in range(1, self.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)

            if use_full_batch:
                data_batch = target_points
            else:
                perm = torch.randperm(target_points.shape[0], device=self.device)[:self.data_batch_size]
                data_batch = target_points[perm]

            model_points, model_measure = self._character_forward(action)
            loss = self._soft_mm_loss(model_points, model_measure, data_batch, step)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % self.eval_interval == 0 or step == 1 or step == self.max_steps:
                with torch.no_grad():
                    act_np = np.clip(action.detach().cpu().numpy(), -1.0, 1.0)
                self.estimator.reset()
                self.estimator.current_dividing_level = -1
                self.estimator.parse(action=act_np)
                self.estimator.generate(current_dividing_level=-1)
                score = float(self.estimator.get_score())
                sub_record.update(score, self.estimator)
                self.record.update(sub_record, 1)

                if score > best_score + 1e-8:
                    best_score = score
                    best_action = act_np.copy()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                print(f"GD Step: {step}/{self.max_steps}, Loss: {loss.item():.6f}, Score: {score:.6f}",
                      end="\r", flush=True)

                if self.patience > 0 and steps_without_improvement >= self.patience:
                    print(f"\nEarly stop at step {step}, best={best_score:.4f}")
                    break

        if best_action is not None:
            self.estimator.reset()
            self.estimator.current_dividing_level = -1
            self.estimator.parse(action=best_action)
            self.estimator.generate(current_dividing_level=-1)
            score = float(self.estimator.get_score())
            if score > best_score:
                best_score = score
        return best_score

    # ══════════════════════════════════════════════════════════════════

    def fit(self):
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
