"""SPSA (Simultaneous Perturbation Stochastic Approximation) Fitter.

直接估计 MM 评分的梯度，无需 Chamfer 距离或 soft proxy。
每步仅需 2 次 MM 评估（与维度无关），天然适合 7D 圆柱等低维全局参数模型。

对标 CS 基线，但评估效率高 100+ 倍。

Reference: Spall, J. C. (1992). "Multivariate stochastic approximation using a
simultaneous perturbation gradient approximation."
"""

from copy import deepcopy

import numpy as np

from core.record import Record, SubRecord
from tools.tool import get_seeds, init_device, set_seed


class Fitter:
    """SPSA 优化器 —— 用随机扰动估计 MM 梯度，直接优化 MM 评分。"""

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
        self.action_dim = self.estimator.num_variables()  # 圆柱=7, NURBS=144

        # SPSA 超参
        self.max_evals = int(fitter_cfg.get("max_episode", 5000))       # 总评估次数
        self.a = float(fitter_cfg.get("spsa_a", 0.1))                    # 初始步长
        self.A = float(fitter_cfg.get("spsa_A", 100))                    # 步长衰减: a/(A+iter)^α
        self.c = float(fitter_cfg.get("spsa_c", 0.05))                   # 扰动幅度初值
        self.gamma = float(fitter_cfg.get("spsa_gamma", 0.2))            # 扰动衰减指数
        self.alpha = float(fitter_cfg.get("spsa_alpha", 0.602))          # 步长衰减指数
        self.eval_interval = int(fitter_cfg.get("spsa_eval_interval", 2))  # 每N步评估一次

        # 初始化模式
        self.init_mode = str(fitter_cfg.get("gd_init", "svd"))
        self.num_restarts = int(fitter_cfg.get("gd_num_restarts", 1))

    # ------------------------------------------------------------------
    def _evaluate_action(self, action):
        """单次 MM 评估，返回 score。action ∈ [-1, 1]^d。"""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.estimator.reset()
        self.estimator.current_dividing_level = -1
        self.estimator.parse(action=action)
        self.estimator.generate(current_dividing_level=-1)
        return float(self.estimator.get_score())

    # ------------------------------------------------------------------
    def _init_action(self, pts):
        """初始化：SVD 或随机。复用 gd_fitter 的逻辑。"""
        init_mode = self.init_mode
        if init_mode == "svd" and hasattr(self.rule, "lb"):
            # SVD/PCA 初始化圆柱
            pts = np.asarray(pts, dtype=np.float32)
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
            action = np.clip(2.0 * (desired - lb) / np.maximum(ub - lb, 1e-8) - 1.0, -0.999, 0.999)
            return action.astype(np.float32)
        else:
            return np.random.default_rng(42).uniform(-1, 1, self.action_dim).astype(np.float32)

    # ------------------------------------------------------------------
    def _spsa_optimize(self, target_points):
        """单次 SPSA 优化。

        SPSA 梯度估计:
          ĝ_k = (f(θ + c_k·Δ_k) - f(θ - c_k·Δ_k)) / (2·c_k) · Δ_k
        其中 Δ_k 是 Rademacher 随机向量（±1 各 50%）。

        参数更新:
          a_k = a / (A + k)^α
          θ_{k+1} = θ_k + a_k · ĝ_k
        """
        init_action = self._init_action(target_points)
        theta = init_action.copy()
        best_action = theta.copy()
        best_score = float("-inf")

        sub_record = SubRecord(self.cfg, env_id=0)
        sub_record.data_cloud = self.record.data_cloud

        # SPSA 每步 2 次评估，总步数 = max_evals // 2
        n_iters = max(1, self.max_evals // 2)

        for k in range(1, n_iters + 1):
            # ── 扰动向量 Δ: Rademacher (±1) ──
            delta = np.random.choice([-1.0, 1.0], size=self.action_dim).astype(np.float32)

            # ── 扰动幅度衰减 ──
            c_k = self.c / (k ** self.gamma)

            # ── 两次 MM 评估 ──
            theta_plus = np.clip(theta + c_k * delta, -1.0, 1.0)
            theta_minus = np.clip(theta - c_k * delta, -1.0, 1.0)

            score_plus = self._evaluate_action(theta_plus)
            score_minus = self._evaluate_action(theta_minus)

            # ── SPSA 梯度估计 ──
            # ĝ_k = (f(θ+) - f(θ-)) / (2·c_k) * Δ_k
            grad_hat = (score_plus - score_minus) / (2.0 * c_k) * delta

            # ── 步长衰减 ──
            a_k = self.a / ((self.A + k) ** self.alpha)

            # ── 参数更新（最大化 → 梯度上升）──
            theta = np.clip(theta + a_k * grad_hat, -1.0, 1.0)

            # ── 记录最优 ──
            for s, act in [(score_plus, theta_plus), (score_minus, theta_minus)]:
                if s > best_score:
                    best_score = s
                    best_action = act.copy()

            # ── 定期评估当前 θ ──
            if k % self.eval_interval == 0 or k == 1:
                score_curr = self._evaluate_action(theta)
                if score_curr > best_score:
                    best_score = score_curr
                    best_action = theta.copy()

                sub_record.update(best_score, self.estimator)
                self.record.update(sub_record, 2)  # 2 evaluations per iter
                print(f"SPSA iter: {k}/{n_iters} (evals: {k*2}), Score: {best_score:.4f}",
                      end="\r", flush=True)

        return best_score, best_action

    # ------------------------------------------------------------------
    def optimize_instance(self):
        """多起点 SPSA 优化"""
        target_points = self.estimator.get_data()

        global_best_score = float("-inf")
        global_best_action = None

        for restart in range(self.num_restarts):
            if restart > 0:
                # 后续重启用随机初始化
                self.init_mode_orig = self.init_mode
                # 留 SVD 不变但靠不同 seed 产生变化
                np.random.seed(np.random.randint(0, 2**31 - 1))

            best_score, best_action = self._spsa_optimize(target_points)

            print(f"\n[SPSA restart {restart+1}/{self.num_restarts}] Best: {best_score:.4f}"
                  + (f" | Global: {global_best_score:.4f}" if self.num_restarts > 1 else ""))

            if best_score > global_best_score:
                global_best_score = best_score
                global_best_action = best_action.copy()

        self.best_action_ = global_best_action
        return global_best_score

    # ------------------------------------------------------------------
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

        print("The SPSA Multi-Instance fitting is finished.")

    def close(self):
        self.record.close()
