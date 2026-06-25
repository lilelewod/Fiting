"""Hierarchical Coarse-to-Fine NURBS Fitting.

Phase 1 — Coarse (4×4=16D):  heuristic global search (CCO/Memetic)
Phase 2 — Knot insertion:    4×4 → target grid, surface preserved exactly
Phase 3 — Fine (e.g. 16×16=256D): gradient refinement from excellent init

    cd /home/m25lll/code/Fiting/fitting
    python entrypoints/fit_point_cloud.py --config configs/fit_mm_compare.yaml --algo hierarchical
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from core.optimizer.nurbs_utils import refine_surface_grid
from core.record import Record, SubRecord
from tools.tool import get_seeds, init_device, set_seed


class HierarchicalFitter:
    """Coarse-to-fine NURBS optimization.

    Config keys (under ``fitter``):
        hier_coarse_algo : str   'cco' | 'memetic' (default 'cco')
        hier_coarse_evals : int  evaluation budget for coarse phase (default 30000)
        hier_fine_steps : int    GD steps for fine phase (default 5000)
        hier_fine_lr : float     learning rate for fine phase (default 0.01)
        hier_smoothness : float  smoothness weight for fine phase (default 0.1)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fc = cfg["fitter"]

        seeds = cfg.get("seeds", None)
        if seeds is None:
            seeds = get_seeds(1)
            cfg["seeds"] = seeds
        set_seed(seeds[-1])

        self.device = init_device(cfg["device"])
        cfg["raw_device"] = deepcopy(cfg["device"])
        cfg["device"] = self.device

        self.estimator = cfg["estimator"]["estimator_class"](cfg)
        self.rule = self.estimator.rule

        data_cloud = self.estimator.get_data()
        self.record = Record(cfg, dimension=data_cloud.shape[1])
        self.record.data_cloud = data_cloud

        mc = cfg.get("model", {})
        self.degree_u = int(mc.get("degree_u", 3))
        self.degree_v = int(mc.get("degree_v", 3))
        self.fine_u = int(mc.get("num_ctrl_u", 8))
        self.fine_v = int(mc.get("num_ctrl_v", 8))
        self.coarse_u = 4
        self.coarse_v = 4
        self.weight_lb = float(mc.get("weight_lb", 0.8))
        self.weight_ub = float(mc.get("weight_ub", 1.2))
        self.data_res = float(self.estimator.data_resolution)

        # ── Hierarchical config ──
        self.coarse_algo = str(fc.get("hier_coarse_algo", "cco")).lower()
        self.coarse_evals = int(fc.get("hier_coarse_evals", 30000))
        self.fine_steps = int(fc.get("hier_fine_steps", 5000))
        self.fine_lr = float(fc.get("hier_fine_lr", 0.01))
        self.smoothness_weight = float(fc.get("hier_smoothness", 0.1))

        self._gd_fitter = None

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 1: Coarse global search
    # ═══════════════════════════════════════════════════════════════════

    def _build_coarse_cfg(self) -> dict:
        """Build a config for the coarse 4×4 grid."""
        cfg = deepcopy(self.cfg)
        cfg["model"]["num_ctrl_u"] = self.coarse_u
        cfg["model"]["num_ctrl_v"] = self.coarse_v
        cfg["fitter"]["algo_name"] = self.coarse_algo
        cfg["fitter"]["max_episode"] = self.coarse_evals
        cfg["fitter"]["num_envs"] = 8
        cfg["fitter"]["episodes_per_env"] = 50
        cfg["seeds"] = None  # let CCO generate its own seeds based on num_envs
        # Ensure estimator class is preserved
        cfg["device"] = cfg["raw_device"]
        return cfg

    def _phase1_coarse(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Run heuristic search on 4×4 grid."""
        print(f"\n{'='*50}")
        print(f"Phase 1: Coarse global search ({self.coarse_algo}, {self.coarse_u}x{self.coarse_v}={self.coarse_u*self.coarse_v*4}D)")
        print(f"{'='*50}")

        coarse_cfg = self._build_coarse_cfg()
        t0 = time.perf_counter()

        if self.coarse_algo == "cco":
            from core.optimizer.cco_fitter import Fitter as CoarseFitter
        elif self.coarse_algo == "memetic":
            from core.optimizer.memetic_fitter import Fitter as CoarseFitter
        else:
            raise ValueError(f"Unknown coarse algo: {self.coarse_algo}")

        fitter = CoarseFitter(coarse_cfg)
        fitter.fit()

        # Extract best control_points + weights from the best model
        best_action = fitter.best_action_.copy()
        # CCO fitter stores estimator as collector.estimator
        coarse_est = getattr(fitter, 'estimator', None) or fitter.collector.estimator
        coarse_est.reset()
        coarse_est.current_dividing_level = -1
        coarse_est.parse(action=best_action)
        coarse_est.generate(current_dividing_level=-1)

        ctrl = coarse_est.rule.trait.control_points.copy()
        weights = coarse_est.rule.trait.weights.copy()
        coarse_score = float(fitter.record.best_score)

        runtime = time.perf_counter() - t0
        fitter.close()
        print(f"  Coarse done: score={coarse_score:.2f}, time={runtime:.1f}s")
        return ctrl, weights, coarse_score

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 2: Knot insertion
    # ═══════════════════════════════════════════════════════════════════

    def _phase2_refine_grid(self, coarse_ctrl, coarse_weights):
        """Refine 4×4 → fine_u × fine_v via knot insertion."""
        print(f"\n{'='*50}")
        print(f"Phase 2: Knot insertion {self.coarse_u}x{self.coarse_v} → {self.fine_u}x{self.fine_v}")
        print(f"{'='*50}")

        fine_ctrl, fine_weights = refine_surface_grid(
            coarse_ctrl, coarse_weights,
            self.coarse_u, self.coarse_v,
            self.fine_u, self.fine_v,
            self.degree_u, self.degree_v,
        )
        print(f"  Refined: ctrl={fine_ctrl.shape}, weights={fine_weights.shape}")
        return fine_ctrl, fine_weights

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 3: Fine gradient refinement
    # ═══════════════════════════════════════════════════════════════════

    def _phase3_fine(self, init_ctrl, init_weights):
        """L-BFGS + periodic hard-projection refinement.

        - L-BFGS: second-order convergence, no manual LR tuning
        - Every K steps: re-project data→model correspondence → hard Chamfer block
        - Final: soft MM refinement for outlier robustness
        """
        REPROJ_EVERY = 20
        HARD_STEPS = 15     # L-BFGS steps with hard Chamfer per projection
        SOFT_STEPS = 10     # final soft MM refinement steps

        print(f"\n{'='*50}")
        print(f"Phase 3: L-BFGS + dynamic projection")
        print(f"  steps={self.fine_steps}  reproj_every={REPROJ_EVERY}  "
              f"smoothness={self.smoothness_weight}")
        print(f"{'='*50}")

        t0 = time.perf_counter()
        from core.optimizer.gd_fitter import Fitter as GDFitter

        fine_cfg = deepcopy(self.cfg)
        if "raw_device" in fine_cfg:
            fine_cfg["device"] = fine_cfg["raw_device"]
        fine_cfg["seeds"] = None
        fine_cfg["fitter"]["max_episode"] = self.fine_steps
        fine_cfg["fitter"]["gd_lr"] = self.fine_lr
        fine_cfg["fitter"]["gd_smoothness_weight"] = self.smoothness_weight

        gd = GDFitter(fine_cfg)
        self._gd_fitter = gd

        target_np = gd._target_points_for_instance()
        target = torch.as_tensor(target_np, dtype=torch.float32, device=gd.device)
        alpha = float(gd.cfg['estimator'].get('regularization_factor', 0.5))
        data_scale = max(float(torch.linalg.norm(gd.data_max - gd.data_min).item()),
                         gd.data_resolution, 1e-8)
        eps = 1e-8

        # ── Flat param for L-BFGS ──
        ctrl_flat = torch.as_tensor(init_ctrl.copy(), dtype=torch.float32, device=gd.device).reshape(-1)
        w_clamped = np.clip((init_weights.copy() - gd.weight_lb) / (gd.weight_ub - gd.weight_lb), 1e-4, 1-1e-4)
        w_flat = torch.as_tensor(w_clamped.reshape(-1), dtype=torch.float32, device=gd.device)
        param = torch.nn.Parameter(torch.cat([ctrl_flat, w_flat]))
        n_ctrl = gd.num_ctrl_u * gd.num_ctrl_v * 3

        def _unflatten(p):
            ctrl = torch.max(torch.min(
                p[:n_ctrl].reshape(gd.num_ctrl_u, gd.num_ctrl_v, 3), gd.ctrl_ub), gd.ctrl_lb)
            w = gd.weight_lb + (gd.weight_ub - gd.weight_lb) * torch.sigmoid(
                p[n_ctrl:].reshape(gd.num_ctrl_u, gd.num_ctrl_v))
            return ctrl, w

        def _smooth_term(ctrl):
            d2u = ctrl[2:, :, :] - 2.0 * ctrl[1:-1, :, :] + ctrl[:-2, :, :]
            d2v = ctrl[:, 2:, :] - 2.0 * ctrl[:, 1:-1, :] + ctrl[:, :-2, :]
            return (d2u.norm(dim=-1).mean() + d2v.norm(dim=-1).mean()) / data_scale

        # ── Tracking ──
        best_score = float("-inf")
        best_ctrl, best_weights = None, None
        total_step = 0

        def _eval():
            nonlocal best_score, best_ctrl, best_weights
            with torch.no_grad():
                c, w = _unflatten(param)
                c_np, w_np = c.detach().cpu().numpy(), w.detach().cpu().numpy()
            s = gd._evaluate_candidate(c_np, w_np)
            if s > best_score + 1e-8:
                best_score = s
                best_ctrl, best_weights = c_np.copy(), w_np.copy()
            return s

        # ── Phase 3a: Hard Chamfer blocks with re-projection ──
        num_blocks = self.fine_steps // REPROJ_EVERY
        for block in range(num_blocks):
            # Re-project: find closest model point for each data point
            with torch.no_grad():
                ctrl, w = _unflatten(param)
                model_pts = gd._sample_surface(ctrl, w)
                d2m_idx = torch.cdist(target, model_pts).argmin(dim=1)

            def _hard_closure():
                opt_hard.zero_grad()
                c, w = _unflatten(param)
                mpts = gd._sample_surface(c, w)
                meas = gd._compute_measure(c, w)
                err = ((mpts[d2m_idx] - target) ** 2).sum(-1).sqrt().mean()
                loss = err / (meas.clamp(min=eps) ** alpha + eps)
                loss = loss + gd.smoothness_weight * _smooth_term(c)
                loss.backward()
                return loss

            opt_hard = torch.optim.LBFGS([param], lr=1.0, history_size=20,
                                         max_iter=20, line_search_fn="strong_wolfe")
            for _ in range(HARD_STEPS):
                try:
                    opt_hard.step(_hard_closure)
                except Exception:
                    break
                with torch.no_grad():
                    param.clamp_(-1.0, 1.0)
                total_step += 1

            score = _eval()
            elapsed = time.perf_counter() - t0
            print(f"  [hard block {block+1}/{num_blocks}] step={total_step}  "
                  f"score={score:.4f}  best={best_score:.4f}  time={elapsed:.0f}s",
                  end="\r", flush=True)

        # ── Phase 3b: Final soft MM refinement ──
        print(f"\n  Soft refinement ({SOFT_STEPS} L-BFGS steps)...", end="\r")
        def _soft_closure():
            opt_soft.zero_grad()
            c, w = _unflatten(param)
            mpts = gd._sample_surface(c, w)
            meas = gd._compute_measure(c, w)
            loss = gd._soft_mm_loss(mpts, meas, target, total_step)
            loss = loss + gd.smoothness_weight * _smooth_term(c)
            loss.backward()
            return loss

        opt_soft = torch.optim.LBFGS([param], lr=1.0, history_size=20,
                                     max_iter=20, line_search_fn="strong_wolfe")
        for _ in range(SOFT_STEPS):
            try:
                opt_soft.step(_soft_closure)
            except Exception:
                break
            with torch.no_grad():
                param.clamp_(-1.0, 1.0)
            total_step += 1

        score = _eval()
        runtime = time.perf_counter() - t0
        print(f"\n  Fine done: best={best_score:.4f}  total_steps={total_step}  time={runtime:.1f}s")
        return best_score

    # ═══════════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════════

    def fit(self) -> None:
        print("=" * 60)
        print(f"Hierarchical NURBS Fitting: 4×4 → {self.fine_u}×{self.fine_v}")
        print(f"Phase 1: {self.coarse_algo.upper()} on {self.coarse_u}x{self.coarse_v} ({self.coarse_u*self.coarse_v*4}D)")
        print(f"Phase 2: Knot insertion → {self.fine_u}x{self.fine_v}")
        print(f"Phase 3: GD ({self.fine_steps} steps)")
        print("=" * 60)

        t_start = time.perf_counter()

        # Phase 1
        coarse_ctrl, coarse_weights, coarse_score = self._phase1_coarse()

        # Phase 2
        fine_ctrl, fine_weights = self._phase2_refine_grid(coarse_ctrl, coarse_weights)

        # Phase 3
        fine_score = self._phase3_fine(fine_ctrl, fine_weights)

        total_time = time.perf_counter() - t_start

        print(f"\n{'='*60}")
        print(f"Hierarchical Fitting Complete")
        print(f"{'='*60}")
        print(f"  Coarse ({self.coarse_algo}, {self.coarse_u}x{self.coarse_v}): {coarse_score:.2f}")
        print(f"  Fine   (GD,       {self.fine_u}x{self.fine_v}): {fine_score:.2f}")
        print(f"  Delta:  {fine_score - coarse_score:+.2f}")
        print(f"  Time:   {total_time:.1f}s")
        print(f"{'='*60}")

    def close(self) -> None:
        if self._gd_fitter is not None:
            self._gd_fitter.close()
        self.record.close()
