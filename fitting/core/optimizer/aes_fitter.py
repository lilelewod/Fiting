"""AES-Opt Fitter — framework integration layer.

Plugs AESOptimizer into the existing Record/SubRecord output system,
producing the same record.json + evolution JSON + PLY as other fitters.

Usage:
    from core.optimizer.aes_fitter import Fitter
    fitter = Fitter(cfg)
    fitter.fit()
    fitter.close()
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d

from core.optimizer.aes_optimizer import AESOptimizer, AESConfig
from core.record import Record, SubRecord
from tools.tool import get_seeds, init_device, set_seed, current_timestamp


class Fitter:
    """AES-Opt Fitter — black-box optimization with framework output."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
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

        fc = cfg["fitter"]
        self.action_dim = self.estimator.num_variables()
        self.max_evals = int(fc.get("max_episode", 30000))
        self.pop_size = int(fc.get("aes_pop_size", 20))
        self.skeleton_size = int(fc.get("aes_skeleton_size", 10))
        self.refine_steps = int(fc.get("aes_refine_steps", 0))
        self.noise_scale = float(fc.get("aes_noise_scale", 0.08))
        self.restart_patience = int(fc.get("aes_restart_patience", 20))
        self.verbose = bool(fc.get("aes_verbose", True))
        self.num_iters = int(fc.get("aes_num_iters",
                                    max(50, self.max_evals // self.pop_size)))
        self._refiner = None

    # ── Evaluator ──
    def _eval_single(self, action: np.ndarray) -> float:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.estimator.reset()
        self.estimator.current_dividing_level = -1
        self.estimator.parse(action=action)
        self.estimator.generate(current_dividing_level=-1)
        return -float(self.estimator.get_score())  # AES minimizes

    # ── Gradient refiner (real gradient, not zero-order) ──
    def _setup_refiner(self):
        """Build AdamRefiner with differentiable forward for supported models."""
        import torch
        self._refiner = None
        model_type = str(self.cfg.get('model', {}).get('type', '')).lower()

        if model_type == 'superquadric':
            rule = self.estimator.rule
            rule._init_bounds()  # lazy init
            data_t = torch.as_tensor(self.estimator.get_data(),
                                     dtype=torch.float32, device=self.device)
            # Differentiable superquadric forward
            n_eta, n_omega = 64, 64
            eta = torch.linspace(-np.pi, np.pi, n_eta, device=self.device)
            omega = torch.linspace(-np.pi/2, np.pi/2, n_omega, device=self.device)
            eta_g, omega_g = torch.meshgrid(eta, omega, indexing='ij')

            cos_e = torch.cos(eta_g)
            sin_e = torch.sin(eta_g)
            cos_o = torch.cos(omega_g)
            sin_o = torch.sin(omega_g)

            lb_t = torch.as_tensor(rule.lb, dtype=torch.float32, device=self.device)
            ub_t = torch.as_tensor(rule.ub, dtype=torch.float32, device=self.device)

            def fwd(a):
                # a ∈ [-1,1]^11 → [lb, ub] → parameters
                p = lb_t + (ub_t - lb_t) * (a + 1.0) / 2.0
                s1, s2, s3 = p[3], p[4], p[5]
                e1, e2 = p[6], p[7]
                # Spherical product (differentiable)
                c1 = torch.sign(cos_e) * torch.abs(cos_e) ** e1 * \
                     torch.sign(cos_o) * torch.abs(cos_o) ** e2
                c2 = torch.sign(sin_e) * torch.abs(sin_e) ** e1 * \
                     torch.sign(cos_o) * torch.abs(cos_o) ** e2
                c3 = torch.sign(sin_o) * torch.abs(sin_o) ** e2
                pts = torch.stack([s1*c1, s2*c2, s3*c3], dim=-1).reshape(-1, 3)
                # Rotation
                rx, ry, rz = p[8], p[9], p[10]
                cx, sx = torch.cos(rx), torch.sin(rx)
                cy, sy = torch.cos(ry), torch.sin(ry)
                cz, sz = torch.cos(rz), torch.sin(rz)
                Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], device=self.device)
                Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], device=self.device)
                Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], device=self.device)
                R = Rz @ Ry @ Rx
                pts = pts @ R.T + p[0:3]
                # Measure: triangle area sum
                g = pts.reshape(n_eta, n_omega, 3)
                p00, p10 = g[:-1,:-1], g[1:,:-1]
                p01, p11 = g[:-1,1:], g[1:,1:]
                a1 = 0.5 * torch.linalg.norm(torch.cross(p10-p00, p01-p00, dim=-1), dim=-1)
                a2 = 0.5 * torch.linalg.norm(torch.cross(p11-p10, p01-p10, dim=-1), dim=-1)
                meas = a1.sum() + a2.sum()
                return pts, meas

            from core.optimizer.memetic_fitter import AdamRefiner
            mm_alpha = float(self.cfg.get("estimator", {}).get("regularization_factor", 0.5))
            self._refiner = AdamRefiner(
                forward_fn=fwd, data_tensor=data_t,
                data_resolution=float(self.estimator.data_resolution),
                lr=0.01, max_steps=self.refine_steps, method='adam',
                device=self.device, alpha=mm_alpha,
            )

        elif model_type == 'character' or 'CharacterRule' in type(self.rule).__name__:
            # Reuse GD fitter's character forward
            from core.optimizer.gd_fitter import Fitter as GDFitter
            ref_cfg = deepcopy(self.cfg)
            if "raw_device" in ref_cfg:
                ref_cfg["device"] = ref_cfg["raw_device"]
            ref_cfg["seeds"] = None
            gd = GDFitter(ref_cfg)
            self._gd_ref = gd
            data_t = torch.as_tensor(self.estimator.get_data(),
                                     dtype=torch.float32, device=self.device)
            def fwd(a):
                pts, meas = gd._character_forward(a)
                return pts, meas, None
            from core.optimizer.memetic_fitter import AdamRefiner
            mm_alpha = float(self.cfg.get("estimator", {}).get("regularization_factor", 0.5))
            self._refiner = AdamRefiner(
                forward_fn=fwd, data_tensor=data_t,
                data_resolution=float(self.estimator.data_resolution),
                lr=0.01, max_steps=self.refine_steps, method='adam',
                device=self.device, alpha=mm_alpha,
            )

        elif model_type == 'nurbs_surface':
            # Reuse memetic's _setup_refiner logic
            from core.optimizer.memetic_fitter import Fitter as MemFitter
            ref_cfg = deepcopy(self.cfg)
            if "raw_device" in ref_cfg:
                ref_cfg["device"] = ref_cfg["raw_device"]
            ref_cfg["seeds"] = None
            ref_cfg["fitter"]["max_episode"] = self.refine_steps
            mem = MemFitter.__new__(MemFitter)
            mem.cfg = ref_cfg
            mem.device = self.device
            mem.estimator = self.estimator
            mem.data_res = float(self.estimator.data_resolution)
            mem._model_type = 'nurbs_surface'
            mem.num_ctrl_u = int(self.cfg['model']['num_ctrl_u'])
            mem.num_ctrl_v = int(self.cfg['model']['num_ctrl_v'])
            from core.optimizer.gd_fitter import Fitter as GDFitter
            ref_cfg2 = deepcopy(ref_cfg)
            ref_cfg2["device"] = ref_cfg["raw_device"]
            ref_cfg2["seeds"] = None
            ref_cfg2["fitter"]["max_episode"] = self.refine_steps
            gd = GDFitter(ref_cfg2)
            mem._gd_ref = gd
            import torch
            data_t = torch.as_tensor(self.estimator.get_data(),
                                     dtype=torch.float32, device=self.device)
            def fwd(a):
                lb_t = torch.as_tensor(gd.rule.lb, dtype=torch.float32, device=gd.device)
                ub_t = torch.as_tensor(gd.rule.ub, dtype=torch.float32, device=gd.device)
                trait_flat = lb_t + (ub_t - lb_t) * (a + 1.0) / 2.0
                xyz_size = gd.num_ctrl_u * gd.num_ctrl_v * 3
                control = trait_flat[:xyz_size].reshape(gd.num_ctrl_u, gd.num_ctrl_v, 3)
                weights = trait_flat[xyz_size:].reshape(gd.num_ctrl_u, gd.num_ctrl_v)
                pts = gd._sample_surface(control, weights)
                meas = gd._compute_measure(control, weights)
                return pts, meas, control
            from core.optimizer.memetic_fitter import AdamRefiner
            mm_alpha = float(self.cfg.get("estimator", {}).get("regularization_factor", 0.5))
            smooth_w = float(self.cfg.get("estimator", {}).get("control_smoothness_penalty_factor", 0.0))
            data_scale = float(torch.linalg.norm(gd.data_max - gd.data_min).item())
            self._refiner = AdamRefiner(
                forward_fn=fwd, data_tensor=data_t,
                data_resolution=float(self.estimator.data_resolution),
                lr=0.01, max_steps=self.refine_steps, method='adam',
                device=self.device, alpha=mm_alpha,
                smoothness_weight=smooth_w, data_scale=data_scale,
            )

    def _refine_gradient(self, action: np.ndarray) -> np.ndarray:
        """Real-gradient refinement via AdamRefiner (not zero-order)."""
        if self._refiner is None:
            return action
        try:
            return self._refiner.refine(action)
        except Exception:
            return action

    # ── Optimization ──
    def optimize_instance(self) -> float:
        if self.refine_steps > 0:
            self._setup_refiner()

        init = np.zeros(self.action_dim, dtype=np.float32)
        grad_fn = None
        if self._refiner is not None:
            import torch
            def grad_fn(action):
                """Compute soft-MM gradient direction via refiner's forward."""
                a = torch.as_tensor(action, dtype=torch.float32, device=self.device)
                a.requires_grad_(True)
                pts, meas = self._refiner.forward_fn(a)[:2]  # first two returns
                loss = self._refiner._loss(pts, meas)
                loss.backward()
                g = a.grad.detach().cpu().numpy().astype(np.float32)
                # Normalize to unit direction
                gn = np.linalg.norm(g)
                if gn > 1e-8:
                    g /= gn
                return g

        cfg = AESConfig(
            dim=self.action_dim, pop_size=self.pop_size,
            skeleton_size=self.skeleton_size, refine_steps=0,
            noise_scale_init=self.noise_scale,
            restart_patience=self.restart_patience, verbose=False,
            gradient_guided=(self._refiner is not None),
        )
        aes = AESOptimizer(init, self._eval_single, cfg, gradient_fn=grad_fn)

        sub = SubRecord(self.cfg, env_id=0)
        sub.data_cloud = self.record.data_cloud
        best_score = float("-inf")
        best_action = None

        # Progress callback
        def _record_cb(force=False):
            nonlocal best_score, best_action
            action = aes.get_best()
            self._eval_single(action)
            score = -float(aes.best_loss)
            sub.update(score, self.estimator)
            self.record.update(sub, 1)
            if score > best_score + 1e-8:
                best_score = score
                best_action = action.copy()

        t0 = time.perf_counter()
        last_record = 0
        last_refine = 0
        _record_cb()  # save initial state immediately
        for it in range(self.num_iters):
            prev_best = aes.best_loss
            aes.step()

            # Gradient refine every 10 iters when improved
            if (self._refiner is not None and aes.best_loss < prev_best
                    and (it - last_refine) >= 10):
                refined = self._refine_gradient(aes.get_best())
                refined_loss = self._eval_single(refined)
                if refined_loss < aes.best_loss:
                    aes.best_loss = refined_loss
                    aes.best_theta = refined.copy()
                    aes._update_skeleton(refined, refined_loss)
                last_refine = it

            # Record every N steps (always saves PLY for monitoring)
            if aes.best_loss < prev_best or (it - last_record) >= 20:
                _record_cb()
                last_record = it

            if self.verbose and (it + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                score = -aes.best_loss
                print(f"  AES iter {it+1:>4}/{self.num_iters}  "
                      f"score={score:.4f}  noise={aes.noise_scale:.3f}  "
                      f"stall={aes._stall_counter}  time={elapsed:.0f}s")

        result = aes.optimize(max_iters=0)  # just get final state
        best_score = -float(aes.best_loss)
        best_action = aes.get_best()

        # Final eval
        self._eval_single(best_action)
        sub.update(best_score, self.estimator)
        self.record.update(sub, 1)
        return best_score

    # ── Public API ──
    def fit(self) -> None:
        for i in range(self.cfg["fitter"].get("num_instances", 1)):
            self.record.token_index = i
            self.record.best_score = 0.0
            self.record.best_sub_record = -1
            self.estimator.instance_index = i
            self.estimator.reset()
            print(f"\nAES fitting instance {i}...")
            score = self.optimize_instance()
            print(f"\nAES done: best={score:.4f}")
        print("AES fitting finished.")

    def close(self) -> None:
        self.record.close()
