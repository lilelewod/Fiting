"""Adaptive Energy-guided Skeleton Optimizer (AES-Opt).

A black-box optimizer for high-dimensional geometric fitting that combines:
  1. Energy-aware perturbation — noise scaled by local loss sensitivity
  2. Skeleton-guided search  — low-dimensional subspace from history of best solutions
  3. Local gradient refinement — Adam/L-BFGS on best candidate

Design goals (vs baselines):
  - More robust than pure Adam (skeleton memory escapes local minima)
  - Fewer evaluations than memetic/NL-SHADE (skeleton direction + gradient steps)
  - More stable than CS/CCO (energy-aware step-size adaptation)

Reference implementation for the framework — plugs into the existing
``optimize_instance()`` pattern used by memetic_fitter / cco_fitter / gd_fitter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AESConfig:
    """Hyperparameters for AES-Opt.

    Attributes
    ----------
    dim : int
        Parameter-space dimensionality.
    pop_size : int
        Number of candidates generated per iteration (default 20).
    skeleton_size : int
        Number of historical best solutions stored in skeleton memory.
    top_k : int
        Number of elite candidates selected per iteration.
    noise_scale_init : float
        Initial perturbation magnitude (fraction of bound range).
    noise_scale_min / noise_scale_max : float
        Adaptive noise bounds.
    energy_sensitivity : float
        Weight for energy-aware perturbation (0 = pure random, 1 = full gradient-guided).
    restart_patience : int
        If no improvement for this many iterations, inject random restart noise.
    restart_noise_scale : float
        Perturbation scale during restart (larger than normal).
    refine_steps : int
        Number of Adam/L-BFGS steps on the best candidate per iteration.
    refine_lr : float
        Learning rate for local refinement.
    refine_method : str
        'adam' or 'lbfgs'.
    device : torch.device | None
    bounds : tuple[float, float]
        Parameter bounds for clamp. Default (-1, 1).
    verbose : bool
    """
    dim: int
    pop_size: int = 20
    skeleton_size: int = 10
    top_k: int = 3
    noise_scale_init: float = 0.05
    noise_scale_min: float = 0.001
    noise_scale_max: float = 0.3
    energy_sensitivity: float = 0.3
    restart_patience: int = 20
    restart_noise_scale: float = 0.15
    refine_steps: int = 30
    refine_lr: float = 0.01
    refine_method: str = "adam"
    gradient_guided: bool = False  # use gradient direction instead of skeleton PCA
    device: torch.device | None = None
    bounds: tuple[float, float] = (-1.0, 1.0)
    verbose: bool = True


# ═══════════════════════════════════════════════════════════════════════
#  AES-Opt core
# ═══════════════════════════════════════════════════════════════════════

class AESOptimizer:
    """Adaptive Energy-guided Skeleton Optimizer.

    Parameters
    ----------
    init_theta : np.ndarray  (dim,)
    evaluator : callable     theta → scalar loss (lower is better in internal
                              representation; the framework scores are higher=better,
                              so wrap with sign flip before passing)
    config : AESConfig | None
    """

    def __init__(
        self,
        init_theta: np.ndarray,
        evaluator: Callable[[np.ndarray], float],
        config: AESConfig | None = None,
        gradient_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.dim = init_theta.shape[0]
        self.cfg = config or AESConfig(dim=self.dim)
        self.evaluator = evaluator
        self.gradient_fn = gradient_fn  # theta → grad (numpy array)
        self.device = self.cfg.device or torch.device("cpu")

        lb, ub = self.cfg.bounds
        self._lb = lb
        self._ub = ub
        self._bound_range = ub - lb

        # ── State ──
        self.best_theta = init_theta.copy().astype(np.float32)
        self.best_loss = float(evaluator(init_theta))

        # Skeleton memory: ring buffer of past best solutions
        self._skeleton: list[np.ndarray] = [init_theta.copy()]
        self._skeleton_losses: list[float] = [self.best_loss]

        # Adaptive noise
        self.noise_scale = self.cfg.noise_scale_init
        self._loss_history: list[float] = [self.best_loss]
        self._stall_counter = 0
        self._iteration = 0

        # Energy gradient estimate (for energy-aware perturbation)
        self._energy_grad: np.ndarray | None = None

        # Track for reporting
        self.history: list[tuple[int, float]] = [(0, self.best_loss)]

    # ═══════════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════════

    def step(self) -> float:
        """One AES iteration. Returns current best loss."""
        cfg = self.cfg
        self._iteration += 1

        # ── 1. Energy-aware perturbation ──
        candidates = self._generate_candidates()

        # ── 2. Evaluate ──
        losses = np.array([self.evaluator(c) for c in candidates], dtype=np.float32)

        # ── 3. Select elites ──
        elite_idx = np.argpartition(losses, cfg.top_k)[:cfg.top_k]
        elite_thetas = candidates[elite_idx]
        elite_losses = losses[elite_idx]

        best_of_gen_idx = int(np.argmin(losses))
        best_of_gen_theta = candidates[best_of_gen_idx]
        best_of_gen_loss = float(losses[best_of_gen_idx])

        # ── 4. Update global best ──
        improved = False
        if best_of_gen_loss < self.best_loss:
            self.best_loss = best_of_gen_loss
            self.best_theta = best_of_gen_theta.copy()
            improved = True

        # ── 5. Local gradient refinement on best ──
        if cfg.refine_steps > 0 and improved:
            try:
                refined_theta, refined_loss = self._local_refine(self.best_theta)
                if refined_loss < self.best_loss:
                    self.best_loss = refined_loss
                    self.best_theta = refined_theta
            except Exception:
                pass

        # ── 6. Update skeleton memory ──
        if improved:
            self._update_skeleton(self.best_theta, self.best_loss)

        # ── 7. Adaptive noise & restart ──
        self._update_noise_scale(improved)
        self._check_restart()

        # ── 8. Record ──
        self._loss_history.append(self.best_loss)
        if improved:
            self.history.append((self._iteration, self.best_loss))

        return self.best_loss

    def optimize(self, max_iters: int) -> dict:
        """Main optimization loop.

        Returns dict with keys: best_theta, best_loss, history, evals, runtime.
        """
        t0 = time.perf_counter()
        total_evals = 1  # initial eval

        for it in range(max_iters):
            prev_best = self.best_loss
            self.step()
            # Each step does pop_size + maybe refine evals
            total_evals += self.cfg.pop_size

            if self.cfg.verbose and (it + 1) % 5 == 0:
                improved = self.best_loss < prev_best
                marker = "↓" if improved else " "
                print(f"  AES iter {it+1:>4}/{max_iters}: loss={self.best_loss:.6f}  "
                      f"noise={self.noise_scale:.4f}  stall={self._stall_counter} {marker}",
                      end="\r", flush=True)

        runtime = time.perf_counter() - t0
        if self.cfg.verbose:
            print(f"\n  AES done: loss={self.best_loss:.6f}, evals≈{total_evals}, "
                  f"time={runtime:.1f}s")

        return {
            "best_theta": self.best_theta.copy(),
            "best_loss": self.best_loss,
            "history": self.history,
            "evals": total_evals,
            "runtime": runtime,
        }

    def get_best(self) -> np.ndarray:
        return self.best_theta.copy()

    # ═══════════════════════════════════════════════════════════════════
    #  Internal: Candidate generation
    # ═══════════════════════════════════════════════════════════════════

    def _generate_candidates(self) -> np.ndarray:
        """Generate candidate population via energy-aware perturbation.

        θ_i = θ_best + isotropic_noise + α · skeleton_direction + β · energy_grad
        """
        cfg = self.cfg
        n = cfg.pop_size
        rng = np.random.default_rng()

        # ── Isotropic noise (scaled by adaptive noise_scale) ──
        noise = rng.normal(0, self.noise_scale * self._bound_range, (n, self.dim))

        # ── Guided direction ──
        guide_dir = None
        if cfg.gradient_guided and self.gradient_fn is not None:
            try:
                guide_dir = self.gradient_fn(self.best_theta)
                if guide_dir is not None and np.linalg.norm(guide_dir) > 1e-8:
                    guide_dir = guide_dir / (np.linalg.norm(guide_dir) + 1e-8)
                    guide_dir = guide_dir.astype(np.float32)
            except Exception:
                guide_dir = None

        if guide_dir is None:
            guide_dir = self._skeleton_direction()

        if guide_dir is not None:
            alpha = 0.3 * self.noise_scale
            noise[: n // 2] += alpha * guide_dir[np.newaxis, :]

        # ── Energy-aware perturbation ──
        if self._energy_grad is not None and cfg.energy_sensitivity > 0:
            # Perturb more in dimensions where energy gradient is large
            grad_abs = np.abs(self._energy_grad) + 1e-8
            grad_weights = grad_abs / grad_abs.sum()
            # Bias noise toward high-gradient dimensions
            beta = cfg.energy_sensitivity * self.noise_scale
            energy_bias = beta * self._energy_grad / (grad_abs.max() + 1e-8)
            noise += energy_bias[np.newaxis, :]

        candidates = self.best_theta[np.newaxis, :] + noise.astype(np.float32)
        candidates = np.clip(candidates, self._lb, self._ub)
        return candidates.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    #  Internal: Skeleton memory
    # ═══════════════════════════════════════════════════════════════════

    def _update_skeleton(self, theta: np.ndarray, loss: float) -> None:
        """Insert new best into ring buffer. Maintains sorted order."""
        self._skeleton.append(theta.copy())
        self._skeleton_losses.append(loss)
        if len(self._skeleton) > self.cfg.skeleton_size:
            # Remove worst (highest loss)
            worst_idx = int(np.argmax(self._skeleton_losses))
            self._skeleton.pop(worst_idx)
            self._skeleton_losses.pop(worst_idx)

    def _skeleton_direction(self) -> np.ndarray | None:
        """Compute principal direction from skeleton history via weighted PCA.

        Uses the difference vectors from the current best to each skeleton point,
        weighted by inverse loss (better solutions contribute more).
        """
        if len(self._skeleton) < 2:
            return None

        skel = np.array(self._skeleton, dtype=np.float32)  # (M, dim)
        deltas = skel - self.best_theta[np.newaxis, :]     # (M, dim)
        losses = np.array(self._skeleton_losses, dtype=np.float32)

        # Weight by inverse relative loss
        w = 1.0 / (losses - losses.min() + 1e-8)
        w = w / w.sum()

        # Weighted mean direction
        direction = (deltas * w[:, np.newaxis]).sum(axis=0)

        # Normalize
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return None
        return (direction / norm).astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    #  Internal: Energy gradient estimation
    # ═══════════════════════════════════════════════════════════════════

    def _estimate_energy_gradient(self) -> None:
        """Estimate per-dimension loss sensitivity via finite differences.

        Samples a few random directions around best_theta and computes
        loss differences. Aggregates into an approximate gradient.
        """
        eps = max(self.noise_scale * self._bound_range * 0.1, 1e-6)
        n_samples = min(8, self.cfg.pop_size)
        rng = np.random.default_rng()

        grad = np.zeros(self.dim, dtype=np.float32)
        base_loss = self.best_loss

        for _ in range(n_samples):
            direction = rng.normal(0, 1, self.dim).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-8

            theta_plus = np.clip(self.best_theta + eps * direction, self._lb, self._ub)
            delta_loss = self.evaluator(theta_plus) - base_loss

            # Gradient estimate: (Δloss / ε) · direction
            grad += (delta_loss / eps) * direction

        self._energy_grad = (grad / n_samples).astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    #  Internal: Adaptive mechanisms
    # ═══════════════════════════════════════════════════════════════════

    def _update_noise_scale(self, improved: bool) -> None:
        """Adapt noise scale based on improvement rate.

        - Loss decreasing fast → shrink noise (exploit)
        - Loss stagnating     → grow noise (explore)
        """
        cfg = self.cfg
        window = min(len(self._loss_history), 10)
        recent = self._loss_history[-window:]

        if window >= 3:
            # Trend: negative slope = improving
            x = np.arange(window, dtype=np.float32)
            slope = float(np.polyfit(x, np.array(recent, dtype=np.float32), 1)[0])

            if slope < -1e-6:  # improving
                self.noise_scale = max(cfg.noise_scale_min, self.noise_scale * 0.9)
                self._stall_counter = 0
            else:  # stagnant or worsening
                self.noise_scale = min(cfg.noise_scale_max, self.noise_scale * 1.1)
                self._stall_counter += 1

    def _check_restart(self) -> None:
        """Periodic strong perturbation to escape local minima."""
        cfg = self.cfg
        if self._stall_counter > cfg.restart_patience:
            # Estimate energy gradient before restart
            if self._iteration % 5 == 0:
                self._estimate_energy_gradient()

            # Hard restart: add strong random noise
            rng = np.random.default_rng()
            restart_noise = rng.normal(0, cfg.restart_noise_scale * self._bound_range,
                                       self.dim).astype(np.float32)
            candidate = np.clip(self.best_theta + restart_noise, self._lb, self._ub)
            candidate_loss = self.evaluator(candidate)

            if candidate_loss < self.best_loss:
                self.best_loss = candidate_loss
                self.best_theta = candidate
                self._stall_counter = 0
                self.noise_scale = cfg.noise_scale_init  # reset
                self._update_skeleton(candidate, candidate_loss)

            if self.cfg.verbose:
                print(f"\n  [restart] loss={self.best_loss:.6f}, noise reset to "
                      f"{self.noise_scale:.4f}", end="\r", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    #  Internal: Local gradient refinement
    # ═══════════════════════════════════════════════════════════════════

    def _local_refine(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        """Adam/L-BFGS refinement on a single candidate.

        Uses a simple differentiable proxy: we assume the evaluator
        is available but non-differentiable. We use a zero-order
        (finite-difference) gradient for the refinement.
        """
        cfg = self.cfg
        action = torch.nn.Parameter(
            torch.as_tensor(theta, dtype=torch.float32, device=self.device))

        if cfg.refine_method == "lbfgs":
            opt = torch.optim.LBFGS([action], lr=cfg.refine_lr, history_size=10,
                                    max_iter=20, line_search_fn="strong_wolfe")
        else:
            opt = torch.optim.Adam([action], lr=cfg.refine_lr)

        best_action = theta.copy()
        best_loss = self.best_loss

        eps = max(self.noise_scale * self._bound_range * 0.05, 1e-5)

        for _ in range(cfg.refine_steps):
            opt.zero_grad()

            # Zero-order gradient: symmetric finite differences
            grad = np.zeros(self.dim, dtype=np.float32)
            base = action.detach().cpu().numpy()
            base_l = self.evaluator(base)

            # Sample a subset of dimensions for efficiency
            n_sample = min(self.dim, max(10, self.dim // 10))
            dims = np.random.choice(self.dim, n_sample, replace=False)

            for d in dims:
                plus = base.copy()
                plus[d] = np.clip(plus[d] + eps, self._lb, self._ub)
                grad[d] = (self.evaluator(plus) - base_l) / eps

            # Apply gradient
            action.grad = torch.as_tensor(grad, dtype=torch.float32, device=self.device)
            opt.step()

            with torch.no_grad():
                action.clamp_(self._lb, self._ub)
                new_theta = action.detach().cpu().numpy()
                new_loss = self.evaluator(new_theta)
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_action = new_theta.copy()

        return best_action.astype(np.float32), best_loss


# ═══════════════════════════════════════════════════════════════════════
#  Factory: convenient creation
# ═══════════════════════════════════════════════════════════════════════

def create_aes(
    dim: int,
    evaluator: Callable[[np.ndarray], float],
    init_theta: np.ndarray | None = None,
    pop_size: int = 20,
    skeleton_size: int = 10,
    refine_steps: int = 30,
    noise_scale: float = 0.05,
    device: torch.device | None = None,
    bounds: tuple[float, float] = (-1.0, 1.0),
    **kwargs,
) -> AESOptimizer:
    """Factory function for AES-Opt with sensible defaults."""
    if init_theta is None:
        rng = np.random.default_rng()
        init_theta = rng.uniform(bounds[0], bounds[1], dim).astype(np.float32)

    config = AESConfig(
        dim=dim,
        pop_size=pop_size,
        skeleton_size=skeleton_size,
        refine_steps=refine_steps,
        noise_scale_init=noise_scale,
        device=device,
        bounds=bounds,
        **kwargs,
    )
    return AESOptimizer(init_theta, evaluator, config)


# ═══════════════════════════════════════════════════════════════════════
#  Quick unit test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Test on a simple benchmark: Rastrigin in 20D
    D = 20
    rng = np.random.default_rng(42)
    init = rng.uniform(-1, 1, D).astype(np.float32)

    def rastrigin(x):
        A = 10.0
        return float(A * D + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

    aes = create_aes(dim=D, evaluator=rastrigin, init_theta=init,
                     pop_size=15, refine_steps=20, noise_scale=0.1,
                     bounds=(-5.12, 5.12))
    result = aes.optimize(max_iters=100)
    print(f"\nRastrigin {D}D: best_loss={result['best_loss']:.4f}  "
          f"(global opt = 0.0)  evals≈{result['evals']}  time={result['runtime']:.1f}s")
