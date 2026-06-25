"""
Memetic Optimization Framework: NL-SHADE + Adam/L-BFGS

Unified global-local optimization for robust multi-model geometric fitting.

Modules:
  NL_SHADE    — Success-History Adaptive DE (vectorized population search)
  AdamRefiner — Gradient-based local refinement with early stopping
  MemeticOpt  — Orchestrator with adaptive scheduling
  Fitter      — Framework integration layer (MM estimator API)

Reference:
  Tanabe & Fukunaga, "Success-History based Parameter Adaptation for DE", CEC 2013
"""

from __future__ import annotations

import time
from copy import deepcopy
from multiprocessing import get_context
from typing import Callable, Optional

import cloudpickle
import numpy as np
import torch

from core.record import Record, SubRecord
from tools.tool import get_seeds, init_device, set_seed


# ═══════════════════════════════════════════════════════════════════════
#  Multiprocessing worker pool (module-level for spawn picklability)
# ═══════════════════════════════════════════════════════════════════════

_MP_WORKER_ESTIMATOR = None  # per-process estimator instance


def _mp_worker_initializer(config_pkl: bytes) -> None:
    """Spawn-safe worker initializer: create estimator once per process (CPU-only)."""
    global _MP_WORKER_ESTIMATOR
    import os, sys
    torch.set_num_threads(1)
    # Suppress worker logs
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    config = cloudpickle.loads(config_pkl)
    config["device"] = {"train_device": "cpu", "cuda_deterministic": False}
    config["record"]["verbose"] = False
    estimator_cls = config["estimator"]["estimator_class"]
    _MP_WORKER_ESTIMATOR = estimator_cls(config)


def _mp_eval_single(action: np.ndarray) -> float:
    """Evaluate one action using the worker's persistent estimator."""
    global _MP_WORKER_ESTIMATOR
    action = np.clip(action, -1.0, 1.0).astype(np.float32)
    _MP_WORKER_ESTIMATOR.reset()
    _MP_WORKER_ESTIMATOR.current_dividing_level = -1
    _MP_WORKER_ESTIMATOR.parse(action=action)
    _MP_WORKER_ESTIMATOR.generate(current_dividing_level=-1)
    return float(_MP_WORKER_ESTIMATOR.get_score())


# ═══════════════════════════════════════════════════════════════════════
#  NL_SHADE — Vectorized Success-History Adaptive DE
# ═══════════════════════════════════════════════════════════════════════

class NL_SHADE:
    """NL-SHADE with linear population size reduction.

    Fully vectorized mutation/crossover/selection.  External archive for
    diversity maintenance.  Success-history memory for F and CR adaptation.

    Parameters
    ----------
    dim : int                  Search-space dimensionality.
    bounds : (float, float)    Parameter bounds.
    pop_size : int             Initial population size.
    min_pop_size : int         Minimum after linear reduction.
    memory_size : int          SHADE history memory slots (H in the paper).
    p_best : float             Fraction for current-to-pbest/1 mutation.
    archive_rate : float       Archive size multiplier relative to |P|.
    seed : int | None          RNG seed.
    """

    def __init__(
        self,
        dim: int,
        bounds: tuple[float, float] = (-1.0, 1.0),
        pop_size: int = 200,
        min_pop_size: int = 20,
        memory_size: int = 6,
        p_best: float = 0.11,
        archive_rate: float = 1.4,
        seed: int | None = None,
    ):
        self.dim = dim
        self.lb, self.ub = bounds
        self.pop_size = pop_size
        self.init_pop_size = pop_size
        self.min_pop_size = min_pop_size
        self.memory_size = memory_size
        self.p_best = p_best
        self.archive_rate = archive_rate

        # SHADE memory: M_F (Cauchy location), M_CR (Normal location)
        self.M_F = np.full(memory_size, 0.5, dtype=np.float32)
        self.M_CR = np.full(memory_size, 0.5, dtype=np.float32)
        self.k = 0  # memory write cursor

        # Archive (ring buffer for replaced inferior solutions)
        self._archive_data = np.zeros((int(pop_size * archive_rate), dim), dtype=np.float32)
        self._archive_size = 0
        self._archive_capacity = int(pop_size * archive_rate)

        # Population state
        self._rng = np.random.default_rng(seed)
        self.population: np.ndarray = self._rng.uniform(self.lb, self.ub, (pop_size, dim)).astype(np.float32)
        self.fitness: np.ndarray = np.zeros(pop_size, dtype=np.float32)
        self._best_idx: int = 0
        self.generations: int = 0
        self.evaluations: int = 0

    # -- archive ops --------------------------------------------------------
    def _archive_insert(self, x: np.ndarray) -> None:
        if self._archive_size < self._archive_capacity:
            self._archive_data[self._archive_size] = x
            self._archive_size += 1
        else:
            self._archive_data[self._rng.integers(0, self._archive_capacity)] = x

    def _archive_sample(self, n: int) -> np.ndarray:
        if self._archive_size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        idx = self._rng.integers(0, self._archive_size, size=n)
        return self._archive_data[idx]

    # -- F / CR generation --------------------------------------------------
    def _sample_F_CR(self) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized F (Cauchy) and CR (Normal) sampling from SHADE memory."""
        ri = self._rng.integers(0, self.memory_size, size=self.pop_size)
        mu_F = self.M_F[ri]
        mu_CR = self.M_CR[ri]

        F = mu_F + 0.1 * np.random.standard_cauchy(self.pop_size).astype(np.float32)
        F = np.clip(F, 0.0, 1.0)
        F[F <= 0] = mu_F[F <= 0]  # fix negative Cauchy draws by re-draw (simple fallback)

        CR = mu_CR + 0.1 * np.random.standard_normal(self.pop_size).astype(np.float32)
        CR = np.clip(CR, 0.0, 1.0)

        return F, CR

    def _update_memory(self, S_F: np.ndarray, S_CR: np.ndarray, delta_f: np.ndarray) -> None:
        """Update SHADE memory with weighted Lehmer mean (F) and arithmetic mean (CR)."""
        if len(S_F) == 0:
            return
        w = delta_f / (delta_f.sum() + 1e-12)
        mean_F = (w * S_F ** 2).sum() / (w * S_F).sum() if (w * S_F).sum() > 1e-12 else 0.5
        mean_CR = (w * S_CR).sum()
        self.M_F[self.k] = float(np.clip(mean_F, 0.0, 1.0))
        self.M_CR[self.k] = float(np.clip(mean_CR, 0.0, 1.0))
        self.k = (self.k + 1) % self.memory_size

    # -- mutation -----------------------------------------------------------
    def _mutation_current_to_pbest(self, F: np.ndarray) -> np.ndarray:
        """Vectorized current-to-pbest/1 mutation with archive support.

        v_i = x_i + F_i*(x_pbest - x_i) + F_i*(x_r1 - x_r2)
        """
        n = self.pop_size
        # p-best pool: top p% of population
        p_num = max(2, int(round(n * self.p_best)))
        pbest_pool = np.argpartition(-self.fitness, p_num - 1)[:p_num]

        # For each i: pick random pbest, random r1 (≠i), random r2 (≠i,≠r1, from P∪A)
        pb = self.population[self._rng.choice(pbest_pool, size=n)]  # (n, dim)
        r1_idx = np.array([self._rng.choice([j for j in range(n) if j != i]) for i in range(n)])
        x_r1 = self.population[r1_idx]

        # r2: sample from union P ∪ A
        n_union = n + self._archive_size
        # Pre-allocate r2
        x_r2 = np.empty_like(self.population)
        for i in range(n):
            while True:
                r2_pool_idx = self._rng.integers(0, n_union)
                if r2_pool_idx < n:
                    if r2_pool_idx != i and r2_pool_idx != r1_idx[i]:
                        x_r2[i] = self.population[r2_pool_idx]
                        break
                elif self._archive_size > 0:
                    arch_idx = r2_pool_idx - n
                    x_r2[i] = self._archive_data[arch_idx]
                    break

        F2 = F[:, np.newaxis]  # (n, 1)
        return self.population + F2 * (pb - self.population) + F2 * (x_r1 - x_r2)

    # -- crossover ----------------------------------------------------------
    def _crossover(self, mutant: np.ndarray, CR: np.ndarray) -> np.ndarray:
        """Vectorized binomial crossover with boundary repair."""
        n = self.pop_size
        trial = self.population.copy()
        j_rand = self._rng.integers(0, self.dim, size=n)
        mask = np.random.random((n, self.dim)) < CR[:, np.newaxis]
        mask[np.arange(n), j_rand] = True
        trial[mask] = mutant[mask]

        # Boundary repair: random re-initialization for out-of-bounds values
        below = trial < self.lb
        above = trial > self.ub
        trial[below] = self._rng.uniform(self.lb, self.ub, size=below.sum()).astype(np.float32)
        trial[above] = self._rng.uniform(self.lb, self.ub, size=above.sum()).astype(np.float32)
        return trial

    # -- generation ---------------------------------------------------------
    def step(self, trial_fitness: np.ndarray, F: np.ndarray, CR: np.ndarray) -> int:
        """Greedy selection: trial replaces parent if fitter. Returns #success."""
        n = self.pop_size
        better = trial_fitness >= self.fitness
        S_F, S_CR, delta_f = [], [], []

        for i in range(n):
            if better[i]:
                self._archive_insert(self.population[i].copy())
                delta = trial_fitness[i] - self.fitness[i]
                S_F.append(F[i])
                S_CR.append(CR[i])
                delta_f.append(delta)

        self.population[better] = self._population_trials[better].copy()
        self.fitness[better] = trial_fitness[better]

        if S_F:
            self._update_memory(np.array(S_F, dtype=np.float32),
                                np.array(S_CR, dtype=np.float32),
                                np.array(delta_f, dtype=np.float32))
        self._best_idx = int(np.argmax(self.fitness))
        self.generations += 1
        return int(better.sum())

    def generate_trials(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One NL-SHADE generation → (trial_population, F, CR)."""
        F, CR = self._sample_F_CR()
        mutant = self._mutation_current_to_pbest(F)
        self._population_trials = self._crossover(mutant, CR)
        return self._population_trials, F, CR

    # -- population reduction -----------------------------------------------
    def reduce_population(self, target: int) -> None:
        """Linear population size reduction: drop worst individuals."""
        if target >= self.pop_size:
            return
        n_remove = self.pop_size - target
        worst = np.argpartition(self.fitness, n_remove)[:n_remove]
        keep = np.ones(self.pop_size, dtype=bool)
        keep[worst] = False
        self.population = self.population[keep]
        self.fitness = self.fitness[keep]
        self.pop_size = target

    # -- accessors ----------------------------------------------------------
    @property
    def best(self) -> tuple[np.ndarray, float]:
        return self.population[self._best_idx].copy(), float(self.fitness[self._best_idx])

    @property
    def worst_idx(self) -> int:
        return int(np.argmin(self.fitness))

    def target_population(self, progress: float) -> int:
        """Linear schedule: pop_size → min_pop_size as progress 0→1."""
        return max(self.min_pop_size, int(round(self.init_pop_size + (self.min_pop_size - self.init_pop_size) * progress)))


# ═══════════════════════════════════════════════════════════════════════
#  AdamRefiner — Gradient-based local refinement
# ═══════════════════════════════════════════════════════════════════════

class AdamRefiner:
    """Gradient-based local refinement with early stopping.

    Supports: Adam, L-BFGS, and cascade (Adam → L-BFGS) modes.

    Parameters
    ----------
    forward_fn : callable    action_tensor → (model_points, measure)
    data_tensor : Tensor     target points (M, dim)
    data_resolution : float  for temperature annealing
    lr : float               Adam learning rate
    max_steps : int          Maximum refinement steps per call
    early_stop_tol : float   Stop if loss improvement < tol for patience steps
    patience : int           Early-stop patience (steps)
    method : str             'adam' | 'lbfgs' | 'cascade'
    device : torch.device
    """

    def __init__(
        self,
        forward_fn: Callable,
        data_tensor: torch.Tensor,
        data_resolution: float = 0.01,
        lr: float = 0.01,
        max_steps: int = 100,
        early_stop_tol: float = 1e-6,
        patience: int = 10,
        method: str = "adam",
        device: torch.device | None = None,
        alpha: float = 1.05,
        smoothness_weight: float = 0.0,
        data_scale: float = 1.0,
    ):
        self.forward_fn = forward_fn
        self.data = data_tensor
        self.data_res = data_resolution
        self.lr = lr
        self.max_steps = max_steps
        self.early_stop_tol = early_stop_tol
        self.patience = patience
        self.method = method
        self.device = device or torch.device("cpu")
        self.alpha = alpha  # MM measure exponent
        self.smoothness_weight = smoothness_weight
        self.data_scale = data_scale
        self._n_calls = 0

    @property
    def n_calls(self) -> int:
        return self._n_calls

    def refine(self, action_np: np.ndarray) -> np.ndarray:
        """Refine action ∈ [-1,1] space. Returns best action found."""
        self._n_calls += 1
        if self.method == "cascade":
            half = self.max_steps // 2
            action_np = self._refine(action_np, "adam", half)
            action_np = self._refine(action_np, "lbfgs", half)
            return action_np
        return self._refine(action_np, self.method, self.max_steps)

    def _refine(self, action_np: np.ndarray, method: str, steps: int) -> np.ndarray:
        action = torch.nn.Parameter(torch.as_tensor(action_np, dtype=torch.float32, device=self.device))

        if method == "lbfgs":
            opt = torch.optim.LBFGS(
                [action], lr=self.lr, history_size=10,
                max_iter=20, line_search_fn="strong_wolfe",
            )
        else:
            opt = torch.optim.Adam([action], lr=self.lr, betas=(0.9, 0.999))

        best_loss = float("inf")
        best_action = action_np.copy()
        no_improve = 0

        def _closure():
            opt.zero_grad()
            result = self.forward_fn(action)
            loss = self._loss(*result)
            loss.backward()
            return loss

        for _step in range(steps):
            if method == "lbfgs":
                loss = opt.step(_closure)
            else:
                opt.zero_grad()
                result = self.forward_fn(action)
                loss = self._loss(*result)
                loss.backward()
                opt.step()

            with torch.no_grad():
                action.clamp_(-1.0, 1.0)
                val = loss.item()
                if best_loss - val > self.early_stop_tol:
                    best_loss = val
                    best_action = action.detach().cpu().numpy().copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break

        return np.clip(best_action, -1.0, 1.0).astype(np.float32)

    def _loss(self, pts: torch.Tensor, meas: torch.Tensor, control: torch.Tensor | None = None) -> torch.Tensor:
        """Soft MM loss: soft_model→data_error / |M|^α + smoothness.

        When control_points is provided and smoothness_weight > 0, adds
        thin-plate smoothness (2nd-order finite difference) to align the
        refiner gradient with the hard MM scoring.
        """
        eps = 1e-8
        tau = max(self.data_res * 0.5, 0.005)
        diff = pts.unsqueeze(1) - self.data.unsqueeze(0)  # (N, M, D)
        dist = torch.sqrt((diff ** 2).sum(-1) + eps)
        logits = -dist / tau
        soft = torch.softmax(logits, dim=1)
        err = (soft * dist).sum(dim=1).mean()
        safe_m = meas.clamp(min=eps)
        loss = err / (safe_m ** self.alpha + eps)

        if control is not None and self.smoothness_weight > 0:
            second_u = control[2:, :, :] - 2.0 * control[1:-1, :, :] + control[:-2, :, :]
            second_v = control[:, 2:, :] - 2.0 * control[:, 1:-1, :] + control[:, :-2, :]
            smoothness = (
                second_u.norm(dim=-1).mean() + second_v.norm(dim=-1).mean()
            ) / max(self.data_scale, self.data_res, np.finfo(np.float32).eps)
            loss = loss + self.smoothness_weight * smoothness

        return loss


# ═══════════════════════════════════════════════════════════════════════
#  MemeticOpt — Orchestrator
# ═══════════════════════════════════════════════════════════════════════

class MemeticOpt:
    """NL-SHADE + Adam/L-BFGS with adaptive refinement scheduling.

    Parameters
    ----------
    dim : int
    bounds : (float, float)
    de : NL_SHADE             Pre-configured DE instance.
    refiner : AdamRefiner | None
    refine_every : int        Generations between refinement calls.
    refine_top_k : int        Number of elite individuals to refine.
    adaptive_K : bool         Increase K when convergence slows.
    """

    def __init__(
        self,
        dim: int,
        bounds: tuple[float, float],
        de: NL_SHADE,
        refiner: AdamRefiner | None = None,
        refine_every: int = 10,
        refine_top_k: int = 1,
        adaptive_K: bool = True,
    ):
        self.dim = dim
        self.de = de
        self.refiner = refiner
        self.refine_every = refine_every
        self.base_refine_every = refine_every
        self.refine_top_k = refine_top_k
        self.adaptive_K = adaptive_K

        # Tracking
        self.best_score = float("-inf")
        self.best_action: np.ndarray | None = None
        self.history: list[tuple[int, float]] = []  # (generation, best_score)
        self._last_improvement_gen = 0
        self._stall_window = 0

    def _evaluate(self, eval_fn: Callable, population: np.ndarray) -> np.ndarray:
        """Evaluate population. eval_fn: (N, dim) → (N,) scores."""
        return np.array([eval_fn(ind) for ind in population], dtype=np.float32)

    def _evaluate_single(self, eval_fn: Callable, individual: np.ndarray) -> float:
        return float(eval_fn(individual))

    def _should_refine(self, gen: int) -> bool:
        if self.refiner is None:
            return False
        if gen % self.refine_every == 0:
            return True
        return False

    def _adapt_K(self) -> None:
        """Increase K (refine less often) as optimization slows."""
        if not self.adaptive_K:
            return
        stall = self.de.generations - self._last_improvement_gen
        if stall > 50:
            self.refine_every = min(self.base_refine_every * 4, 40)
        elif stall > 20:
            self.refine_every = min(self.base_refine_every * 2, 20)
        else:
            self.refine_every = self.base_refine_every

    def optimize(self, eval_fn: Callable, max_evals: int, verbose: bool = True,
                 batch_eval_fn: Callable | None = None,
                 record_cb: Callable | None = None) -> dict:
        """Run memetic optimization.

        Parameters
        ----------
        eval_fn : callable
            Individual evaluation: action (D,) → score (float).
        max_evals : int
            Total evaluation budget.
        verbose : bool

        Returns
        -------
        dict with keys: best_action, best_score, history, evals, runtime
        """
        t0 = time.perf_counter()
        de = self.de
        refiner = self.refiner

        # Initial evaluation
        if batch_eval_fn is not None:
            de.fitness = batch_eval_fn(de.population)
        else:
            de.fitness = self._evaluate(eval_fn, de.population)
        de._best_idx = int(np.argmax(de.fitness))
        total_evals = de.pop_size

        self.best_score = float(de.fitness[de._best_idx])
        self.best_action = de.population[de._best_idx].copy()
        self.history = [(total_evals, self.best_score)]
        self._last_improvement_gen = 0
        if record_cb is not None:
            record_cb(total_evals, self.best_score, self.best_action)

        while total_evals < max_evals:
            gen = de.generations

            # ── NL-SHADE generation ──
            trial, F, CR = de.generate_trials()
            if batch_eval_fn is not None:
                trial_fitness = batch_eval_fn(trial)
            else:
                trial_fitness = self._evaluate(eval_fn, trial)
            total_evals += len(trial)
            de.step(trial_fitness, F, CR)

            # Update best
            if de.fitness[de._best_idx] > self.best_score:
                self.best_score = float(de.fitness[de._best_idx])
                self.best_action = de.population[de._best_idx].copy()
                self.history.append((total_evals, self.best_score))
                self._last_improvement_gen = gen
                if record_cb is not None:
                    record_cb(total_evals, self.best_score, self.best_action)

            # ── Adaptive K scheduling ──
            self._adapt_K()

            # ── Local refinement ──
            if self._should_refine(gen):
                elite_indices = np.argpartition(-de.fitness, min(self.refine_top_k, de.pop_size - 1))[:self.refine_top_k]
                for idx in elite_indices:
                    old_score = float(de.fitness[idx])
                    refined = refiner.refine(de.population[idx])
                    new_score = self._evaluate_single(eval_fn, refined)
                    total_evals += 1

                    if new_score > old_score:
                        de.population[idx] = refined
                        de.fitness[idx] = new_score
                        if new_score > self.best_score:
                            self.best_score = new_score
                            self.best_action = refined.copy()
                            self.history.append((total_evals, self.best_score))
                            self._last_improvement_gen = gen
                            if record_cb is not None:
                                record_cb(total_evals, self.best_score, self.best_action)

            # ── Population reduction ──
            progress = total_evals / max_evals
            target = de.target_population(progress)
            de.reduce_population(target)

            if verbose and gen % 5 == 0:
                print(f"  gen {gen:>4}: evals={total_evals}/{max_evals}, "
                      f"pop={de.pop_size}, best={self.best_score:.4f}")

        runtime = time.perf_counter() - t0
        if verbose:
            print(f"\n  Done: best={self.best_score:.4f}, evals={total_evals}, time={runtime:.1f}s")

        return {
            "best_action": self.best_action,
            "best_score": self.best_score,
            "history": self.history,
            "evals": total_evals,
            "runtime": runtime,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Fitter — Framework integration
# ═══════════════════════════════════════════════════════════════════════

class Fitter:
    """Memetic Fitter — integrates with the Fiting MM estimator API.

    Supports: NURBS surface, cylinder, character.
    """

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
        mc = cfg.get("model", {})

        # ── Model detection ──
        self._model_type = str(mc.get("type", "nurbs_surface")).lower()
        if "CharacterRule" in type(self.rule).__name__:
            self._model_type = "character"

        self.action_dim = self.estimator.num_variables()
        self.data_res = float(self.estimator.data_resolution)

        # ── Config ──
        self.max_evals = int(fc.get("max_episode", 50000))
        self.pop_size = int(fc.get("mem_pop_size", 200))
        self.min_pop = int(fc.get("mem_min_pop", 20))
        self.refine_every = int(fc.get("mem_refine_every", 10))
        self.refine_steps = int(fc.get("mem_refine_steps", 100))
        self.refine_method = str(fc.get("mem_refine_method", "adam"))
        self.refine_top_k = int(fc.get("mem_refine_top_k", 1))
        self.adaptive_K = bool(fc.get("mem_adaptive_K", True))
        self.lr = float(fc.get("gd_lr", 0.01))
        self.num_workers = int(fc.get("mem_num_workers", 1))

        # ── Multiprocess pool for parallel population evaluation ──
        self._mp_pool = None
        self._mp_config_pkl: bytes | None = None
        if self.num_workers > 1:
            self._setup_mp_pool()

        # ── Refiner setup ──
        self._refiner: AdamRefiner | None = None
        self._gd_ref = None
        self._setup_refiner()

    def _setup_refiner(self) -> None:
        """Build AdamRefiner using the differentiable forward of GD fitter."""
        if self._model_type not in ("cylinder", "character", "nurbs_surface"):
            return  # Unknown model type

        from core.optimizer.gd_fitter import Fitter as GDFitter

        ref_cfg = deepcopy(self.cfg)
        if "raw_device" in ref_cfg:
            ref_cfg["device"] = ref_cfg["raw_device"]
        ref_cfg["seeds"] = None
        ref_cfg["fitter"]["max_episode"] = self.refine_steps

        gd = GDFitter(ref_cfg)
        self._gd_ref = gd
        data_t = torch.as_tensor(self.estimator.get_data(), dtype=torch.float32, device=self.device)

        if self._model_type == "cylinder":
            def fwd(a):
                pts, meas = gd._cylinder_forward(torch.tanh(a))
                return pts, meas, None
        elif self._model_type == "character":
            def fwd(a):
                pts, meas = gd._character_forward(a)
                return pts, meas, None
        else:  # nurbs_surface
            def fwd(a):
                # a ∈ [-1,1] → [lb, ub] → control_points + weights
                lb_t = torch.as_tensor(gd.rule.lb, dtype=torch.float32, device=gd.device)
                ub_t = torch.as_tensor(gd.rule.ub, dtype=torch.float32, device=gd.device)
                trait_flat = lb_t + (ub_t - lb_t) * (a + 1.0) / 2.0
                xyz_size = gd.num_ctrl_u * gd.num_ctrl_v * 3
                control = trait_flat[:xyz_size].reshape(gd.num_ctrl_u, gd.num_ctrl_v, 3)
                weights = trait_flat[xyz_size:].reshape(gd.num_ctrl_u, gd.num_ctrl_v)
                pts = gd._sample_surface(control, weights)
                meas = gd._compute_measure(control, weights)
                return pts, meas, control

        mm_alpha = float(self.cfg.get("estimator", {}).get("regularization_factor", 1.05))
        smooth_w = float(self.cfg.get("estimator", {}).get("control_smoothness_penalty_factor", 0.0))
        data_scale = float(
            torch.linalg.norm(gd.data_max - gd.data_min).item()
        ) if self._model_type == "nurbs_surface" else 1.0
        self._refiner = AdamRefiner(
            forward_fn=fwd,
            data_tensor=data_t,
            data_resolution=self.data_res,
            lr=self.lr,
            max_steps=self.refine_steps,
            method=self.refine_method,
            device=self.device,
            alpha=mm_alpha,
            smoothness_weight=smooth_w,
            data_scale=data_scale,
        )

    # ── Multiprocess evaluation pool ────────────────────────────────────
    def _setup_mp_pool(self) -> None:
        """Create spawn-context process pool for parallel population evaluation.

        Each worker initializes its own CPU estimator via ``_mp_worker_initializer``
        to avoid GPU contention with the main-process refiner.
        """
        self._mp_config_pkl = cloudpickle.dumps(self.cfg)
        ctx = get_context("spawn")
        self._mp_pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_mp_worker_initializer,
            initargs=(self._mp_config_pkl,),
        )

    def _eval_batch_mp(self, actions: np.ndarray) -> np.ndarray:
        """Evaluate a population in parallel using the process pool."""
        if self._mp_pool is None:
            return np.array([self._eval_single(a) for a in actions], dtype=np.float32)
        results = self._mp_pool.map(_mp_eval_single, actions)
        return np.array(results, dtype=np.float32)

    # ── Evaluation bridge ──
    def _eval_single(self, action: np.ndarray) -> float:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.estimator.reset()
        self.estimator.current_dividing_level = -1
        self.estimator.parse(action=action)
        self.estimator.generate(current_dividing_level=-1)
        return float(self.estimator.get_score())

    # ── Optimization ──
    def optimize_instance(self) -> float:
        de = NL_SHADE(
            dim=self.action_dim,
            bounds=(-1.0, 1.0),
            pop_size=self.pop_size,
            min_pop_size=self.min_pop,
            seed=int(np.random.randint(0, 2**31)),
        )
        opt = MemeticOpt(
            dim=self.action_dim,
            bounds=(-1.0, 1.0),
            de=de,
            refiner=self._refiner,
            refine_every=self.refine_every,
            refine_top_k=self.refine_top_k,
            adaptive_K=self.adaptive_K,
        )

        # Real-time record callback
        sub = SubRecord(self.cfg, env_id=0)
        sub.data_cloud = self.record.data_cloud

        def _on_improve(evals, score, action):
            # Re-evaluate best action to set estimator state for record
            self._eval_single(action)
            sub.update(score, self.estimator)
            self.record.update(sub, 1)

        result = opt.optimize(eval_fn=self._eval_single, max_evals=self.max_evals,
                              batch_eval_fn=self._eval_batch_mp if self.num_workers > 1 else None,
                              record_cb=_on_improve,
                              verbose=True)

        # Final evaluation
        final = self._eval_single(result["best_action"])
        if final > result["best_score"]:
            result["best_score"] = final
        self.best_action_ = result["best_action"].copy()
        return result["best_score"]

    # ── Public API ──
    def fit(self) -> None:
        for i in range(self.cfg["fitter"].get("num_instances", 1)):
            self.record.token_index = i
            self.record.best_score = 0.0
            self.record.best_sub_record = -1
            self.estimator.instance_index = i
            self.estimator.reset()
            print(f"Fitting for the model instance {i} begins")
            score = self.optimize_instance()
            print(f"Fitting for the model instance {i} finished. Best Score: {score}\n")
        print("The Memetic Multi-Instance fitting is finished.")

    def close(self) -> None:
        if self._mp_pool is not None:
            self._mp_pool.terminate()
            self._mp_pool.join()
        if self._gd_ref is not None:
            self._gd_ref.close()
        self.record.close()
