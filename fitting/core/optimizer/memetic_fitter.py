"""
Memetic Optimizer: NL-SHADE + Adam/L-BFGS

NL-SHADE (Non-Linear population size reduction SHADE):
  - current-to-pbest/1 mutation
  - external archive
  - adaptive F/CR memory (Success-History)
  - linear population size reduction
  - greedy selection

Local refinement:
  - Every K generations, refine the best individual
  - Adam or L-BFGS on soft MM loss
  - Refined individual replaces worst in population

References:
  - Tanabe & Fukunaga, "Success-History based Parameter Adaptation for DE" (SHADE), CEC 2013
  - Brest et al., "jSO" (single-objective real-parameter optimization), CEC 2017
  - Stanovov et al., "NL-SHADE-LBC" (Linear Bias Reduction), CEC 2018
  - Mohamed et al., "LSHADE-SPACMA" (SPSA + CMA-ES hybrid), CEC 2017
"""

from copy import deepcopy

import numpy as np
import torch

from core.record import Record, SubRecord
from tools.tool import get_seeds, init_device, set_seed


# ═══════════════════════════════════════════════════════════════════════
#  Archive
# ═══════════════════════════════════════════════════════════════════════

class Archive:
    """External archive for maintaining diversity in DE mutation.

    Stores replaced (inferior) individuals. Bounded circular buffer.
    When full, randomly replaces an existing entry.
    """

    def __init__(self, max_size, dim):
        self.max_size = max_size
        self.dim = dim
        self.data = np.zeros((max_size, dim), dtype=np.float32)
        self.size = 0

    def insert(self, individual):
        if self.size < self.max_size:
            self.data[self.size] = individual
            self.size += 1
        else:
            idx = np.random.randint(0, self.max_size)
            self.data[idx] = individual

    def sample(self, n):
        if self.size == 0:
            return None
        idx = np.random.randint(0, self.size, size=n)
        return self.data[idx]

    def clear(self):
        self.size = 0


# ═══════════════════════════════════════════════════════════════════════
#  NL-SHADE
# ═══════════════════════════════════════════════════════════════════════

class NLSHADE:
    """NL-SHADE: Success-History Adaptive DE with Linear Population Reduction.

    Key features:
    - current-to-pbest/1 mutation with archive
    - binomial crossover
    - adaptive F/CR via historical memory (SHADE)
    - linear population size reduction
    - greedy selection
    """

    def __init__(self, dim, bounds=(-1.0, 1.0), pop_size=200, memory_size=6,
                 p_best=0.11, archive_rate=1.4):
        self.dim = dim
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.pop_size = pop_size
        self.init_pop_size = pop_size
        self.memory_size = memory_size
        self.p_best = p_best  # fraction for p-best mutation
        self.archive_rate = archive_rate

        # SHADE memory: store successful F and CR values
        self.M_F = np.full(memory_size, 0.5, dtype=np.float32)   # F memory
        self.M_CR = np.full(memory_size, 0.5, dtype=np.float32)  # CR memory
        self.mem_idx = 0  # current memory position

        # Archive
        archive_max = int(round(pop_size * archive_rate))
        self.archive = Archive(archive_max, dim)

        # State
        self.population = None       # (pop_size, dim)
        self.fitness = None          # (pop_size,)
        self.generations = 0
        self.evaluations = 0
        self.best_idx = 0
        self._init_population()

    def _init_population(self):
        rng = np.random.default_rng()
        self.population = rng.uniform(self.lb, self.ub, (self.pop_size, self.dim)).astype(np.float32)

    def set_fitness(self, fitness):
        """Set fitness after external evaluation."""
        self.fitness = np.asarray(fitness, dtype=np.float32)
        self.best_idx = int(np.argmax(self.fitness))

    def _generate_F_CR(self, pop_size):
        """Generate F and CR for each individual from SHADE memory."""
        rng = np.random.default_rng()
        F = np.zeros(pop_size, dtype=np.float32)
        CR = np.zeros(pop_size, dtype=np.float32)

        for i in range(pop_size):
            ri = rng.integers(0, self.memory_size)
            # Sample from Cauchy (for F) and Normal (for CR) around memory values
            mu_F = self.M_F[ri]
            F[i] = mu_F + 0.1 * np.random.standard_cauchy()
            while F[i] <= 0:
                F[i] = mu_F + 0.1 * np.random.standard_cauchy()
            F[i] = min(F[i], 1.0)

            mu_CR = self.M_CR[ri]
            CR[i] = mu_CR + 0.1 * np.random.standard_normal()
            CR[i] = np.clip(CR[i], 0.0, 1.0)

        return F, CR

    def _update_memory(self, S_F, S_CR, delta_f):
        """Update SHADE memory with successful F/CR values."""
        if len(S_F) == 0:
            return

        # Lehmer mean for F, arithmetic mean for CR
        if np.sum(S_F) > 0:
            weight = delta_f / np.sum(delta_f) if np.sum(delta_f) > 0 else np.ones_like(delta_f) / len(delta_f)
            mean_F = np.sum(weight * S_F ** 2) / np.sum(weight * S_F) if np.sum(weight * S_F) > 0 else 0.5
            mean_CR = np.sum(weight * S_CR)

            self.M_F[self.mem_idx] = mean_F
            self.M_CR[self.mem_idx] = mean_CR
            self.mem_idx = (self.mem_idx + 1) % self.memory_size

    def _mutation_current_to_pbest(self, F):
        """current-to-pbest/1 mutation with archive."""
        pop_size = self.pop_size
        p_num = max(2, int(round(pop_size * self.p_best)))
        mutant = np.zeros_like(self.population)
        rng = np.random.default_rng()

        for i in range(pop_size):
            # Select p-best individual
            pbest_pool = np.argpartition(-self.fitness, p_num - 1)[:p_num]
            pbest_idx = rng.choice(pbest_pool)

            # Select r1 from population (r1 != i)
            candidates = [j for j in range(pop_size) if j != i]
            r1 = rng.choice(candidates)

            # Select r2 from population ∪ archive (r2 != i, r2 != r1)
            arch_sample = self.archive.sample(1)
            union = list(range(pop_size))
            if arch_sample is not None and self.archive.size > 0:
                union = list(range(pop_size)) + list(range(pop_size, pop_size + self.archive.size))

            while True:
                r2_pool = rng.choice(union, size=min(10, len(union)), replace=False)
                found = False
                for r2_raw in r2_pool:
                    if r2_raw < pop_size:
                        r2 = r2_raw
                    else:
                        r2 = -1  # flag for archive
                    if r2 != i and (r2 != r1 or r2 == -1):
                        found = True
                        break
                if found:
                    break

            if r2 >= 0 and r2 < pop_size:
                x_r2 = self.population[r2]
            elif arch_sample is not None and self.archive.size > 0:
                x_r2 = self.archive.sample(1)[0]
            else:
                x_r2 = self.population[rng.choice([j for j in range(pop_size) if j != i and j != r1])]

            x_pbest = self.population[pbest_idx]
            x_i = self.population[i]
            x_r1 = self.population[r1]

            # current-to-pbest/1
            mutant[i] = x_i + F[i] * (x_pbest - x_i) + F[i] * (x_r1 - x_r2)

        return mutant

    def _crossover(self, mutant, CR):
        """Binomial crossover."""
        pop_size = self.pop_size
        trial = np.copy(self.population)
        rng = np.random.default_rng()

        for i in range(pop_size):
            j_rand = rng.integers(0, self.dim)
            mask = rng.random(self.dim) < CR[i]
            mask[j_rand] = True
            trial[i, mask] = mutant[i, mask]

        # Boundary repair: reflect back
        below = trial < self.lb
        above = trial > self.ub
        trial[below] = self.lb + rng.random(np.sum(below)) * (self.ub - self.lb)
        trial[above] = self.lb + rng.random(np.sum(above)) * (self.ub - self.lb)

        return trial

    def generate_trials(self):
        """Generate trial vectors for one generation. Returns (trial_pop, F, CR)."""
        pop_size = self.pop_size
        F, CR = self._generate_F_CR(pop_size)
        mutant = self._mutation_current_to_pbest(F)
        trial = self._crossover(mutant, CR)
        return trial, F, CR

    def select(self, trial, trial_fitness, F, CR):
        """Greedy selection: replace if trial is better. Update archive and memory."""
        pop_size = self.pop_size
        S_F, S_CR, delta_f = [], [], []
        rng = np.random.default_rng()

        for i in range(pop_size):
            if trial_fitness[i] >= self.fitness[i]:
                # Success: archive old individual, update memory data
                self.archive.insert(self.population[i].copy())
                delta = trial_fitness[i] - self.fitness[i]
                S_F.append(F[i])
                S_CR.append(CR[i])
                delta_f.append(delta)
                self.population[i] = trial[i].copy()
                self.fitness[i] = trial_fitness[i]

        self._update_memory(np.array(S_F, dtype=np.float32),
                           np.array(S_CR, dtype=np.float32),
                           np.array(delta_f, dtype=np.float32))

        self.best_idx = int(np.argmax(self.fitness))
        self.generations += 1

    def reduce_population(self, target_pop):
        """Linear population size reduction. Remove worst individuals."""
        if target_pop >= self.pop_size:
            return
        n_remove = self.pop_size - target_pop
        worst_idx = np.argpartition(self.fitness, n_remove)[:n_remove]
        keep = np.ones(self.pop_size, dtype=bool)
        keep[worst_idx] = False
        self.population = self.population[keep]
        self.fitness = self.fitness[keep]
        self.pop_size = target_pop
        # Update archive max size
        self.archive = Archive(int(round(target_pop * self.archive_rate)), self.dim)

    def get_best(self):
        return self.population[self.best_idx].copy(), self.fitness[self.best_idx]

    def get_population(self):
        return self.population.copy()

    def get_worst_idx(self):
        return int(np.argmin(self.fitness))


# ═══════════════════════════════════════════════════════════════════════
#  LocalRefiner — Adam/L-BFGS on soft MM loss
# ═══════════════════════════════════════════════════════════════════════

class LocalRefiner:
    """Gradient-based local refinement using Adam or L-BFGS on soft MM loss.

    Requires a differentiable forward function: action → (model_points, measure).
    """

    def __init__(self, differentiable_forward, data_points, data_resolution,
                 max_steps=200, lr=0.01, method='adam', device='cpu'):
        """
        Args:
            differentiable_forward: fn(action_tensor) -> (points, measure)
            data_points: torch.Tensor (M, dim) target points
            data_resolution: float
            max_steps: refinement steps
            lr: learning rate
            method: 'adam' or 'lbfgs'
            device: torch device
        """
        self.forward_fn = differentiable_forward
        self.data = torch.as_tensor(data_points, dtype=torch.float32, device=device)
        self.data_res = data_resolution
        self.max_steps = max_steps
        self.lr = lr
        self.method = method
        self.device = device
        self.alpha = 1.2  # MM regularization factor

    def refine(self, action_np):
        """Refine action in [-1,1] space. Returns refined action, score.

        cascade mode: Adam (fast, momentum) → L-BFGS (precise, second-order).
        """
        if self.method == 'cascade':
            # Phase 1: Adam — 快速粗调
            action_np = self._refine_with(action_np, 'adam', self.max_steps // 2)
            # Phase 2: L-BFGS — 精确终调
            action_np = self._refine_with(action_np, 'lbfgs', self.max_steps // 2)
            return action_np
        return self._refine_with(action_np, self.method, self.max_steps)

    def _refine_with(self, action_np, method, steps):
        action = torch.nn.Parameter(
            torch.as_tensor(action_np, dtype=torch.float32, device=self.device))

        if method == 'lbfgs':
            optimizer = torch.optim.LBFGS([action], lr=self.lr, history_size=10,
                                          max_iter=20, line_search_fn='strong_wolfe')
        else:
            optimizer = torch.optim.Adam([action], lr=self.lr)

        best_loss = float('inf')
        best_action = action_np.copy()

        def closure():
            optimizer.zero_grad()
            model_pts, measure = self.forward_fn(action)
            loss = self._soft_mm_loss(model_pts, measure)
            loss.backward()
            return loss

        for step in range(steps):
            if method == 'lbfgs':
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                model_pts, measure = self.forward_fn(action)
                loss = self._soft_mm_loss(model_pts, measure)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                action.clamp_(-1.0, 1.0)
                loss_val = loss.item()
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_action = action.detach().cpu().numpy().copy()

        return np.clip(best_action, -1.0, 1.0).astype(np.float32)

    def _soft_mm_loss(self, model_points, model_measure):
        eps = 1e-8
        tau = self.data_res * 0.5  # small tau for local refinement

        diff = model_points.unsqueeze(1) - self.data.unsqueeze(0)
        dist = torch.sqrt((diff ** 2).sum(-1) + eps)

        logits = -dist / tau
        soft_assign = torch.softmax(logits, dim=1)
        soft_error = (soft_assign * dist).sum(dim=1).mean()

        safe_measure = model_measure.clamp(min=eps)
        return soft_error / (safe_measure ** self.alpha + eps)


# ═══════════════════════════════════════════════════════════════════════
#  MemeticFitter — NL-SHADE + Local Refinement
# ═══════════════════════════════════════════════════════════════════════

class Fitter:
    """Memetic Optimizer: NL-SHADE global search + Adam/L-BFGS local refinement.

    Integrates with the Fiting framework via the estimator API.
    Supports: NURBS surface, cylinder, character.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        seeds = self.cfg.get("seeds", None)
        if seeds is None:
            seeds = get_seeds(1)
            self.cfg["seeds"] = seeds
        set_seed(seeds[-1])

        self.device = init_device(cfg["device"])
        cfg["raw_device"] = deepcopy(cfg["device"])
        cfg["device"] = self.device

        self.estimator = self.cfg["estimator"]["estimator_class"](self.cfg)
        self.rule = self.estimator.rule

        data_cloud = self.estimator.get_data()
        self.dim = data_cloud.shape[1]
        self.record = Record(cfg, dimension=self.dim)
        self.record.data_cloud = data_cloud

        fitter_cfg = cfg["fitter"]
        model_cfg = cfg.get("model", {})

        # Memetic parameters
        self.max_evals = int(fitter_cfg.get("max_episode", 100000))
        self.init_pop_size = int(fitter_cfg.get("mem_pop_size", 200))
        self.min_pop_size = int(fitter_cfg.get("mem_min_pop", 20))
        self.refine_every = int(fitter_cfg.get("mem_refine_every", 10))  # generations
        self.refine_steps = int(fitter_cfg.get("mem_refine_steps", 100))
        self.refine_method = str(fitter_cfg.get("mem_refine_method", "adam"))
        self.data_batch_size = int(fitter_cfg.get("gd_data_batch_size", 0))
        self.refine_init = str(fitter_cfg.get("gd_init", "svd"))

        # Model type detection
        self._model_type = str(model_cfg.get('type', 'nurbs_surface')).lower()
        if 'CharacterRule' in type(self.rule).__name__:
            self._model_type = 'character'
        self.action_dim = self.estimator.num_variables()
        self.data_resolution = float(self.estimator.data_resolution)

        # Setup differentiable forward for local refinement
        self._target_points = None
        self._setup_refiner()

    def _setup_refiner(self):
        """Setup the differentiable forward function for the LocalRefiner."""
        if self._model_type not in ('cylinder', 'character'):
            self._refiner = None
            self._gd_ref = None
            return

        from core.optimizer.gd_fitter import Fitter as GDFitter

        # Restore device to dict form for GDFitter init
        ref_cfg = deepcopy(self.cfg)
        if 'raw_device' in ref_cfg:
            ref_cfg['device'] = ref_cfg['raw_device']
        ref_cfg['seeds'] = None
        ref_cfg['fitter']['max_episode'] = self.refine_steps

        gd = GDFitter(ref_cfg)
        self._gd_ref = gd
        pts = self.estimator.get_data()

        if self._model_type == 'cylinder':
            def forward_fn(action):
                return gd._cylinder_forward(torch.tanh(action))
        else:  # character
            def forward_fn(action):
                return gd._character_forward(action)

        self._refiner = LocalRefiner(
            forward_fn, pts, self.data_resolution,
            max_steps=self.refine_steps, method=self.refine_method,
            device=self.device)

    def _evaluate_population(self, population):
        """Evaluate a population of individuals using the MM estimator."""
        scores = np.zeros(len(population), dtype=np.float32)
        for i, action in enumerate(population):
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            self.estimator.reset()
            self.estimator.current_dividing_level = -1
            self.estimator.parse(action=action)
            self.estimator.generate(current_dividing_level=-1)
            scores[i] = float(self.estimator.get_score())
        return scores

    def _evaluate_single(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.estimator.reset()
        self.estimator.current_dividing_level = -1
        self.estimator.parse(action=action)
        self.estimator.generate(current_dividing_level=-1)
        return float(self.estimator.get_score())

    def _init_population(self):
        """Initialize population with SVD init + random perturbation."""
        rng = np.random.default_rng()
        pop = rng.uniform(-1.0, 1.0, (self.init_pop_size, self.action_dim)).astype(np.float32)

        # Try to seed with SVD/geometric init if available
        if self._model_type == 'cylinder' and self._gd_ref is not None:
            try:
                svd_action = self._gd_ref._init_cylinder_action(
                    np.asarray(self.estimator.get_data(), dtype=np.float32))
                svd_action = np.tanh(svd_action)  # raw → [-1,1]
                pop[0] = svd_action
                # Rest near SVD
                for i in range(1, min(20, self.init_pop_size)):
                    pop[i] = np.clip(svd_action + rng.normal(0, 0.2, self.action_dim), -1.0, 1.0)
            except Exception:
                pass

        return pop

    def optimize_instance(self):
        """Run memetic optimization: NL-SHADE + periodic local refinement."""
        target_evals = self.max_evals
        pop_size = self.init_pop_size
        min_pop = self.min_pop_size

        # Initialize
        de = NLSHADE(self.action_dim, bounds=(-1.0, 1.0),
                     pop_size=pop_size, p_best=0.11)
        population = self._init_population()
        de.population = population
        de.pop_size = pop_size

        # Initial evaluation
        fitness = self._evaluate_population(population)
        de.set_fitness(fitness)
        total_evals = pop_size

        sub_record = SubRecord(self.cfg, env_id=0)
        sub_record.data_cloud = self.record.data_cloud
        best_score = float(de.fitness[de.best_idx])
        best_action = de.population[de.best_idx].copy()

        # Target population sizes for linear reduction
        target_pops = np.linspace(pop_size, min_pop,
                                  num=max(1, target_evals // (2 * pop_size)), dtype=int)

        gen = 0
        while total_evals < target_evals:
            # ── NL-SHADE generation ──
            trial, F, CR = de.generate_trials()
            trial_fitness = self._evaluate_population(trial)
            total_evals += len(trial)
            de.select(trial, trial_fitness, F, CR)

            gen += 1

            # Update best
            if de.fitness[de.best_idx] > best_score:
                best_score = float(de.fitness[de.best_idx])
                best_action = de.population[de.best_idx].copy()

            # ── Local refinement every K generations ──
            if gen % self.refine_every == 0 and self._refiner is not None:
                curr_best, _ = de.get_best()
                refined = self._refiner.refine(curr_best)
                refined_score = self._evaluate_single(refined)
                total_evals += 1

                if refined_score > best_score:
                    best_score = refined_score
                    best_action = refined.copy()

                # Replace worst individual with refined
                worst_idx = de.get_worst_idx()
                de.population[worst_idx] = refined
                de.fitness[worst_idx] = refined_score

            # ── Linear population reduction ──
            target_pop = int(round(
                self.init_pop_size - (self.init_pop_size - min_pop) * total_evals / target_evals
            ))
            target_pop = max(min_pop, target_pop)
            if target_pop < de.pop_size:
                de.reduce_population(target_pop)

            # ── Logging ──
            sub_record.update(best_score, self.estimator)
            self.record.update(sub_record, len(trial) + 1)
            print(f"Memetic gen {gen}: evals={total_evals}/{target_evals}, "
                  f"pop={de.pop_size}, best={best_score:.4f}",
                  end="\r", flush=True)

        # Final evaluation of best
        final_score = self._evaluate_single(best_action)
        if final_score > best_score:
            best_score = final_score

        print(f"\nMemetic finished: best={best_score:.4f}, evals={total_evals}")
        return best_score

    def fit(self):
        for i in range(self.cfg["fitter"].get("num_instances", 1)):
            self.record.token_index = i
            self.record.best_score = 0.0
            self.record.best_sub_record = -1
            self.record.base_cloud = None
            self.record.base_color = None

            self.estimator.instance_index = i
            self.estimator.reset()

            print(f"Fitting for the model instance {i} begins")
            best_score = self.optimize_instance()
            print(f"\nFitting for the model instance {i} finished. Best Score: {best_score}\n")

        print("The Memetic Multi-Instance fitting is finished.")

    def close(self):
        if hasattr(self, '_gd_ref') and self._gd_ref is not None:
            self._gd_ref.close()
        self.record.close()
