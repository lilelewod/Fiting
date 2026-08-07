"""Particle swarm optimizer baseline for black-box model fitting."""

from copy import deepcopy
import pickle

import numpy as np

from core.collector import Collector
from core.record import Record
from tools.tool import get_seeds, init_device, set_seed
from tools.superquadric_initialization import guided_population


class Fitter:
    """PSO baseline using the same normalized action space and evaluator as CCO."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = int(cfg['fitter']['num_envs'])
        seeds = cfg.get('seeds')
        if seeds is None:
            seeds = get_seeds(self.num_envs + 1)
            cfg['seeds'] = seeds
            cfg['raw_seeds'] = None
        set_seed(seeds[-1])

        self.device = init_device(cfg['device'])
        cfg['raw_device'] = deepcopy(cfg['device'])
        cfg['device'] = self.device
        self.collector = Collector(cfg, self.num_envs)
        self.action_dim = self.collector.get_action_dim()
        self.episodes_per_env = int(cfg['fitter']['episodes_per_env'])
        self.population_size = self.num_envs * self.episodes_per_env
        if self.population_size < 4:
            raise ValueError('PSO comparison requires a population of at least 4')

        data_cloud = self.collector.launch()
        self.record = Record(cfg, dimension=data_cloud.shape[1])
        self.record.data_cloud = data_cloud
        self.record.action_dim = self.action_dim

    def estimate(self, solutions):
        if solutions.shape != (self.population_size, self.action_dim):
            raise ValueError('PSO always evaluates one complete population')
        scores = np.zeros(self.population_size)
        for env_id in range(self.num_envs):
            begin = env_id * self.episodes_per_env
            end = begin + self.episodes_per_env
            self.collector.estimate(env_id=env_id, actions=solutions[begin:end])
        for env_id in range(self.num_envs):
            begin = env_id * self.episodes_per_env
            end = begin + self.episodes_per_env
            try:
                scores[begin:end], record = self.collector.receive(env_id)
                self.record.update(record, self.episodes_per_env)
            except pickle.UnpicklingError:
                raise RuntimeError('Failed to receive PSO fitness values')
        return scores

    def optimize_instance(self):
        n, dim = self.population_size, self.action_dim
        budget = int(self.cfg['fitter']['max_episode'])
        if budget < n:
            raise ValueError('Function-evaluation budget must be at least the population size')

        lower = np.full(dim, -1.0)
        upper = np.full(dim, 1.0)
        if self.cfg['fitter'].get('pso_guided_initialization', False):
            rule = self.collector.estimator.rule
            if self.cfg.get('model', {}).get('type') != 'superquadric' or dim != 11:
                raise ValueError('PSO guided initialization currently supports only 11D superquadrics')
            rule._init_bounds()
            position, initialization_info = guided_population(
                self.record.data_cloud,
                rule.lb,
                rule.ub,
                population_size=n,
                rng=np.random,
                guided_fraction=float(self.cfg['fitter'].get('pso_guided_fraction', 0.75)),
                jitter=float(self.cfg['fitter'].get('pso_guided_jitter', 0.04)),
                extent_quantile=float(self.cfg['fitter'].get('pso_guided_extent_quantile', 0.005)),
                support_fraction=float(self.cfg['fitter'].get('pso_guided_support_fraction', 1.0)),
                support_neighbors=int(self.cfg['fitter'].get('pso_guided_support_neighbors', 8)),
            )
            self.record.guided_initialization = initialization_info
            print(f'PSO guided initialization: {initialization_info}')
        elif (
            self.cfg['fitter'].get('template_guided_initialization', False)
            or self.cfg['fitter'].get('pso_template_guided_initialization', False)
        ):
            guided_fraction = float(
                self.cfg['fitter'].get(
                    'template_guided_fraction',
                    self.cfg['fitter'].get('pso_template_guided_fraction', 0.5),
                )
            )
            guided_sigma = float(
                self.cfg['fitter'].get(
                    'template_guided_sigma',
                    self.cfg['fitter'].get('pso_template_guided_sigma', 0.15),
                )
            )
            if not 0.0 < guided_fraction <= 1.0:
                raise ValueError('pso_template_guided_fraction must be in (0, 1]')
            if guided_sigma <= 0.0:
                raise ValueError('pso_template_guided_sigma must be positive')
            guided_count = max(1, min(n, int(round(n * guided_fraction))))
            position = np.random.uniform(lower, upper, size=(n, dim)).astype(np.float32)
            position[0] = 0.0
            if guided_count > 1:
                position[1:guided_count] = np.clip(
                    np.random.normal(0.0, guided_sigma, size=(guided_count - 1, dim)),
                    lower,
                    upper,
                ).astype(np.float32)
            initialization_info = {
                'mode': 'template_zero_action_with_gaussian_neighborhood',
                'guided_count': guided_count,
                'random_count': n - guided_count,
                'guided_fraction': guided_fraction,
                'guided_sigma': guided_sigma,
            }
            self.record.guided_initialization = initialization_info
            print(f'PSO template-guided initialization: {initialization_info}')
        else:
            position = np.random.uniform(lower, upper, size=(n, dim)).astype(np.float32)
        velocity_scale = float(self.cfg['fitter'].get('pso_velocity_scale', 0.1))
        velocity = np.random.uniform(-velocity_scale, velocity_scale, size=(n, dim)).astype(np.float32)

        fitness = self.estimate(position)
        evaluations = n
        personal_position = position.copy()
        personal_fitness = fitness.copy()
        best_idx = int(np.argmax(fitness))
        best_position = position[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        inertia_start = float(self.cfg['fitter'].get('pso_inertia_start', 0.9))
        inertia_end = float(self.cfg['fitter'].get('pso_inertia_end', 0.4))
        cognitive = float(self.cfg['fitter'].get('pso_cognitive', 2.0))
        social = float(self.cfg['fitter'].get('pso_social', 2.0))
        velocity_limit = float(self.cfg['fitter'].get('pso_velocity_limit', 0.2))

        while evaluations + n <= budget:
            progress = evaluations / max(budget, 1)
            inertia = inertia_start + (inertia_end - inertia_start) * progress
            r1 = np.random.random((n, dim))
            r2 = np.random.random((n, dim))
            velocity = (
                inertia * velocity
                + cognitive * r1 * (personal_position - position)
                + social * r2 * (best_position - position)
            )
            velocity = np.clip(velocity, -velocity_limit, velocity_limit)
            position = np.clip(position + velocity, lower, upper).astype(np.float32)
            fitness = self.estimate(position)
            evaluations += n

            improved = fitness >= personal_fitness
            personal_fitness[improved] = fitness[improved]
            personal_position[improved] = position[improved]
            best_idx = int(np.argmax(personal_fitness))
            if personal_fitness[best_idx] >= best_fitness:
                best_fitness = float(personal_fitness[best_idx])
                best_position = personal_position[best_idx].copy()
            print(f'Evaluations: {evaluations}/{budget}, Best Score: {best_fitness:.4f}', end='\r', flush=True)

        self.best_action_ = best_position
        self.evaluations_ = evaluations
        self.record.num_evaluations = evaluations
        return best_fitness

    def fit(self):
        for i in range(self.cfg['fitter']['num_instances']):
            self.record.token_index = i
            self.record.best_score = 0.0
            self.record.best_sub_record = -1
            self.record.base_cloud = None
            self.record.base_color = None
            if i > 0:
                self.collector.update(self.record)
            print(f'Fitting for the model instance {i} begins')
            best_score = self.optimize_instance()
            print(f'\nFitting for the model instance {i} finished. Best Score: {best_score}\n')
        print('The PSO Multi-Instance fitting is finished.')

    def close(self):
        self.collector.close()
        self.record.close()
