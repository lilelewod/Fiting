"""Differential evolution baseline for black-box model fitting."""

from copy import deepcopy
import pickle

import numpy as np

from core.collector import Collector
from core.record import Record
from tools.tool import get_seeds, init_device, set_seed


class Fitter:
    """DE/rand/1/bin with greedy selection and a fixed evaluation budget."""

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
            raise ValueError('DE requires a population of at least 4')

        data_cloud = self.collector.launch()
        self.record = Record(cfg, dimension=data_cloud.shape[1])
        self.record.data_cloud = data_cloud

    def estimate(self, solutions):
        if solutions.shape != (self.population_size, self.action_dim):
            raise ValueError('DE always evaluates one complete population')
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
                raise RuntimeError('Failed to receive DE fitness values')
        return scores

    def optimize_instance(self):
        n, dim = self.population_size, self.action_dim
        budget = int(self.cfg['fitter']['max_episode'])
        if budget < n:
            raise ValueError('Function-evaluation budget must be at least the population size')

        population = np.random.uniform(-1.0, 1.0, size=(n, dim)).astype(np.float32)
        fitness = self.estimate(population)
        evaluations = n
        differential_weight = float(self.cfg['fitter'].get('de_weight', 0.5))
        crossover_rate = float(self.cfg['fitter'].get('de_crossover_rate', 0.9))

        best_idx = int(np.argmax(fitness))
        best_position = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        while evaluations + n <= budget:
            trials = np.empty_like(population)
            indices = np.arange(n)
            for i in range(n):
                candidates = indices[indices != i]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                mutant = population[r1] + differential_weight * (population[r2] - population[r3])
                mutant = np.clip(mutant, -1.0, 1.0)
                mask = np.random.random(dim) < crossover_rate
                mask[np.random.randint(dim)] = True
                trials[i] = np.where(mask, mutant, population[i])

            trial_fitness = self.estimate(trials)
            evaluations += n
            improved = trial_fitness >= fitness
            population[improved] = trials[improved]
            fitness[improved] = trial_fitness[improved]
            best_idx = int(np.argmax(fitness))
            if fitness[best_idx] >= best_fitness:
                best_fitness = float(fitness[best_idx])
                best_position = population[best_idx].copy()
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
        print('The DE Multi-Instance fitting is finished.')

    def close(self):
        self.collector.close()
        self.record.close()
