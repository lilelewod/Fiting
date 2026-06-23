import numpy as np
import math
from copy import deepcopy
import pickle

from core.record import Record
from core.collector import Collector
from tools.tool import set_seed, init_device, get_seeds


def simple_bounds(s, lb, ub):
    index = s < lb
    s[index] = lb[index]
    index = s > ub
    s[index] = ub[index]
    return s


def get_cuckoos(nest, best, lb, ub):
    new_nest = deepcopy(nest)
    n = nest.shape[0]
    beta = 3 / 2
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    for i in range(n):
        s = nest[i, :]
        u = np.random.standard_normal(s.shape) * sigma
        v = np.random.standard_normal(s.shape)
        step = u / np.abs(v) ** (1 / beta)
        step_size = 0.01 * step * (s - best)
        s = s + step_size * np.random.standard_normal(s.shape)
        new_nest[i, :] = simple_bounds(s, lb, ub)
    return new_nest


def empty_nests(nest, lb, ub, pa):
    n = nest.shape[0]
    k = np.random.random_sample(nest.shape) > pa
    step_size = np.random.random_sample() * (nest[np.random.permutation(n), :] - nest[np.random.permutation(n), :])
    new_nest = nest + step_size * k
    for i in range(n):
        s = new_nest[i, :]
        new_nest[i, :] = simple_bounds(s, lb, ub)
    return new_nest


class Fitter:
    """
    基于 CS (Cuckoo Search) 的多实例拟合器 (Baseline)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = int(cfg['fitter']['num_envs'])
        seeds = self.cfg.get('seeds', None)
        if seeds is None:
            seeds = get_seeds(self.num_envs + 1)
            self.cfg['seeds'] = seeds
            self.cfg['raw_seeds'] = None
        set_seed(seeds[-1])

        self.device = init_device(cfg['device'])
        cfg['raw_device'] = deepcopy(cfg['device'])
        cfg['device'] = self.device

        self.collector = Collector(cfg, self.num_envs)
        self.action_dim = self.collector.get_action_dim()
        self.episodes_per_env = int(self.cfg['fitter']['episodes_per_env'])
        self.population_size = self.num_envs * self.episodes_per_env

        data_cloud = self.collector.launch()
        self.record = Record(cfg, dimension=data_cloud.shape[1])
        self.record.data_cloud = data_cloud

    def estimate(self, solutions):
        assert solutions.shape[0] == self.population_size
        scores = np.zeros(self.population_size)
        for env_id in range(self.num_envs):
            actions = solutions[env_id * self.episodes_per_env: (env_id + 1) * self.episodes_per_env]
            self.collector.estimate(env_id=env_id, actions=actions)
        for env_id in range(self.num_envs):
            try:
                scores[env_id * self.episodes_per_env: (
                                                                   env_id + 1) * self.episodes_per_env], record = self.collector.receive(
                    env_id)
                self.record.update(record, self.episodes_per_env)
            except pickle.UnpicklingError as e:
                assert False
        return scores

    def _get_best_nest(self, nest, new_nest, fitness):
        """张老师原版 get_best_nest: 评估+贪心选择"""
        scores = self.estimate(solutions=new_nest)
        mask = scores >= fitness
        fitness[mask] = scores[mask]
        nest[mask, :] = new_nest[mask, :]
        max_index = np.argmax(fitness)
        return fitness[max_index], nest[max_index, :], nest, fitness

    def optimize_instance(self):
        """张老师原版 CS: initialize → cuckoo → empty_nests 循环"""
        dim = self.action_dim
        lb = np.full(dim, -1.)
        ub = np.full(dim, 1.)
        n = self.population_size
        pa = 0.25

        # initialize — 支持 warm-start: 从给定 action 邻域初始化种群
        warm_action = self.cfg['fitter'].get('cs_warm_start_action', None)
        warm_noise = float(self.cfg['fitter'].get('cs_warm_start_noise', 0.05))

        nest = np.zeros((n, dim), dtype=np.float32)
        if warm_action is not None:
            warm_action = np.asarray(warm_action, dtype=np.float32)
            for i in range(n):
                nest[i, :] = np.clip(warm_action + warm_noise * np.random.randn(dim), -1.0, 1.0)
            print(f'[CS] warm-start from action (noise={warm_noise}), norm={np.linalg.norm(warm_action):.3f}')
        else:
            for i in range(n):
                nest[i, :] = lb + (ub - lb) * np.random.random_sample(dim)
        fitness = self.estimate(solutions=nest)

        best_score = 0.
        best_action = nest[np.argmax(fitness), :].copy()  # 初始最优
        max_estimations = self.cfg['fitter']['max_episode']
        estimations = n

        while estimations < max_estimations:
            best_nest = nest[np.argmax(fitness), :]
            new_nest = get_cuckoos(nest, best_nest, lb, ub)
            f_new, best_nest, nest, fitness = self._get_best_nest(nest, new_nest, fitness)
            new_nest = empty_nests(nest, lb, ub, pa)
            f_new, best_nest, nest, fitness = self._get_best_nest(nest, new_nest, fitness)
            estimations += 2 * n
            if f_new > best_score:
                best_score = f_new
                best_action = best_nest.copy()  # 持久化最优nest

        self.best_action_ = best_action  # 供GD warm-start获取
        return best_score

    def fit(self):
        for i in range(self.cfg['fitter']['num_instances']):
            self.record.token_index = i
            print(f'Fitting for the model instance {i} begins')
            best_score = self.optimize_instance()
            print(f'\nFitting for the model instance {i} finished. Best Score: {best_score}\n\n')
            self.collector.update(self.record)
        print('The CS Multi-Instance fitting is finished.')

    def close(self):
        self.collector.close()
        self.record.close()
