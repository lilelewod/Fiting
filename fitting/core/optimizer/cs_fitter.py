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

    def optimize_instance(self):
        dim = self.action_dim
        lower_bound = np.full(dim, -1.)
        upper_bound = np.full(dim, 1.)
        n = self.population_size
        pa = 0.25

        nest = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            nest[i, :] = lower_bound + (upper_bound - lower_bound) * np.random.random_sample(dim)

        max_iteration = self.cfg['fitter']['max_episode']
        iteration = 0
        
        # 初始化当前种群的得分
        fitness = self.estimate(solutions=nest)
        iteration += n

        while iteration < max_iteration:
            # 获取当前最优
            best_nest = nest[np.argmax(fitness), :]

            # --- 第一阶段：莱维飞行 ---
            new_nest = get_cuckoos(nest, best_nest, lower_bound, upper_bound)
            new_scores = self.estimate(solutions=new_nest)
            # 贪心选择：只保留更好的解
            mask = new_scores >= fitness
            fitness[mask] = new_scores[mask]
            nest[mask, :] = new_nest[mask, :]
            iteration += n

            if iteration >= max_iteration:
                break

            # 重新获取当前最优 (因为上一步可能更新了)
            best_nest = nest[np.argmax(fitness), :]

            # --- 第二阶段：发现并丢弃外来蛋 ---
            new_nest = empty_nests(nest, lower_bound, upper_bound, pa)
            new_scores = self.estimate(solutions=new_nest)
            # 再次贪心选择：只保留更好的解
            mask = new_scores >= fitness
            fitness[mask] = new_scores[mask]
            nest[mask, :] = new_nest[mask, :]
            iteration += n

            info = (f'CS Evaluations: {iteration}/{max_iteration}, Best Score: {np.max(fitness):.4f}')
            print(info, end="\r", flush=True)

        return np.max(fitness)
        dim = self.action_dim
        lower_bound = np.full(dim, -1.)
        upper_bound = np.full(dim, 1.)
        n = self.population_size
        pa = 0.25

        nest = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            nest[i, :] = lower_bound + (upper_bound - lower_bound) * np.random.random_sample(dim)

        max_iteration = self.cfg['fitter']['max_episode']
        iteration = 0

        while iteration < max_iteration:
            # 1. 评估当前巢穴
            scores = self.estimate(solutions=nest)
            iteration += n

            # 2. 贪心获取最优
            best_nest = nest[np.argmax(scores), :]

            # 3. 莱维飞行 (获取新巢穴)
            nest = get_cuckoos(nest, best_nest, lower_bound, upper_bound)

            # 4. 评估新巢穴
            scores = self.estimate(solutions=nest)

            # 5. 发现并丢弃外来蛋 (概率 p_a)
            nest = empty_nests(nest, lower_bound, upper_bound, pa)
            iteration += n

            info = (f'CS Evaluations: {iteration}/{max_iteration}, Best Score: {np.max(scores):.4f}')
            print(info, end="\r", flush=True)

        return np.max(scores)

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
