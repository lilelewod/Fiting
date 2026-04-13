import numpy as np
import math
import pickle
from copy import deepcopy

from core.record import Record
from core.collector import Collector
from tools.tool import set_seed, init_device, get_seeds


def levy(d):
    """莱维飞行 (Levy flight) 函数"""
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
            math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.power(np.abs(v), 1 / beta)
    return step


class Fitter:
    """
    基于 ALA (Artificial Lemming Algorithm) 的多实例拟合器
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
        """完全对标 CS 的并行评估逻辑"""
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
        """ALA 核心寻优逻辑"""
        dim = self.action_dim
        # 完美对标框架：搜索空间固定在 [-1, 1]
        lower_bound = np.full(dim, -1.)
        upper_bound = np.full(dim, 1.)
        n = self.population_size

        # 1. 种群初始化
        X = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            X[i, :] = lower_bound + (upper_bound - lower_bound) * np.random.random_sample(dim)

        max_evaluations = self.cfg['fitter']['max_episode']
        evaluations = 0

        # 评估初始种群
        fitness = self.estimate(solutions=X)
        evaluations += n

        # 记录全局最优
        best_idx = np.argmax(fitness)
        Position = X[best_idx].copy()
        Score = fitness[best_idx]

        max_iter = max_evaluations // n
        current_iter = 1

        while evaluations < max_evaluations:
            RB = np.random.randn(n, dim)
            F = np.random.choice([1, -1])

            # ALA 核心时间动态参数
            theta = 2 * math.atan(1 - current_iter / max_iter)
            Xnew = np.zeros((n, dim), dtype=np.float32)

            for i in range(n):
                E = 2 * math.log(1 / max(np.random.rand(), 1e-10)) * theta

                if E > 1:
                    if np.random.rand() < 0.3:
                        r1 = 2 * np.random.rand(dim) - 1
                        rand_idx = np.random.randint(n)
                        Xnew[i] = Position + F * RB[i] * (r1 * (Position - X[i]) + (1 - r1) * (X[i] - X[rand_idx]))
                    else:
                        r2 = np.random.rand() * (1 + math.sin(0.5 * current_iter))
                        rand_idx = np.random.randint(n)
                        Xnew[i] = X[i] + F * r2 * (Position - X[rand_idx])
                else:
                    if np.random.rand() < 0.5:
                        radius = np.sqrt(np.sum((Position - X[i]) ** 2))
                        r3 = np.random.rand()
                        spiral = radius * (math.sin(2 * math.pi * r3) + math.cos(2 * math.pi * r3))
                        Xnew[i] = Position + F * X[i] * spiral * np.random.rand()
                    else:
                        G = 2 * np.sign(np.random.rand() - 0.5) * (1 - current_iter / max_iter)
                        Xnew[i] = Position + F * G * levy(dim) * (Position - X[i])


            # 越界修正
            # Xnew = np.clip(Xnew, lower_bound, upper_bound)

            out_of_bounds = (Xnew < lower_bound) | (Xnew > upper_bound)
            random_respawn = lower_bound + (upper_bound - lower_bound) * np.random.rand(n, dim)
            Xnew = np.where(out_of_bounds, random_respawn, Xnew)

            # 评估新种群
            newPopfit = self.estimate(solutions=Xnew)
            evaluations += n
            current_iter += 1

            # 贪心更新种群 (求最大化，保留更高分)
            better_mask = newPopfit > fitness
            X[better_mask] = Xnew[better_mask]
            fitness[better_mask] = newPopfit[better_mask]

            # 更新全局最优
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > Score:
                Position = X[current_best_idx].copy()
                Score = fitness[current_best_idx]

            info = (f'ALA Evaluations: {evaluations}/{max_evaluations}, Best Score: {Score:.4f}')
            print(info, end="\r", flush=True)

        return Score

    def fit(self):
        for i in range(self.cfg['fitter']['num_instances']):
            print(f'Fitting for the model instance {i} begins')
            best_score = self.optimize_instance()
            print(f'\nFitting for the model instance {i} finished. Best Score: {best_score}\n\n')
            self.collector.update(self.record.best_estimator)
        print('The ALA Multi-Instance fitting is finished.')

    def close(self):
        self.collector.close()
        self.record.close()