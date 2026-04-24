import numpy as np
import math
from copy import deepcopy
import pickle
from core.record import Record
from core.collector import Collector
from tools.tool import set_seed, init_device, get_seeds


def simple_bounds(s, lb, ub):
    """边界约束函数：将越界的参数强行拉回边界内"""
    index = s < lb
    s[index] = lb[index]
    index = s > ub
    s[index] = ub[index]
    return s


def lev(n, m):
    """CCO 算法专属：计算莱维飞行步长"""
    Beta = 1.5
    num = math.gamma(1 + Beta) * np.sin(np.pi * Beta / 2)
    den = math.gamma((1 + Beta) / 2) * Beta * 2 ** ((Beta - 1) / 2)
    sigma_u = (num / den) ** (1 / Beta)

    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    levy = 0.05 * u / (np.abs(v) ** (-Beta))
    return levy


class Fitter:
    """
    基于 CCO (Cuckoo Catfish Optimizer) 的多实例拟合器
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

        # 初始化 Collector 用于与底层环境交互评估
        self.collector = Collector(cfg, self.num_envs)
        self.action_dim = self.collector.get_action_dim()
        self.episodes_per_env = int(self.cfg['fitter']['episodes_per_env'])
        self.population_size = self.num_envs * self.episodes_per_env

        data_cloud = self.collector.launch()
        self.record = Record(cfg, dimension=data_cloud.shape[1])
        self.record.data_cloud = data_cloud

    def estimate(self, solutions):
        """并行批量评估种群的适应度 (NPRE 得分)"""
        assert solutions.shape[0] == self.population_size
        scores = np.zeros(self.population_size)

        # 分发任务
        for env_id in range(self.num_envs):
            actions = solutions[env_id * self.episodes_per_env: (env_id + 1) * self.episodes_per_env]
            self.collector.estimate(env_id=env_id, actions=actions)

        # 接收并行计算结果
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
        """核心 CCO 算法：用于寻找单一模型实例的最优参数"""
        dim = self.action_dim
        # 在连续参数空间中，框架将其统一归一化为 [-1, 1]
        lower_bound = np.full(dim, -1.)
        upper_bound = np.full(dim, 1.)
        nPop = self.population_size

        # --- 1. CCO 种群初始化 ---
        PopPos = np.zeros((nPop, dim), dtype=np.float32)
        for i in range(nPop):
            PopPos[i, :] = lower_bound + (upper_bound - lower_bound) * np.random.random_sample(dim)

        # 预计算 CCO 螺旋搜索路径的基准参数 x, y
        alpha, beta_param = 1.34, 0.3
        x, y = np.zeros(nPop), np.zeros(nPop)
        for i in range(nPop):
            theta = (1 - 10 * (i + 1) / nPop) * np.pi
            r = alpha * np.exp(beta_param * theta / 3)
            x[i], y[i] = r * np.cos(theta), r * np.sin(theta)

        max_iteration = self.cfg['fitter']['max_episode']
        iteration = 0

        # 初始批量评估
        PopFit = self.estimate(solutions=PopPos)
        iteration += nPop

        best_idx = np.argmax(PopFit)
        BestF = PopFit[best_idx]
        BestX = PopPos[best_idx, :].copy()

        gen = 0
        s_counter, z_counter, t_counter = 0, 0, 0
        eps = np.finfo(float).eps

        # --- 2. CCO 迭代主循环 ---
        while iteration < max_iteration:
            gen += 1
            current_ratio = iteration / max_iteration
            C = 1 - current_ratio
            T = (1 - (np.sin((np.pi * current_ratio) / 2))) ** max(current_ratio, eps)

            # 动态死亡淘汰概率
            if t_counter < 15:
                die = 0.02 * T
            else:
                die = 0.02
                C = 0.8

            windex = np.argmin(PopFit)
            WorstX = PopPos[windex, :].copy()
            Dis = np.abs(np.mean((PopPos - BestX) / (WorstX - BestX + eps)))
            Lx = np.abs(np.random.randn()) * np.random.rand()

            newPopPos = np.zeros_like(PopPos)
            Q_flags = np.zeros(nPop, dtype=int)

            # (A) CCO 策略生成新位置
            for i in range(nPop):
                F = np.sign(0.5 - np.random.rand())
                E = 1 * T + np.random.rand()
                R1, R4 = np.random.rand(dim), np.random.rand(dim)
                r1, r2 = np.random.rand(), np.random.rand()
                S_vec = np.sin(np.pi * R4 * C)

                k = np.random.permutation(nPop)
                PopPosrand, PopFitrand = PopPos[k, :], PopFit[k]

                if np.random.rand() > C:
                    J_i = np.abs(np.mean((PopPos[i, :] - BestX) / (WorstX - BestX + eps)))
                    if np.random.rand() > C:
                        Cy = 1 / (np.pi * (1 + C ** 2))
                        if J_i > Dis:
                            newPopPos[i, :] = BestX + F * S_vec * (BestX - PopPos[i, :])
                        else:
                            if Dis * Lx < J_i:
                                newPopPos[i, :] = BestX * (1 + T ** 5 * Cy * E) + F * (S_vec * (BestX - PopPos[i, :]))
                            else:
                                newPopPos[i, :] = BestX * (1 + T ** 5 * np.random.normal(0, C ** 2)) + F * (
                                            S_vec * (BestX - PopPos[i, :]))
                    else:
                        if np.random.rand() > C:
                            if (i + 1) % 2 == 1:
                                r3 = np.random.rand()
                                step = (BestX - E * PopPos[i, :])
                                newPopPos[i, :] = C / gen * (r1 * BestX - r3 * PopPos[i, :]) + T ** 2 * lev(1,
                                                                                                            dim).flatten() * np.abs(
                                    step)
                            else:
                                R2, R3 = np.random.rand(dim), np.random.rand(dim)
                                step = PopPos[i, :] - E * BestX
                                DE = C * F
                                newPopPos[i, :] = 0.5 * (BestX + PopPosrand[0, :]) + DE * (
                                            2 * R1 * step - R2 / 2 * (DE * R3 - 1))
                        else:
                            if np.random.rand() < np.random.rand():
                                if J_i < Dis:
                                    V = 2 * (np.random.rand() * (
                                                np.mean(PopPos, axis=0) - PopPos[i, :]) + np.random.rand() * (
                                                         BestX - PopPos[i, :]))
                                else:
                                    V = 2 * (np.random.rand() * (
                                                PopPosrand[1, :] - PopPosrand[2, :]) + np.random.rand() * (
                                                         PopPosrand[0, :] - PopPos[i, :]))

                                step = PopPos[i, :] - E * PopPosrand[i, :] if PopFit[i] >= PopFitrand[i] else \
                                PopPosrand[i, :] - E * PopPos[i, :]
                                base_pos = PopPos[i, :] if PopFit[i] >= PopFitrand[i] else PopPosrand[i, :]
                                xy_factor = y[i] if (i + 1) % 2 == 1 else x[i]

                                newPopPos[i, :] = (base_pos + T ** 2 * xy_factor * (1 - R1) * np.abs(
                                    step)) + F * R1 * step / 2 + V * J_i / gen

                                s_counter += 1
                                if s_counter > 10:
                                    idx1, idx2 = np.random.choice(nPop), np.random.choice(nPop)
                                    lesp1 = r1 * PopPos[idx1, :] + (1 - r1) * PopPos[idx2, :]
                                    newPopPos[i, :] = np.round(lesp1) + F * r1 * R1 / (gen ** 4) * newPopPos[i, :]
                                    s_counter = 0
                            else:
                                index = np.argsort(PopFit)[::-1]
                                A2, A1 = np.random.choice(4), np.random.choice(4)
                                B = np.vstack((PopPos[index[0:3], :], np.mean(PopPos, axis=0)))[A1, :]

                                Rt1, Rt2 = np.random.choice(360, dim,
                                                            replace=(dim > 360)) * np.pi / 360, np.random.choice(360,
                                                                                                                 dim,
                                                                                                                 replace=(
                                                                                                                             dim > 360)) * np.pi / 360
                                w = 1 - ((np.exp(current_ratio) - 1) / (np.exp(1) - 1)) ** 2

                                rand_val = np.random.rand()
                                if rand_val < 0.33:
                                    newPopPos[i, :] = B + 2 * w * F * np.cos(Rt1) * np.sin(Rt2) * (B - PopPos[i, :])
                                elif rand_val < 0.66:
                                    newPopPos[i, :] = B + 2 * w * F * np.sin(Rt1) * np.cos(Rt2) * (B - PopPos[i, :])
                                else:
                                    newPopPos[i, :] = B + 2 * w * F * np.cos(Rt2) * (B - PopPos[i, :])

                                if A2 == 3: Q_flags[i] = 1

                                z_counter += 1
                                if z_counter > 5:
                                    newPopPos[i, :] = BestX * (
                                                1 - (1 - 1 / (PopPos[np.random.choice(nPop), :] + eps)) * R1)
                                    z_counter = 0
                else:
                    if np.random.rand() > C:
                        if np.random.rand() > C:
                            newPopPos[i, :] = PopPosrand[2, :] + np.abs(np.random.randn()) * (
                                        BestX - PopPos[i, :] + PopPosrand[0, :] - PopPosrand[1, :])
                        else:
                            Z2 = (np.random.rand(dim) < np.random.rand()).astype(float)
                            newPopPos[i, :] = Z2 * (PopPosrand[2, :] + np.abs(np.random.randn()) * (
                                        PopPosrand[0, :] - PopPosrand[1, :])) + (1 - Z2) * PopPos[i, :]
                    else:
                        Z1 = float(np.random.rand() < np.random.rand())
                        newPopPos[i, :] = PopPos[i, :] + (Z1 * np.abs(np.random.randn()) * (
                                    (BestX + PopPosrand[0, :]) / 2 - PopPosrand[1, :]) + np.random.rand() / 2 * (
                                                                      PopPosrand[2, :] - PopPosrand[3, :]))

                    if np.random.rand() > C or t_counter > 0.8 * nPop:
                        mask = np.random.rand(dim) >= 0.2 * C + 0.2
                        newPopPos[i, mask] = PopPos[i, mask]

                # (B) CCO 阶段三：鲶鱼死亡机制
                if np.random.rand() < die:
                    if np.random.rand() > C:
                        newPopPos[i, :] = np.random.uniform(lower_bound, upper_bound)
                    else:
                        best = BestX * (lev(1, 1)[0, 0] * (r1 > r2) + np.abs(np.random.randn()) * (r1 <= r2))
                        newPopPos[i, :] = np.random.uniform(np.min(best), np.max(best), dim)

                newPopPos[i, :] = simple_bounds(newPopPos[i, :], lower_bound, upper_bound)

            # --- 3. 批量评估新种群 ---
            newPopFit = self.estimate(solutions=newPopPos)
            iteration += nPop
            info = (f'Evaluations: {iteration}/{max_iteration}, Best Score: {BestF:.4f}')
            print(info, end="\r", flush=True)

            # --- 4. 贪心选择与最优解更新 ---
            for i in range(nPop):
                if newPopFit[i] >= PopFit[i]:
                    PopFit[i] = newPopFit[i]
                    PopPos[i, :] = newPopPos[i, :].copy()

                    if Q_flags[i] == 1:
                        index = np.argsort(PopFit)[::-1]
                        PopPos[index[-1], :] = PopPos[i, :].copy()
                        PopFit[index[-1]] = PopFit[i]
                    t_counter = 0
                else:
                    t_counter += 1

                if PopFit[i] >= BestF:
                    BestF = PopFit[i]
                    BestX = PopPos[i, :].copy()

        # 循环结束，返回找到的最优得分
        return BestF

    def fit(self):
        """顺序拟合主循环：拟合一个实例后，将其在点云中剔除，继续拟合下一个"""
        for i in range(self.cfg['fitter']['num_instances']):
            self.record.token_index = i
            print(f'Fitting for the model instance {i} begins')

            # 使用 CCO 对当前数据寻找最佳模型实例，并返回最佳得分
            best_score = self.optimize_instance()

            print(f'\nFitting for the model instance {i} finished. Best Score: {best_score}\n\n')

            # 更新底层采集器：剔除已找到的模型对应的数据点（解决重叠问题的关键机制）
            self.collector.update(self.record)

        print('The CCO Multi-Instance fitting is finished.')

    def close(self):
        self.collector.close()
        self.record.close()
