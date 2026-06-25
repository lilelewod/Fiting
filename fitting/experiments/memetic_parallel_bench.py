"""Memetic 多进程并行评估加速比测试 — character (46D)

    cd /home/m25lll/code/Fiting/fitting && python experiments/memetic_parallel_bench.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.optimizer.memetic_fitter import Fitter


def build_cfg(num_workers):
    """构建字符测试配置"""
    with open(PROJECT_ROOT / 'configs/fit_character.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    from core.estimator.mm_estimator import MeanMeasureEstimator
    from models.character.character_rule import CharacterRule as Rule
    from tools.data_tool import load_image_data as load_data

    cfg['fitter']['algo_name'] = 'memetic'
    cfg['fitter']['num_instances'] = 1
    cfg['fitter']['num_envs'] = 4
    cfg['fitter']['max_episode'] = 2000
    cfg['fitter']['mem_pop_size'] = 100
    cfg['fitter']['mem_min_pop'] = 20
    cfg['fitter']['mem_refine_every'] = 100  # 几乎不做refine
    cfg['fitter']['mem_refine_steps'] = 10
    cfg['fitter']['mem_refine_method'] = 'adam'
    cfg['fitter']['mem_refine_top_k'] = 1
    cfg['fitter']['mem_adaptive_K'] = False
    cfg['fitter']['mem_num_workers'] = num_workers
    cfg['fitter']['gd_lr'] = 0.01

    run_id = cfg['run_id']; test_id = cfg['test_id']
    nt = cfg['noise_type']; nl = cfg['noise_level']
    cfg['rule']['token_file'] = f"datasets/character/test/run{run_id}_test{test_id}_1.mat"
    cfg['estimator']['data_file'] = f"datasets/character/test/{nt}/{nl}/{test_id - 1}/noisy_{run_id}.png"
    cfg['record']['root_dir'] = f"/home/m25lll/code/Fiting/outputs/memetic_bench/{num_workers}w/"
    cfg['estimator']['rule_class'] = Rule
    cfg['estimator']['estimator_class'] = MeanMeasureEstimator
    cfg['estimator']['estimator_instance'] = None
    cfg['estimator']['load_data_fn'] = load_data
    cfg['record']['visualization'] = None
    return cfg


def benchmark_population_eval(num_workers, n_repeat=5):
    """单独测试种群评估速度 (不跑完整优化)"""
    cfg = build_cfg(num_workers)

    # 创建 fitter (含mp pool初始化)
    t0 = time.perf_counter()
    fitter = Fitter(cfg)
    init_time = time.perf_counter() - t0

    # 生成随机种群
    rng = np.random.default_rng(42)
    population = rng.uniform(-1, 1, (100, fitter.action_dim)).astype(np.float32)

    # 预热 (触发worker创建和首次evaluation)
    _ = fitter._eval_batch_mp(population[:min(num_workers, 4)])

    # 计时
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = fitter._eval_batch_mp(population)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    fitter.close()
    return np.mean(times), np.std(times), init_time


def main():
    print("=" * 60)
    print("Memetic 多进程并行加速比 — Character (46D)")
    print("=" * 60)
    print(f"{'Workers':>8}  {'Eval time (s)':>14}  {'Speedup':>8}  {'Init (s)':>10}")
    print("-" * 50)

    workers_list = [1, 2, 4, 8]
    results = {}

    for w in workers_list:
        mean_t, std_t, init_t = benchmark_population_eval(w, n_repeat=3)
        results[w] = (mean_t, std_t, init_t)

    baseline = results[1][0]

    for w in workers_list:
        mean_t, std_t, init_t = results[w]
        speedup = baseline / mean_t if mean_t > 0 else float('inf')
        print(f"{w:>8}  {mean_t:>10.2f} ± {std_t:.2f}  {speedup:>7.2f}x  {init_t:>8.2f}")

    print("-" * 50)
    print(f"测试: 100个随机action的MM评估 (字符46D, CPU workers, spawn context)")

    efficiency = {}
    for w, (mean_t, _, _) in results.items():
        efficiency[w] = baseline / (mean_t * w) if mean_t > 0 else 0
    print(f"\n并行效率 (ideal=1.0):")
    for w in workers_list:
        print(f"  {w} workers: {efficiency[w]:.2f}")


if __name__ == '__main__':
    main()
