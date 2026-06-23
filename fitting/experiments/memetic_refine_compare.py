"""Memetic 局部精调对比: Adam vs L-BFGS vs Cascade(Adam+L-BFGS)

    cd /home/m25lll/code/Fiting && python fitting/experiments/memetic_refine_compare.py
"""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.estimator.mm_estimator import MeanMeasureEstimator
from models.surface.cylinder_rule import CylinderRule
from tools.data_tool import load_ply_data as load_data
from core.optimizer.memetic_fitter import Fitter


BASE_CFG = dict(
    task_type='3d',
    run_id=1,
    data_file=str(PROJECT_ROOT / 'datasets/synthetic/cylinder_4k.ply'),
    model=dict(type='cylinder'),
    device=dict(train_device='cuda:1', cuda_deterministic=True),
    seeds=None,
    estimator=dict(
        type='mm',
        data_resolution=0.01,
        model_resolution=0.004,
        regularization_factor=1.2,
        early_rejection=True,
        use_faiss=True,
        data_file=str(PROJECT_ROOT / 'datasets/synthetic/cylinder_4k.ply'),
        rule_class=CylinderRule,
        estimator_class=MeanMeasureEstimator,
        estimator_instance=None,
        load_data_fn=load_data,
    ),
    fitter=dict(
        algo_name='memetic',
        num_instances=1,
        max_episode=20000,            # 之前确认的收敛点
        mem_pop_size=200,
        mem_min_pop=30,
        mem_refine_every=10,
        mem_refine_steps=100,
        num_envs=4,
        episodes_per_env=50,
    ),
    record=dict(
        pulse_size=2000,
        verbose=False,
        trim_final_mesh=True,
    ),
)

METHODS = ['lbfgs', 'cascade']
N_RUNS = 3  # 补跑: lbfgs需3, cascade需4(先跑3)


def run_one(method, run_idx):
    cfg = deepcopy(BASE_CFG)
    cfg['fitter']['mem_refine_method'] = method
    cfg['record']['root_dir'] = str(
        PROJECT_ROOT.parent / f'outputs/memetic_refine_test/{method}/run_{run_idx}/')
    print(f'[{method}] run {run_idx+1}/{N_RUNS} ...', end=' ', flush=True)
    fitter = Fitter(cfg)
    fitter.fit()
    score = float(fitter.record.best_score)
    fitter.close()
    print(f'score={score:.4f}')
    return score


def main():
    results = {}
    for method in METHODS:
        print(f'\n{"="*50}')
        print(f'Testing: {method}')
        print(f'{"="*50}')
        scores = []
        for i in range(N_RUNS):
            scores.append(run_one(method, i))
        results[method] = np.array(scores)

    print('\n' + '=' * 60)
    print('Memetic Refine Method Comparison — Cylinder Clean')
    print('=' * 60)
    print(f'{"Method":>12}  {"Mean":>8}  {"Std":>8}  {"Min":>8}  {"Max":>8}')
    print('-' * 50)
    for method in METHODS:
        s = results[method]
        print(f'{method:>12}  {s.mean():>8.2f}  {s.std():>8.2f}  {s.min():>8.2f}  {s.max():>8.2f}')

    # 统计检验
    from scipy import stats
    print('\n--- pairwise t-test ---')
    for i, m1 in enumerate(METHODS):
        for m2 in METHODS[i+1:]:
            t, p = stats.ttest_ind(results[m1], results[m2])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f'{m1} vs {m2}: t={t:.2f}, p={p:.4f} {sig}')


if __name__ == '__main__':
    main()
