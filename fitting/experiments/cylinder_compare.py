"""CS+MM vs GD+MM 圆柱拟合对比 — Fiting 框架
    cd /home/m25lll/code/Fiting && python fitting/experiments/cylinder_compare.py
"""

import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # fitting/ 包目录
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.tool import current_timestamp, get_seeds, set_seed

DATA_DIR = PROJECT_ROOT / 'datasets/synthetic'
DATA_FILES = {
    'clean':   str(DATA_DIR / 'cylinder_4k.ply'),
    'noise':   str(DATA_DIR / 'cylinder_noise_4k.ply'),
    'outlier': str(DATA_DIR / 'cylinder_outlier_4k.ply'),
}


def build_config(data_file, algo, run_id=1):
    """构建与 fit_mm_compare.yaml 一致的配置"""
    timestamp = current_timestamp()

    # CS 需要更多评估次数（种群迭代），GD 是梯度步数
    if algo == 'cs':
        max_episode = 500000  # 大盘看收敛点
        lr = 0.01
        fitter_extra = dict(num_envs=4, episodes_per_env=50)
    elif algo == 'gd':
        max_episode = 5000    # 每次重启的梯度步数（少量，靠多起点+早停）
        lr = 0.01
        fitter_extra = dict(
            gd_init='svd',
            gd_smoothness_weight=0.05,
            gd_patience=15,
            gd_eval_interval=200,
            # ★ 低维GD改进
            gd_num_restarts=20,
            gd_restart_noise=0.3,
            gd_param_noise_std=0.02,
            gd_tau_schedule='plateau',
            gd_two_phase=True,
            gd_lr_restarts=3,
        )
    else:
        max_episode = 200000
        lr = 0.01
        fitter_extra = dict(num_envs=4, episodes_per_env=50)

    cfg = dict(
        task_type='3d',
        run_id=run_id,
        data_file=data_file,
        model=dict(
            type='cylinder',
            num_ctrl_u=6, num_ctrl_v=6,
            degree_u=3, degree_v=3,
            sample_u=60, sample_v=60,
            weight_lb=0.8, weight_ub=1.2,
        ),
        device=dict(
            train_device='cuda:1',
            cuda_deterministic=True,
        ),
        seeds=None,
        estimator=dict(
            type='mm',
            data_resolution=0.01,
            model_resolution=0.004,
            regularization_factor=1.2,
            incremental_coverage=True,
            outlier_distance_factor=2.5,
            outlier_penalty_factor=0.0,
            bbox_margin_factor=1.0,
            bbox_penalty_factor=0.0,
            overlap_penalty_factor=0.0,
            control_smoothness_penalty_factor=0.0,
            mm_bbox_penalty_factor=0.0,
            early_rejection=True,
            use_faiss=True,
        ),
        fitter=dict(
            algo_name=algo,
            num_instances=1,
            max_episode=max_episode,
            gd_lr=lr,
            gd_lr_min_factor=0.1,
            gd_eval_interval=200,
            gd_data_batch_size=0,
            **fitter_extra,
        ),
        record=dict(
            root_dir=f'/home/m25lll/code/Fiting/outputs/cylinder/{algo}/',
            pulse_size=2000,
            verbose=False,
            trim_final_mesh=True,
            timestamp=timestamp,
        ),
    )
    # 补充 estimator 需要的字段
    from core.estimator.mm_estimator import MeanMeasureEstimator
    from models.surface.cylinder_rule import CylinderRule
    from tools.data_tool import load_ply_data as load_data
    cfg['estimator']['data_file'] = data_file
    cfg['estimator']['rule_class'] = CylinderRule
    cfg['estimator']['estimator_class'] = MeanMeasureEstimator
    cfg['estimator']['estimator_instance'] = None
    cfg['estimator']['load_data_fn'] = load_data
    return cfg


def run_one(data_label, data_file, algo):
    """跑单组实验，返回 best_score"""
    cfg = build_config(data_file, algo)
    set_seed(get_seeds(1)[-1])

    if algo == 'cs':
        from core.optimizer.cs_fitter import Fitter
    elif algo == 'gd':
        from core.optimizer.gd_fitter import Fitter
    else:
        raise ValueError(f"Unknown algo: {algo}")

    print(f'\n{"="*60}')
    print(f'{algo.upper()}+MM  {data_label}: {Path(data_file).name}')
    print(f'{"="*60}')
    fitter = Fitter(deepcopy(cfg))
    fitter.fit()
    score = float(fitter.record.best_score)
    fitter.close()
    return score


def main():
    results = {}
    for data_label, data_file in DATA_FILES.items():
        for algo in ['cs', 'gd']:
            score = run_one(data_label, data_file, algo)
            results[(algo, data_label)] = score

    # 打印对比表
    print('\n' + '=' * 70)
    print('圆柱 MM 评分对比 — Fiting 框架')
    print('=' * 70)
    print(f"{'Dataset':<12} {'CS+MM':>12} {'GD+MM':>12} {'Δ(GD-CS)':>12}")
    print('-' * 48)
    for data_label in ['clean', 'noise', 'outlier']:
        cs_score = results.get(('cs', data_label), float('nan'))
        gd_score = results.get(('gd', data_label), float('nan'))
        delta = gd_score - cs_score
        print(f"{data_label:<12} {cs_score:12.4f} {gd_score:12.4f} {delta:+12.4f}")
    print('-' * 48)
    cs_avg = sum(results.get(('cs', d), 0) for d in ['clean', 'noise', 'outlier']) / 3
    gd_avg = sum(results.get(('gd', d), 0) for d in ['clean', 'noise', 'outlier']) / 3
    print(f"{'avg':<12} {cs_avg:12.4f} {gd_avg:12.4f} {gd_avg-cs_avg:+12.4f}")
    print('=' * 70)


if __name__ == '__main__':
    main()
