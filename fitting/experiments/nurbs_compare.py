"""CS+MM vs GD+MM NURBS 鞍面拟合对比
    cd /home/m25lll/code/Fiting && python fitting/experiments/nurbs_compare.py
"""

import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / 'datasets/synthetic'
DATA_FILES = {
    'clean':   str(DATA_DIR / 'saddle_3k.ply'),
    'noise_low':  str(DATA_DIR / 'saddle_noise_low_3k.ply'),
    'noise_mid':  str(DATA_DIR / 'saddle_noise_mid_3k.ply'),
    'noise_high': str(DATA_DIR / 'saddle_noise_high_3k.ply'),
    'outlier': str(DATA_DIR / 'saddle_outlier_3k.ply'),
}


def build_config(data_file, algo):
    from core.estimator.mm_estimator import MeanMeasureEstimator
    from models.surface.nurbs_surface_rule import NURBSSurfaceRule
    from tools.data_tool import load_ply_data as load_data

    max_episode = 200000 if algo == 'cs' else 10000

    return dict(
        task_type='3d',
        data_file=data_file,
        seeds=None,
        model=dict(
            type='nurbs_surface',
            num_ctrl_u=6, num_ctrl_v=6,
            degree_u=3, degree_v=3,
            sample_u=60, sample_v=60,
            weight_lb=0.8, weight_ub=1.2,
        ),
        device=dict(train_device='cuda:1', cuda_deterministic=True),
        estimator=dict(
            type='mm',
            data_resolution=0.01,
            model_resolution=0.004,
            regularization_factor=1.2,
            early_rejection=True,
            use_faiss=True,
            data_file=data_file,
            rule_class=NURBSSurfaceRule,
            estimator_class=MeanMeasureEstimator,
            estimator_instance=None,
            load_data_fn=load_data,
        ),
        fitter=dict(
            algo_name=algo,
            num_instances=1,
            max_episode=max_episode,
            gd_lr=0.01,
            gd_lr_min_factor=0.1,
            gd_eval_interval=200,
            gd_data_batch_size=4096,  # batch 加速，0=全量
            gd_init='svd',
            gd_smoothness_weight=0.05,
            gd_patience=10,  # 10×200=2000步不涨就停
            num_envs=4,
            episodes_per_env=50,
        ),
        record=dict(
            root_dir=f'/home/m25lll/code/Fiting/outputs/nurbs_saddle/{algo}/',
            pulse_size=2000,
            verbose=True,
            trim_final_mesh=True,
        ),
    )


def run_one(data_label, data_file, algo):
    cfg = build_config(data_file, algo)
    print(f'\n{"="*60}')
    print(f'{algo.upper()}+MM  {data_label}: {Path(data_file).name}')
    print(f'{"="*60}')

    if algo == 'cs':
        from core.optimizer.cs_fitter import Fitter
    else:
        from core.optimizer.gd_fitter import Fitter

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

    print('\n' + '=' * 70)
    print('NURBS 鞍面 MM 评分对比')
    print('=' * 70)
    print(f"{'Dataset':<16} {'CS+MM':>12} {'GD+MM':>12} {'Δ(GD-CS)':>12}")
    print('-' * 52)
    for data_label in DATA_FILES:
        cs = results.get(('cs', data_label), float('nan'))
        gd = results.get(('gd', data_label), float('nan'))
        print(f"{data_label:<16} {cs:12.4f} {gd:12.4f} {gd-cs:+12.4f}")
    print('=' * 70)


if __name__ == '__main__':
    main()
