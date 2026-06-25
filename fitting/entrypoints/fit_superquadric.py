#!/usr/bin/env python3
"""超二次曲面拟合 — 专用入口 (11D: center×3 + scale×3 + shape×2 + rotation×3)

用法:
    python entrypoints/fit_superquadric.py --shape ellipsoid                    # 走fit_superquadric.yaml
    python entrypoints/fit_superquadric.py --shape box --algo cco --runs 5
    python entrypoints/fit_superquadric.py --all --algo memetic --runs 3        # 全部5种形状
    python entrypoints/fit_superquadric.py --shape diamond --condition noise

配置文件: configs/fit_superquadric.yaml  (修改 algo、pop_size、max_episode 等)
命令行 --shape/--algo/--condition/--runs 会覆盖 config 对应字段
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.tool import current_timestamp, set_project_root_as_working_directory
set_project_root_as_working_directory(__file__)

from core.estimator.mm_estimator import MeanMeasureEstimator
from models.surface.superquadric_rule import SuperquadricRule
from tools.data_tool import load_ply_data as load_data

SHAPES = {
    'ellipsoid': 'superq_ellipsoid_3k.ply',
    'box':       'superq_box_3k.ply',
    'cylinder':  'superq_cylinder_3k.ply',
    'diamond':   'superq_diamond_3k.ply',
    'pillow':    'superq_pillow_3k.ply',
}

CONDITIONS = ['clean', 'noise', 'outlier']


def load_base_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_config(base_cfg: dict, shape_name: str, condition: str, algo: str, run_id: int) -> dict:
    """Merge YAML base config with CLI overrides and auto-detect paths."""
    cfg = deepcopy(base_cfg)

    # ── data file ──
    base_name = SHAPES[shape_name]
    if condition == 'clean':
        data_file = PROJECT_ROOT / 'datasets/synthetic' / base_name
    else:
        stem = base_name.replace('_3k.ply', f'_{condition}_3k.ply')
        data_file = PROJECT_ROOT / 'datasets/synthetic' / stem
    cfg['data_file'] = str(data_file)

    # ── model ──
    cfg.setdefault('model', {})['type'] = 'superquadric'

    # ── estimator ──
    cfg.setdefault('estimator', {})
    cfg['estimator']['rule_class'] = SuperquadricRule
    cfg['estimator']['estimator_class'] = MeanMeasureEstimator
    cfg['estimator']['estimator_instance'] = None
    cfg['estimator']['load_data_fn'] = load_data
    cfg['estimator']['data_file'] = str(data_file)

    # ── fitter ──
    cfg.setdefault('fitter', {})
    cfg['fitter']['algo_name'] = algo

    # ── record ──
    cfg.setdefault('record', {})
    cfg['record']['root_dir'] = str(
        PROJECT_ROOT.parent / f'outputs/superquadric/{algo}/{shape_name}/{condition}/'
    )
    cfg['record']['timestamp'] = current_timestamp()
    cfg['run_id'] = run_id

    return cfg


def get_fitter(algo: str):
    if algo == 'cco':
        from core.optimizer.cco_fitter import Fitter
    elif algo == 'gd':
        from core.optimizer.gd_fitter import Fitter
    elif algo == 'cs':
        from core.optimizer.cs_fitter import Fitter
    elif algo == 'ala':
        from core.optimizer.ala_fitter import Fitter
    elif algo == 'memetic':
        from core.optimizer.memetic_fitter import Fitter
    elif algo == 'aes':
        from core.optimizer.aes_fitter import Fitter
    elif algo == 'hierarchical':
        from core.optimizer.hierarchical_fitter import HierarchicalFitter as Fitter
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    return Fitter


def run_one(cfg: dict) -> dict:
    t0 = time.perf_counter()

    base_root = cfg['record']['root_dir']
    cfg['record']['root_dir'] = str(Path(base_root) / f'run_{cfg["run_id"]}')

    fitter = get_fitter(cfg['fitter']['algo_name'])(cfg)
    fitter.fit()
    fitter.close()

    # Read score from record or evolution
    score = float('nan')
    for dirpath, _, filenames in os.walk(cfg['record']['root_dir']):
        rec_path = os.path.join(dirpath, 'record.json')
        if os.path.exists(rec_path):
            with open(rec_path) as f:
                s = json.load(f).get('best_score', 0)
            if s and s > 0:
                score = max(score, float(s))
        evo_path = os.path.join(dirpath, 'evolution_of_round_0_instance_0.json')
        if os.path.exists(evo_path) and (np.isnan(score) or score <= 0):
            with open(evo_path) as f:
                evo = json.load(f)
            if evo:
                s = evo[-1].get('score', 0)
                if s and s > 0:
                    score = max(score, float(s))

    elapsed = time.perf_counter() - t0
    return {'score': score, 'time': elapsed}


def main():
    parser = argparse.ArgumentParser(
        description='超二次曲面拟合 — 走 configs/fit_superquadric.yaml',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python entrypoints/fit_superquadric.py --shape ellipsoid
  python entrypoints/fit_superquadric.py --shape box --algo cco --runs 5
  python entrypoints/fit_superquadric.py --all --algo memetic --runs 3
  python entrypoints/fit_superquadric.py --shape diamond --algo cco --condition noise
        ''',
    )
    parser.add_argument('--config', type=str, default='configs/fit_superquadric.yaml')
    parser.add_argument('--shape', type=str, default='ellipsoid',
                        choices=list(SHAPES.keys()), help='超二次曲面形状')
    parser.add_argument('--all', action='store_true', help='遍历全部5种形状')
    parser.add_argument('--algo', type=str, default=None,
                        choices=['cco', 'gd', 'cs', 'ala', 'memetic', 'hierarchical', 'aes'])
    parser.add_argument('--condition', type=str, default=None,
                        choices=CONDITIONS)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--max-episode', type=int, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    base_cfg = load_base_config(args.config)

    # CLI overrides
    if args.algo:
        base_cfg.setdefault('fitter', {})['algo_name'] = args.algo
    if args.condition:
        cond = args.condition
    else:
        cond = 'clean'
    if args.device:
        base_cfg.setdefault('device', {})['train_device'] = args.device
    if args.max_episode:
        base_cfg.setdefault('fitter', {})['max_episode'] = args.max_episode

    algo = base_cfg.get('fitter', {}).get('algo_name', 'memetic')
    shapes_to_run = list(SHAPES.keys()) if args.all else [args.shape]

    all_results = {}

    for shape_name in shapes_to_run:
        print(f"\n{'='*60}")
        print(f"  {shape_name}  {algo}  {cond}  ×{args.runs}")
        print(f"{'='*60}")

        scores = []
        times = []
        for i in range(args.runs):
            cfg = prepare_config(base_cfg, shape_name, cond, algo, run_id=i + 1)
            cfg['record']['verbose'] = args.verbose

            print(f"  [{shape_name}] Run {i+1}/{args.runs}...", end=' ', flush=True)
            result = run_one(cfg)
            scores.append(result['score'])
            times.append(result['time'])
            print(f"score={result['score']:.4f}  time={result['time']:.0f}s")

        arr = np.array(scores)
        tarr = np.array(times)
        all_results[shape_name] = {'scores': arr, 'times': tarr}

        print(f"  => {arr.mean():.4f} ± {arr.std():.4f}  [{arr.min():.4f}, {arr.max():.4f}]  "
              f"avg {tarr.mean():.0f}s/run")

    if len(shapes_to_run) > 1:
        print(f"\n{'='*60}")
        print(f"  汇总 ({algo} / {cond})")
        print(f"{'='*60}")
        print(f"{'Shape':<15s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Time':>8s}")
        print("-" * 55)
        for shape_name in shapes_to_run:
            r = all_results[shape_name]
            print(f"  {shape_name:<15s} {r['scores'].mean():>8.4f} {r['scores'].std():>8.4f} "
                  f"{r['scores'].min():>8.4f} {r['scores'].max():>8.4f} "
                  f"{r['times'].mean():>7.0f}s")


if __name__ == '__main__':
    main()
