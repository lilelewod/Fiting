import os
import time
import argparse
import yaml
from copy import deepcopy
import numpy as np

from tools.tool import current_timestamp, set_project_root_as_working_directory

set_project_root_as_working_directory(__file__)


def build_config_paths(cfg):
    task_type = cfg.get('task_type', 'character')
    algo = cfg['fitter']['algo_name']

    if task_type == 'character':
        from models.character.character_rule import CharacterRule as Rule
        run_id = cfg['run_id']
        test_id = cfg['test_id']
        noise_type = cfg['noise_type']
        noise_level = cfg['noise_level']

        cfg['rule']['token_file'] = f"datasets/character/test/run{run_id}_test{test_id}_1.mat"
        cfg['estimator'][
            'data_file'] = f"datasets/character/test/{noise_type}/{noise_level}/{test_id - 1}/noisy_{run_id}.png"
        cfg['record'][
            'root_dir'] = f"./outputs/{algo}/character/{noise_type}/{noise_level}/{test_id - 1}/noisy_{run_id}/"

    elif task_type == 'line':
        from fitting.models.line_segment.line_segment_2d_open3d import LineSegment2dRule as Rule
        cfg['estimator']['data_file'] = cfg['data_file']
        data_name = cfg['data_file'].split('/')[-1].split('.')[0]
        cfg['record']['root_dir'] = f"./outputs/{algo}/line/{data_name}/run_{cfg['run_id']}/"

    from fitting.core.estimator import Estimator
    from tools.data_tool import load_image_data as load_data

    cfg['estimator']['rule_class'] = Rule
    cfg['estimator']['estimator_class'] = Estimator
    cfg['estimator']['estimator_instance'] = None
    cfg['estimator']['load_data_fn'] = load_data

    return cfg


def run_experiment(cfg):
    algo = cfg['fitter']['algo_name'].lower()
    if algo == 'cco':
        from core.optimizers.cco_fitter import Fitter
    elif algo == 'cs':
        from core.optimizers.cs_fitter import Fitter
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    fitter = Fitter(cfg)
    fitter.fit()
    fitter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/fit_line.yaml')
    # parser.add_argument('--config', type=str, default='configs/fit_character.yaml')
    parser.add_argument('--algo', type=str, default=None, choices=['cco', 'cs'])
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    if args.algo:
        base_cfg['fitter']['algo_name'] = args.algo

    base_cfg = build_config_paths(base_cfg)

    print("=" * 60)
    print(f"Algorithm: {base_cfg['fitter']['algo_name'].upper()} | Config: {args.config} | Runs: {args.runs}")
    print("=" * 60)

    for i in range(args.runs):
        cfg = deepcopy(base_cfg)
        timestamp = current_timestamp()
        cfg["record"]["timestamp"] = timestamp
        print(f"\n[{timestamp}] Start run ({i + 1}/{args.runs})")
        run_experiment(cfg)