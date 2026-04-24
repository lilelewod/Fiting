import os
import argparse
import sys
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.tool import current_timestamp, set_project_root_as_working_directory

set_project_root_as_working_directory(__file__)

from core.estimator.npre_estimator import NPREEstimator
from core.estimator.mm_estimator import MeanMeasureEstimator
from models.character.character_rule import CharacterRule as Rule
from tools.data_tool import load_image_data as load_data


def run_experiment(cfg):
    algo = cfg['fitter']['algo_name'].lower()
    if algo == 'cco':
        from core.optimizer.cco_fitter import Fitter
    elif algo == 'cs':
        from core.optimizer.cs_fitter import Fitter
    elif algo == 'ala':
        from core.optimizer.ala_fitter import Fitter
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    fitter = Fitter(cfg)
    fitter.fit()
    fitter.close()


def get_estimator_class(cfg):

    est_type = cfg['estimator'].get('type', 'npre').lower()

    if est_type == 'npre':
        return NPREEstimator
    elif est_type in ['mm', 'mean measure']:
        return MeanMeasureEstimator
    else:
        raise ValueError(f"Unknown estimator type specified in config: {est_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/fit_character.yaml')
    parser.add_argument('--algo', type=str, default=None, choices=['cco', 'cs', 'ala'])
    parser.add_argument('--estimator', type=str, default=None, choices=['npre', 'mm'])
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    # 命令行参数覆盖 YAML 配置
    if args.algo:
        base_cfg['fitter']['algo_name'] = args.algo
    if args.estimator:
        if 'estimator' not in base_cfg:
            base_cfg['estimator'] = {}
        base_cfg['estimator']['type'] = args.estimator

    # 专门为字符任务构建路径
    run_id = base_cfg['run_id']
    test_id = base_cfg['test_id']
    noise_type = base_cfg['noise_type']
    noise_level = base_cfg['noise_level']
    algo = base_cfg['fitter']['algo_name']
    # 获取当前的 estimator_type，用于打印日志
    est_type = base_cfg['estimator'].get('type', 'npre')

    base_cfg['rule']['token_file'] = f"datasets/character/test/run{run_id}_test{test_id}_1.mat"
    base_cfg['estimator'][
        'data_file'] = f"datasets/character/test/{noise_type}/{noise_level}/{test_id - 1}/noisy_{run_id}.png"
    base_cfg['record'][
        'root_dir'] = f"./outputs/{algo}/character/{noise_type}/{noise_level}/{test_id - 1}/noisy_{run_id}/"
    base_cfg['estimator']['rule_class'] = Rule
    base_cfg['estimator']['estimator_class'] = get_estimator_class(base_cfg)
    base_cfg['estimator']['estimator_instance'] = None
    base_cfg['estimator']['load_data_fn'] = load_data

    print("=" * 70)
    print(f"Task: CHARACTER | Algorithm: {algo.upper()} | Estimator: {est_type.upper()}")
    print(f"Config: {args.config}")
    print("=" * 70)

    for i in range(args.runs):
        cfg = deepcopy(base_cfg)
        timestamp = current_timestamp()
        cfg["record"]["timestamp"] = timestamp
        print(f"\n[{timestamp}] Start run ({i + 1}/{args.runs})")
        run_experiment(cfg)
