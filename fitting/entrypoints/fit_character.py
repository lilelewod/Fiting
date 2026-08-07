import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
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
    elif algo == 'pso':
        from core.optimizer.pso_fitter import Fitter
    elif algo == 'ala':
        from core.optimizer.ala_fitter import Fitter
    elif algo == 'gd':
        from core.optimizer.gd_fitter import Fitter
    elif algo == 'memetic':
        from core.optimizer.memetic_fitter import Fitter
    elif algo == 'aes':
        from core.optimizer.aes_fitter import Fitter
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
    parser.add_argument('--algo', type=str, default=None, choices=['cco', 'cs', 'pso', 'ala', 'gd', 'memetic', 'aes'])
    parser.add_argument('--estimator', type=str, default=None, choices=['npre', 'mm'])
    parser.add_argument(
        '--nearest-neighbor-backend',
        type=str,
        default=None,
        choices=['legacy', 'sklearn', 'faiss', 'torch_cuda'],
        help='Override the MM nearest-neighbor backend for timing/equivalence checks.',
    )
    parser.add_argument('--run-id', type=int, default=None)
    parser.add_argument('--test-id', type=int, default=None)
    parser.add_argument(
        '--template-test-id',
        type=int,
        default=None,
        help='Candidate-template class id; defaults to --test-id for matched fitting.',
    )
    parser.add_argument('--num-envs', type=int, default=None)
    parser.add_argument('--episodes-per-env', type=int, default=None)
    parser.add_argument('--max-episode', type=int, default=None)
    parser.add_argument('--pso-template-guided-initialization', action='store_true')
    parser.add_argument('--pso-template-guided-fraction', type=float, default=None)
    parser.add_argument('--pso-template-guided-sigma', type=float, default=None)
    parser.add_argument(
        '--template-guided-initialization',
        action='store_true',
        help='Use the shared zero-action template warm start (PSO or CS).',
    )
    parser.add_argument('--template-guided-fraction', type=float, default=None)
    parser.add_argument('--template-guided-sigma', type=float, default=None)
    parser.add_argument('--visualization', type=str, default=None, choices=['parallel', 'non-parallel', 'none'])
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None,
                        help='Base seed; run i uses seed + i for reproducible paired experiments.')
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
    if args.nearest_neighbor_backend:
        base_cfg.setdefault('estimator', {})
        base_cfg['estimator']['nearest_neighbor_backend'] = args.nearest_neighbor_backend
    if args.run_id is not None:
        base_cfg['run_id'] = args.run_id
    if args.test_id is not None:
        base_cfg['test_id'] = args.test_id
    if args.num_envs is not None:
        base_cfg['fitter']['num_envs'] = args.num_envs
    if args.episodes_per_env is not None:
        base_cfg['fitter']['episodes_per_env'] = args.episodes_per_env
    if args.max_episode is not None:
        base_cfg['fitter']['max_episode'] = args.max_episode
    if args.pso_template_guided_initialization:
        base_cfg['fitter']['pso_template_guided_initialization'] = True
    if args.pso_template_guided_fraction is not None:
        base_cfg['fitter']['pso_template_guided_fraction'] = args.pso_template_guided_fraction
    if args.pso_template_guided_sigma is not None:
        base_cfg['fitter']['pso_template_guided_sigma'] = args.pso_template_guided_sigma
    if args.template_guided_initialization:
        base_cfg['fitter']['template_guided_initialization'] = True
    if args.template_guided_fraction is not None:
        base_cfg['fitter']['template_guided_fraction'] = args.template_guided_fraction
    if args.template_guided_sigma is not None:
        base_cfg['fitter']['template_guided_sigma'] = args.template_guided_sigma
    if args.visualization is not None:
        base_cfg.setdefault('record', {})
        base_cfg['record']['visualization'] = None if args.visualization == 'none' else args.visualization

    # 专门为字符任务构建路径
    run_id = base_cfg['run_id']
    test_id = base_cfg['test_id']
    template_test_id = args.template_test_id if args.template_test_id is not None else test_id
    noise_type = base_cfg['noise_type']
    noise_level = base_cfg['noise_level']
    algo = base_cfg['fitter']['algo_name']
    # 获取当前的 estimator_type，用于打印日志
    est_type = base_cfg['estimator'].get('type', 'npre')

    token_file = PROJECT_ROOT / f"datasets/character/test/run{run_id}_test{template_test_id}_1.mat"
    data_file = PROJECT_ROOT / f"datasets/character/test/{noise_type}/{noise_level}/{test_id - 1}/noisy_{run_id}.png"
    if not token_file.is_file():
        raise FileNotFoundError(f"Character token file not found: {token_file}")
    if not data_file.is_file():
        raise FileNotFoundError(f"Character image not found: {data_file}")

    base_cfg['rule']['token_file'] = str(token_file)
    base_cfg['estimator']['data_file'] = str(data_file)
    if args.template_test_id is None:
        record_root = (
            PROJECT_ROOT.parent / 'outputs' / algo / 'character' / noise_type
            / str(noise_level) / str(test_id - 1) / f'noisy_{run_id}'
        )
    else:
        record_root = (
            PROJECT_ROOT.parent / 'outputs' / algo / 'character_classification'
            / noise_type / str(noise_level) / f'run_{run_id}'
            / f'observation_{test_id}' / f'candidate_{template_test_id}'
        )
    base_cfg['record']['root_dir'] = str(record_root)
    base_cfg['classification'] = {
        'run_id': run_id,
        'observation_test_id': test_id,
        'template_test_id': template_test_id,
        'is_matched_candidate': template_test_id == test_id,
    }
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
        if args.seed is not None:
            seed_sequence = np.random.SeedSequence(args.seed + i)
            cfg['seeds'] = [
                int(x) for x in seed_sequence.generate_state(int(cfg['fitter']['num_envs']) + 1)
            ]
        timestamp = current_timestamp()
        cfg["record"]["timestamp"] = timestamp
        print(f"\n[{timestamp}] Start run ({i + 1}/{args.runs})")
        run_experiment(cfg)
