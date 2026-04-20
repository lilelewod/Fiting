import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from tools.tool import current_timestamp, set_project_root_as_working_directory

set_project_root_as_working_directory(__file__)

from core.estimator.npre_estimator import NPREEstimator
from core.estimator.mm_estimator import MeanMeasureEstimator
from tools.data_tool import load_3d_pointcloud_data as load_data


def get_rule_class(cfg):
    model_cfg = cfg.get('model', {})
    model_type = str(model_cfg.get('type', 'curve')).lower()

    if model_type == 'curve':
        from models.road_curve.curve_rule import CurveRule

        return CurveRule
    if model_type == 'surface':
        from models.surface_patch.surface_patch_rule import SurfacePatchRule

        return SurfacePatchRule
    raise ValueError(f"Unknown 3D model type: {model_type}")


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
    if est_type in ['mm', 'mean measure']:
        return MeanMeasureEstimator
    raise ValueError(f"Unknown estimator type specified in config: {est_type}")


def prepare_3d_cfg(base_cfg):
    cfg = deepcopy(base_cfg)
    cfg['task_type'] = '3d'
    cfg.setdefault('model', {})

    algo = cfg['fitter']['algo_name']
    data_file = cfg['data_file']
    run_id = cfg['run_id']
    data_path = Path(data_file)
    model_type = str(cfg['model'].get('type', 'curve')).lower()

    cfg['estimator']['data_file'] = data_file
    cfg['record']['root_dir'] = (
        f"./outputs/{algo}/3d/{model_type}/{data_path.parent.name}/{data_path.stem}/run_{run_id}/"
    )
    cfg['estimator']['rule_class'] = get_rule_class(cfg)
    cfg['estimator']['estimator_class'] = get_estimator_class(cfg)
    cfg['estimator']['estimator_instance'] = None
    cfg['estimator']['load_data_fn'] = load_data
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/fit_point_cloud.yaml')
    parser.add_argument('--algo', type=str, default=None, choices=['cco', 'cs', 'ala'])
    parser.add_argument('--estimator', type=str, default=None, choices=['npre', 'mm'])
    parser.add_argument('--model', type=str, default=None, choices=['curve', 'surface'])
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--num-instances', type=int, default=None)
    parser.add_argument('--num-envs', type=int, default=None)
    parser.add_argument('--episodes-per-env', type=int, default=None)
    parser.add_argument('--max-episode', type=int, default=None)
    parser.add_argument('--data-resolution', type=float, default=None)
    parser.add_argument('--model-resolution', type=float, default=None)
    parser.add_argument('--visualization', type=str, default=None, choices=['parallel', 'non-parallel', 'none'])
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    if args.algo:
        base_cfg['fitter']['algo_name'] = args.algo
    if args.estimator:
        base_cfg.setdefault('estimator', {})
        base_cfg['estimator']['type'] = args.estimator
    if args.model:
        base_cfg.setdefault('model', {})
        base_cfg['model']['type'] = args.model
    if args.data_file:
        base_cfg['data_file'] = args.data_file
    if args.num_instances is not None:
        base_cfg['fitter']['num_instances'] = args.num_instances
    if args.num_envs is not None:
        base_cfg['fitter']['num_envs'] = args.num_envs
    if args.episodes_per_env is not None:
        base_cfg['fitter']['episodes_per_env'] = args.episodes_per_env
    if args.max_episode is not None:
        base_cfg['fitter']['max_episode'] = args.max_episode
    if args.data_resolution is not None:
        base_cfg.setdefault('estimator', {})
        base_cfg['estimator']['data_resolution'] = args.data_resolution
    if args.model_resolution is not None:
        base_cfg.setdefault('estimator', {})
        base_cfg['estimator']['model_resolution'] = args.model_resolution
    if args.visualization is not None:
        base_cfg.setdefault('record', {})
        base_cfg['record']['visualization'] = None if args.visualization == 'none' else args.visualization

    base_cfg = prepare_3d_cfg(base_cfg)

    algo = base_cfg['fitter']['algo_name']
    est_type = base_cfg['estimator'].get('type', 'npre')

    print("=" * 70)
    model_type = base_cfg.get('model', {}).get('type', 'curve')
    print(f"Task: 3D POINT CLOUD FITTING | Model: {str(model_type).upper()} | Algorithm: {algo.upper()} | Estimator: {est_type.upper()}")
    print(f"Config: {args.config}")
    print("=" * 70)

    for i in range(args.runs):
        cfg = deepcopy(base_cfg)
        timestamp = current_timestamp()
        cfg["record"]["timestamp"] = timestamp
        print(f"\n[{timestamp}] Start run ({i + 1}/{args.runs})")
        run_experiment(cfg)


if __name__ == "__main__":
    main()
