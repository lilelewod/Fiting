import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, choices=['auto', 'character', 'road'], default='auto')
    parser.add_argument('--envs', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--algo', type=str, default=None, choices=['cco', 'cs', 'ala'])
    parser.add_argument('--estimator', type=str, default=None, choices=['npre', 'mm'])
    parser.add_argument(
        '--population-size',
        type=int,
        default=None,
        help='Keep total population size fixed across num_envs. Defaults to num_envs * episodes_per_env from the config.',
    )
    parser.add_argument('--max-episode', type=int, default=None)
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument(
        '--keep-visualization',
        action='store_true',
        help='Keep the original visualization setting. By default visualization is disabled to avoid affecting timing.',
    )
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def current_timestamp():
    now = datetime.now()
    return f'{now.year}-{now.month:02d}{now.day:02d}/{now.hour:02d}{now.minute:02d}-{now.second:02d}'


def get_seeds(num_seeds):
    rng = np.random.default_rng()
    return rng.choice(100000, size=num_seeds, replace=False).tolist()


def infer_task(cfg):
    task_type = cfg.get('task_type')
    if task_type == 'character':
        return 'character'
    if task_type == 'road':
        return 'road'
    if {'run_id', 'test_id', 'noise_type', 'noise_level'}.issubset(cfg):
        return 'character'
    data_file = str(cfg.get('data_file', ''))
    if data_file.endswith('.ply'):
        return 'road'
    raise ValueError('Unable to infer task type from config. Please pass --task character or --task road.')


def load_base_config(config_path, algo=None, estimator=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg['fitter']['algo_name'] = str(algo or cfg['fitter']['algo_name']).lower()
    if estimator is not None:
        cfg.setdefault('estimator', {})
        cfg['estimator']['type'] = estimator.lower()
    elif 'estimator' in cfg and 'type' in cfg['estimator']:
        cfg['estimator']['type'] = str(cfg['estimator']['type']).lower()

    return cfg


def prepare_character_cfg(base_cfg, load_runtime):
    cfg = deepcopy(base_cfg)
    run_id = cfg['run_id']
    test_id = cfg['test_id']
    noise_type = cfg['noise_type']
    noise_level = cfg['noise_level']
    algo = cfg['fitter']['algo_name']

    cfg['rule']['token_file'] = f'datasets/character/test/run{run_id}_test{test_id}_1.mat'
    cfg['estimator']['data_file'] = f'datasets/character/test/{noise_type}/{noise_level}/{test_id - 1}/noisy_{run_id}.png'
    cfg['record']['root_dir'] = f'./outputs/{algo}/character/{noise_type}/{noise_level}/{test_id - 1}/noisy_{run_id}/'
    if not load_runtime:
        return cfg, None

    import fit_character as task_module

    cfg['estimator']['rule_class'] = task_module.Rule
    cfg['estimator']['estimator_class'] = task_module.get_estimator_class(cfg)
    cfg['estimator']['estimator_instance'] = None
    cfg['estimator']['load_data_fn'] = task_module.load_data
    return cfg, task_module.run_experiment


def prepare_road_cfg(base_cfg, load_runtime):
    cfg = deepcopy(base_cfg)
    algo = cfg['fitter']['algo_name']
    data_file = cfg['data_file']
    run_id = cfg['run_id']
    data_path = Path(data_file)

    cfg['estimator']['data_file'] = data_file
    cfg['record']['root_dir'] = f'./outputs/{algo}/road/{data_path.parent.name}/{data_path.stem}/run_{run_id}/'
    if not load_runtime:
        return cfg, None

    import fit_road as task_module

    cfg['estimator']['rule_class'] = task_module.Rule
    cfg['estimator']['estimator_class'] = task_module.get_estimator_class(cfg)
    cfg['estimator']['estimator_instance'] = None
    cfg['estimator']['load_data_fn'] = task_module.load_data
    return cfg, task_module.run_experiment


def prepare_task(base_cfg, task, load_runtime):
    if task == 'character':
        return prepare_character_cfg(base_cfg, load_runtime)
    if task == 'road':
        return prepare_road_cfg(base_cfg, load_runtime)
    raise ValueError(f'Unsupported task: {task}')


def build_experiment_root(base_root, experiment_name, num_envs, repeat_index):
    root = Path(base_root)
    run_root = root / 'parallel_comparison' / experiment_name / f'num_envs={num_envs}' / f'repeat_{repeat_index + 1}'
    return run_root.as_posix().rstrip('/') + '/'


def build_seed_list(seed_reference, num_envs):
    return seed_reference[:num_envs] + [seed_reference[-1]]


def normalize_envs(envs):
    values = sorted(set(envs))
    if not values:
        raise ValueError('At least one num_envs value is required.')
    if values[0] <= 0:
        raise ValueError('num_envs must be positive.')
    return values


def main():
    args = parse_args()
    config_path = str(Path(args.config))
    base_cfg = load_base_config(config_path, algo=args.algo, estimator=args.estimator)
    task = infer_task(base_cfg) if args.task == 'auto' else args.task
    prepared_cfg, run_experiment = prepare_task(base_cfg, task, load_runtime=not args.dry_run)

    env_values = normalize_envs(args.envs)
    reference_population = args.population_size
    if reference_population is None:
        reference_population = int(prepared_cfg['fitter']['num_envs']) * int(prepared_cfg['fitter']['episodes_per_env'])
    if reference_population <= 0:
        raise ValueError('population_size must be positive.')

    experiment_name = args.experiment_name or datetime.now().strftime('parallel-%Y%m%d-%H%M%S')
    summary = {
        'config': config_path,
        'task': task,
        'algo': prepared_cfg['fitter']['algo_name'],
        'estimator': prepared_cfg.get('estimator', {}).get('type'),
        'envs': env_values,
        'runs': args.runs,
        'population_size': reference_population,
        'max_episode': args.max_episode or prepared_cfg['fitter']['max_episode'],
        'experiment_name': experiment_name,
        'records': [],
    }

    print('=' * 72)
    print(f'Parallel comparison | task={task} | algo={prepared_cfg["fitter"]["algo_name"]}')
    print(f'config={config_path}')
    print(f'envs={env_values} | runs={args.runs} | population_size={reference_population}')
    print(f'experiment_name={experiment_name}')
    print('=' * 72)

    for repeat_index in range(args.runs):
        seed_reference = get_seeds(max(env_values) + 1)
        print(f'\n[repeat {repeat_index + 1}/{args.runs}] reference seeds: {seed_reference}')

        for num_envs in env_values:
            if reference_population % num_envs != 0:
                raise ValueError(
                    f'population_size={reference_population} is not divisible by num_envs={num_envs}. '
                    'Choose env counts that evenly divide the total population.'
                )

            cfg = deepcopy(prepared_cfg)
            cfg['fitter']['num_envs'] = num_envs
            cfg['fitter']['episodes_per_env'] = reference_population // num_envs
            if args.max_episode is not None:
                cfg['fitter']['max_episode'] = args.max_episode

            if not args.keep_visualization:
                cfg['record']['visualization'] = None

            cfg['collector']['parallel'] = num_envs > 1
            cfg['seeds'] = build_seed_list(seed_reference, num_envs)
            cfg['record']['root_dir'] = build_experiment_root(
                prepared_cfg['record']['root_dir'],
                experiment_name,
                num_envs,
                repeat_index,
            )

            timestamp = 'DRY_RUN' if args.dry_run else current_timestamp()
            cfg['record']['timestamp'] = timestamp
            cfg['experiment'] = {
                'name': experiment_name,
                'task': task,
                'num_envs': num_envs,
                'repeat_index': repeat_index,
                'population_size': reference_population,
                'episodes_per_env': cfg['fitter']['episodes_per_env'],
                'visualization_enabled': cfg['record']['visualization'] is not None,
            }

            record_file = str(Path(cfg['record']['root_dir']) / timestamp / 'record.json')
            summary['records'].append(
                {
                    'num_envs': num_envs,
                    'repeat_index': repeat_index,
                    'record_file': record_file,
                    'seeds': cfg['seeds'],
                }
            )

            print(
                f'num_envs={num_envs:>2} | episodes_per_env={cfg["fitter"]["episodes_per_env"]:>3} | '
                f'max_episode={cfg["fitter"]["max_episode"]} | record={record_file}'
            )

            if not args.dry_run:
                run_experiment(cfg)

    experiment_root = Path(prepared_cfg['record']['root_dir']) / 'parallel_comparison' / experiment_name
    if not args.dry_run:
        experiment_root.mkdir(parents=True, exist_ok=True)
        summary_file = experiment_root / 'experiment_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f'\nSaved experiment summary to {summary_file}')

    print(f'\nExperiment root: {experiment_root}')
    print(f'Plot command: python tools/comparison.py --root {experiment_root}')


if __name__ == '__main__':
    main()
