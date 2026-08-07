"""Run a reproducible CS/PSO/DE/CCO comparison under one evaluation budget."""

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entrypoints.fit_point_cloud import prepare_3d_cfg, run_experiment
from tools.data_tool import read_point_cloud
from tools.superquadric_evaluation import geometric_metrics, load_trait, sample_trait
from tools.tool import json_default


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/fit_superquadric.yaml')
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--ground-truth', default=None, help='Optional clean point cloud used only for final evaluation.')
    parser.add_argument('--ground-truth-trait', default=None, help='JSON trait used to generate an area-uniform analytic reference.')
    parser.add_argument('--algorithms', nargs='+', default=['cs', 'pso', 'de', 'cco'], choices=['cs', 'pso', 'de', 'cco'])
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--base-seed', type=int, default=20260714)
    parser.add_argument('--seed-list', nargs='+', type=int, default=None, help='Explicit paired base seeds; overrides --runs/--base-seed.')
    parser.add_argument('--population-size', type=int, default=16)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--max-evaluations', type=int, default=10000)
    parser.add_argument('--data-resolution', type=float, default=None, help='Fixed estimator resolution; recommended for robustness matrices.')
    parser.add_argument('--model-resolution', type=float, default=None, help='Defaults to 0.45 * data-resolution when omitted.')
    parser.add_argument('--success-chamfer', type=float, default=0.05)
    parser.add_argument('--gt-threshold', type=float, default=0.05, help='Fixed distance threshold for clean-reference F-score.')
    parser.add_argument('--evaluation-points', type=int, default=20000)
    parser.add_argument('--evaluation-grid', type=int, default=256)
    parser.add_argument('--evaluation-seed', type=int, default=20260716)
    parser.add_argument('--output-root', default=None)
    parser.add_argument('--resume', action='store_true', help='Resume an interrupted output root and skip completed seed/algorithm pairs.')
    parser.add_argument('--pso-guided-initialization', action='store_true')
    parser.add_argument('--pso-guided-fraction', type=float, default=0.75)
    parser.add_argument('--pso-guided-jitter', type=float, default=0.04)
    parser.add_argument('--pso-guided-extent-quantile', type=float, default=0.005)
    parser.add_argument('--pso-guided-support-fraction', type=float, default=1.0)
    parser.add_argument('--pso-guided-support-neighbors', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs <= 0 or args.num_envs <= 0 or args.population_size < 4:
        raise ValueError('runs and num-envs must be positive; population-size must be at least 4')
    if args.evaluation_points <= 0 or args.evaluation_grid < 8:
        raise ValueError('evaluation-points must be positive and evaluation-grid must be at least 8')
    if not 0.0 < args.pso_guided_fraction <= 1.0:
        raise ValueError('--pso-guided-fraction must lie in (0, 1]')
    if args.pso_guided_jitter < 0.0:
        raise ValueError('--pso-guided-jitter must be nonnegative')
    if not 0.0 <= args.pso_guided_extent_quantile < 0.25:
        raise ValueError('--pso-guided-extent-quantile must lie in [0, 0.25)')
    if args.seed_list is not None and len(set(args.seed_list)) != len(args.seed_list):
        raise ValueError('--seed-list must not contain duplicates')
    if not 0.0 < args.pso_guided_support_fraction <= 1.0:
        raise ValueError('--pso-guided-support-fraction must lie in (0, 1]')
    if args.pso_guided_support_neighbors < 2:
        raise ValueError('--pso-guided-support-neighbors must be at least 2')
    if args.ground_truth_trait and not args.ground_truth:
        raise ValueError('--ground-truth-trait requires --ground-truth for an explicit evaluation protocol')
    if args.population_size % args.num_envs:
        raise ValueError('population-size must be divisible by num-envs')
    if args.max_evaluations < args.population_size or (
        args.max_evaluations - args.population_size
    ) % (2 * args.population_size):
        raise ValueError(
            'Exact fairness requires max-evaluations = population-size + k * '
            '(2 * population-size). With population 16, use 10000.'
        )

    with open(args.config, encoding='utf-8') as stream:
        base_cfg = yaml.safe_load(stream)
    base_cfg['data_file'] = str(Path(args.data_file).resolve())
    base_cfg.setdefault('model', {})['type'] = 'superquadric'
    base_cfg['fitter']['num_envs'] = args.num_envs
    base_cfg['fitter']['episodes_per_env'] = args.population_size // args.num_envs
    base_cfg['fitter']['max_episode'] = args.max_evaluations
    base_cfg['fitter']['pso_guided_initialization'] = args.pso_guided_initialization
    base_cfg['fitter']['pso_guided_fraction'] = args.pso_guided_fraction
    base_cfg['fitter']['pso_guided_jitter'] = args.pso_guided_jitter
    base_cfg['fitter']['pso_guided_extent_quantile'] = args.pso_guided_extent_quantile
    base_cfg['fitter']['pso_guided_support_fraction'] = args.pso_guided_support_fraction
    base_cfg['fitter']['pso_guided_support_neighbors'] = args.pso_guided_support_neighbors
    if args.data_resolution is not None:
        if args.data_resolution <= 0.0:
            raise ValueError('data-resolution must be positive')
        base_cfg['estimator']['data_resolution'] = args.data_resolution
        base_cfg['estimator']['model_resolution'] = (
            args.model_resolution if args.model_resolution is not None
            else 0.45 * args.data_resolution
        )
    elif args.model_resolution is not None:
        raise ValueError('--model-resolution requires --data-resolution')
    base_cfg.setdefault('record', {})['visualization'] = None

    experiment_name = datetime.now().strftime('optimizer-%Y%m%d-%H%M%S')
    output_root = Path(args.output_root or (PROJECT_ROOT.parent / 'outputs' / 'optimizer_comparison' / experiment_name))
    output_root.mkdir(parents=True, exist_ok=True)
    results_file = output_root / 'results.json'
    if results_file.exists() and not args.resume:
        raise FileExistsError(f'{results_file} already exists; use --resume or choose a new output root')
    gt_cloud = read_point_cloud(args.ground_truth) if args.ground_truth else None
    evaluation_reference_mode = None
    if args.ground_truth_trait:
        gt_trait = load_trait(args.ground_truth_trait)
        gt_cloud = sample_trait(
            gt_trait,
            count=args.evaluation_points,
            seed=args.evaluation_seed,
            grid_resolution=args.evaluation_grid,
        )
        evaluation_reference_mode = 'analytic-area-uniform'
    elif gt_cloud is not None:
        evaluation_reference_mode = 'provided-point-cloud-density-dependent'
        print(
            'WARNING: --ground-truth-trait was not supplied; the reference-side '
            'metric remains dependent on the provided point-cloud density.'
        )

    if args.resume and results_file.exists():
        with open(results_file, encoding='utf-8') as stream:
            rows = json.load(stream)
        print(f'Resuming {output_root}: loaded {len(rows)} completed rows')
    else:
        rows = []
    completed = {(int(row['seed']), row['algorithm']) for row in rows}
    run_seeds = args.seed_list if args.seed_list is not None else [args.base_seed + i for i in range(args.runs)]
    for repeat, base_seed in enumerate(run_seeds):
        seed_sequence = np.random.SeedSequence(base_seed)
        shared_seeds = [int(x) for x in seed_sequence.generate_state(args.num_envs + 1)]
        for algorithm in args.algorithms:
            completion_key = (base_seed, algorithm)
            if completion_key in completed:
                print(f'[{repeat + 1}/{len(run_seeds)}] {algorithm.upper()} | already complete, skipping')
                continue
            cfg = deepcopy(base_cfg)
            cfg['fitter']['algo_name'] = algorithm
            cfg['seeds'] = shared_seeds.copy()
            cfg = prepare_3d_cfg(cfg)
            cfg['record']['root_dir'] = (output_root / algorithm / f'repeat_{repeat + 1:02d}').as_posix() + '/'
            cfg['record']['timestamp'] = datetime.now().strftime('%Y-%m%d/%H%M-%S-%f')
            cfg['experiment'] = {
                'comparison': 'CS-PSO-DE-CCO',
                'repeat': repeat + 1,
                'base_seed': base_seed,
                'shared_seeds': shared_seeds,
                'population_size': args.population_size,
                'max_evaluations': args.max_evaluations,
                'data_resolution': cfg['estimator']['data_resolution'],
                'model_resolution': cfg['estimator']['model_resolution'],
                'evaluation_points': args.evaluation_points,
                'evaluation_grid': args.evaluation_grid,
                'evaluation_reference_seed': args.evaluation_seed,
                'evaluation_model_seed': args.evaluation_seed + 1,
                'evaluation_reference_mode': evaluation_reference_mode,
                'pso_guided_initialization': args.pso_guided_initialization,
                'pso_guided_fraction': args.pso_guided_fraction,
                'pso_guided_jitter': args.pso_guided_jitter,
                'pso_guided_extent_quantile': args.pso_guided_extent_quantile,
                'pso_guided_support_fraction': args.pso_guided_support_fraction,
                'pso_guided_support_neighbors': args.pso_guided_support_neighbors,
            }

            print(f'\n[{repeat + 1}/{len(run_seeds)}] {algorithm.upper()} | seeds={shared_seeds}')
            started = time.perf_counter()
            record = run_experiment(cfg)
            wall_time = time.perf_counter() - started
            row = {
                'repeat': repeat + 1,
                'algorithm': algorithm,
                'pso_guided_initialization': bool(
                    args.pso_guided_initialization and algorithm == 'pso'
                ),
                'pso_guided_fraction': args.pso_guided_fraction if algorithm == 'pso' else None,
                'pso_guided_jitter': args.pso_guided_jitter if algorithm == 'pso' else None,
                'pso_guided_extent_quantile': (
                    args.pso_guided_extent_quantile if algorithm == 'pso' else None
                ),
                'pso_guided_support_fraction': (
                    args.pso_guided_support_fraction if algorithm == 'pso' else None
                ),
                'pso_guided_support_neighbors': (
                    args.pso_guided_support_neighbors if algorithm == 'pso' else None
                ),
                'seed': base_seed,
                'evaluations': int(getattr(record, 'num_evaluations', args.max_evaluations)),
                'wall_time_s': wall_time,
                'best_score': float(record.best_score),
                'input_chamfer': float(record.chamfer),
                'input_d2m': float(record.d2m),
                'input_m2d': float(record.m2d),
                'input_fscore': float(record.f5),
                'input_metric_threshold': float(record.metric_threshold),
                'trait': record.best_token_set[0].trait if record.best_token_set[0] is not None else None,
                'record_file': str(Path(record.out_json_file_name).resolve()),
            }
            best_token = record.best_token_set[0] if record.best_token_set else None
            if gt_cloud is not None and best_token is not None and best_token.trait is not None:
                evaluation_cloud = sample_trait(
                    best_token.trait,
                    count=args.evaluation_points,
                    seed=args.evaluation_seed + 1,
                    grid_resolution=args.evaluation_grid,
                )
                row.update(geometric_metrics(gt_cloud, evaluation_cloud, args.gt_threshold))
                row['gt_metric_threshold'] = args.gt_threshold
                row['evaluation_points'] = args.evaluation_points
                row['evaluation_grid'] = args.evaluation_grid
                row['evaluation_reference_seed'] = args.evaluation_seed
                row['evaluation_model_seed'] = args.evaluation_seed + 1
                row['evaluation_reference_mode'] = evaluation_reference_mode
                row['success'] = int(row['gt_chamfer'] <= args.success_chamfer)
            rows.append(row)
            completed.add(completion_key)

            with open(results_file, 'w', encoding='utf-8') as stream:
                json.dump(rows, stream, default=json_default, indent=2)
            scalar_keys = [key for key, value in rows[0].items() if key != 'trait' and not isinstance(value, (list, dict))]
            with open(output_root / 'results.csv', 'w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)

    numeric_metrics = [
        'best_score', 'wall_time_s', 'input_chamfer', 'input_fscore',
        'gt_chamfer', 'gt_fscore', 'success',
    ]
    summaries = []
    for algorithm in args.algorithms:
        algorithm_rows = [row for row in rows if row['algorithm'] == algorithm]
        summary = {'algorithm': algorithm, 'runs': len(algorithm_rows)}
        for metric in numeric_metrics:
            values = np.asarray([row[metric] for row in algorithm_rows if metric in row], dtype=float)
            if values.size == 0:
                continue
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_std'] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            summary[f'{metric}_median'] = float(np.median(values))
            summary[f'{metric}_iqr'] = float(np.percentile(values, 75) - np.percentile(values, 25))
        summaries.append(summary)
    summary_keys = list(dict.fromkeys(key for summary in summaries for key in summary))
    with open(output_root / 'summary.csv', 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_keys)
        writer.writeheader()
        writer.writerows(summaries)
    with open(output_root / 'summary.json', 'w', encoding='utf-8') as stream:
        json.dump(summaries, stream, indent=2)

    print(f'\nSaved comparison to: {output_root}')


if __name__ == '__main__':
    main()
