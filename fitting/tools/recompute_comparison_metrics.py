"""Recompute clean-reference metrics from saved comparison point clouds."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.data_tool import read_point_cloud
from tools.superquadric_evaluation import geometric_metrics, load_trait, sample_trait, trait_from_mapping
from tools.tool import json_default


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row if key != 'trait'))
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--ground-truth', required=True)
    parser.add_argument('--ground-truth-trait', default=None)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--success-chamfer', type=float, default=0.05)
    parser.add_argument('--evaluation-points', type=int, default=20000)
    parser.add_argument('--evaluation-grid', type=int, default=256)
    parser.add_argument('--evaluation-seed', type=int, default=20260716)
    args = parser.parse_args()

    root = Path(args.root)
    with open(root / 'results.json', encoding='utf-8') as stream:
        rows = json.load(stream)
    if args.ground_truth_trait:
        reference = sample_trait(
            load_trait(args.ground_truth_trait), args.evaluation_points,
            args.evaluation_seed, args.evaluation_grid,
        )
        reference_mode = 'analytic-area-uniform'
    else:
        reference = read_point_cloud(args.ground_truth)
        reference_mode = 'provided-point-cloud-density-dependent'
        print(
            'WARNING: --ground-truth-trait was not supplied; the reference-side '
            'metric remains dependent on the provided point-cloud density.'
        )
    for row in rows:
        if not row.get('trait'):
            raise ValueError(f"row has no fitted trait: {row.get('record_file')}")
        model = sample_trait(
            trait_from_mapping(row['trait']), args.evaluation_points,
            args.evaluation_seed + 1, args.evaluation_grid,
        )
        row['input_metric_threshold'] = row.pop('metric_threshold', row.get('input_metric_threshold'))
        row.update(geometric_metrics(reference, model, args.threshold))
        row.update({
            'gt_metric_threshold': args.threshold,
            'evaluation_points': args.evaluation_points,
            'evaluation_grid': args.evaluation_grid,
            'evaluation_reference_seed': args.evaluation_seed,
            'evaluation_model_seed': args.evaluation_seed + 1,
            'evaluation_reference_mode': reference_mode,
        })
        row['success'] = int(row['gt_chamfer'] <= args.success_chamfer)

    with open(root / 'results.json', 'w', encoding='utf-8') as stream:
        json.dump(rows, stream, default=json_default, indent=2)
    write_csv(root / 'results.csv', rows)

    metric_names = ['best_score', 'wall_time_s', 'input_chamfer', 'input_fscore', 'gt_chamfer', 'gt_fscore', 'success']
    summaries = []
    for algorithm in dict.fromkeys(row['algorithm'] for row in rows):
        selected = [row for row in rows if row['algorithm'] == algorithm]
        summary = {'algorithm': algorithm, 'runs': len(selected)}
        for name in metric_names:
            values = np.asarray([row[name] for row in selected], dtype=float)
            summary[f'{name}_mean'] = float(np.mean(values))
            summary[f'{name}_std'] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            summary[f'{name}_median'] = float(np.median(values))
            summary[f'{name}_iqr'] = float(np.percentile(values, 75) - np.percentile(values, 25))
        summaries.append(summary)
    with open(root / 'summary.json', 'w', encoding='utf-8') as stream:
        json.dump(summaries, stream, indent=2)
    write_csv(root / 'summary.csv', summaries)


if __name__ == '__main__':
    main()
