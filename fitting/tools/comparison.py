import argparse
import json
from collections import defaultdict
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot


max_episode = int(1e5)

DEFAULT_CURVES = [
    (
        '/home/m25lll/code/Fiting/fitting/outputs/cs/character/saltpepper_noise/0.6/1/noisy_1/2026-0402/1616-57/record.json',
        2,
        'num_envs=1',
    ),
    (
        '/home/m25lll/code/Fiting/fitting/outputs/cs/character/saltpepper_noise/0.6/1/noisy_1/2026-0402/1635-49/record.json',
        1,
        'num_envs=8',
    ),
]


def resolve_record_files(record_source):
    if isinstance(record_source, str):
        return sorted(glob(record_source, recursive=True))
    return list(record_source)


def load_record(record_file, scores_list, x_values_list, x_key):
    with open(record_file) as f:
        record = json.load(f)
        scores = np.array(record['evolving_scores'], dtype=np.float32)
        x_values = np.array(record[x_key], dtype=np.float64)
        scores_list.append(scores)
        x_values_list.append(x_values)
        assert len(scores) == len(x_values)


def compute(scores_list, x_values_list, min_last_x=None):
    assert len(scores_list) == len(x_values_list)
    if not scores_list:
        raise ValueError('No records were loaded for comparison.')

    all_x_values = np.unique(np.concatenate(x_values_list))
    if min_last_x is not None and all_x_values[-1] < min_last_x:
        all_x_values = np.append(all_x_values, min_last_x)

    new_scores_array = np.zeros((len(scores_list), all_x_values.size), dtype=np.float32)
    for i, (scores, x_values) in enumerate(zip(scores_list, x_values_list)):
        scores = np.insert(scores, 0, 0.0)
        x_values = np.insert(x_values, 0, 0.0)
        positions = np.searchsorted(x_values, all_x_values, side='right') - 1
        new_scores_array[i, :] = scores[positions]

    score_std = np.std(new_scores_array, axis=0)
    score_mean = np.mean(new_scores_array, axis=0)
    return all_x_values, score_mean, score_std


def draw(ax, record_source, color_index, label, x_key, min_last_x=None):
    scores_list, x_values_list = [], []
    file_names = resolve_record_files(record_source)
    if not file_names:
        raise ValueError(f'No record files matched: {record_source}')

    for file_name in file_names:
        load_record(file_name, scores_list, x_values_list, x_key)

    all_x_values, score_mean, score_std = compute(scores_list, x_values_list, min_last_x)
    color = palette(color_index % palette.N)
    ax.plot(all_x_values, score_mean, color=color, label=label, linewidth=3.5)
    ax.fill_between(
        all_x_values,
        score_mean - score_std,
        score_mean + score_std,
        color=color,
        alpha=0.2,
    )


def draw_comparison(curves, output_path, x_key, x_label, min_last_x=None):
    fig, ax = plt.subplots(figsize=(20, 10))
    for record_source, color_index, label in curves:
        draw(ax, record_source, color_index, label, x_key, min_last_x)

    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlabel(x_label, fontsize=32)
    ax.set_ylabel('Fitness', fontsize=32)
    ax.legend(loc='lower right', prop=font1)
    fig.savefig(output_path)
    return fig


def discover_curves(search_root):
    grouped_records = defaultdict(list)
    discovered_max_episode = None
    pattern = str(Path(search_root) / '**' / 'record.json')

    for record_file in sorted(glob(pattern, recursive=True)):
        with open(record_file) as f:
            record = json.load(f)

        fitter_cfg = record.get('cfg', {}).get('fitter', {})
        num_envs = fitter_cfg.get('num_envs')
        if num_envs is None:
            continue

        grouped_records[int(num_envs)].append(record_file)

        current_max_episode = fitter_cfg.get('max_episode')
        if current_max_episode is not None:
            current_max_episode = int(current_max_episode)
            if discovered_max_episode is None or current_max_episode > discovered_max_episode:
                discovered_max_episode = current_max_episode

    if not grouped_records:
        raise ValueError(f'No record.json with cfg.fitter.num_envs found under: {search_root}')

    curves = []
    for color_index, num_envs in enumerate(sorted(grouped_records)):
        runs = len(grouped_records[num_envs])
        label = f'num_envs={num_envs} (runs={runs})'
        curves.append((grouped_records[num_envs], color_index, label))

    return curves, discovered_max_episode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        default=None,
        help='Recursively search record.json under this directory and group curves by cfg.fitter.num_envs.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for saved comparison figures. Defaults to --root when provided, otherwise the current directory.',
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='comparison',
        help='Prefix for output figure names.',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display figures after saving them.',
    )
    return parser.parse_args()


plt.style.use('seaborn-v0_8-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 32,
}


def main():
    args = parse_args()

    if args.root:
        curves, discovered_max_episode = discover_curves(args.root)
        eval_max_episode = discovered_max_episode
    else:
        curves = DEFAULT_CURVES
        eval_max_episode = max_episode

    output_dir = Path(args.output_dir or args.root or '.')
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output = output_dir / f'{args.output_prefix}.pdf'
    time_output = output_dir / f'{args.output_prefix}_time.pdf'

    draw_comparison(curves, eval_output, 'evolving_episodes', 'Number of fitness evaluations', eval_max_episode)
    draw_comparison(curves, time_output, 'evolving_times', 'Elapsed time (s)')

    print(f'Saved evaluation comparison to {eval_output}')
    print(f'Saved time comparison to {time_output}')

    if args.show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
