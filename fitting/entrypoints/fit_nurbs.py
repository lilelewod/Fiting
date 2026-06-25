"""NURBS surface fitting entrypoint.

支持算法: cco, cs, ala, gd, memetic, hierarchical
默认配置: configs/fit_hierarchical.yaml

    cd /home/m25lll/code/Fiting/fitting

    # 层次化优化 (推荐)
    python entrypoints/fit_nurbs.py --algo hierarchical --runs 10

    # 单算法对比
    python entrypoints/fit_nurbs.py --algo memetic --runs 10
    python entrypoints/fit_nurbs.py --algo cco --runs 10
    python entrypoints/fit_nurbs.py --algo gd --runs 10

    # 自定义数据 & 网格
    python entrypoints/fit_nurbs.py --data-file datasets/synthetic/cylinder_4k.ply --grid 12 12
"""

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

from core.estimator.mm_estimator import MeanMeasureEstimator
from models.surface.nurbs_surface_rule import NURBSSurfaceRule as Rule
from tools.data_tool import load_ply_data as load_data


def run_experiment(cfg):
    algo = cfg['fitter']['algo_name'].lower()
    if algo == 'cco':
        from core.optimizer.cco_fitter import Fitter
    elif algo == 'cs':
        from core.optimizer.cs_fitter import Fitter
    elif algo == 'ala':
        from core.optimizer.ala_fitter import Fitter
    elif algo == 'gd':
        from core.optimizer.gd_fitter import Fitter
    elif algo == 'memetic':
        from core.optimizer.memetic_fitter import Fitter
    elif algo == 'hierarchical':
        from core.optimizer.hierarchical_fitter import HierarchicalFitter as Fitter
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    fitter = Fitter(cfg)
    fitter.fit()
    fitter.close()


def prepare_cfg(base_cfg):
    """组装 NURBS 实验配置"""
    cfg = deepcopy(base_cfg)
    cfg['task_type'] = '3d'

    algo = cfg['fitter']['algo_name']
    data_file = cfg['data_file']
    run_id = cfg['run_id']
    data_path = Path(data_file)

    grid_u = cfg['model']['num_ctrl_u']
    grid_v = cfg['model']['num_ctrl_v']

    cfg['estimator']['data_file'] = data_file
    cfg['record']['root_dir'] = (
        f"{PROJECT_ROOT.parent}/outputs/{algo}/3d/nurbs_surface/"
        f"{data_path.parent.name}/{data_path.stem}/"
        f"{grid_u}x{grid_v}/run_{run_id}/"
    )
    cfg['estimator']['rule_class'] = Rule
    cfg['estimator']['estimator_class'] = MeanMeasureEstimator
    cfg['estimator']['estimator_instance'] = None
    cfg['estimator']['load_data_fn'] = load_data
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="NURBS surface fitting with MM estimator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python entrypoints/fit_nurbs.py --algo hierarchical --runs 10
  python entrypoints/fit_nurbs.py --algo memetic --runs 10
  python entrypoints/fit_nurbs.py --algo cco --runs 5 --data-file datasets/synthetic/cylinder_4k.ply
  python entrypoints/fit_nurbs.py --algo gd --grid 12 12
        """,
    )
    parser.add_argument('--config', type=str, default='configs/fit_hierarchical.yaml',
                        help='YAML config file')
    parser.add_argument('--algo', type=str, default=None,
                        choices=['cco', 'cs', 'ala', 'gd', 'memetic', 'hierarchical'])
    parser.add_argument('--data-file', type=str, default=None,
                        help='Point cloud .ply file')
    parser.add_argument('--grid', type=int, nargs=2, default=None, metavar=('U', 'V'),
                        help='Control grid size (e.g. --grid 8 8)')
    parser.add_argument('--max-evals', type=int, default=None,
                        help='Total evaluation budget')
    parser.add_argument('--coarse-evals', type=int, default=None,
                        help='Coarse phase evals (hierarchical only)')
    parser.add_argument('--fine-steps', type=int, default=None,
                        help='Fine phase GD steps (hierarchical only)')
    parser.add_argument('--smoothness', type=float, default=None,
                        help='Smoothness weight (hierarchical fine phase)')
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    # ── 命令行覆盖 ──
    if args.algo:
        base_cfg['fitter']['algo_name'] = args.algo
    if args.data_file:
        base_cfg['data_file'] = args.data_file
    if args.grid:
        base_cfg.setdefault('model', {})
        base_cfg['model']['num_ctrl_u'] = args.grid[0]
        base_cfg['model']['num_ctrl_v'] = args.grid[1]
    if args.max_evals is not None:
        base_cfg['fitter']['max_episode'] = args.max_evals
    if args.coarse_evals is not None:
        base_cfg['fitter']['hier_coarse_evals'] = args.coarse_evals
    if args.fine_steps is not None:
        base_cfg['fitter']['hier_fine_steps'] = args.fine_steps
    if args.smoothness is not None:
        base_cfg['fitter']['hier_smoothness'] = args.smoothness

    base_cfg = prepare_cfg(base_cfg)

    algo = base_cfg['fitter']['algo_name']
    grid_u = base_cfg['model']['num_ctrl_u']
    grid_v = base_cfg['model']['num_ctrl_v']
    dim = grid_u * grid_v * 4
    data_path = Path(base_cfg['data_file'])

    print("=" * 70)
    print(f"Task: NURBS FITTING | {grid_u}×{grid_v} ({dim}D)")
    print(f"Data: {data_path.stem} | Algorithm: {algo.upper()} | Estimator: MM")
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
