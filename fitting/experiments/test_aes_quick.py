#!/usr/bin/env python3
"""AES-Opt 快速测试 — 超二次椭球 11D  vs CCO / Memetic 对照"""
import sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.tool import set_project_root_as_working_directory
set_project_root_as_working_directory(__file__)

import yaml, torch
from core.estimator.mm_estimator import MeanMeasureEstimator
from models.surface.superquadric_rule import SuperquadricRule
from tools.data_tool import load_ply_data
from core.optimizer.aes_optimizer import AESOptimizer, AESConfig

# ── Setup: same config as fit_superquadric ──
with open('configs/fit_superquadric.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['estimator']['rule_class'] = SuperquadricRule
cfg['estimator']['estimator_class'] = MeanMeasureEstimator
cfg['estimator']['estimator_instance'] = None
cfg['estimator']['load_data_fn'] = load_ply_data
cfg['estimator']['data_file'] = cfg['data_file']

estimator = MeanMeasureEstimator(cfg)
dim = estimator.num_variables()
print(f"Model: superquadric ellipsoid  |  Dim: {dim}  |  Data: superq_ellipsoid_3k.ply")

# ── Evaluator: negate MM score (AES minimizes loss) ──
def evaluator(action):
    action = np.clip(action, -1.0, 1.0).astype(np.float32)
    estimator.reset()
    estimator.current_dividing_level = -1
    estimator.parse(action=action)
    estimator.generate(current_dividing_level=-1)
    score = float(estimator.get_score())
    return -score  # minimize negative score = maximize MM

# ── Initial theta (zeros = mid-range params) ──
init_theta = np.zeros(dim, dtype=np.float32)

# ── Run AES ──
aes_cfg = AESConfig(dim=dim, pop_size=15, skeleton_size=8, refine_steps=0,
                    noise_scale_init=0.08, restart_patience=15, verbose=True)
aes = AESOptimizer(init_theta, evaluator, aes_cfg)

t0 = time.perf_counter()
result = aes.optimize(max_iters=100)
elapsed = time.perf_counter() - t0

score = -result['best_loss']
print(f"\nAES score: {score:.4f}  |  time: {elapsed:.0f}s  |  evals≈{result['evals']}")
print(f"Reference: CCO=1.28 (48s)  Memetic=1.25 (38s)  Ground truth≈1.27")

# ── Save best point cloud ──
import open3d as o3d
out_dir = Path('/home/m25lll/code/Fiting/outputs/aes_test')
out_dir.mkdir(parents=True, exist_ok=True)
best_action = result['best_theta']
estimator.reset()
estimator.current_dividing_level = -1
estimator.parse(action=best_action)
pts = estimator.rule.generate()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
ply_path = out_dir / 'best_cloud.ply'
o3d.io.write_point_cloud(str(ply_path), pcd)
print(f"Saved: {ply_path}")
print(f"\nEvaluate: python tools/eval_fit.py {out_dir}/")
