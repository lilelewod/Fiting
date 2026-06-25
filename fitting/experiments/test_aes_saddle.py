#!/usr/bin/env python3
"""AES-Opt on NURBS saddle (1024D) vs Memetic / Hierarchical"""
import sys, time
from pathlib import Path
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.tool import set_project_root_as_working_directory
set_project_root_as_working_directory(__file__)

from core.estimator.mm_estimator import MeanMeasureEstimator
from models.surface.nurbs_surface_rule import NURBSSurfaceRule
from tools.data_tool import load_ply_data
from core.optimizer.aes_optimizer import AESOptimizer, AESConfig

# ── Setup ──
cfg = {
    'task_type': '3d', 'run_id': 1, 'seeds': None,
    'data_file': str(PROJECT_ROOT / 'datasets/synthetic/saddle_3k.ply'),
    'model': {'type': 'nurbs_surface', 'num_ctrl_u': 16, 'num_ctrl_v': 16,
              'degree_u': 3, 'degree_v': 3, 'sample_u': 100, 'sample_v': 100,
              'weight_lb': 0.8, 'weight_ub': 1.2},
    'device': {'train_device': 'cuda:1', 'cuda_deterministic': True},
    'estimator': {
        'type': 'mm', 'data_resolution': 0.01, 'model_resolution': 0.004,
        'regularization_factor': 0.5, 'incremental_coverage': True,
        'outlier_distance_factor': 2.5, 'outlier_penalty_factor': 0.0,
        'bbox_margin_factor': 1.0, 'bbox_penalty_factor': 0.0,
        'overlap_penalty_factor': 0.0, 'control_smoothness_penalty_factor': 0.0,
        'mm_bbox_penalty_factor': 0.0, 'early_rejection': True, 'use_faiss': True,
        'rule_class': NURBSSurfaceRule, 'estimator_class': MeanMeasureEstimator,
        'estimator_instance': None, 'load_data_fn': load_ply_data,
        'data_file': str(PROJECT_ROOT / 'datasets/synthetic/saddle_3k.ply'),
    },
}

estimator = MeanMeasureEstimator(cfg)
dim = estimator.num_variables()
print(f"Model: NURBS saddle 16×16  |  Dim: {dim}  |  Data: saddle_3k.ply")

def evaluator(action):
    action = np.clip(action, -1.0, 1.0).astype(np.float32)
    estimator.reset()
    estimator.current_dividing_level = -1
    estimator.parse(action=action)
    estimator.generate(current_dividing_level=-1)
    return -float(estimator.get_score())

init_theta = np.zeros(dim, dtype=np.float32)

# 1024D: larger pop, more skeleton, no refine (zero-order too slow)
aes_cfg = AESConfig(dim=dim, pop_size=30, skeleton_size=15,
                    refine_steps=0, noise_scale_init=0.1,
                    restart_patience=25, verbose=True)
aes = AESOptimizer(init_theta, evaluator, aes_cfg)

t0 = time.perf_counter()
result = aes.optimize(max_iters=200)
elapsed = time.perf_counter() - t0

score = -result['best_loss']
print(f"\nAES score: {score:.4f}  |  time: {elapsed:.0f}s  |  evals≈{result['evals']}")
print(f"Reference: Memetic=3.38 (160s)  Hierarchical=7.75 (8s)")

# Save PLY
out_dir = Path('/home/m25lll/code/Fiting/outputs/aes_saddle')
out_dir.mkdir(parents=True, exist_ok=True)
best_action = result['best_theta']
estimator.reset()
estimator.current_dividing_level = -1
estimator.parse(action=best_action)
pts = estimator.rule.generate()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
o3d.io.write_point_cloud(str(out_dir / 'best_cloud.ply'), pcd)
print(f"Saved: {out_dir}/best_cloud.ply")
