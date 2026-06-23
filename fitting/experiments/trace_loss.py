"""追踪 soft loss / hard loss / MM score 在训练过程中的演化"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from copy import deepcopy
from core.estimator.mm_estimator import MeanMeasureEstimator
from models.surface.nurbs_surface_rule import NURBSSurfaceRule
from tools.data_tool import load_ply_data as load_data
from core.optimizer.gd_fitter import Fitter, _inverse_rescale

cfg = {
    'estimator': {
        'type': 'mm', 'data_resolution': 0.01, 'model_resolution': 0.004,
        'regularization_factor': 1.2, 'early_rejection': False, 'use_faiss': False,
        'rule_class': NURBSSurfaceRule, 'estimator_class': MeanMeasureEstimator,
        'estimator_instance': None, 'load_data_fn': load_data,
        'data_file': 'fitting/datasets/synthetic/saddle_3k.ply',
    },
    'device': {'train_device': 'cuda:1', 'cuda_deterministic': True},
    'fitter': {'gd_init': 'svd', 'gd_data_batch_size': 4096, 'gd_lr': 0.01,
               'gd_lr_min_factor': 0.1, 'gd_smoothness_weight': 0.05,
               'num_instances': 1, 'max_episode': 5000,
               'algo_name': 'gd', 'num_envs': 1, 'episodes_per_env': 1},
    'model': {'type': 'nurbs_surface', 'num_ctrl_u': 6, 'num_ctrl_v': 6,
              'degree_u': 3, 'degree_v': 3, 'sample_u': 60, 'sample_v': 60,
              'weight_lb': 0.8, 'weight_ub': 1.2},
    'record': {'root_dir': '/tmp/gd_trace/', 'pulse_size': 2000, 'verbose': False, 'trim_final_mesh': False},
    'seeds': [42], 'task_type': '3d',
}

fitter = Fitter(deepcopy(cfg))
pts_all = fitter.estimator.get_data()
target_points = torch.as_tensor(pts_all, dtype=torch.float32, device=fitter.device)

init_ctrl = fitter._initial_control_grid(pts_all)
control_points = torch.nn.Parameter(
    torch.as_tensor(init_ctrl, dtype=torch.float32, device=fitter.device))
init_weights = torch.ones((6, 6), dtype=torch.float32, device=fitter.device)
weights_raw = torch.nn.Parameter(
    torch.logit(((init_weights - 0.8) / (1.2 - 0.8)).clamp(1e-4, 1 - 1e-4)))

opt = torch.optim.Adam([control_points, weights_raw], lr=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=0.001)

eps = 1e-8
alpha = 1.2
res = 0.01
prev_score = 0
best_score = 0
best_step = 0

print(f"{'step':>5}  {'soft_loss':>10}  {'hard_loss':>10}  {'MM':>10}  {'tau':>8}  {'measure':>10}")
for step in range(1, 5001):
    opt.zero_grad()
    control = torch.max(torch.min(control_points, fitter.ctrl_ub), fitter.ctrl_lb)
    weights = 0.8 + (1.2 - 0.8) * torch.sigmoid(weights_raw)
    model_pts = fitter._sample_surface(control, weights)
    raw_meas = fitter._compute_measure(control, weights)
    perm = torch.randperm(target_points.shape[0], device=fitter.device)[:4096]
    data_batch = target_points[perm]

    tau_start = res * 3.0
    tau_end = res * 0.5
    progress = (step - 1) / 4999
    tau = tau_start * (1 - progress) + tau_end * progress

    diff = model_pts.unsqueeze(1) - data_batch.unsqueeze(0)
    dist = torch.sqrt((diff ** 2).sum(-1) + eps)

    # hard loss (monitoring only)
    hard_dist, _ = dist.min(dim=1)
    hard_error = hard_dist.mean()
    safe_meas = raw_meas.clamp(min=eps)
    hard_loss_val = hard_error / (safe_meas ** alpha + eps)

    # soft loss (training)
    logits = -dist / tau
    soft_assign = torch.softmax(logits, dim=1)
    soft_error = (soft_assign * dist).sum(dim=1).mean()
    soft_loss_val = soft_error / (safe_meas ** alpha + eps)

    # smoothness
    second_u = control[2:, :, :] - 2 * control[1:-1, :, :] + control[:-2, :, :]
    second_v = control[:, 2:, :] - 2 * control[:, 1:-1, :] + control[:, :-2, :]
    smoothness = (second_u.norm(dim=-1).mean() + second_v.norm(dim=-1).mean()) / max(
        float(torch.linalg.norm(fitter.data_max - fitter.data_min).item()), res, eps)

    loss = soft_loss_val + 0.05 * smoothness
    loss.backward()
    opt.step()
    sched.step()

    if step == 1 or step % 200 == 0 or step == 5000:
        control_np = control.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        trait_flat = np.concatenate(
            [control_np.reshape(-1), weights_np.reshape(-1)]).astype(np.float32)
        action = _inverse_rescale(trait_flat, fitter.rule.lb, fitter.rule.ub)
        fitter.estimator.reset()
        fitter.estimator.current_dividing_level = -1
        fitter.estimator.parse(action=action)
        fitter.estimator.generate(current_dividing_level=-1)
        mm_score = float(fitter.estimator.get_score())

        flag = ''
        if mm_score > best_score + 0.001:
            best_score = mm_score
            best_step = step
            flag = ' *** NEW BEST'
        elif mm_score < prev_score - 0.01:
            flag = ' (degrading)'
        prev_score = mm_score

        print(f"{step:>5}  {soft_loss_val.item():>10.6f}  {hard_loss_val.item():>10.6f}  "
              f"{mm_score:>10.4f}  {tau:>8.4f}  {safe_meas.item():>10.1f}{flag}")

print(f"\nBest MM={best_score:.4f} at step {best_step}")
