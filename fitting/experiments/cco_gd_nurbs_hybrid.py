"""CCO → GD 混合 NURBS 拟合实验

CCO 全局探索 + GD 梯度精调，解决纯CCO的离群点和表面凌乱问题。

    cd /home/m25lll/code/Fiting/fitting && python experiments/cco_gd_nurbs_hybrid.py
"""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_nurbs_cfg(algo='cco'):
    """构建 NURBS 实验配置"""
    with open(PROJECT_ROOT / 'configs/fit_mm_compare.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    from core.estimator.mm_estimator import MeanMeasureEstimator
    from models.surface.nurbs_surface_rule import NURBSSurfaceRule
    from tools.data_tool import load_ply_data as load_data

    cfg['fitter']['algo_name'] = algo
    cfg['fitter']['max_episode'] = 50000 if algo == 'cco' else 5000  # CCO粗搜 + GD精调
    cfg['fitter']['num_envs'] = 8
    cfg['fitter']['episodes_per_env'] = 50
    cfg['fitter']['gd_smoothness_weight'] = 0.05
    cfg['fitter']['gd_lr'] = 0.01

    model_type = cfg['model']['type']
    data_file = cfg['data_file']
    run_id = cfg['run_id']
    data_path = Path(data_file)

    cfg['estimator']['data_file'] = data_file
    cfg['estimator']['rule_class'] = NURBSSurfaceRule
    cfg['estimator']['estimator_class'] = MeanMeasureEstimator
    cfg['estimator']['estimator_instance'] = None
    cfg['estimator']['load_data_fn'] = load_data
    cfg['record']['root_dir'] = (
        f"/home/m25lll/code/Fiting/outputs/hybrid_nurbs/{algo}/"
        f"{model_type}/{data_path.parent.name}/{data_path.stem}/run_{run_id}/"
    )
    cfg['record']['visualization'] = None
    return cfg


def run_cco_global(cfg):
    """Phase 1: CCO 全局搜索"""
    from core.optimizer.cco_fitter import Fitter as CCOFitter

    print("  [Phase 1] CCO global search (50k evals)...")
    t0 = time.perf_counter()
    fitter = CCOFitter(cfg)
    fitter.fit()
    runtime = time.perf_counter() - t0

    best_action = fitter.best_action_.copy()
    cco_score = float(fitter.record.best_score)

    # 提取 best 的 control_points 和 weights
    fitter.estimator.reset()
    fitter.estimator.current_dividing_level = -1
    fitter.estimator.parse(action=best_action)
    fitter.estimator.generate(current_dividing_level=-1)
    rule = fitter.estimator.rule
    best_ctrl = rule.trait.control_points.copy()
    best_weights = rule.trait.weights.copy()

    fitter.close()
    print(f"  CCO done: score={cco_score:.2f}, ctrl={best_ctrl.shape}, time={runtime:.1f}s")
    return best_action, best_ctrl, best_weights, cco_score


def run_gd_refine(base_cfg, init_ctrl, init_weights, gd_steps=5000):
    """Phase 2: GD 从 CCO 结果出发精调 NURBS"""
    print(f"  [Phase 2] GD refinement ({gd_steps} steps)...")
    t0 = time.perf_counter()

    cfg = deepcopy(base_cfg)
    cfg['fitter']['algo_name'] = 'gd'
    cfg['fitter']['max_episode'] = gd_steps
    cfg['record']['root_dir'] = cfg['record']['root_dir'].replace('/cco/', '/cco_gd_hybrid/')

    from core.optimizer.gd_fitter import Fitter as GDFitter
    from core.record import Record, SubRecord

    gd = GDFitter(cfg)

    # Override init: use CCO's control_points + weights
    orig_optimize_nurbs = gd._optimize_nurbs

    def _optimize_nurbs_with_init():
        target_points_np = gd._target_points_for_instance()
        target_points = torch.as_tensor(target_points_np, dtype=torch.float32, device=gd.device)
        use_full_batch = gd.data_batch_size <= 0 or target_points.shape[0] <= gd.data_batch_size

        # ── 用 CCO 的结果替代 SVD init ──
        control_points = torch.nn.Parameter(
            torch.as_tensor(init_ctrl, dtype=torch.float32, device=gd.device))
        w_clamped = np.clip((init_weights - gd.weight_lb) / (gd.weight_ub - gd.weight_lb), 1e-4, 1 - 1e-4)
        weights_raw = torch.nn.Parameter(
            torch.as_tensor(np.log(w_clamped / (1 - w_clamped)), dtype=torch.float32, device=gd.device))

        optimizer = torch.optim.Adam([control_points, weights_raw], lr=gd.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(gd.max_steps, 1), eta_min=gd.lr * gd.lr_min_factor)

        sub_record = SubRecord(gd.cfg, env_id=0)
        sub_record.data_cloud = gd.record.data_cloud
        best_score = float("-inf")
        best_ctrl = None
        best_weights = None
        steps_without_improvement = 0

        for step in range(1, gd.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)

            if use_full_batch:
                data_batch = target_points
            else:
                perm = torch.randperm(target_points.shape[0], device=gd.device)[:gd.data_batch_size]
                data_batch = target_points[perm]

            control = torch.max(torch.min(control_points, gd.ctrl_ub), gd.ctrl_lb)
            weights = gd.weight_lb + (gd.weight_ub - gd.weight_lb) * torch.sigmoid(weights_raw)
            model_points = gd._sample_surface(control, weights)
            raw_measure = gd._compute_measure(control, weights)

            loss = gd._soft_mm_loss(model_points, raw_measure, data_batch, step)

            second_u = control[2:, :, :] - 2.0 * control[1:-1, :, :] + control[:-2, :, :]
            second_v = control[:, 2:, :] - 2.0 * control[:, 1:-1, :] + control[:, :-2, :]
            smoothness = (second_u.norm(dim=-1).mean() + second_v.norm(dim=-1).mean()) / max(
                float(torch.linalg.norm(gd.data_max - gd.data_min).item()),
                gd.data_resolution, np.finfo(np.float32).eps)
            loss = loss + gd.smoothness_weight * smoothness

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % gd.eval_interval == 0 or step == 1 or step == gd.max_steps:
                with torch.no_grad():
                    control_eval = torch.max(torch.min(control_points, gd.ctrl_ub), gd.ctrl_lb).detach().cpu().numpy()
                    weights_eval = (gd.weight_lb + (gd.weight_ub - gd.weight_lb) *
                                    torch.sigmoid(weights_raw)).detach().cpu().numpy()
                score = gd._evaluate_candidate(control_eval, weights_eval)
                sub_record.update(score, gd.estimator)
                gd.record.update(sub_record, 1)

                if score > best_score + 1e-8:
                    best_score = score
                    best_ctrl = control_eval.copy()
                    best_weights = weights_eval.copy()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                print(f"  GD Step: {step}/{gd.max_steps}, Loss: {loss.item():.6f}, Score: {score:.4f}",
                      end="\r", flush=True)

                if gd.patience > 0 and steps_without_improvement >= gd.patience:
                    print(f"\n  Early stop at step {step}, best={best_score:.4f}")
                    break

        if best_ctrl is not None and best_weights is not None:
            score = gd._evaluate_candidate(best_ctrl, best_weights)
            if score > best_score:
                best_score = score
        return best_score

    gd._optimize_nurbs = _optimize_nurbs_with_init
    gd._optimize = lambda: gd._optimize_nurbs()  # route to nurbs

    # Run GD refinement directly
    gd.estimator.instance_index = 0
    gd.estimator.reset()
    hybrid_score = gd._optimize_nurbs()
    runtime = time.perf_counter() - t0

    gd.close()
    print(f"\n  GD done: hybrid_score={hybrid_score:.2f}, time={runtime:.1f}s")
    return hybrid_score


def main():
    print("=" * 60)
    print("CCO → GD 混合 NURBS 拟合 (8×8 grid, 256D)")
    print("=" * 60)

    # Phase 1: CCO
    cco_cfg = build_nurbs_cfg('cco')
    _, init_ctrl, init_weights, cco_score = run_cco_global(cco_cfg)

    # Phase 2: GD from CCO init
    gd_cfg = build_nurbs_cfg('gd')
    hybrid_score = run_gd_refine(gd_cfg, init_ctrl, init_weights, gd_steps=5000)

    # ── Pure GD baseline (SVD init) ──
    print(f"\n  [Baseline] Pure GD (SVD init, 5000 steps)...")
    t0 = time.perf_counter()
    from core.optimizer.gd_fitter import Fitter as GDFitter
    gd_pure_cfg = build_nurbs_cfg('gd')
    gd_pure = GDFitter(gd_pure_cfg)
    gd_pure.record.token_index = 0
    gd_pure.record.best_score = 0.0
    gd_pure.record.best_sub_record = -1
    gd_pure.estimator.instance_index = 0
    gd_pure.estimator.reset()
    gd_pure_score = gd_pure._optimize_nurbs()
    gd_pure_time = time.perf_counter() - t0
    gd_pure.close()
    print(f"  Pure GD: score={gd_pure_score:.2f}, time={gd_pure_time:.1f}s")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("NURBS 混合优化结果 (saddle_3k, 8×8 grid)")
    print("=" * 60)
    print(f"{'Method':>20}  {'Score':>10}")
    print("-" * 35)
    print(f"{'CCO only':>20}  {cco_score:>10.2f}")
    print(f"{'Pure GD (SVD)':>20}  {gd_pure_score:>10.2f}")
    print(f"{'CCO → GD hybrid':>20}  {hybrid_score:>10.2f}")


if __name__ == '__main__':
    main()
