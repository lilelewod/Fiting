"""CCO + Memetic 混合优化实验 — Character (46D)

CCO 全局探索 → Adam/L-BFGS 局部精调

    cd /home/m25lll/code/Fiting/fitting && python experiments/cco_memetic_hybrid.py
"""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_cfg(algo, refine_method="adam", refine_steps=200, num_workers=4):
    with open(PROJECT_ROOT / 'configs/fit_character.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    from core.estimator.mm_estimator import MeanMeasureEstimator
    from models.character.character_rule import CharacterRule as Rule
    from tools.data_tool import load_image_data as load_data

    cfg['fitter']['algo_name'] = algo
    cfg['fitter']['num_instances'] = 1
    cfg['fitter']['num_envs'] = 8
    cfg['fitter']['episodes_per_env'] = 10
    cfg['fitter']['max_episode'] = 50000
    cfg['fitter']['mem_pop_size'] = 100
    cfg['fitter']['mem_min_pop'] = 20
    cfg['fitter']['mem_refine_every'] = 999  # 不做refine，只用CCO
    cfg['fitter']['mem_refine_steps'] = refine_steps
    cfg['fitter']['mem_refine_method'] = refine_method
    cfg['fitter']['mem_refine_top_k'] = 1
    cfg['fitter']['mem_adaptive_K'] = False
    cfg['fitter']['mem_num_workers'] = num_workers
    cfg['fitter']['gd_lr'] = 0.01

    run_id = cfg['run_id']; test_id = cfg['test_id']
    nt = cfg['noise_type']; nl = cfg['noise_level']
    algo_name = algo.upper()
    cfg['rule']['token_file'] = f"datasets/character/test/run{run_id}_test{test_id}_1.mat"
    cfg['estimator']['data_file'] = f"datasets/character/test/{nt}/{nl}/{test_id - 1}/noisy_{run_id}.png"
    cfg['record']['root_dir'] = f"/home/m25lll/code/Fiting/outputs/hybrid_test/{algo_name}_{refine_method}/"
    cfg['estimator']['rule_class'] = Rule
    cfg['estimator']['estimator_class'] = MeanMeasureEstimator
    cfg['estimator']['estimator_instance'] = None
    cfg['estimator']['load_data_fn'] = load_data
    cfg['record']['visualization'] = None
    return cfg


def cco_global_search(cfg):
    """Run CCO global search, return best action + score."""
    from core.optimizer.cco_fitter import Fitter as CCOFitter

    print("  [Phase 1] CCO global search (50k evals)...")
    t0 = time.perf_counter()
    fitter = CCOFitter(cfg)
    fitter.fit()
    runtime = time.perf_counter() - t0

    best_action = fitter.best_action_
    best_score = fitter.record.best_score
    fitter.close()
    print(f"  CCO done: score={best_score:.2f}, time={runtime:.1f}s")
    return best_action, best_score, runtime


def memetic_refine(cfg, init_action, refine_method="adam", refine_steps=200):
    """Adam/L-BFGS local refinement from given init action."""
    import torch
    from core.optimizer.gd_fitter import Fitter as GDFitter
    from core.optimizer.memetic_fitter import AdamRefiner

    print(f"  [Phase 2] {refine_method.upper()} refinement ({refine_steps} steps)...")
    t0 = time.perf_counter()

    # Build GDFitter for forward function
    ref_cfg = deepcopy(cfg)
    if "raw_device" in ref_cfg:
        ref_cfg["device"] = ref_cfg["raw_device"]
    ref_cfg["seeds"] = None
    ref_cfg["fitter"]["max_episode"] = refine_steps

    gd = GDFitter(ref_cfg)
    data_t = torch.as_tensor(gd.estimator.get_data(), dtype=torch.float32, device=gd.device)

    def fwd(a):
        return gd._character_forward(a)

    refiner = AdamRefiner(
        forward_fn=fwd,
        data_tensor=data_t,
        data_resolution=float(gd.estimator.data_resolution),
        lr=0.01,
        max_steps=refine_steps,
        method=refine_method,
        device=gd.device,
    )

    refined = refiner.refine(init_action.copy())

    # Evaluate refined action
    gd.estimator.reset()
    gd.estimator.current_dividing_level = -1
    gd.estimator.parse(action=np.clip(refined, -1, 1).astype(np.float32))
    gd.estimator.generate(current_dividing_level=-1)
    refined_score = float(gd.estimator.get_score())

    runtime = time.perf_counter() - t0
    gd.close()
    print(f"  Refine done: {init_action.shape} → refined, score={refined_score:.2f}, time={runtime:.1f}s")
    return refined, refined_score, runtime


def main():
    print("=" * 60)
    print("CCO + Memetic 混合优化 — Character (46D)")
    print("=" * 60)

    methods = [
        ("adam", 200),
        ("lbfgs", 200),
        ("cascade", 200),
    ]

    results = {}
    for method, steps in methods:
        print(f"\n{'─'*50}")
        print(f"Testing: CCO → {method.upper()}")
        print(f"{'─'*50}")

        cfg = build_cfg("cco", refine_method=method, refine_steps=steps)
        best_action, cco_score, cco_time = cco_global_search(cfg)
        refined_action, hybrid_score, refine_time = memetic_refine(
            cfg, best_action, refine_method=method, refine_steps=steps
        )
        results[method] = {
            "cco_score": cco_score,
            "hybrid_score": hybrid_score,
            "delta": hybrid_score - cco_score,
            "cco_time": cco_time,
            "refine_time": refine_time,
            "total_time": cco_time + refine_time,
        }

    # ── Summary ──
    print("\n" + "=" * 65)
    print("CCO + Memetic 混合优化结果")
    print("=" * 65)
    print(f"{'Method':>20}  {'CCO':>8}  {'Hybrid':>8}  {'Δ':>8}  {'Time':>8}")
    print("-" * 60)
    for method, r in results.items():
        print(f"{'CCO → '+method.upper():>20}  {r['cco_score']:>8.1f}  {r['hybrid_score']:>8.1f}  {r['delta']:>+7.1f}  {r['total_time']:>7.1f}s")


if __name__ == '__main__':
    main()
