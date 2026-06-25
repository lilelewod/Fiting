#!/usr/bin/env python3
"""生成超二次曲面合成点云数据集

    python datasets/synthetic/generate_superquadric.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import open3d as o3d
from models.surface.superquadric_rule import SuperquadricRule, SuperquadricTrait

OUT_DIR = PROJECT_ROOT / 'datasets/synthetic'


def save_ply(path: str, pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(path, pcd)


def generate(target_n: int, a1: float, a2: float, a3: float,
             e1: float, e2: float, cx: float = 0., cy: float = 0., cz: float = 0.,
             rx: float = 0., ry: float = 0., rz: float = 0.) -> np.ndarray:
    """Generate superquadric point cloud."""
    from scipy.spatial.transform import Rotation

    # Compute n_eta, n_omega to get ~target_n points (n_eta * n_omega ≈ target_n)
    n = int(np.sqrt(target_n))
    n_eta, n_omega = n, n

    pts, _ = SuperquadricRule._spherical_product(a1, a2, a3, e1, e2, n_eta, n_omega)
    rot = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix().astype(np.float32)
    pts = pts @ rot.T + np.array([cx, cy, cz], dtype=np.float32)
    return pts.astype(np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = {
        # name: (a1, a2, a3, e1, e2, rx, ry, rz)
        'superq_ellipsoid': (1.5, 1.0, 0.8, 1.0, 1.0, 0., 0., 0.),
        'superq_box':      (1.5, 1.0, 0.8, 0.2, 0.2, 0., 0., 0.),
        'superq_cylinder': (1.2, 1.2, 1.5, 1.0, 0.1, 0., 0., 0.),
        'superq_diamond':  (1.5, 1.0, 0.8, 1.8, 1.8, 0., 0., 0.),
        'superq_pillow':   (1.5, 1.0, 0.7, 1.3, 1.3, 0.3, 0.2, 0.),
    }

    for name, (a1, a2, a3, e1, e2, rx, ry, rz) in variants.items():
        # Clean
        pts = generate(3000, a1, a2, a3, e1, e2, rx=rx, ry=ry, rz=rz)
        path = OUT_DIR / f'{name}_3k.ply'
        save_ply(str(path), pts)
        print(f"  {path.name}: {len(pts)} points  ε=({e1},{e2})  scale=({a1},{a2},{a3})")

        # Noisy (Gaussian noise 1% of data extent)
        noise_std = 0.01 * float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        pts_noise = pts + np.random.randn(*pts.shape).astype(np.float32) * noise_std
        path_n = OUT_DIR / f'{name}_noise_3k.ply'
        save_ply(str(path_n), pts_noise)
        print(f"  {path_n.name}: {len(pts_noise)} points  noise_std={noise_std:.4f}")

        # Outlier (10% scattered points)
        bbox = pts.max(axis=0) - pts.min(axis=0)
        n_out = 300
        outliers = pts.min(axis=0) + np.random.rand(n_out, 3).astype(np.float32) * bbox * 2.0
        pts_ol = np.vstack([pts, outliers]).astype(np.float32)
        path_o = OUT_DIR / f'{name}_outlier_3k.ply'
        save_ply(str(path_o), pts_ol)
        print(f"  {path_o.name}: {len(pts_ol)} points  ({n_out} outliers)")

    print("\nDone.")


if __name__ == '__main__':
    main()
