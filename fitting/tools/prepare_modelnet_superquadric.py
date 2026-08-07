"""Prepare a reproducible ModelNet40 subset for single-superquadric fitting."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


DEFAULT_CATEGORIES = ('bottle', 'bowl', 'cone', 'flower_pot', 'glass_box', 'vase', 'xbox')


def write_ply(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertices['x'], vertices['y'], vertices['z'] = points.T.astype(np.float32)
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(path))


def sample_surface(vertices, triangles, count, rng):
    tri = vertices[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = np.isfinite(areas) & (areas > np.finfo(np.float64).eps)
    if not np.any(valid):
        raise ValueError('mesh contains no positive-area triangles')
    tri = tri[valid]
    probabilities = areas[valid] / areas[valid].sum()
    chosen = rng.choice(tri.shape[0], size=count, replace=True, p=probabilities)
    selected = tri[chosen]
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    return (
        (1.0 - r1)[:, None] * selected[:, 0]
        + (r1 * (1.0 - r2))[:, None] * selected[:, 1]
        + (r1 * r2)[:, None] * selected[:, 2]
    )


def load_and_normalize_mesh(path, rng, random_rotation):
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    # ModelNet OFF files may contain isolated vertices that are not referenced
    # by any face. They must not influence normalization of the sampled surface.
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.shape[0] < 4 or triangles.shape[0] < 4:
        raise ValueError('mesh is empty or too small')
    if not np.all(np.isfinite(vertices)):
        raise ValueError('mesh contains non-finite vertices')
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = 0.5 * (lo + hi)
    diagonal = float(np.linalg.norm(hi - lo))
    if diagonal <= np.finfo(np.float64).eps:
        raise ValueError('mesh has zero bounding-box diagonal')
    vertices = (vertices - center) / diagonal
    matrix = Rotation.random(random_state=rng).as_matrix() if random_rotation else np.eye(3)
    vertices = vertices @ matrix.T
    return vertices, triangles, center, diagonal, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--categories', nargs='+', default=list(DEFAULT_CATEGORIES))
    parser.add_argument('--models-per-category', type=int, default=20)
    parser.add_argument('--fit-points', type=int, default=5000)
    parser.add_argument('--reference-points', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260715)
    parser.add_argument('--no-random-rotation', action='store_true')
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)
    rows = []

    for category in args.categories:
        candidates = sorted((source_root / category / 'test').glob('*.off'))
        if len(candidates) < args.models_per_category:
            raise ValueError(f'{category} has {len(candidates)} test models, fewer than requested')
        selected_ids = master_rng.choice(len(candidates), args.models_per_category, replace=False)
        for index in sorted(selected_ids):
            source = candidates[int(index)]
            model_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            rng = np.random.default_rng(model_seed)
            row = {'category': category, 'model': source.stem, 'source': str(source), 'seed': model_seed}
            try:
                vertices, triangles, center, diagonal, matrix = load_and_normalize_mesh(
                    source, rng, not args.no_random_rotation
                )
                fit_points = sample_surface(vertices, triangles, args.fit_points, rng)
                reference_points = sample_surface(vertices, triangles, args.reference_points, rng)
                model_root = output_root / category / source.stem
                fit_file = model_root / 'fit_clean.ply'
                reference_file = model_root / 'reference.ply'
                write_ply(fit_file, fit_points)
                write_ply(reference_file, reference_points)
                row.update({
                    'status': 'ok',
                    'fit_file': str(fit_file),
                    'reference_file': str(reference_file),
                    'vertices': int(vertices.shape[0]),
                    'triangles': int(triangles.shape[0]),
                    'fit_points': args.fit_points,
                    'reference_points': args.reference_points,
                    'source_bbox_center': center.tolist(),
                    'source_bbox_diagonal': diagonal,
                    'rotation_matrix': matrix.tolist(),
                })
            except Exception as exc:
                row.update({'status': 'error', 'error': str(exc)})
            rows.append(row)
            print(f"{category}/{source.stem}: {row['status']}")

    with open(output_root / 'manifest.json', 'w', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2)
    csv_fields = ['category', 'model', 'status', 'seed', 'source', 'fit_file', 'reference_file',
                  'vertices', 'triangles', 'fit_points', 'reference_points', 'source_bbox_diagonal', 'error']
    with open(output_root / 'manifest.csv', 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(row['status'] != 'ok' for row in rows)
    print(f'Prepared {len(rows) - failures}/{len(rows)} models in {output_root}; failures={failures}')


if __name__ == '__main__':
    main()
