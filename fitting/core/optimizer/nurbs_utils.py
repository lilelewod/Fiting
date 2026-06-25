"""NURBS knot insertion for coarse-to-fine refinement.

Refinement from 4×4 → 16×16 control grid while preserving surface shape exactly.
"""

import numpy as np


def _open_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    """Open-uniform knot vector for n control points of given degree."""
    n_knots = n_ctrl + degree + 1
    knots = np.zeros(n_knots, dtype=np.float64)
    n_inner = n_knots - 2 * (degree + 1)
    if n_inner > 0:
        knots[degree + 1 : degree + 1 + n_inner] = np.linspace(0, 1, n_inner + 2)[1:-1]
    knots[-(degree + 1):] = 1.0
    return knots


def _insert_knots_1d(Pw, degree, old_knots, new_knots):
    """Insert knots into a 1D rational B-spline (homogeneous coords).

    Parameters
    ----------
    Pw : (n, D) float
        Homogeneous control points [w*P, w].
    degree : int
    old_knots : (n + degree + 1,) float
    new_knots : (m,) float
        Knots to insert (sorted). Must lie within [old_knots[0], old_knots[-1]].

    Returns
    -------
    Qw : (n + len(new_knots), D) float
        New homogeneous control points.
    final_knots : (n + len(new_knots) + degree + 1,) float
    """
    Qw = np.asarray(Pw, dtype=np.float64).copy()
    knots = np.asarray(old_knots, dtype=np.float64).copy()
    n = len(Qw)

    for u_new in new_knots:
        n = len(Qw)
        # Find knot span index j: knots[j] <= u_new < knots[j+1]
        j = np.searchsorted(knots, u_new, side="right") - 1
        j = np.clip(j, degree, n - 1)

        # New CP array (size = n + 1)
        Qw_new = np.zeros((n + 1, Qw.shape[1]), dtype=np.float64)

        # For i <= j - degree: Qw_new[i] = Qw[i] (unchanged)
        Qw_new[: j - degree + 1] = Qw[: j - degree + 1]

        # For i = j - degree + 1, ..., j: blend
        for i in range(j - degree + 1, j + 1):
            alpha = (u_new - knots[i]) / (knots[i + degree] - knots[i] + 1e-12)
            alpha = np.clip(alpha, 0.0, 1.0)
            Qw_new[i] = (1.0 - alpha) * Qw[i - 1] + alpha * Qw[i]

        # For i >= j + 1: Qw_new[i] = Qw[i - 1] (shifted right)
        Qw_new[j + 1:] = Qw[j:]

        # Insert knot
        knots = np.insert(knots, j + 1, u_new)
        Qw = Qw_new

    return Qw, knots


def refine_surface_grid(
    control_points, weights,
    old_u, old_v,
    new_u, new_v,
    degree_u=3, degree_v=3,
):
    """Refine a NURBS surface control grid via knot insertion.

    Surface shape is EXACTLY preserved (new CPs are linear combinations of old).

    Parameters
    ----------
    control_points : (U, V, 3) float
    weights : (U, V) float
    old_u, old_v : int
        Original control grid dimensions.
    new_u, new_v : int
        Target control grid dimensions.
    degree_u, degree_v : int

    Returns
    -------
    new_ctrl : (new_u, new_v, 3) float
    new_weights : (new_u, new_v) float
    """
    old_knots_u = _open_uniform_knots(old_u, degree_u)
    old_knots_v = _open_uniform_knots(old_v, degree_v)
    new_knots_u = _open_uniform_knots(new_u, degree_u)
    new_knots_v = _open_uniform_knots(new_v, degree_v)

    # Knots to insert
    insert_u = np.setdiff1d(new_knots_u, old_knots_u)
    insert_v = np.setdiff1d(new_knots_v, old_knots_v)
    insert_u.sort()
    insert_v.sort()

    # Convert to homogeneous: [w*Px, w*Py, w*Pz, w]
    Pw = np.zeros((old_u, old_v, 4), dtype=np.float64)
    Pw[..., :3] = control_points * weights[..., np.newaxis]
    Pw[..., 3] = weights

    # ── Step 1: insert knots in u-direction (process each v-row) ──
    Pw_u = np.zeros((new_u, old_v, 4), dtype=np.float64)
    for vidx in range(old_v):
        Qw_row, _ = _insert_knots_1d(Pw[:, vidx, :], degree_u, old_knots_u, insert_u)
        Pw_u[:, vidx, :] = Qw_row

    # ── Step 2: insert knots in v-direction (process each u-column) ──
    Pw_uv = np.zeros((new_u, new_v, 4), dtype=np.float64)
    for uidx in range(new_u):
        Qw_col, _ = _insert_knots_1d(Pw_u[uidx, :, :], degree_v, old_knots_v, insert_v)
        Pw_uv[uidx, :, :] = Qw_col

    # Convert back from homogeneous
    new_ctrl = np.zeros((new_u, new_v, 3), dtype=np.float64)
    new_weights = np.zeros((new_u, new_v), dtype=np.float64)
    w = Pw_uv[..., 3]
    eps = 1e-12
    for d in range(3):
        new_ctrl[..., d] = Pw_uv[..., d] / (w + eps)
    new_weights = w

    return new_ctrl.astype(np.float32), new_weights.astype(np.float32)


def _test_refinement():
    """Quick test: refine 4×4 plane → 8×8, verify surface unchanged."""
    ctrl = np.zeros((4, 4, 3), dtype=np.float32)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), indexing="ij")
    ctrl[..., 0] = xx
    ctrl[..., 1] = yy
    ctrl[..., 2] = 0.0
    weights = np.ones((4, 4), dtype=np.float32)

    new_ctrl, new_w = refine_surface_grid(ctrl, weights, 4, 4, 8, 8)

    # Sample at parameter (0.5, 0.5) using B-spline evaluation
    from models.surface.nurbs_surface_rule import _basis_functions

    old_knots_u = _open_uniform_knots(4, 3)
    old_knots_v = _open_uniform_knots(4, 3)
    new_knots_u = _open_uniform_knots(8, 3)
    new_knots_v = _open_uniform_knots(8, 3)

    u_test, v_test = np.array([0.4]), np.array([0.6])
    b_u_old = _basis_functions(u_test, 4, 3, old_knots_u)
    b_v_old = _basis_functions(v_test, 4, 3, old_knots_v)
    b_u_new = _basis_functions(u_test, 8, 3, new_knots_u)
    b_v_new = _basis_functions(v_test, 8, 3, new_knots_v)

    # Evaluate
    old_pt = np.zeros(3)
    for i in range(4):
        for j in range(4):
            old_pt += b_u_old[0, i] * b_v_old[0, j] * ctrl[i, j] * weights[i, j]
    old_pt /= np.dot(b_u_old[0], np.dot(weights, b_v_old[0]))

    new_pt = np.zeros(3)
    for i in range(8):
        for j in range(8):
            new_pt += b_u_new[0, i] * b_v_new[0, j] * new_ctrl[i, j] * new_w[i, j]
    new_pt /= np.dot(b_u_new[0], np.dot(new_w, b_v_new[0]))

    print(f"Old point at (0.4, 0.6): {old_pt}")
    print(f"New point at (0.4, 0.6): {new_pt}")
    print(f"Error: {np.linalg.norm(old_pt - new_pt):.2e}")


if __name__ == "__main__":
    _test_refinement()
