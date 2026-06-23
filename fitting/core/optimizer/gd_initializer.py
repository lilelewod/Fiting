import math
import numpy as np
import torch


def _torch_signed_power(x, exponent):
    return torch.sign(x) * torch.pow(torch.clamp(torch.abs(x), min=1e-8), exponent)


def _euler_xyz_matrix(rotation):
    rx, ry, rz = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    row0 = torch.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], dim=-1)
    row1 = torch.stack([sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], dim=-1)
    row2 = torch.stack([-sy, cy * sx, cy * cx], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _sample_superquadric_batch(traits, eta_count, omega_count):
    device = traits.device
    dtype = traits.dtype
    translation = traits[:, 0:3]
    rotation = traits[:, 3:6]
    scale = torch.clamp(traits[:, 6:9], min=1e-6)
    eps1 = torch.clamp(traits[:, 9], min=0.05)
    eps2 = torch.clamp(traits[:, 10], min=0.05)
    eta_start = traits[:, 11]
    eta_end = traits[:, 12]
    omega_start = traits[:, 13]
    omega_end = traits[:, 14]

    eta_unit = torch.linspace(0.0, 1.0, eta_count, device=device, dtype=dtype)
    omega_unit = torch.linspace(0.0, 1.0, omega_count, device=device, dtype=dtype)
    eta = eta_start[:, None] + (eta_end - eta_start)[:, None] * eta_unit[None, :]
    omega = omega_start[:, None] + (omega_end - omega_start)[:, None] * omega_unit[None, :]
    eta_grid = eta[:, :, None]
    omega_grid = omega[:, None, :]

    cos_eta = _torch_signed_power(torch.cos(eta_grid), eps1[:, None, None])
    x = scale[:, 0, None, None] * cos_eta * _torch_signed_power(torch.cos(omega_grid), eps2[:, None, None])
    y = scale[:, 1, None, None] * cos_eta * _torch_signed_power(torch.sin(omega_grid), eps2[:, None, None])
    z = scale[:, 2, None, None] * _torch_signed_power(torch.sin(eta_grid), eps1[:, None, None])
    z = z.expand_as(x)
    local = torch.stack([x, y, z], dim=-1).reshape(traits.shape[0], -1, 3)
    rot = _euler_xyz_matrix(rotation)
    return torch.matmul(local, rot.transpose(1, 2)) + translation[:, None, :]


def _actions_to_traits(actions, lb, ub):
    return (ub - lb) * (actions + 1.0) * 0.5 + lb


def _canonicalize_patch_bounds(traits, min_width=1e-5):
    out = traits.clone()
    eta_pair, _ = torch.sort(out[:, 11:13], dim=1)
    eta_start = torch.clamp(eta_pair[:, 0], -math.pi / 2.0, math.pi / 2.0 - min_width)
    eta_end = torch.clamp(eta_pair[:, 1], -math.pi / 2.0 + min_width, math.pi / 2.0)
    eta_end = torch.maximum(eta_end, eta_start + min_width)
    omega_pair, _ = torch.sort(out[:, 13:15], dim=1)
    omega_start = torch.clamp(omega_pair[:, 0], -math.pi, math.pi - min_width)
    omega_end = torch.clamp(omega_pair[:, 1], -math.pi + min_width, math.pi)
    omega_end = torch.maximum(omega_end, omega_start + min_width)
    out[:, 11] = eta_start
    out[:, 12] = eta_end
    out[:, 13] = omega_start
    out[:, 14] = omega_end
    return out


def _traits_to_actions(traits, lb, ub):
    actions = 2.0 * (traits - lb) / torch.clamp(ub - lb, min=1e-12) - 1.0
    return torch.clamp(actions, -1.0, 1.0)


def gradient_initialize_actions(estimator, num_actions, cfg, rng=None):
    if num_actions <= 0:
        return np.empty((0, estimator.num_variables()), dtype=np.float64)
    rng = np.random.default_rng() if rng is None else rng
    gd_cfg = cfg.get('gradient_initializer', {})
    enabled = gd_cfg.get('enabled', False)
    if not enabled:
        return np.empty((0, estimator.num_variables()), dtype=np.float64)

    device_name = gd_cfg.get('device', None) or cfg.get('estimator', {}).get('device', None)
    if device_name is None:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    dtype = torch.float32
    dim = estimator.num_variables()
    lb_np = estimator.rule.lb.astype(np.float32)
    ub_np = estimator.rule.ub.astype(np.float32)
    lb = torch.as_tensor(lb_np, device=device, dtype=dtype)
    ub = torch.as_tensor(ub_np, device=device, dtype=dtype)
    data = np.asarray(estimator.data, dtype=np.float32)
    max_data_points = int(gd_cfg.get('max_data_points', 4096))
    if data.shape[0] > max_data_points:
        idx = rng.choice(data.shape[0], size=max_data_points, replace=False)
        data = data[idx]
    data_t = torch.as_tensor(data, device=device, dtype=dtype)

    num_candidates = int(gd_cfg.get('num_candidates', max(num_actions, 16)))
    steps = int(gd_cfg.get('steps', 80))
    lr = float(gd_cfg.get('lr', 0.04))
    eta_count = int(gd_cfg.get('eta_samples', 10))
    omega_count = int(gd_cfg.get('omega_samples', 20))
    batch_size = int(gd_cfg.get('batch_size', min(num_candidates, 16)))
    best_actions = []
    best_losses = []

    for begin in range(0, num_candidates, batch_size):
        cur = min(batch_size, num_candidates - begin)
        init = rng.uniform(-1.0, 1.0, size=(cur, dim)).astype(np.float32)
        z0 = np.arctanh(np.clip(init, -0.999, 0.999))
        z = torch.nn.Parameter(torch.as_tensor(z0, device=device, dtype=dtype))
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            actions = torch.tanh(z)
            traits = _canonicalize_patch_bounds(_actions_to_traits(actions, lb, ub))
            points = _sample_superquadric_batch(traits, eta_count, omega_count)
            dist = torch.cdist(points, data_t[None, :, :].expand(cur, -1, -1))
            loss_each = torch.min(dist, dim=2).values.mean(dim=1)
            loss = loss_each.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            actions = torch.tanh(z)
            traits = _canonicalize_patch_bounds(_actions_to_traits(actions, lb, ub))
            actions = _traits_to_actions(traits, lb, ub)
            points = _sample_superquadric_batch(traits, eta_count, omega_count)
            dist = torch.cdist(points, data_t[None, :, :].expand(cur, -1, -1))
            losses = torch.min(dist, dim=2).values.mean(dim=1)
            best_actions.append(actions.detach().cpu().numpy())
            best_losses.append(losses.detach().cpu().numpy())

    actions = np.vstack(best_actions)
    losses = np.concatenate(best_losses)
    order = np.argsort(losses)[:num_actions]
    return actions[order].astype(np.float64)
