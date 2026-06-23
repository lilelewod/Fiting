"""CylinderRule — 圆柱模型（经典基元，低维验证用）

参数 (7维): x0, y0, z0, azimuth, elevation, radius, length
对比 NURBS (144维): CCO/CS 在低维有效，高维失效。
"""

import numpy as np
from easydict import EasyDict

from ..rule import ModelRule, Token
from tools.tool import rescale


class CylinderTrait(EasyDict):
    def __init__(self):
        EasyDict.__init__(self)
        self.x0 = 0.        # 底面中心 x
        self.y0 = 0.        # 底面中心 y
        self.z0 = 0.        # 底面中心 z
        self.azimuth = 0.   # 轴方位角 [0, 2π)
        self.elevation = 0. # 轴仰角 [-π/2, π/2]
        self.radius = 1.    # 半径
        self.length = 1.    # 高度


def cylinder_sample(trait, sample_u=40, sample_v=20):
    """从圆柱面均匀采样。

    圆柱轴: (cos(azimuth)*cos(elevation), sin(azimuth)*cos(elevation), sin(elevation))
    采样: u∈[0,2π) 绕轴旋转, v∈[0,1] 沿轴方向
    """
    # 轴方向
    az, el = trait.azimuth, trait.elevation
    axis = np.array([np.cos(az) * np.cos(el),
                     np.sin(az) * np.cos(el),
                     np.sin(el)])
    axis = axis / np.linalg.norm(axis)

    # 构造两个正交方向
    if np.abs(axis[2]) < 0.9:
        ref = np.array([0., 0., 1.])
    else:
        ref = np.array([1., 0., 0.])
    u_dir = np.cross(axis, ref)
    u_dir = u_dir / np.linalg.norm(u_dir)
    v_dir = np.cross(axis, u_dir)
    v_dir = v_dir / np.linalg.norm(v_dir)

    base = np.array([trait.x0, trait.y0, trait.z0])

    # u: 绕轴角度, v: 沿轴距离
    u = np.linspace(0, 2 * np.pi, sample_u)
    v = np.linspace(0, trait.length, sample_v)
    uu, vv = np.meshgrid(u, v, indexing='ij')

    radius_vec = trait.radius * (np.cos(uu[..., None]) * u_dir +
                                  np.sin(uu[..., None]) * v_dir)
    axis_vec = vv[..., None] * axis
    points = base + radius_vec + axis_vec

    return points.reshape(-1, 3)


class CylinderRule(ModelRule):
    name = "cylinder"

    def __init__(self, estimator=None):
        ModelRule.__init__(self, estimator)
        self.trait = None
        self.action = None
        self.lb = None
        self.ub = None
        self.set_trait_range()

    def set_trait_range(self):
        # 鲁棒包围盒：5%/95%分位数，抗离群点
        data = self.estimator.get_data()
        lo = np.percentile(data, 5, axis=0)
        hi = np.percentile(data, 95, axis=0)
        extent = hi - lo
        padding = 0.25 * np.maximum(extent, self.estimator.resolution)

        lb_arr = np.zeros(7, dtype=np.float32)
        ub_arr = np.zeros(7, dtype=np.float32)

        lb_arr[0:3] = lo - padding
        ub_arr[0:3] = hi + padding
        lb_arr[3], ub_arr[3] = 0.0, 2.0 * np.pi
        lb_arr[4], ub_arr[4] = -np.pi / 2, np.pi / 2
        # 紧bounds: r ≤ min(水平范围)/2, h ≤ 垂直范围
        lb_arr[5] = 0.02
        ub_arr[5] = 0.6 * max(extent[0], extent[1])
        lb_arr[6] = 0.02
        ub_arr[6] = 1.5 * extent[2]

        self.lb = lb_arr
        self.ub = ub_arr

    def get_num_variables(self):
        return 7

    def parse(self, **kwargs):
        action = kwargs['action']
        assert action.size == 7
        flat = rescale(action, self.lb, self.ub).astype(float)

        trait = CylinderTrait()
        trait.x0 = flat[0]
        trait.y0 = flat[1]
        trait.z0 = flat[2]
        trait.azimuth = flat[3]
        trait.elevation = flat[4]
        trait.radius = flat[5]
        trait.length = flat[6]

        self.trait = trait
        self.action = action
        self.compute_top_dividing_level()
        return trait

    @staticmethod
    def measure(trait):
        """圆柱侧面积 = 2π × r × h"""
        return 2.0 * np.pi * trait.radius * trait.length

    def compute_top_dividing_level(self):
        self.top_level = np.asarray([4, 3], dtype=np.int64)

    def sample(self):
        level = self.compute_current_dividing_level().astype(np.int64)
        su = min(80, max(12, 2 ** (int(level[0]) + 1)))
        sv = min(40, max(6, 2 ** (int(level[1]) + 1)))
        return cylinder_sample(self.trait, sample_u=su, sample_v=sv)

    def generate(self):
        cloud = self.sample()
        token = Token(self.estimator.dimension)
        token.points = cloud
        token.trait = self.trait
        token.measure = self.measure(self.trait)
        token.action = self.action
        self.estimator.add_token(token)
        return cloud
