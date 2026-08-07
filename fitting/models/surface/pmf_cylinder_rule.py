"""PMF-style partial-cylinder rule.

The seven optimized variables are the cylinder base location (3), radius,
height, start angle, and angular span.  The end angle is start + span.  This
matches the dimensional structure of model M5 in Zhang et al. (PR 2019) while
using a span internally to avoid invalid end-before-start candidates.
"""

import numpy as np
from easydict import EasyDict

from ..rule import ModelRule, Token
from tools.tool import rescale


class PMFCylinderTrait(EasyDict):
    def __init__(self):
        super().__init__()
        self.x0 = 0.0
        self.y0 = 0.0
        self.z0 = 0.0
        self.radius = 1.0
        self.height = 1.0
        self.start_angle = -np.pi
        self.angular_span = 2.0 * np.pi
        self.end_angle = np.pi


def sample_partial_cylinder(trait, sample_angle=64, sample_height=32):
    """Sample midpoint locations on the lateral surface without seam duplicates."""
    du = trait.angular_span / sample_angle
    dz = trait.height / sample_height
    u = trait.start_angle + (np.arange(sample_angle) + 0.5) * du
    z = trait.z0 + (np.arange(sample_height) + 0.5) * dz
    uu, zz = np.meshgrid(u, z, indexing="ij")
    return np.column_stack(
        (
            trait.x0 + trait.radius * np.cos(uu).ravel(),
            trait.y0 + trait.radius * np.sin(uu).ravel(),
            zz.ravel(),
        )
    ).astype(np.float32)


class PMFCylinderRule(ModelRule):
    name = "pmf_cylinder"

    def __init__(self, estimator=None):
        super().__init__(estimator)
        self.trait = None
        self.action = None
        self.set_trait_range()

    def set_trait_range(self):
        data = np.asarray(self.estimator.get_data(), dtype=float)
        model_cfg = self.estimator.cfg.get("model", {})
        bounds = model_cfg.get("parameter_bounds", {})

        lo = np.percentile(data, 2.5, axis=0)
        hi = np.percentile(data, 97.5, axis=0)
        extent = np.maximum(hi - lo, self.estimator.resolution)
        center_pad = 0.15 * extent

        center_lb = np.asarray(bounds.get("center_lb", lo - center_pad), dtype=float)
        center_ub = np.asarray(bounds.get("center_ub", hi + center_pad), dtype=float)
        radius_lb = float(bounds.get("radius_lb", 0.05 * max(extent[0], extent[1])))
        radius_ub = float(bounds.get("radius_ub", 0.75 * max(extent[0], extent[1])))
        height_lb = float(bounds.get("height_lb", 0.05 * extent[2]))
        height_ub = float(bounds.get("height_ub", 1.25 * extent[2]))
        span_lb = float(bounds.get("span_lb", np.deg2rad(20.0)))
        span_ub = float(bounds.get("span_ub", 2.0 * np.pi))

        self.lb = np.asarray(
            [*center_lb, radius_lb, height_lb, -np.pi, span_lb], dtype=np.float32
        )
        self.ub = np.asarray(
            [*center_ub, radius_ub, height_ub, np.pi, span_ub], dtype=np.float32
        )
        if np.any(self.ub <= self.lb):
            raise ValueError("Every PMF cylinder upper bound must exceed its lower bound.")

    def get_num_variables(self):
        return 7

    def parse(self, **kwargs):
        action = np.asarray(kwargs["action"])
        if action.size != self.get_num_variables():
            raise ValueError("PMFCylinderRule expects seven normalized variables.")
        values = rescale(action, self.lb, self.ub).astype(float)
        trait = PMFCylinderTrait()
        trait.x0, trait.y0, trait.z0 = values[0:3]
        trait.radius = values[3]
        trait.height = values[4]
        trait.start_angle = values[5]
        trait.angular_span = values[6]
        trait.end_angle = trait.start_angle + trait.angular_span
        self.trait = trait
        self.action = action.copy()
        self.top_level = np.asarray([5, 4], dtype=np.int64)
        return trait

    @staticmethod
    def measure(trait):
        return trait.radius * trait.angular_span * trait.height

    def sample(self):
        level = self.compute_current_dividing_level().astype(np.int64)
        sample_angle = max(8, 2 ** (int(level[0]) + 1))
        sample_height = max(4, 2 ** (int(level[1]) + 1))
        return sample_partial_cylinder(trait=self.trait,
                                       sample_angle=sample_angle,
                                       sample_height=sample_height)

    def generate(self):
        cloud = self.sample()
        token = Token(self.estimator.dimension)
        token.points = cloud
        token.trait = self.trait
        token.measure = self.measure(self.trait)
        token.action = self.action
        self.estimator.add_token(token)
        return cloud
