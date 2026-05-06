from .point_cloud_core import PointCloudEstimatorCore


class GDEstimator:
    def __init__(self, cfg):
        object.__setattr__(self, "cfg", cfg)
        core = PointCloudEstimatorCore(cfg, owner=None, estimator_type="gd")
        object.__setattr__(self, "_core", core)
        core.set_rule(owner=self)

    def __getattr__(self, name):
        return getattr(self._core, name)

    def __setattr__(self, name, value):
        if name == "_core" or "_core" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._core, name, value)
