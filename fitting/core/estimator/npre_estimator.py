from copy import deepcopy

from .point_cloud_core import PointCloudEstimatorCore, npre


class NPREEstimator:
    def __init__(self, cfg):
        object.__setattr__(self, "cfg", cfg)
        core = PointCloudEstimatorCore(cfg, owner=None, estimator_type="npre")
        object.__setattr__(self, "_core", core)
        core.set_rule(owner=self)

    def __getattr__(self, name):
        return getattr(self._core, name)

    def __setattr__(self, name, value):
        if name == "_core" or "_core" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._core, name, value)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        object.__setattr__(result, "cfg", deepcopy(self.cfg, memo))
        core = deepcopy(self._core, memo)
        object.__setattr__(result, "_core", core)
        if getattr(core, "rule", None) is not None:
            core.rule.estimator = result
        return result
