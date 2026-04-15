from .npre_estimator import NPREEstimator, npre

try:
    from .mm_estimator import MeanMeasureEstimator
except ModuleNotFoundError:
    MeanMeasureEstimator = None

Estimator = NPREEstimator

__all__ = ["Estimator", "NPREEstimator", "npre"]
if MeanMeasureEstimator is not None:
    __all__.append("MeanMeasureEstimator")
