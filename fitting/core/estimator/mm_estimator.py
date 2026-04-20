from ._base_estimator import BaseEstimator


class MeanMeasureEstimator(BaseEstimator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.estimator_type = "mean measure"
