from ._base_estimator import BaseEstimator, npre


class NPREEstimator(BaseEstimator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.estimator_type = "npre"
