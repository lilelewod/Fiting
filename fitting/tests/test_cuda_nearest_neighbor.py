import numpy as np
import pytest
from sklearn.neighbors import KDTree

torch = pytest.importorskip("torch")

from core.estimator.mm_estimator import MeanMeasureEstimator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_cuda_selected_neighbors_preserve_mean_distance():
    rng = np.random.default_rng(20260722)
    data = rng.normal(size=(1024, 3)).astype(np.float32)
    model = rng.normal(size=(1536, 3)).astype(np.float32)
    estimator = MeanMeasureEstimator.__new__(MeanMeasureEstimator)
    estimator.data = data.astype(np.float64)
    estimator.cfg = {"device": torch.device("cuda:0")}
    estimator._torch_device = None
    estimator._torch_data = None
    estimator._torch_cached_data_to_model_errors = None

    cuda_errors, indexes = estimator._torch_cuda_bidirectional_nearest(model)
    cpu_errors = KDTree(data).query(model, k=1)[0].reshape(-1)

    assert indexes.shape == (len(model),)
    assert abs(float(cuda_errors.mean()) - float(cpu_errors.mean())) <= 1e-6
    assert estimator._torch_cached_data_to_model_errors.shape == (len(data),)
