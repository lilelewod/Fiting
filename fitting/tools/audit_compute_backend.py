"""Record the requested and effective compute backends used by formal fitting."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.estimator.mm_estimator as mm_module
from core.optimizer.pso_fitter import Fitter as PSOFitter


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\code\Fiting\outputs\environment\compute_backend_audit.json"),
    )
    args = parser.parse_args()

    pso_source = inspect.getsource(PSOFitter)
    mm_source = inspect.getsource(mm_module.MeanMeasureEstimator)
    errors = []
    # init_device records the configured torch device, but the formal PSO state,
    # velocity and fitness arrays are NumPy arrays and never move to that device.
    pso_has_tensor_compute = "torch." in pso_source or ".to(self.device" in pso_source
    pso_device_references = pso_source.count("self.device")
    if pso_has_tensor_compute or pso_device_references != 2 or "np.random" not in pso_source:
        errors.append("PSO execution-path audit no longer proves a NumPy-only search")

    faiss_available = mm_module.faiss is not None
    faiss_gpu_path = "index_cpu_to_gpu" in mm_source or "StandardGpuResources" in mm_source
    if faiss_gpu_path:
        errors.append("Mean-measure estimator contains an unaccounted FAISS GPU path")
    if "KDTree" not in mm_source:
        errors.append("Mean-measure estimator no longer exposes the audited CPU KDTree path")

    cuda_available = bool(torch.cuda.is_available())
    cuda_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    report = {
        "status": "PASS" if not errors else "FAIL",
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "torch_cuda_devices": cuda_devices,
        "formal_execution_path": {
            "configured_torch_device": "cuda:0",
            "pso_search_backend": "CPU NumPy",
            "pso_tensor_compute_detected": pso_has_tensor_compute,
            "mean_measure_nearest_neighbor_backend": (
                "CPU FAISS IndexFlatL2" if faiss_available else "CPU sklearn KDTree"
            ),
            "faiss_import_available": faiss_available,
            "faiss_gpu_path_detected": faiss_gpu_path,
            "independent_evaluation_backend": "CPU NumPy/scikit-learn KDTree",
            "note": (
                "A CUDA device may be initialized by inherited configuration, but the audited "
                "formal PSO and geometric scoring path does not execute tensor or FAISS-GPU kernels."
            ),
        },
        "errors": errors,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
