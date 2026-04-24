import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from entrypoints.fit_point_cloud import (
    get_estimator_class,
    get_rule_class,
    load_data,
    main,
    run_experiment,
)

Rule = get_rule_class({'model': {'type': 'curve'}})


if __name__ == "__main__":
    main()
