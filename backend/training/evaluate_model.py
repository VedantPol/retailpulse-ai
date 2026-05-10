from __future__ import annotations

import json
from pathlib import Path


def load_latest_metrics(metrics_path: Path = Path("backend/artifacts/metrics/model_metrics.json")) -> dict[str, object]:
    """Small helper used by demos and CI to display saved model metrics."""

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    print(json.dumps(load_latest_metrics(), indent=2))
