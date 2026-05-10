from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def load_joblib(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)

