from __future__ import annotations

from typing import Any
import numpy as np
import pathlib
import math


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert common scientific Python objects into JSON-safe types.

    - numpy scalars -> Python scalars
    - numpy arrays -> lists
    - NaN/Inf -> None
    - pathlib.Path -> str
    """
    if obj is None:
        return None

    if isinstance(obj, pathlib.Path):
        return str(obj)

    if isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    if isinstance(obj, (np.ndarray,)):
        return [sanitize_for_json(x) for x in obj.tolist()]

    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    # dataclasses etc.
    if hasattr(obj, "__dict__"):
        return sanitize_for_json(vars(obj))

    # fallback
    return str(obj)
