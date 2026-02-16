from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _vlen_str_dtype():
    import h5py

    return h5py.string_dtype(encoding="utf-8")


def _write_scalar(group, name: str, value: Any) -> None:
    """Write a scalar value as a dataset (robust for strings)."""
    import h5py

    if value is None:
        group.attrs[name] = "__None__"
        return

    if isinstance(value, (str, bytes)):
        dt = _vlen_str_dtype()
        group.create_dataset(name, data=np.array([str(value)], dtype=object), dtype=dt)
        return

    # JSON for complex types
    if isinstance(value, (dict, list, tuple)):
        dt = _vlen_str_dtype()
        group.create_dataset(name, data=np.array([json.dumps(value)], dtype=object), dtype=dt)
        return

    # numbers / booleans
    try:
        group.create_dataset(name, data=value)
    except Exception:
        # fallback to string
        dt = _vlen_str_dtype()
        group.create_dataset(name, data=np.array([str(value)], dtype=object), dtype=dt)


def _write_ndarray(group, name: str, arr: Optional[np.ndarray], compression: str = "gzip") -> None:
    if arr is None:
        return
    arr = np.asarray(arr)
    group.create_dataset(name, data=arr, compression=compression, shuffle=True)


def _write_dataframe(group, name: str, df, compression: str = "gzip") -> None:
    """Store a pandas DataFrame inside a group.

    We intentionally avoid pandas.HDFStore / PyTables to keep the dependency surface minimal.
    """

    import pandas as pd

    if df is None:
        return
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    g = group.create_group(name)
    g.create_dataset("index", data=df.index.to_numpy(), compression=compression, shuffle=True)
    dt = _vlen_str_dtype()
    g.create_dataset("columns", data=np.array(df.columns.astype(str).tolist(), dtype=object), dtype=dt)

    colgrp = g.create_group("data")
    for col in df.columns:
        s = df[col]
        if s.dtype.kind in "iub":
            colgrp.create_dataset(col, data=s.to_numpy(), compression=compression, shuffle=True)
        elif s.dtype.kind == "f":
            colgrp.create_dataset(col, data=s.to_numpy(dtype=np.float64), compression=compression, shuffle=True)
        else:
            colgrp.create_dataset(col, data=np.array(s.astype(str).to_list(), dtype=object), dtype=dt)


def _write_environment(group) -> None:
    import numpy as np

    _write_scalar(group, "created_at", datetime.now().isoformat(timespec="seconds"))
    _write_scalar(group, "python", sys.version)
    _write_scalar(group, "platform", platform.platform())
    _write_scalar(group, "executable", sys.executable)
    _write_scalar(group, "cwd", os.getcwd())
    _write_scalar(group, "numpy", np.__version__)

    # Optional libs
    try:
        import pandas as pd

        _write_scalar(group, "pandas", pd.__version__)
    except Exception:
        pass
    try:
        import scipy

        _write_scalar(group, "scipy", scipy.__version__)
    except Exception:
        pass
    try:
        import napari

        _write_scalar(group, "napari", getattr(napari, "__version__", "unknown"))
    except Exception:
        pass
    try:
        import h5py

        _write_scalar(group, "h5py", h5py.__version__)
    except Exception:
        pass


def save_workspace_h5(
    path: str | Path,
    *,
    labelis_version: Optional[str] = None,
    config,
    summary: Dict[str, Any],
    debug_lines: list[str],
    roi_polygon_nm: Optional[list[tuple[float, float]]],
    df_input,
    df_filtered,
    df_npc_final,
    images: Dict[str, Optional[np.ndarray]],
    detections: Dict[str, Optional[np.ndarray]],
    npc_records: Optional[list[Any]] = None,
    derived: Optional[Dict[str, Any]] = None,
    compression: str = "gzip",
) -> Path:
    """Save a MATLAB-like "workspace" in a single HDF5 file.

    This is meant to replace the original MATLAB `save(f_out)` behavior.
    """

    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as h5:
        h5.attrs["labelis_version"] = str(labelis_version or "unknown")
        h5.attrs["schema"] = "labelis_workspace_v1"

        g_env = h5.create_group("environment")
        _write_environment(g_env)

        g_cfg = h5.create_group("config")
        try:
            cfg_dict = asdict(config)
        except Exception:
            cfg_dict = getattr(config, "__dict__", {})
        _write_scalar(g_cfg, "json", cfg_dict)

        g_sum = h5.create_group("summary")
        _write_scalar(g_sum, "json", summary)

        if derived:
            g_der = h5.create_group("derived")
            _write_scalar(g_der, "json", derived)

        # Debug
        g_dbg = h5.create_group("debug")
        dt = _vlen_str_dtype()
        g_dbg.create_dataset("lines", data=np.array(debug_lines, dtype=object), dtype=dt)

        # ROI
        if roi_polygon_nm:
            g_roi = h5.create_group("roi")
            poly = np.asarray(roi_polygon_nm, dtype=float)
            g_roi.create_dataset("polygon_nm", data=poly, compression=compression, shuffle=True)

        # Tables
        g_data = h5.create_group("data")
        _write_dataframe(g_data, "input_table", df_input, compression=compression)
        _write_dataframe(g_data, "filtered_table", df_filtered, compression=compression)
        _write_dataframe(g_data, "npc_final_table", df_npc_final, compression=compression)

        # Arrays
        g_img = h5.create_group("images")
        for k, v in images.items():
            _write_ndarray(g_img, k, v, compression=compression)

        g_det = h5.create_group("detections")
        for k, v in detections.items():
            _write_ndarray(g_det, k, v, compression=compression)

        # Per-NPC records (optional, but matches MATLAB workspace spirit)
        if npc_records:
            g_npcs = h5.create_group("npcs")
            for rec in npc_records:
                try:
                    npc_id = int(getattr(rec, "npc_id"))
                except Exception:
                    continue
                gr = g_npcs.create_group(f"npc_{npc_id:05d}")
                for field in [
                    "npc_center_nm",
                    "npc_status",
                    "idx_all",
                    "idx_good",
                    "idx_tooclose",
                    "idx_toofar",
                    "x_nm",
                    "y_nm",
                    "sigma_nm",
                    "frame",
                    "uncertainty_nm",
                    "intensity",
                    "npc_n_locs",
                    "npc_sigma_nm_mean",
                    "npc_uncertainty_mean",
                    "center_fitcircle_step1_nm",
                    "radius_fitcircle_step1_nm",
                    "center_fitcircle_step2_nm",
                    "radius_fitcircle_step2_nm",
                    "rotation_rad",
                    "ele",
                    "locs_per_corner",
                    "sum_distance_initial",
                    "sum_distance_minimized",
                    "sum_distance_initial_all",
                    "sum_distance_minimized_all",
                    "coords_polar_rho_all",
                    "coords_polar_theta_alligned_all",
                ]:
                    if not hasattr(rec, field):
                        continue
                    val = getattr(rec, field)
                    if isinstance(val, np.ndarray):
                        _write_ndarray(gr, field, val, compression=compression)
                    elif isinstance(val, (list, tuple)):
                        _write_ndarray(gr, field, np.asarray(val), compression=compression)
                    else:
                        _write_scalar(gr, field, val)

    return path
