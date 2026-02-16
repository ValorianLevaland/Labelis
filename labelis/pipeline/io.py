from __future__ import annotations

from typing import Dict, Optional
import pathlib
import json
import pandas as pd
import numpy as np

# -----------------------------
# Column mapping helpers
# -----------------------------

# Very permissive aliases (ThunderSTORM, SMAP, custom exports, etc.)
_ALIASES = {
    "x_nm_": ["x_nm_", "x [nm]", "x (nm)", "x_nm", "x", "x_nm__"],
    "y_nm_": ["y_nm_", "y [nm]", "y (nm)", "y_nm", "y", "y_nm__"],
    "sigma_nm_": ["sigma_nm_", "sigma [nm]", "sigma (nm)", "sigma_nm", "sigma", "psf_sigma_nm", "sigma1_nm_"],
    "uncertainty_xy_nm_": ["uncertainty_xy_nm_", "uncertainty [nm]", "uncertainty (nm)", "uncertainty_nm", "uncertainty", "uncertainty_xy", "uncertainty_xy_nm"],
    "frame": ["frame", "frame_number", "t", "time", "frame_"],
    "intensity_photon_": ["intensity_photon_", "intensity [photon]", "intensity (photon)", "photons", "intensity", "intensity_photons"],
    # optional fields often present in ThunderSTORM / SMAP exports
    "offset_photon_": ["offset_photon_", "offset [photon]", "offset (photon)", "offset", "background", "bg"],
    "bkgstd_photon_": [
        "bkgstd_photon_",
        "bkgstd [photon]",
        "bkgstd (photon)",
        "bkgstd",
        "background_std",
        "bgstd",
    ],
    "id": ["id", "molecule", "particle", "spot_id"],
}


def _normalize_colname(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("\u00b5", "u")
        .replace("µ", "u")
        .replace("__", "_")
        .replace("  ", " ")
    )


def _find_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    # exact match first
    cols_norm = {_normalize_colname(c): c for c in df.columns}
    for alias in _ALIASES.get(canonical, []):
        a = _normalize_colname(alias)
        if a in cols_norm:
            return cols_norm[a]
    # fuzzy contains match
    target_tokens = set(_normalize_colname(canonical).replace("_", " ").split())
    best = None
    best_score = 0
    for c in df.columns:
        toks = set(_normalize_colname(c).replace("_", " ").split())
        score = len(target_tokens & toks)
        if score > best_score:
            best_score = score
            best = c
    if best_score >= 1:
        return best
    return None


def load_localizations(path: str | pathlib.Path) -> pd.DataFrame:
    """Load a localization table from CSV/TSV.

    The function tries to robustly map columns to the canonical names used
    by the legacy MATLAB pipeline:
        x_nm_, y_nm_, sigma_nm_, uncertainty_xy_nm_, frame, intensity_photon_

    Returns
    -------
    df : pandas.DataFrame
        Always contains at least x_nm_, y_nm_, sigma_nm_, uncertainty_xy_nm_, frame.
        intensity_photon_ is created if missing (filled with NaN).
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # delimiter sniffing (very lightweight)
    head = path.read_text(errors="ignore", encoding="utf-8")[:20000]
    sep = "," if head.count(",") >= head.count("\t") else "\t"

    df = pd.read_csv(path, sep=sep, engine="python")
    if df.empty:
        raise ValueError(f"Empty localization table: {path}")

    colmap: Dict[str, str] = {}
    for canonical in [
        "x_nm_",
        "y_nm_",
        "sigma_nm_",
        "uncertainty_xy_nm_",
        "frame",
        "intensity_photon_",
        "offset_photon_",
        "bkgstd_photon_",
        "id",
    ]:
        found = _find_column(df, canonical)
        if found is not None:
            colmap[found] = canonical

    df = df.rename(columns=colmap)

    required = ["x_nm_", "y_nm_", "sigma_nm_", "uncertainty_xy_nm_", "frame"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in localization table: "
            + ", ".join(missing)
            + f"\nAvailable columns: {list(df.columns)}"
        )

    if "intensity_photon_" not in df.columns:
        df["intensity_photon_"] = np.nan
    if "offset_photon_" not in df.columns:
        df["offset_photon_"] = np.nan
    if "bkgstd_photon_" not in df.columns:
        df["bkgstd_photon_"] = np.nan

    # Ensure numeric
    for c in required + ["intensity_photon_", "offset_photon_", "bkgstd_photon_"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x_nm_", "y_nm_", "sigma_nm_", "uncertainty_xy_nm_", "frame"])

    # Frame should be integer-like
    df["frame"] = df["frame"].astype(int)

    # If id missing, create stable id
    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=int)

    return df


def save_npc_table(df_npc: pd.DataFrame, out_csv: str | pathlib.Path) -> None:
    out_csv = pathlib.Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_npc.to_csv(out_csv, index=False)


def save_summary(summary: dict, out_json: str | pathlib.Path) -> None:
    out_json = pathlib.Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
