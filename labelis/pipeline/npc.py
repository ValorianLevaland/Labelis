from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class NPCRecord:
    npc_id: int
    npc_center_nm: Tuple[float, float]
    npc_status: int = 1  # 1=ok, 2=too small radius, 3=too big radius, 4=too many close, 5=too many far, 6=manual reject

    # raw locs (all within extraction region)
    idx_all: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    x_nm: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    y_nm: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    sigma_nm: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    frame: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    uncertainty_nm: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    intensity: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))

    npc_n_locs: int = 0
    npc_sigma_nm_mean: float = float("nan")
    npc_uncertainty_mean: float = float("nan")
    npc_intensity_mean: float = float("nan")

    # radius fit step1 (free radius)
    center_fitcircle_step1_nm: Optional[Tuple[float, float]] = None
    radius_fitcircle_step1_nm: Optional[float] = None
    residual_fitcircle_step1: Optional[float] = None

    # radius fit step2 (fixed radius recentering)
    center_fitcircle_step2_nm: Optional[Tuple[float, float]] = None
    radius_fitcircle_step2_nm: Optional[float] = None

    # classification of locs by distance-to-center
    idx_good: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    idx_tooclose: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    idx_toofar: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))

    n_locs_good: int = 0
    n_locs_tooclose: int = 0
    n_locs_toofar: int = 0

    # alignment / ELE
    rotation_rad: float = 0.0
    ele: int = 0
    locs_per_corner: Optional[np.ndarray] = None

    sum_distance_initial: Optional[float] = None
    sum_distance_minimized: Optional[float] = None
    sum_distance_initial_all: Optional[float] = None
    sum_distance_minimized_all: Optional[float] = None

    # for downstream plots (optional)
    coords_polar_rho_all: Optional[np.ndarray] = None
    coords_polar_theta_alligned_all: Optional[np.ndarray] = None

    def finalize_stats(self) -> None:
        self.npc_n_locs = int(self.x_nm.size)
        self.npc_sigma_nm_mean = float(np.nanmean(self.sigma_nm)) if self.sigma_nm.size else float("nan")
        self.npc_uncertainty_mean = float(np.nanmean(self.uncertainty_nm)) if self.uncertainty_nm.size else float("nan")
        self.npc_intensity_mean = float(np.nanmean(self.intensity)) if self.intensity.size else float("nan")


def extract_npcs_square(
    df: pd.DataFrame,
    centers_px: np.ndarray,
    pixel_size_nm: float,
    box_radius_nm: float,
    start_id: int = 0,
) -> List[NPCRecord]:
    centers_px = np.asarray(centers_px, dtype=float)
    out: List[NPCRecord] = []
    for i in range(centers_px.shape[0]):
        cx_px, cy_px = float(centers_px[i, 0]), float(centers_px[i, 1])
        center_nm = (cx_px * pixel_size_nm, cy_px * pixel_size_nm)
        x_min = np.floor(center_nm[0] - box_radius_nm)
        x_max = np.ceil(center_nm[0] + box_radius_nm)
        y_min = np.floor(center_nm[1] - box_radius_nm)
        y_max = np.ceil(center_nm[1] + box_radius_nm)

        mask = (
            (df["x_nm_"] > x_min) & (df["x_nm_"] < x_max) &
            (df["y_nm_"] > y_min) & (df["y_nm_"] < y_max)
        )
        sub = df.loc[mask]
        rec = NPCRecord(
            npc_id=int(i + start_id + 1),
            npc_center_nm=center_nm,
            idx_all=sub.index.to_numpy(dtype=int),
            x_nm=sub["x_nm_"].to_numpy(dtype=float),
            y_nm=sub["y_nm_"].to_numpy(dtype=float),
            sigma_nm=sub["sigma_nm_"].to_numpy(dtype=float),
            frame=sub["frame"].to_numpy(dtype=int),
            uncertainty_nm=sub["uncertainty_xy_nm_"].to_numpy(dtype=float),
            intensity=sub["intensity_photon_"].to_numpy(dtype=float),
            npc_status=1,
        )
        rec.finalize_stats()
        out.append(rec)
    return out


def extract_npcs_by_radius(
    df: pd.DataFrame,
    centers_px: np.ndarray,
    pixel_size_nm: float,
    radius_nm: float,
    start_id: int = 0,
) -> List[NPCRecord]:
    centers_px = np.asarray(centers_px, dtype=float)
    out: List[NPCRecord] = []
    r2 = float(radius_nm) * float(radius_nm)
    for i in range(centers_px.shape[0]):
        cx_px, cy_px = float(centers_px[i, 0]), float(centers_px[i, 1])
        center_nm = (cx_px * pixel_size_nm, cy_px * pixel_size_nm)
        dx = df["x_nm_"].to_numpy(dtype=float) - center_nm[0]
        dy = df["y_nm_"].to_numpy(dtype=float) - center_nm[1]
        mask = (dx*dx + dy*dy) < r2
        sub = df.loc[mask]
        rec = NPCRecord(
            npc_id=int(i + start_id + 1),
            npc_center_nm=center_nm,
            idx_all=sub.index.to_numpy(dtype=int),
            x_nm=sub["x_nm_"].to_numpy(dtype=float),
            y_nm=sub["y_nm_"].to_numpy(dtype=float),
            sigma_nm=sub["sigma_nm_"].to_numpy(dtype=float),
            frame=sub["frame"].to_numpy(dtype=int),
            uncertainty_nm=sub["uncertainty_xy_nm_"].to_numpy(dtype=float),
            intensity=sub["intensity_photon_"].to_numpy(dtype=float),
            npc_status=1,
        )
        rec.finalize_stats()
        out.append(rec)
    return out


def npc_list_to_dataframe(npcs: Sequence[NPCRecord]) -> pd.DataFrame:
    rows = []
    for n in npcs:
        row = dict(
            npc_id=n.npc_id,
            status=n.npc_status,
            center_nm_x=n.npc_center_nm[0],
            center_nm_y=n.npc_center_nm[1],
            n_locs=n.npc_n_locs,
            mean_sigma_nm=n.npc_sigma_nm_mean,
            mean_uncertainty_nm=n.npc_uncertainty_mean,
            mean_intensity=n.npc_intensity_mean,
            center_step1_x=(n.center_fitcircle_step1_nm[0] if n.center_fitcircle_step1_nm else np.nan),
            center_step1_y=(n.center_fitcircle_step1_nm[1] if n.center_fitcircle_step1_nm else np.nan),
            radius_step1_nm=(n.radius_fitcircle_step1_nm if n.radius_fitcircle_step1_nm is not None else np.nan),
            residual_step1=(n.residual_fitcircle_step1 if n.residual_fitcircle_step1 is not None else np.nan),
            center_step2_x=(n.center_fitcircle_step2_nm[0] if n.center_fitcircle_step2_nm else np.nan),
            center_step2_y=(n.center_fitcircle_step2_nm[1] if n.center_fitcircle_step2_nm else np.nan),
            radius_step2_nm=(n.radius_fitcircle_step2_nm if n.radius_fitcircle_step2_nm is not None else np.nan),
            n_locs_good=n.n_locs_good,
            n_locs_tooclose=n.n_locs_tooclose,
            n_locs_toofar=n.n_locs_toofar,
            rotation_rad=n.rotation_rad,
            ELE=n.ele,
            sum_distance_initial=(n.sum_distance_initial if n.sum_distance_initial is not None else np.nan),
            sum_distance_minimized=(n.sum_distance_minimized if n.sum_distance_minimized is not None else np.nan),
        )
        if n.locs_per_corner is not None:
            for k in range(len(n.locs_per_corner)):
                row[f"locs_corner_{k+1}"] = int(n.locs_per_corner[k])
        rows.append(row)
    return pd.DataFrame(rows)
