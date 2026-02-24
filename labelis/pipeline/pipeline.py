from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import json
import pathlib

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path as MplPath

from .. import __version__ as LABELIS_VERSION

from .config import PipelineConfig
from .debug import DebugTrace
from .ele import fit_ele_bootstrap
from .figures import (
    save_figure_ele,
    save_figure_info_npc,
    save_figure_npc_image,
    save_figure_npc_image_intensity_profile,
)
from .filters import (
    filter_npcs_by_radius,
    filter_npcs_min_locs_sigma_unc,
    recenter_and_filter_distribution,
    remove_close_npcs,
    update_loc_table_remove_used,
)
from .io import load_localizations, save_npc_table
from .npc import NPCRecord, extract_npcs_by_radius, extract_npcs_square, npc_list_to_dataframe
from .render import have_numba, render_dispatch
from .segmentation import locate_npcs_dispatch, prepare_image_for_segmentation
from .alignment import align_and_count_corners
from .utils import sanitize_for_json
from .workspace_h5 import save_workspace_h5


@dataclass
class PipelineResult:
    config: PipelineConfig
    # offsets applied to coordinates (after x/y filtering)
    x_offset_nm: float
    y_offset_nm: float

    # images
    image_rendered: Optional[np.ndarray] = None
    image_segment: Optional[np.ndarray] = None
    image_rendered_final_ok: Optional[np.ndarray] = None
    image_rendered_final_rejected: Optional[np.ndarray] = None

    # raw detections
    all_centers_px: Optional[np.ndarray] = None
    all_radii_px: Optional[np.ndarray] = None
    all_metric: Optional[np.ndarray] = None

    # NPC lists
    npc_all: Optional[List[NPCRecord]] = None
    npc_ok: Optional[List[NPCRecord]] = None
    npc_rejected: Optional[List[NPCRecord]] = None
    npc_final: Optional[List[NPCRecord]] = None

    # ELE fit
    p_label: float = float("nan")
    p_label_error: float = float("nan")

    # tables
    df_npc_final: Optional[pd.DataFrame] = None

    # derived parameters (auto-tuned)
    derived: Dict[str, object] = None  # type: ignore

    # outputs
    workspace_h5: Optional[str] = None
    debug_log: Optional[str] = None
    figure_paths: Dict[str, str] = None  # type: ignore
    final_localizations_csv: Optional[str] = None

    # summary
    summary: Dict[str, object] = None  # type: ignore

    cpsam_instance_labels: Optional[np.ndarray] = None


class PipelineAborted(RuntimeError):
    """Raised when a user aborts an interactive (step-by-step) run."""


def _apply_cfg_updates(cfg: PipelineConfig, updates: Dict[str, object]) -> None:
    """Best-effort application of config updates coming from GUI checkpoints."""
    for k, v in (updates or {}).items():
        if not hasattr(cfg, k):
            continue
        try:
            setattr(cfg, k, v)
        except Exception:
            # do not hard-fail on a single bad update
            continue


def _ensure_dir(p: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig_multi(fig: "plt.Figure", path_png: pathlib.Path) -> None:
    """Save a matplotlib figure as PNG + TIFF side-by-side."""
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, bbox_inches="tight")
    # TIFF copy (same stem)
    path_tif = path_png.with_suffix(".tif")
    fig.savefig(path_tif, bbox_inches="tight")


def _save_image_png(img: np.ndarray, path: pathlib.Path, title: str = "", vmax_quantile: float = 0.999) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111)
    vmax = float(np.quantile(img, vmax_quantile)) if img.size else 1.0
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    _save_fig_multi(fig, path)
    plt.close(fig)


def _save_overlay_png(
    img: np.ndarray,
    centers_px: np.ndarray,
    radii_px: np.ndarray,
    path: pathlib.Path,
    title: str = "",
    vmax_quantile: float = 0.999,
    add_radius_px: float = 0.0,
    edgecolors: Optional[Sequence[Tuple[float, float, float]]] = None,
    edgecolor_default: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    linewidth: float = 1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111)
    vmax = float(np.quantile(img, vmax_quantile)) if img.size else 1.0
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)

    if edgecolors is not None:
        cols = list(edgecolors)
        if len(cols) != len(centers_px):
            # fallback to default if lengths mismatch
            cols = [edgecolor_default] * len(centers_px)
    else:
        cols = [edgecolor_default] * len(centers_px)

    for (cx, cy), r, col in zip(centers_px, radii_px, cols):
        ax.add_patch(Circle((float(cx), float(cy)), float(r) + float(add_radius_px), fill=False, linewidth=float(linewidth), edgecolor=col))
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    _save_fig_multi(fig, path)
    plt.close(fig)


def _save_roi_png(
    img: np.ndarray,
    polygon_nm: Sequence[Tuple[float, float]],
    pixel_size_nm: float,
    path: pathlib.Path,
    title: str = "ROI",
    vmax_quantile: float = 0.999,
) -> None:
    """Save ROI polygon overlay on an image (PNG)."""

    poly = np.asarray(polygon_nm, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        return
    poly_px = poly / float(pixel_size_nm)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111)
    vmax = float(np.quantile(img, vmax_quantile)) if img.size else 1.0
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)
    ax.add_patch(Polygon(poly_px, closed=True, fill=False, linewidth=2.0, edgecolor=(1.0, 0.0, 0.0)))
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    _save_fig_multi(fig, path)
    plt.close(fig)


def _apply_roi(df: pd.DataFrame, polygon_nm: Sequence[Tuple[float, float]]) -> pd.DataFrame:
    poly = np.asarray(polygon_nm, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        return df
    path = MplPath(poly, closed=True)
    pts = df[["x_nm_", "y_nm_"]].to_numpy(dtype=float)
    inside = path.contains_points(pts)
    return df.loc[inside]


def _export_final_localizations(df_filtered: pd.DataFrame, npc_final: Sequence[NPCRecord], out_csv: pathlib.Path) -> None:
    """Export the localizations used for the final NPCs.

    This mirrors the MATLAB pipeline's final export (see the v7 script).
    """

    if not npc_final:
        out_csv.write_text("", encoding="utf-8")
        return

    idx = []
    for n in npc_final:
        # Prefer "good" localizations if available.
        if getattr(n, "idx_good", None) is not None and n.idx_good.size:
            idx.append(n.idx_good.astype(int))
        else:
            idx.append(n.idx_all.astype(int))

    if not idx:
        out_csv.write_text("", encoding="utf-8")
        return

    idx_u = np.unique(np.concatenate(idx, axis=0))
    df_out = df_filtered.loc[idx_u].copy()

    # Keep a predictable front-matter column order if present
    preferred = [
        "id",
        "frame",
        "x_nm_",
        "y_nm_",
        "sigma_nm_",
        "intensity_photon_",
        "uncertainty_xy_nm_",
        "offset_photon_",
        "bkgstd_photon_",
    ]
    cols = [c for c in preferred if c in df_out.columns] + [c for c in df_out.columns if c not in preferred]
    df_out = df_out[cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)



def _npc_center_nm(n) -> tuple[float, float]:
    """Robustly get an NPC center in nm for overlays.

    Priority (closest to MATLAB downstream behavior):
    1) refined center from step2 (fixed-radius recenter)
    2) fitted center from step1 (free-radius fit)
    3) initial detection center
    """
    if getattr(n, "center_fitcircle_step2_nm", None) is not None:
        c = n.center_fitcircle_step2_nm
        return (float(c[0]), float(c[1]))
    if getattr(n, "center_fitcircle_step1_nm", None) is not None:
        c = n.center_fitcircle_step1_nm
        return (float(c[0]), float(c[1]))
    c = getattr(n, "npc_center_nm", (0.0, 0.0))
    return (float(c[0]), float(c[1]))


def run_pipeline(
    config: PipelineConfig,
    log_cb: Optional[Callable[[str], None]] = None,
    df0: Optional[pd.DataFrame] = None,
    checkpoint_cb: Optional[Callable[[str, Dict[str, object]], Dict[str, object]]] = None,
) -> PipelineResult:
    """Run the Labelis NPC pipeline.

    Key UX constraints (matching the user's workflow):
    - ROI is assumed to have been drawn in Napari *before* calling this.
    - After ROI confirmation, the user still provides analysis parameters.
    """

    cfg = config
    dbg = DebugTrace(log_cb=log_cb)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    if cfg.compute_engine == "numba" and not have_numba():
        dbg.log(0, "Numba not available - falling back to reference_bruteforce")
        cfg.compute_engine = "reference_bruteforce"

    out_dir = _ensure_dir(cfg.output_dir)
    fig_dir = _ensure_dir(cfg.fig_dir or cfg.output_dir)

    dbg.log(0, f"Labelis {LABELIS_VERSION} | {datetime.now().isoformat(timespec='seconds')}")
    dbg.log(0, f"Input: {cfg.input_path}")
    dbg.log(0, f"Output: {out_dir}")
    dbg.log(0, f"Figures: {fig_dir}")

    # ------------------------------------------------------------------
    # DEBUG(0) - data management / filtering
    # ------------------------------------------------------------------
    dbg.log(0, "Loading localization table")
    if df0 is None:
        df_in = load_localizations(cfg.input_path)
    else:
        df_in = df0.copy()
        dbg.log(0, f"Using preloaded table: {len(df_in)} rows")

    n_input = int(len(df_in))
    if n_input == 0:
        raise ValueError("Input localization table is empty")

    # X/Y window selection
    df = df_in.copy()
    df = df.loc[
        (df["x_nm_"] >= cfg.min_x_nm)
        & (df["x_nm_"] <= cfg.max_x_nm)
        & (df["y_nm_"] >= cfg.min_y_nm)
        & (df["y_nm_"] <= cfg.max_y_nm)
    ].copy()
    if len(df) == 0:
        raise ValueError("No localizations left after X/Y window filtering")

    # Shift to origin (MATLAB behavior)
    x_offset_nm = float(df["x_nm_"].min())
    y_offset_nm = float(df["y_nm_"].min())
    df["x_nm_"] = df["x_nm_"] - x_offset_nm
    df["y_nm_"] = df["y_nm_"].copy() - y_offset_nm
    dbg.log(0, f"X/Y filter: {len(df)} / {n_input} kept")
    dbg.log(0, f"X/Y offsets: ({x_offset_nm:.3f} nm, {y_offset_nm:.3f} nm)")

    # ROI selection
    roi_polygon_nm = cfg.roi_polygon_nm if (cfg.use_roi and cfg.roi_polygon_nm) else None
    if roi_polygon_nm:
        n_before = len(df)
        df = _apply_roi(df, roi_polygon_nm)
        dbg.log(0, f"ROI filter: {len(df)} / {n_before} kept")
    else:
        dbg.log(0, "ROI filter: disabled")

    # sigma filter
    n_before = len(df)
    df = df.loc[(df["sigma_nm_"] >= cfg.min_sigma_nm) & (df["sigma_nm_"] <= cfg.max_sigma_nm)].copy()
    dbg.log(0, f"Sigma filter [{cfg.min_sigma_nm:g}, {cfg.max_sigma_nm:g}] nm: {len(df)} / {n_before} kept")

    # uncertainty filter
    n_before = len(df)
    df = df.loc[
        (df["uncertainty_xy_nm_"] >= cfg.min_uncertainty_nm)
        & (df["uncertainty_xy_nm_"] <= cfg.max_uncertainty_nm)
    ].copy()
    dbg.log(
        0,
        f"Uncertainty filter [{cfg.min_uncertainty_nm:g}, {cfg.max_uncertainty_nm:g}] nm: {len(df)} / {n_before} kept",
    )

    # frame filter
    n_before = len(df)
    df = df.loc[(df["frame"] >= cfg.min_frame) & (df["frame"] <= cfg.max_frame)].copy()
    dbg.log(0, f"Frame filter [{cfg.min_frame}, {cfg.max_frame}]: {len(df)} / {n_before} kept")

    if len(df) == 0:
        raise ValueError("No localizations left after filtering")

    df_filtered = df.copy()  # stable view for exports (even if we iterate)
    dbg.log(0, f"Total filtered localizations: {len(df_filtered)}")

    # ------------------------------------------------------------------
    # DEBUG(1/2) - iterative segmentation & filtering (v7 logic)
    # ------------------------------------------------------------------
    df_remaining = df_filtered

    job_ok = False
    cycle_n = 1
    npc_list_final: List[NPCRecord] = []

    all_centers: List[np.ndarray] = []
    all_radii: List[np.ndarray] = []
    all_metric: List[np.ndarray] = []

    start_id = 0
    effective_extract_circle_radius: Optional[float] = None
    effective_radius: Optional[float] = None
    effective_min_radius: Optional[float] = None
    effective_max_radius: Optional[float] = None
    effective_inner_radius: Optional[float] = None
    effective_outer_radius: Optional[float] = None
    effective_npc_radius: Optional[float] = None

    # may become auto
    min_dist_between_npcs_nm = float(cfg.min_dist_between_npcs_nm)

    image_rendered_saved: Optional[np.ndarray] = None
    image_segment_saved: Optional[np.ndarray] = None

    # For workspace export
    sum_img_raw: Optional[np.ndarray] = None
    sum_img_aligned: Optional[np.ndarray] = None

    while (not job_ok) and (len(df_remaining) > 0):
        dbg.log(1, f"--- Cycle {cycle_n} ---")
        dbg.log(1, f"Rendering image (engine={cfg.compute_engine}, px={cfg.render_px_size_nm:g} nm)")

        image_rendered = render_dispatch(
            engine=cfg.compute_engine,
            x_nm=df_remaining["x_nm_"].to_numpy(),
            y_nm=df_remaining["y_nm_"].to_numpy(),
            sigma_nm=df_remaining["sigma_nm_"].to_numpy(),
            pixel_size_nm=cfg.render_px_size_nm,
            grid_size_px=None,
            signal_half_width_px=cfg.signal_half_width_px,
            sigma_is_variance=cfg.sigma_is_variance,
            compat_kernel_crop=cfg.compat_kernel_crop,
            turbo_blur_sigma_px=cfg.turbo_blur_sigma_px,
        )
        if cycle_n == 1:
            image_rendered_saved = image_rendered.copy()

        image_segment = prepare_image_for_segmentation(
            image=image_rendered,
            thres_clip=cfg.thres_clip,
            expected_radius_nm=cfg.expected_npc_radius_nm,
            pixel_size_nm=cfg.render_px_size_nm,
        )

        if cfg.segmentation_engine == "cpsam":
            cpsam_instance_labels = <your label map>
        else:
            cpsam_instance_labels = None
        
        if cycle_n == 1:
            image_segment_saved = image_segment.copy()

        # Save ROI overlays on cycle 1 (rendered + segmentation input)
        if cycle_n == 1 and roi_polygon_nm and image_rendered_saved is not None:
            try:
                _save_roi_png(
                    image_rendered_saved,
                    roi_polygon_nm,
                    pixel_size_nm=cfg.render_px_size_nm,
                    path=fig_dir / "ROI.png",
                    title="ROI overlay (cycle 1 rendered)",
                )
            except Exception:
                # non-critical
                pass

        if cycle_n == 1 and roi_polygon_nm and image_segment_saved is not None:
            try:
                _save_roi_png(
                    image_segment_saved,
                    roi_polygon_nm,
                    pixel_size_nm=cfg.render_px_size_nm,
                    path=fig_dir / "ROI_on_segment_input.png",
                    title="ROI overlay (cycle 1 segmentation input)",
                )
            except Exception:
                pass

        # Determine min distance
        if min_dist_between_npcs_nm == -1:
            if effective_extract_circle_radius is not None:
                effective_min_dist_nm = 1.3 * float(effective_extract_circle_radius)
            else:
                effective_min_dist_nm = 2.0 * float(cfg.extract_circle_radius_nm)
        else:
            effective_min_dist_nm = float(min_dist_between_npcs_nm)

        # ------------------------------------------------------------------
        # Checkpoint (optional): candidate detection preview + parameter tuning
        # ------------------------------------------------------------------
        while True:
            dbg.log(
                1,
                f"Locating NPCs (engine={cfg.segmentation_engine}, sens={cfg.sensitivity:.3f}, minDist={effective_min_dist_nm:.1f} nm)",
            )

            centers_px, radii_px, metric = locate_npcs_dispatch(
                engine=cfg.segmentation_engine,
                image_to_segment=image_segment,
                sensitivity=cfg.sensitivity,
                expected_radius_nm=cfg.expected_npc_radius_nm,
                pixel_size_nm=cfg.render_px_size_nm,
                min_distance_nm=effective_min_dist_nm,
                circle_radius_range_nm=(cfg.circle_min_radius_nm, cfg.circle_max_radius_nm),
            )

            dbg.log(1, f"Found {centers_px.shape[0]} candidate centers")

            if bool(getattr(cfg, "step_by_step", False)) and bool(getattr(cfg, "checkpoint_segmentation", True)) and checkpoint_cb is not None:
                payload: Dict[str, object] = {
                    "cycle": int(cycle_n),
                    "image_rendered": image_rendered,
                    "image_segment": image_segment,
                    "centers_px": np.asarray(centers_px, dtype=np.float32),
                    "radii_px": np.asarray(radii_px, dtype=np.float32),
                    "metric": np.asarray(metric, dtype=np.float32),
                    "n_candidates": int(centers_px.shape[0]),
                    "pixel_size_nm": float(cfg.render_px_size_nm),
                    "segmentation_engine": str(cfg.segmentation_engine),
                    "sensitivity": float(cfg.sensitivity),
                    "expected_npc_radius_nm": float(cfg.expected_npc_radius_nm),
                    "circle_min_radius_nm": float(cfg.circle_min_radius_nm),
                    "circle_max_radius_nm": float(cfg.circle_max_radius_nm),
                    "min_dist_between_npcs_nm": float(min_dist_between_npcs_nm),
                    "effective_min_dist_nm": float(effective_min_dist_nm),
                    "thres_clip": float(cfg.thres_clip),
                }

                resp = checkpoint_cb("segmentation", payload) or {}
                action = str(resp.get("action", "continue")).lower().strip()
                if action == "abort":
                    raise PipelineAborted("Aborted by user at candidate-detection checkpoint")
                if action == "update":
                    updates = resp.get("updates", {})
                    # recompute segmentation input if needed
                    recompute_seg = False
                    if isinstance(updates, dict):
                        recompute_seg = any(k in updates for k in ("expected_npc_radius_nm", "thres_clip"))
                        _apply_cfg_updates(cfg, updates)
                        if "min_dist_between_npcs_nm" in updates:
                            try:
                                min_dist_between_npcs_nm = float(updates["min_dist_between_npcs_nm"])
                            except Exception:
                                pass

                    if recompute_seg:
                        image_segment = prepare_image_for_segmentation(
                            image=image_rendered,
                            thres_clip=cfg.thres_clip,
                            expected_radius_nm=cfg.expected_npc_radius_nm,
                            pixel_size_nm=cfg.render_px_size_nm,
                        )
                        if cycle_n == 1:
                            image_segment_saved = image_segment.copy()

                    # Re-evaluate minDist each time (auto/manual may have changed)
                    if min_dist_between_npcs_nm == -1:
                        if effective_extract_circle_radius is not None:
                            effective_min_dist_nm = 1.3 * float(effective_extract_circle_radius)
                        else:
                            effective_min_dist_nm = 2.0 * float(cfg.extract_circle_radius_nm)
                    else:
                        effective_min_dist_nm = float(min_dist_between_npcs_nm)

                    continue  # rerun detection with updated parameters

                # default: continue
                break

            # no checkpoints
            break

        # Store only the final (accepted) detections of this cycle
        all_centers.append(centers_px)
        all_radii.append(radii_px)
        all_metric.append(metric)

        # Extract in a square (rough)
        npc_list = extract_npcs_square(
            df=df_remaining,
            centers_px=centers_px,
            pixel_size_nm=cfg.render_px_size_nm,
            box_radius_nm=cfg.extract_box_radius_nm,
            start_id=start_id,
        )
        dbg.log(1, f"Extracted {len(npc_list)} NPC ROIs (square)")

        # ------------------------------------------------------------------
        # Checkpoint (optional): threshold tuning (mean sigma / mean uncertainty)
        # ------------------------------------------------------------------
        if bool(getattr(cfg, "step_by_step", False)) and bool(getattr(cfg, "checkpoint_thresholds", True)) and checkpoint_cb is not None and len(npc_list) > 0:
            sigma_means = np.array([n.npc_sigma_nm_mean for n in npc_list], dtype=float)
            unc_means = np.array([n.npc_uncertainty_mean for n in npc_list], dtype=float)
            n_locs = np.array([n.npc_n_locs for n in npc_list], dtype=int)

            while True:
                mask = n_locs >= int(cfg.min_n_locs)
                if np.isfinite(cfg.max_mean_sigma_nm):
                    mask &= sigma_means < float(cfg.max_mean_sigma_nm)
                if np.isfinite(cfg.max_mean_uncertainty_nm):
                    mask &= unc_means < float(cfg.max_mean_uncertainty_nm)
                n_pass = int(np.sum(mask))

                payload_thr: Dict[str, object] = {
                    "cycle": int(cycle_n),
                    "n_total": int(len(npc_list)),
                    "n_pass": int(n_pass),
                    "sigma_means": sigma_means,
                    "unc_means": unc_means,
                    "min_n_locs": int(cfg.min_n_locs),
                    "max_mean_sigma_nm": float(cfg.max_mean_sigma_nm),
                    "max_mean_uncertainty_nm": float(cfg.max_mean_uncertainty_nm),
                }
                resp = checkpoint_cb("thresholds", payload_thr) or {}
                action = str(resp.get("action", "continue")).lower().strip()
                if action == "abort":
                    raise PipelineAborted("Aborted by user at threshold-tuning checkpoint")
                if action == "update":
                    updates = resp.get("updates", {})
                    if isinstance(updates, dict):
                        _apply_cfg_updates(cfg, updates)
                    continue
                break

        # -------------------------
        # DEBUG(2) - filtering
        # -------------------------
        n_before = len(npc_list)
        npc_list = filter_npcs_min_locs_sigma_unc(
            npc_list,
            min_n_locs=cfg.min_n_locs,
            max_mean_sigma_nm=cfg.max_mean_sigma_nm,
            max_mean_uncertainty_nm=cfg.max_mean_uncertainty_nm,
        )
        dbg.log(2, f"Filter step1 (min locs + mean sigma/unc): {len(npc_list)} / {n_before}")

        if len(npc_list) == 0:
            dbg.log(2, "No NPCs left after basic filtering. Stopping.")
            job_ok = True
            break

        # Fit radius (step1) with loose bounds (0..inf) to estimate effective radius
        npc_list = filter_npcs_by_radius(npc_list, min_radius_nm=0.0, max_radius_nm=float("inf"))

        # ------------------------------------------------------------------
        # Checkpoint (optional): centering QC after free-radius fit
        # ------------------------------------------------------------------
        if bool(getattr(cfg, "step_by_step", False)) and bool(getattr(cfg, "checkpoint_centering", True)) and checkpoint_cb is not None and len(npc_list) > 0:
            det_centers_nm = np.array([n.npc_center_nm for n in npc_list], dtype=float)
            fit_centers_nm = np.array([n.center_fitcircle_step1_nm for n in npc_list if n.center_fitcircle_step1_nm is not None], dtype=float)

            # Ensure paired arrays (they should match for all entries, but keep safe)
            if fit_centers_nm.shape[0] == det_centers_nm.shape[0] and fit_centers_nm.size:
                shift_nm = np.sqrt(np.sum((fit_centers_nm - det_centers_nm) ** 2, axis=1))
                payload_center: Dict[str, object] = {
                    "cycle": int(cycle_n),
                    "n": int(shift_nm.size),
                    "pixel_size_nm": float(cfg.render_px_size_nm),
                    "centers_detected_px": det_centers_nm / float(cfg.render_px_size_nm),
                    "centers_fit_px": fit_centers_nm / float(cfg.render_px_size_nm),
                    "shift_nm": shift_nm,
                    "median_shift_nm": float(np.nanmedian(shift_nm)) if shift_nm.size else float("nan"),
                    "p95_shift_nm": float(np.nanpercentile(shift_nm, 95)) if shift_nm.size else float("nan"),
                }
                resp = checkpoint_cb("centering", payload_center) or {}
                action = str(resp.get("action", "continue")).lower().strip()
                if action == "abort":
                    raise PipelineAborted("Aborted by user at centering QC checkpoint")

        radii_step1 = np.array(
            [n.radius_fitcircle_step1_nm for n in npc_list if n.radius_fitcircle_step1_nm is not None], dtype=float
        )

        if cycle_n == 1:
            effective_radius = float(np.nanmedian(radii_step1)) if radii_step1.size else float(cfg.expected_npc_radius_nm)
            dbg.log(2, f"Median fitted radius (step1) ~ {effective_radius:.2f} nm")

            if cfg.min_dist_between_npcs_nm == -1:
                min_dist_between_npcs_nm = 1.3 * 2.0 * float(effective_radius)
                dbg.log(2, f"Auto min distance between NPCs = {min_dist_between_npcs_nm:.2f} nm")

        # Use recentered centers (step1) and re-extract by a radius
        recentered_centers_nm = np.array(
            [n.center_fitcircle_step1_nm for n in npc_list if n.center_fitcircle_step1_nm is not None], dtype=float
        )
        if recentered_centers_nm.size == 0:
            dbg.log(2, "No fitted centers available. Stopping.")
            job_ok = True
            break

        recentered_centers_px = recentered_centers_nm / float(cfg.render_px_size_nm)

        if cycle_n == 1:
            # Tune extraction circle radius: median(radius_step1) + 4*median(mean_uncertainty)
            unc_means = np.array([n.npc_uncertainty_mean for n in npc_list if np.isfinite(n.npc_uncertainty_mean)], dtype=float)
            med_unc = float(np.nanmedian(unc_means)) if unc_means.size else 0.0
            effective_extract_circle_radius = (
                float(np.nanmedian(radii_step1)) if radii_step1.size else float(cfg.extract_circle_radius_nm)
            ) + 4.0 * med_unc
            dbg.log(2, f"Auto extraction circle radius = {effective_extract_circle_radius:.2f} nm")

        extract_r = float(effective_extract_circle_radius if effective_extract_circle_radius is not None else cfg.extract_circle_radius_nm)

        npc_list = extract_npcs_by_radius(
            df=df_remaining,
            centers_px=recentered_centers_px,
            pixel_size_nm=cfg.render_px_size_nm,
            radius_nm=extract_r,
            start_id=start_id,
        )
        dbg.log(2, f"Re-extracted {len(npc_list)} NPC ROIs (circle, r={extract_r:.1f} nm)")

        # Filter again
        n_before = len(npc_list)
        npc_list = filter_npcs_min_locs_sigma_unc(
            npc_list,
            min_n_locs=cfg.min_n_locs,
            max_mean_sigma_nm=cfg.max_mean_sigma_nm,
            max_mean_uncertainty_nm=cfg.max_mean_uncertainty_nm,
        )
        dbg.log(2, f"Filter step1b (after re-extract): {len(npc_list)} / {n_before}")

        if len(npc_list) == 0:
            dbg.log(2, "No NPCs left after recentered extraction + filtering. Stopping.")
            job_ok = True
            break

        # Determine radius bounds (auto) on first cycle
        radii_step1 = np.array(
            [n.radius_fitcircle_step1_nm for n in npc_list if n.radius_fitcircle_step1_nm is not None], dtype=float
        )
        if cycle_n == 1:
            effective_radius = float(np.nanmedian(radii_step1)) if radii_step1.size else float(cfg.expected_npc_radius_nm)

            # median localization precision proxy
            unc_means = np.array([n.npc_uncertainty_mean for n in npc_list if np.isfinite(n.npc_uncertainty_mean)], dtype=float)
            med_unc = float(np.nanmedian(unc_means)) if unc_means.size else 0.0

            effective_min_radius = (
                effective_radius - cfg.min_max_radius_tolerance * med_unc
                if cfg.min_radius_nm == -1
                else float(cfg.min_radius_nm)
            )
            effective_max_radius = (
                effective_radius + cfg.min_max_radius_tolerance * med_unc
                if cfg.max_radius_nm == -1
                else float(cfg.max_radius_nm)
            )
            dbg.log(2, f"Auto radius filter = [{effective_min_radius:.2f}, {effective_max_radius:.2f}] nm")
        else:
            effective_min_radius = float(cfg.min_radius_nm if cfg.min_radius_nm != -1 else 0.0)
            effective_max_radius = float(cfg.max_radius_nm if cfg.max_radius_nm != -1 else float("inf"))

        # Fit + status assignment for radius bounds
        npc_list = filter_npcs_by_radius(npc_list, min_radius_nm=float(effective_min_radius), max_radius_nm=float(effective_max_radius))

        # Determine inner/outer bounds (auto) on first cycle
        if cycle_n == 1:
            radii_step1_all = np.array(
                [n.radius_fitcircle_step1_nm for n in npc_list if n.radius_fitcircle_step1_nm is not None], dtype=float
            )
            effective_npc_radius = (
                float(np.nanmedian(radii_step1_all)) if radii_step1_all.size else float(cfg.expected_npc_radius_nm)
            )

            unc_means_ok = np.array(
                [
                    n.npc_uncertainty_mean
                    for n in npc_list
                    if np.isfinite(n.npc_uncertainty_mean) and n.npc_status == 1
                ],
                dtype=float,
            )
            med_unc_ok = float(np.nanmedian(unc_means_ok)) if unc_means_ok.size else 0.0

            effective_inner_radius = (
                effective_npc_radius - cfg.inner_outer_radius_tolerance * med_unc_ok
                if cfg.inner_radius_nm == -1
                else float(cfg.inner_radius_nm)
            )
            effective_outer_radius = (
                effective_npc_radius + cfg.inner_outer_radius_tolerance * med_unc_ok
                if cfg.outer_radius_nm == -1
                else float(cfg.outer_radius_nm)
            )
            dbg.log(
                2,
                f"Auto loc distribution bounds: inner={effective_inner_radius:.2f} nm, outer={effective_outer_radius:.2f} nm",
            )
        else:
            effective_npc_radius = float(cfg.expected_npc_radius_nm)
            effective_inner_radius = float(cfg.inner_radius_nm if cfg.inner_radius_nm != -1 else 0.0)
            effective_outer_radius = float(cfg.outer_radius_nm if cfg.outer_radius_nm != -1 else float("inf"))

        # Filter by localization distribution (center refinement with fixed radius)
        npc_list = recenter_and_filter_distribution(
            npc_list,
            fix_radius_nm=float(effective_npc_radius),
            max_locs_too_close=cfg.max_locs_too_close,
            max_locs_too_far=cfg.max_locs_too_far,
            inner_radius_nm=float(effective_inner_radius),
            outer_radius_nm=float(effective_outer_radius),
        )

        n_ok_cycle = sum(n.npc_status == 1 for n in npc_list)
        dbg.log(2, f"After distribution filter: {n_ok_cycle} ok / {len(npc_list)} total in cycle")

        # ------------------------------------------------------------------
        # Checkpoint (optional): per-cycle per-NPC QC (centering + 8-corner mapping)
        # ------------------------------------------------------------------
        if bool(getattr(cfg, "step_by_step", False)) and bool(getattr(cfg, "checkpoint_cycle_qc", True)) and checkpoint_cb is not None and len(npc_list) > 0:
            try:
                npc_list_qc = align_and_count_corners(
                    npc_list,
                    method=str(cfg.alignment_method),
                    min_locs_per_corner=int(cfg.min_locs_per_corner),
                )
            except Exception:
                npc_list_qc = npc_list

            keep_default = [int(n.npc_id) for n in npc_list_qc if int(getattr(n, "npc_status", 1)) == 1]
            payload_npc_qc = {
                "cycle": int(cycle_n),
                "title": f"Per-cycle NPC QC (cycle {cycle_n})",
                "npcs": npc_list_qc,
                "pixel_size_nm": float(pixel_size_nm),
                "alignment_method": str(cfg.alignment_method),
                "min_locs_per_corner": int(cfg.min_locs_per_corner),
                "qc_template_radius_nm": float(getattr(cfg, "qc_template_radius_nm", 55.0)),
                "qc_corner_tolerance_nm": float(getattr(cfg, "qc_corner_tolerance_nm", 15.0)),
                "keep_npc_ids": keep_default,
            }

            resp = checkpoint_cb("npc_qc", payload_npc_qc) or {}
            action = str(resp.get("action", "continue")).lower()
            if action == "abort":
                raise PipelineAborted("Aborted by user at per-cycle NPC QC checkpoint.")

            keep_ids = resp.get("keep_npc_ids", None)
            if keep_ids is not None:
                keep_set = {int(x) for x in keep_ids}
                for n in npc_list:
                    if int(n.npc_id) in keep_set:
                        n.npc_status = 1
                    else:
                        n.npc_status = 6  # manual reject
                n_ok_cycle = sum(n.npc_status == 1 for n in npc_list)
                dbg.log(2, f"After manual QC: {n_ok_cycle} ok / {len(npc_list)} total in cycle")

        # Update aggregated list and remove close NPCs
        start_id += len(npc_list)
        npc_list_final = remove_close_npcs(
            npc_list_final + npc_list,
            min_distance_nm=float(min_dist_between_npcs_nm if min_dist_between_npcs_nm != -1 else effective_min_dist_nm),
        )

        dbg.log(2, f"Accumulated NPCs (status=1): {sum(n.npc_status == 1 for n in npc_list_final)}")

        # Stop conditions
        if cycle_n >= int(cfg.max_cycles):
            job_ok = True
        else:
            # remove used localizations and iterate
            df_new = update_loc_table_remove_used(df_remaining, npc_list, mode="by_index")
            if len(df_new) == 0:
                job_ok = True
            else:
                df_remaining = df_new
                cycle_n += 1

    # ------------------------------------------------------------------
    # Final cleanup + alignment/ELE
    # ------------------------------------------------------------------
    npc_ok = remove_close_npcs(
        [n for n in npc_list_final if n.npc_status == 1],
        min_distance_nm=float(min_dist_between_npcs_nm if min_dist_between_npcs_nm != -1 else 0.0),
    )
    npc_rejected = remove_close_npcs(
        [n for n in npc_list_final if n.npc_status != 1],
        min_distance_nm=float(min_dist_between_npcs_nm if min_dist_between_npcs_nm != -1 else 0.0),
    )

    dbg.log(3, f"Final NPCs: {len(npc_ok)} ok, {len(npc_rejected)} rejected")
    dbg.log(3, f"Corner counting / alignment: method={cfg.alignment_method}")

    # ------------------------------------------------------------------
    # Alignment / corner counting (with optional interactive QC)
    # ------------------------------------------------------------------
    # Keep a stable master list for interactive keep/reject decisions.
    npc_ok_master = list(npc_ok)
    keep_ids_set = set(int(n.npc_id) for n in npc_ok_master)

    def _npc_ok_selected() -> List[NPCRecord]:
        return [n for n in npc_ok_master if int(n.npc_id) in keep_ids_set]

    npc_final = align_and_count_corners(
        _npc_ok_selected(),
        method=str(cfg.alignment_method),
        min_locs_per_corner=int(cfg.min_locs_per_corner),
    )

    # ------------------------------------------------------------------
    # Checkpoint (optional): validate alignment + corner model application
    # ------------------------------------------------------------------
    if bool(getattr(cfg, "step_by_step", False)) and bool(getattr(cfg, "checkpoint_model_qc", True)) and checkpoint_cb is not None:
        while True:
            # Recompute alignment on the currently selected subset
            npc_final = align_and_count_corners(
                _npc_ok_selected(),
                method=str(cfg.alignment_method),
                min_locs_per_corner=int(cfg.min_locs_per_corner),
            )

            ele_arr = np.asarray([int(getattr(n, "ele", 0)) for n in npc_final], dtype=int)
            centers_nm = (
                np.asarray([_npc_center_nm(n) for n in npc_final], dtype=float) if npc_final else np.zeros((0, 2), dtype=float)
            )
            centers_px = centers_nm / float(cfg.render_px_size_nm) if centers_nm.size else np.zeros((0, 2), dtype=float)

            payload_model: Dict[str, object] = {
                "cycle": int(cycle_n),
                "title": "Final NPC QC (alignment + 8-corner model)",
                "n_npcs": int(len(npc_final)),
                "ele": ele_arr,
                "alignment_method": str(cfg.alignment_method),
                "min_locs_per_corner": int(cfg.min_locs_per_corner),
                "pixel_size_nm": float(cfg.render_px_size_nm),
                "centers_px": centers_px,
                "npcs": npc_final,
                "keep_npc_ids": sorted(int(i) for i in keep_ids_set),
                "qc_template_radius_nm": float(getattr(cfg, "qc_template_radius_nm", 55.0)),
                "qc_corner_tolerance_nm": float(getattr(cfg, "qc_corner_tolerance_nm", 15.0)),
            }

            resp = checkpoint_cb("model_qc", payload_model) or {}
            action = str(resp.get("action", "continue")).lower().strip()
            if action == "abort":
                raise PipelineAborted("Aborted by user at model/alignment QC checkpoint")

            # Update keep-set if provided
            if "keep_npc_ids" in resp and resp.get("keep_npc_ids") is not None:
                try:
                    keep_ids_set = set(int(x) for x in resp.get("keep_npc_ids"))
                except Exception:
                    pass

            if action == "update":
                updates = resp.get("updates", {})
                if isinstance(updates, dict):
                    _apply_cfg_updates(cfg, updates)
                continue

            break

    # Apply final keep-set to statuses so downstream outputs reflect manual QC
    if not keep_ids_set:
        # If the user deselected everything, it is not meaningful to fit ELE.
        raise ValueError("No NPCs left after QC selection.")

    for n in npc_ok_master:
        if int(n.npc_id) not in keep_ids_set:
            n.npc_status = 6

    # Recompute ok/rejected lists after potential status edits
    npc_ok = remove_close_npcs(
        [n for n in npc_list_final if n.npc_status == 1],
        min_distance_nm=float(min_dist_between_npcs_nm if min_dist_between_npcs_nm != -1 else effective_min_dist_nm),
    )
    npc_rejected = remove_close_npcs(
        [n for n in npc_list_final if n.npc_status != 1],
        min_distance_nm=float(min_dist_between_npcs_nm if min_dist_between_npcs_nm != -1 else effective_min_dist_nm),
    )

    npc_final = align_and_count_corners(
        npc_ok,
        method=str(cfg.alignment_method),
        min_locs_per_corner=int(cfg.min_locs_per_corner),
    )
    # Fit ELE
    ele_list = [n.ele for n in npc_final]
    p_label, p_label_err = fit_ele_bootstrap(
        ele_list=ele_list,
        n_samples=int(cfg.ele_bootstrap_samples),
        bins_to_fit=cfg.ele_bins_to_fit,
        seed=int(cfg.random_seed),
    )
    dbg.log(4, f"ELE fit: p_label={p_label:.3f} +/- {p_label_err:.3f}")

    # Build results table
    df_npc_final = npc_list_to_dataframe(npc_final)

    # ------------------------------------------------------------------
    # Save outputs (JSON/CSV)
    # ------------------------------------------------------------------
    cfg_json = out_dir / "labelis_config.json"
    cfg_json.write_text(json.dumps(sanitize_for_json(cfg), indent=2), encoding="utf-8")
    save_npc_table(df_npc_final, out_dir / "labelis_npcs_final.csv")

    derived = {
        "effective_extract_circle_radius_nm": float(effective_extract_circle_radius) if effective_extract_circle_radius is not None else None,
        "effective_radius_nm": float(effective_radius) if effective_radius is not None else None,
        "effective_min_radius_nm": float(effective_min_radius) if effective_min_radius is not None else None,
        "effective_max_radius_nm": float(effective_max_radius) if effective_max_radius is not None else None,
        "effective_inner_radius_nm": float(effective_inner_radius) if effective_inner_radius is not None else None,
        "effective_outer_radius_nm": float(effective_outer_radius) if effective_outer_radius is not None else None,
        "effective_npc_radius_nm": float(effective_npc_radius) if effective_npc_radius is not None else None,
        "min_dist_between_npcs_nm": float(min_dist_between_npcs_nm),
    }

    summary: Dict[str, object] = {
        "labelis_version": LABELIS_VERSION,
        "input_path": str(cfg.input_path),
        "output_dir": str(out_dir),
        "n_localizations_input": int(n_input),
        "n_localizations_filtered": int(len(df_filtered)),
        "x_offset_nm": float(x_offset_nm),
        "y_offset_nm": float(y_offset_nm),
        "n_npcs_ok": int(len(npc_ok)),
        "n_npcs_rejected": int(len(npc_rejected)),
        "n_npcs_final": int(len(npc_final)),
        "p_label": float(p_label),
        "p_label_error": float(p_label_err),
        "alignment_method": str(cfg.alignment_method),
        "compute_engine": str(cfg.compute_engine),
        "segmentation_engine": str(cfg.segmentation_engine),
        "derived": sanitize_for_json(derived),
    }
    (out_dir / "labelis_summary.json").write_text(json.dumps(sanitize_for_json(summary), indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Save intermediate images (same as v3) + requested MATLAB-like figures
    # ------------------------------------------------------------------
    if cfg.save_intermediate_images and image_rendered_saved is not None:
        _save_image_png(image_rendered_saved, fig_dir / "rendered_cycle1.png", title="Rendered image (cycle 1)")
    if cfg.save_intermediate_images and image_segment_saved is not None:
        _save_image_png(image_segment_saved, fig_dir / "segment_input_cycle1.png", title="Segmentation input (cycle 1)")

    # Overlay all detections (green circles)
    if cfg.save_intermediate_images and image_rendered_saved is not None and len(all_centers) > 0:
        c_all = np.concatenate(all_centers, axis=0) if all_centers else np.zeros((0, 2))
        r_all = np.concatenate(all_radii, axis=0) if all_radii else np.zeros((0,))
        if c_all.size and r_all.size:
            _save_overlay_png(
                image_rendered_saved,
                c_all,
                r_all,
                fig_dir / "overlay_all_candidates.png",
                title="All candidates (rough)",
                edgecolor_default=(0.0, 1.0, 0.0),
                linewidth=1.0,
            )

    # Overlay final OK NPCs on rendered image (color-coded by ELE=1..8)
    if cfg.save_intermediate_images and image_rendered_saved is not None and len(npc_final) > 0:
        import matplotlib.cm as cm

        centers_nm = np.array([_npc_center_nm(n) for n in npc_final], dtype=float)
        centers_px = centers_nm / float(cfg.render_px_size_nm)

        radii_nm = np.array(
            [n.radius_fitcircle_step2_nm if n.radius_fitcircle_step2_nm else (n.radius_fitcircle_step1_nm or cfg.expected_npc_radius_nm) for n in npc_final],
            dtype=float,
        )
        radii_px = radii_nm / float(cfg.render_px_size_nm)

        # Map ELE (1..8) -> colormap
        cmap = cm.get_cmap("viridis", 8)
        ele_vals = np.array([int(getattr(n, "ele", 0)) for n in npc_final], dtype=int)
        ele_vals = np.clip(ele_vals, 1, 8)
        edgecols = [tuple(map(float, cmap(int(v) - 1)[:3])) for v in ele_vals]

        _save_overlay_png(
            image_rendered_saved,
            centers_px,
            radii_px,
            fig_dir / "overlay_ok_npcs.png",
            title="OK NPCs (colored by ELE)",
            edgecolors=edgecols,
            edgecolor_default=(1.0, 1.0, 0.0),
            linewidth=1.2,
        )

    # MATLAB-like figures
    fig_paths: Dict[str, str] = {}
    try:
        dbg.log(5, "Saving diagnostic figures")

        # Figure 1: ELE
        p = save_figure_ele(fig_dir, ele_list=ele_list, p_label=float(p_label), p_label_error=float(p_label_err))
        fig_paths["figure_ELE"] = str(p)
        dbg.log(5, f"Saved {p.name}")

        # Figure 2: NPC info
        p = save_figure_info_npc(
            fig_dir,
            npc_final=npc_final,
            effective_min_radius_nm=float(effective_min_radius) if effective_min_radius is not None else float("nan"),
            effective_max_radius_nm=float(effective_max_radius) if effective_max_radius is not None else float("nan"),
            effective_npc_radius_nm=float(effective_npc_radius) if effective_npc_radius is not None else float(cfg.expected_npc_radius_nm),
            p_label=float(p_label),
        )
        fig_paths["figure_info_NPC"] = str(p)
        dbg.log(5, f"Saved {p.name}")

        # Figure 3: Summed NPC images + Figure 4: intensity profile
        # Use extraction radius as field of view for summed image
        radius_sum = float(effective_extract_circle_radius) if effective_extract_circle_radius is not None else float(cfg.extract_circle_radius_nm)
        p_img, sum_img_aligned = save_figure_npc_image(fig_dir, npc_final=npc_final, radius_nm=radius_sum)
        fig_paths["figure_NPC_image"] = str(p_img)
        dbg.log(6, f"Saved {p_img.name}")

        sum_img_raw, _ = None, None
        # We also store the aligned sum image in the workspace export
        sum_img_aligned = np.asarray(sum_img_aligned, dtype=np.float32)
        # For workspace export we also want raw sum
        try:
            from .figures import construct_summed_images

            sum_img_raw, sum_img_aligned2 = construct_summed_images(
                npc_final, radius_nm=radius_sum, pixel_size_nm=5.0, blur_sigma_px=1.0
            )
            sum_img_raw = np.asarray(sum_img_raw, dtype=np.float32)
            sum_img_aligned = np.asarray(sum_img_aligned2, dtype=np.float32)
        except Exception:
            pass

        sum_img_raw = sum_img_raw
        sum_img_aligned = sum_img_aligned

        p_prof = save_figure_npc_image_intensity_profile(fig_dir, npc_image_aligned=sum_img_aligned)
        fig_paths["figure_NPC_image_intensity_profile"] = str(p_prof)
        dbg.log(6, f"Saved {p_prof.name}")

    except Exception as e:
        dbg.log(5, f"Figure generation failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # DEBUG(7) - save "workspace" into H5
    # ------------------------------------------------------------------
    dbg_path = out_dir / "labelis_debug.log"
    dbg.log(7, f"Debug log file: {dbg_path.name}")

    # Construct images dict for workspace
    images = {
        "rendered_cycle1": image_rendered_saved,
        "segment_input_cycle1": image_segment_saved,
        "sum_npc_raw": sum_img_raw,
        "sum_npc_aligned": sum_img_aligned,
    }
    detections = {
        "centers_px": (np.concatenate(all_centers, axis=0) if all_centers else np.zeros((0, 2), dtype=float)),
        "radii_px": (np.concatenate(all_radii, axis=0) if all_radii else np.zeros((0,), dtype=float)),
        "metric": (np.concatenate(all_metric, axis=0) if all_metric else np.zeros((0,), dtype=float)),
    }

    h5_path = out_dir / "labelis_workspace.h5"
    try:
        save_workspace_h5(
            h5_path,
            labelis_version=LABELIS_VERSION,
            config=cfg,
            summary=summary,
            derived=derived,
            debug_lines=dbg.as_lines(),
            roi_polygon_nm=roi_polygon_nm,
            df_input=df_in,
            df_filtered=df_filtered,
            df_npc_final=df_npc_final,
            images=images,
            detections=detections,
            npc_records=(npc_list_final if npc_list_final else None),
        )
        dbg.log(7, f"Workspace saved: {h5_path.name}")
    except Exception as e:
        dbg.log(7, f"Workspace H5 save failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # DEBUG(8) - export final localizations
    # ------------------------------------------------------------------
    final_locs_path = out_dir / "labelis_final_localizations.csv"
    try:
        _export_final_localizations(df_filtered, npc_final, final_locs_path)
        dbg.log(8, f"Final localization export: {final_locs_path.name}")
    except Exception as e:
        dbg.log(8, f"Final localization export failed (non-fatal): {e}")

    # Persist debug log *after* all stages so it contains Debug(0..8)
    try:
        dbg.save_text(dbg_path)
    except Exception:
        # Non-fatal: the GUI already displayed the logs.
        pass

    # ------------------------------------------------------------------
    # Return results
    # ------------------------------------------------------------------
    res = PipelineResult(
        config=cfg,
        x_offset_nm=x_offset_nm,
        y_offset_nm=y_offset_nm,
        image_rendered=image_rendered_saved,
        image_segment=image_segment_saved,
        all_centers_px=detections["centers_px"],
        all_radii_px=detections["radii_px"],
        all_metric=detections["metric"],
        npc_all=npc_list_final,
        npc_ok=npc_ok,
        npc_rejected=npc_rejected,
        npc_final=npc_final,
        p_label=float(p_label),
        p_label_error=float(p_label_err),
        df_npc_final=df_npc_final,
        derived=derived,
        workspace_h5=str(h5_path),
        debug_log=str(dbg_path),
        figure_paths=fig_paths,
        final_localizations_csv=str(final_locs_path),
        summary=summary,
        cpsam_instance_labels=cpsam_instance_labels,
    )
    return res
