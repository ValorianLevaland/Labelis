from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


from .ele import fun_fit_ele


# -----------------------------
# Styling knobs (edit here)
# -----------------------------
# Change these constants to control the color palette of exported figures.
COLOR_ELE_MODEL_LINE = "orange"      # model curve in figure_ELE
COLOR_ELE_HIST_BARS = "#B0B0B0"      # histogram bars in figure_ELE

COLOR_INFO_ALL_LOCS = "#2C7FB8"      # line for "all locs" in figure_info_NPC
COLOR_INFO_GOOD_LOCS = "#F28E2B"     # line for "good locs" in figure_info_NPC


def save_figure_ele(
    fig_dir: str | Path,
    *,
    ele_list: Sequence[int],
    p_label: float,
    p_label_error: float,
) -> Path:
    """Reproduce MATLAB `figure_ELE` output (bar + model fit)."""

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "figure_ELE.png"

    ele_arr = np.asarray(ele_list, dtype=int)
    edges = np.arange(-0.5, 8.5 + 1e-9, 1.0)
    counts, _ = np.histogram(ele_arr, bins=edges)
    counts = counts.astype(float)
    if counts.sum() > 0:
        counts /= counts.sum()

    bins = np.arange(0, 9, 1)
    model = fun_fit_ele(float(p_label), bins)

    fig = plt.figure(figsize=(6, 4), dpi=150)
    ax = fig.add_subplot(111)
    ax.bar(bins, counts, align="center", color=COLOR_ELE_HIST_BARS, edgecolor="none")
    ax.plot(bins, model, marker="o", linewidth=2, color=COLOR_ELE_MODEL_LINE)
    ax.set_title(f"p_l_a_b_e_l = {p_label:.3f}  +/-  {p_label_error:.3f}")
    ax.set_xlabel("Number of visible corners")
    ax.set_ylabel("Probability")
    ax.set_xticks(bins)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_figure_info_npc(
    fig_dir: str | Path,
    *,
    npc_final,
    effective_min_radius_nm: float,
    effective_max_radius_nm: float,
    effective_npc_radius_nm: float,
    p_label: float,
    add_radius_px: float = 3.0,
) -> Path:
    """Reproduce MATLAB `figure_info_NPC` (2x3 diagnostics)."""

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "figure_info_NPC.png"

    npcs = list(npc_final) if npc_final is not None else []

    n_locs_all = np.array([n.npc_n_locs for n in npcs], dtype=float)
    n_locs_good = np.array([n.n_locs_good for n in npcs], dtype=float)
    ele = np.array([n.ele for n in npcs], dtype=float)
    locs_per_corner = np.concatenate([n.locs_per_corner for n in npcs if n.locs_per_corner is not None], axis=0) if npcs else np.zeros((0,))

    dist_all = []
    dist_good = []
    for n in npcs:
        if n.sum_distance_minimized_all is not None and n.npc_n_locs > 0:
            dist_all.append(float(n.sum_distance_minimized_all) / float(n.npc_n_locs))
        if n.sum_distance_minimized is not None and n.n_locs_good > 0:
            dist_good.append(float(n.sum_distance_minimized) / float(n.n_locs_good))

    dist_all = np.asarray(dist_all, dtype=float)
    dist_good = np.asarray(dist_good, dtype=float)

    rho_all = np.concatenate([n.coords_polar_rho_all for n in npcs if n.coords_polar_rho_all is not None], axis=0) if npcs else np.zeros((0,))
    theta_all = np.concatenate([n.coords_polar_theta_alligned_all for n in npcs if n.coords_polar_theta_alligned_all is not None], axis=0) if npcs else np.zeros((0,))
    r_fit = np.array([n.radius_fitcircle_step1_nm for n in npcs if n.radius_fitcircle_step1_nm is not None], dtype=float)

    fig = plt.figure(figsize=(10, 6), dpi=150)

    # (1) localizations per NPC
    ax = fig.add_subplot(2, 3, 1)
    edges = np.arange(0, 250 + 10, 10)
    bins = edges[:-1] + np.diff(edges) / 2
    if n_locs_all.size:
        h_all, _ = np.histogram(n_locs_all, bins=edges, density=True)
        ax.plot(bins, h_all, linewidth=2, label="all locs", color=COLOR_INFO_ALL_LOCS)
    if n_locs_good.size:
        h_good, _ = np.histogram(n_locs_good, bins=edges, density=True)
        ax.plot(bins, h_good, linewidth=2, label="good locs", color=COLOR_INFO_GOOD_LOCS)
    ax.set_xlabel("# locs per NPC")
    ax.set_ylabel("Probability")
    ax.legend()
    # crude fluorophore estimate as in MATLAB: mean(n_locs_good/(32*ELE))
    denom = 32.0 * np.where(ele > 0, ele, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = np.nanmean(n_locs_good / denom) if denom.size else np.nan
    ax.set_title(f"Mean locs per fluorophore ≈ {est:.3g}")

    # (2) localizations per corner
    ax = fig.add_subplot(2, 3, 2)
    edges = np.arange(0, 50 + 1, 1)
    bins = edges[:-1] + 0.5
    if locs_per_corner.size:
        h, _ = np.histogram(locs_per_corner, bins=edges, density=True)
        ax.plot(bins, h, linewidth=2, color=COLOR_INFO_ALL_LOCS)
    ax.set_xlabel("# locs per corner (good locs only)")
    ax.set_ylabel("Probability")
    ax.set_title("Localizations per corner")

    # (3) distance to template
    ax = fig.add_subplot(2, 3, 3)
    edges = np.arange(0, 30 + 1, 1)
    bins = edges[:-1] + 0.5
    if dist_all.size:
        h_all, _ = np.histogram(dist_all, bins=edges, density=True)
        ax.plot(bins, h_all, linewidth=2, label="all locs", color=COLOR_INFO_ALL_LOCS)
    if dist_good.size:
        h_good, _ = np.histogram(dist_good, bins=edges, density=True)
        ax.plot(bins, h_good, linewidth=2, label="good locs", color=COLOR_INFO_GOOD_LOCS)
    ax.set_xlabel("Avg distance locs→template per NPC")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.set_title("Template match")

    # (4) radius distribution (rho)
    ax = fig.add_subplot(2, 3, 4)
    edges = np.arange(0, 150 + 2, 2)
    if rho_all.size:
        ax.hist(rho_all, bins=edges, density=True, edgecolor="none")
    ax.axvline(40, linewidth=2)
    ax.axvline(70, linewidth=2)
    med_rho = float(np.nanmedian(rho_all)) if rho_all.size else float("nan")
    ax.set_xlabel("Distance to center (nm)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Median distance={med_rho:.2f} nm")

    # (5) theta alignment
    ax = fig.add_subplot(2, 3, 5)
    edges = np.arange(0, 2 * np.pi + (np.pi / 64), np.pi / 64)
    if theta_all.size:
        ax.hist(theta_all, bins=edges, density=True, edgecolor="none")
    for x in np.arange(0, 2 * np.pi + 1e-9, np.pi / 4):
        ax.axvline(x)
    ax.set_title("Theta localizations after alignment")
    ax.set_xlabel("Theta (rad)")
    ax.set_ylabel("Probability")

    # (6) fitted NPC radius
    ax = fig.add_subplot(2, 3, 6)
    if np.isfinite(effective_min_radius_nm) and np.isfinite(effective_max_radius_nm) and effective_max_radius_nm > effective_min_radius_nm:
        edges = np.arange(effective_min_radius_nm, effective_max_radius_nm + 0.5, 0.5)
    else:
        edges = np.arange(0, 150 + 0.5, 0.5)
    if r_fit.size:
        ax.hist(r_fit, bins=edges, density=True, edgecolor="none")
    ax.set_title("Fitted NPC radius")
    ax.set_xlabel("NPC radius (nm)")
    ax.set_ylabel("Probability")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _npc_center_nm(n) -> Tuple[float, float]:
    if getattr(n, "center_fitcircle_step2_nm", None) is not None:
        c = n.center_fitcircle_step2_nm
        return float(c[0]), float(c[1])
    if getattr(n, "center_fitcircle_step1_nm", None) is not None:
        c = n.center_fitcircle_step1_nm
        return float(c[0]), float(c[1])
    c = n.npc_center_nm
    return float(c[0]), float(c[1])


def construct_summed_images(
    npc_final,
    *,
    radius_nm: float,
    pixel_size_nm: float = 5.0,
    blur_sigma_px: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build raw and aligned summed NPC images (MATLAB-like diagnostic)."""

    from scipy.ndimage import gaussian_filter

    npcs = list(npc_final) if npc_final is not None else []
    radius_nm = float(radius_nm)
    px = float(pixel_size_nm)
    size = int(np.ceil(2.0 * radius_nm / px)) + 1
    if size < 8:
        size = 8

    raw = np.zeros((size, size), dtype=np.float32)
    ali = np.zeros((size, size), dtype=np.float32)

    for n in npcs:
        cx, cy = _npc_center_nm(n)

        # ---------------- raw ----------------
        dx = np.asarray(n.x_nm, dtype=float) - cx
        dy = np.asarray(n.y_nm, dtype=float) - cy
        ix = np.floor((dx + radius_nm) / px).astype(int)
        iy = np.floor((dy + radius_nm) / px).astype(int)
        m = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
        if np.any(m):
            np.add.at(raw, (iy[m], ix[m]), 1.0)

        # ---------------- aligned ----------------
        rho = getattr(n, "coords_polar_rho_all", None)
        th = getattr(n, "coords_polar_theta_alligned_all", None)
        if rho is None or th is None:
            # fallback: use raw if alignment vectors not available
            dx2, dy2 = dx, dy
        else:
            rho = np.asarray(rho, dtype=float)
            th = np.asarray(th, dtype=float)
            dx2 = rho * np.cos(th - np.pi)
            dy2 = rho * np.sin(th - np.pi)

        ix2 = np.floor((dx2 + radius_nm) / px).astype(int)
        iy2 = np.floor((dy2 + radius_nm) / px).astype(int)
        m2 = (ix2 >= 0) & (ix2 < size) & (iy2 >= 0) & (iy2 < size)
        if np.any(m2):
            np.add.at(ali, (iy2[m2], ix2[m2]), 1.0)

    if blur_sigma_px and blur_sigma_px > 0:
        raw = gaussian_filter(raw, sigma=float(blur_sigma_px))
        ali = gaussian_filter(ali, sigma=float(blur_sigma_px))

    return raw, ali


def save_figure_npc_image(
    fig_dir: str | Path,
    *,
    npc_final,
    radius_nm: float,
    pixel_size_nm: float = 5.0,
    blur_sigma_px: float = 1.0,
) -> Tuple[Path, np.ndarray]:
    """Save MATLAB-like `figure_NPC_image` (raw + aligned summed NPC)."""

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "figure_NPC_image.png"

    raw, ali = construct_summed_images(
        npc_final,
        radius_nm=float(radius_nm),
        pixel_size_nm=float(pixel_size_nm),
        blur_sigma_px=float(blur_sigma_px),
    )

    fig = plt.figure(figsize=(8, 4), dpi=150)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(raw, cmap="viridis")
    ax1.set_title("Sum NPCs raw")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(ali, cmap="viridis")
    n_npc = len(list(npc_final)) if npc_final is not None else 0
    ax2.set_title(f"Sum NPCs aligned, # NPCs={n_npc}")
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, ali


def save_figure_npc_image_intensity_profile(
    fig_dir: str | Path,
    *,
    npc_image_aligned: np.ndarray,
) -> Path:
    """Save intensity profile figure (MATLAB `figure_NPC_image_intensity_profile`)."""

    from skimage.measure import profile_line

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "figure_NPC_image_intensity_profile.png"

    img = np.asarray(npc_image_aligned, dtype=float)
    h, w = img.shape
    # Match MATLAB-ish geometry: x from 0..W, y from ~0.375H..0.625H
    y0 = 0.375 * float(h)
    y1 = 0.625 * float(h)
    prof = profile_line(img, (y0, 0.0), (y1, float(w - 1)), mode="reflect", reduce_func=None)

    fig = plt.figure(figsize=(6, 3), dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(prof)
    ax.set_title("Intensity profile (aligned summed NPC image)")
    ax.set_xlabel("Profile coordinate (px)")
    ax.set_ylabel("Intensity (a.u.)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
