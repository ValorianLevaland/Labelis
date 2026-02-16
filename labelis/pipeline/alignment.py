from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from .npc import NPCRecord


def fun_match_template_npc(rotation: float, array_radius: np.ndarray, array_theta: np.ndarray) -> float:
    """Port of fun_match_template_NPC_3.m.

    The NPC template has 8 corners at angles pi/8 + k*pi/4 with radius 55 nm.

    Returns
    -------
    sum_total : float
        Sum of abs(signed distances) to the nearest template corner in the corresponding 45° sector.
    """
    NPC_template_radius = np.array([55.0] * 8, dtype=float)
    NPC_template_corner = np.arange(1, 16, 2, dtype=float) * (np.pi / 8.0)  # pi/8, 3pi/8, ..., 15pi/8

    r = np.asarray(array_radius, dtype=float)
    th = np.asarray(array_theta, dtype=float)

    # add the rotation
    corners_rot = th + float(rotation)
    # wrap into [0,2pi)
    corners_rot = np.mod(corners_rot, 2.0 * np.pi)

    # sector boundaries at 0,45,90,...,360
    sector = np.floor(corners_rot / (np.pi / 4.0)).astype(int)  # 0..7
    sector = np.clip(sector, 0, 7)

    # template corner for each loc
    t2 = NPC_template_corner[sector]
    r2 = NPC_template_radius[sector]

    t1 = corners_rot
    r1 = r

    # distance between two polar points
    dist = np.sqrt(r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * np.cos(t1 - t2))

    # sign convention: if t1 - t2 > 0, distance becomes negative
    dist = np.where((t1 - t2) > 0, -dist, dist)

    # sum of absolute values per sector (MATLAB accumulates per sector then sums, equivalent to sum(abs(dist)))
    return float(np.sum(np.abs(dist)))


def _wrap_0_2pi(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    theta = np.mod(theta, 2.0 * np.pi)
    # ensure [0,2pi)
    theta[theta < 0] += 2.0 * np.pi
    return theta


def align_and_count_corners(
    npcs: Sequence[NPCRecord],
    method: str,
    min_locs_per_corner: int,
) -> List[NPCRecord]:
    """Compute polar coords, alignment rotation, and corner counts (ELE) for each NPC."""
    out: List[NPCRecord] = []
    edges = np.arange(0.0, 2.0 * np.pi + 1e-9, np.pi / 4.0)  # 9 edges for 8 bins

    for n in npcs:
        # "good" locs for alignment: those within inner/outer radial bounds (computed earlier)
        if n.center_fitcircle_step2_nm is None:
            center = n.center_fitcircle_step1_nm if n.center_fitcircle_step1_nm else n.npc_center_nm
        else:
            center = n.center_fitcircle_step2_nm

        cx, cy = float(center[0]), float(center[1])

        # good loc indices
        if n.idx_good is not None and n.idx_good.size:
            # subset from raw arrays by indices mapping
            # build a mapping from idx_all to position
            pos = {int(idx): k for k, idx in enumerate(n.idx_all)}
            good_pos = np.array([pos[int(i)] for i in n.idx_good if int(i) in pos], dtype=int)
        else:
            good_pos = np.arange(n.x_nm.size, dtype=int)

        x_good = n.x_nm[good_pos]
        y_good = n.y_nm[good_pos]

        # all locs
        x_all = n.x_nm
        y_all = n.y_nm

        # polar coordinates (MATLAB adds pi to make 0..2pi)
        dxg = x_good - cx
        dyg = y_good - cy
        rho = np.sqrt(dxg * dxg + dyg * dyg)
        theta = np.arctan2(dyg, dxg) + np.pi

        dxa = x_all - cx
        dya = y_all - cy
        rho_all = np.sqrt(dxa * dxa + dya * dya)
        theta_all = np.arctan2(dya, dxa) + np.pi

        # Determine rotation
        rotation = 0.0
        if method == "template":
            # minimize template objective over [0,2pi]
            f = lambda a: fun_match_template_npc(a, rho, theta)
            res = minimize_scalar(f, bounds=(0.0, 2.0 * np.pi), method="bounded")
            rotation = float(res.x)
            theta_aligned = theta + rotation
            theta_aligned_all = theta_all + rotation

        elif method == "smap":
            # SMAP-like: minimize sum(abs(a - mod(theta, pi/4)))
            f = lambda a: float(np.sum(np.abs(a - np.mod(theta, np.pi / 4.0))))
            res = minimize_scalar(f, bounds=(0.0, 2.0 * np.pi), method="bounded")
            rotation = float(res.x) - (np.pi / 8.0)
            theta_aligned = theta - rotation
            theta_aligned_all = theta_all - rotation

        elif method == "none":
            rotation = 0.0
            theta_aligned = theta
            theta_aligned_all = theta_all

        else:
            raise ValueError(f"Unknown alignment method: {method}")

        theta_aligned = _wrap_0_2pi(theta_aligned)
        theta_aligned_all = _wrap_0_2pi(theta_aligned_all)

        # Distances to template (store both good and all, as in MATLAB)
        n.sum_distance_initial = fun_match_template_npc(0.0, rho, theta)
        n.sum_distance_minimized = fun_match_template_npc(rotation, rho, theta)
        n.sum_distance_initial_all = fun_match_template_npc(0.0, rho_all, theta_all)
        n.sum_distance_minimized_all = fun_match_template_npc(rotation, rho_all, theta_all)

        n.rotation_rad = rotation
        n.coords_polar_rho_all = rho_all
        n.coords_polar_theta_alligned_all = theta_aligned_all

        # Corner counts using histogram edges
        counts, _ = np.histogram(theta_aligned, bins=edges)
        n.locs_per_corner = counts.astype(int)
        n.ele = int(np.sum(counts >= int(min_locs_per_corner)))

        out.append(n)

    return out
