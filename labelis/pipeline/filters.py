from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .npc import NPCRecord


def fitcircle(points_xy: np.ndarray, nonlinear: bool = True, maxits: int = 100, tol: float = 1e-5) -> Tuple[np.ndarray, float, float]:
    """Least-squares circle fit (ported from the MATLAB fitcircle.m).

    Parameters
    ----------
    points_xy : (N,2) array
    nonlinear : bool
        If True, perform Gauss-Newton refinement to minimize geometric error.
        If False, return linear (algebraic) fit.

    Returns
    -------
    center : (2,) array [x,y]
    radius : float
    residual : float (2-norm of residual vector)
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must have shape (N,2)")
    if pts.shape[0] < 3:
        raise ValueError("At least 3 points are required to fit a circle")

    x1 = pts[:, 0]
    x2 = pts[:, 1]
    m = pts.shape[0]

    # Linear algebraic fit: Bu=0, ||u||=1, u = [a, b1, b2, c]
    B = np.column_stack([x1 * x1 + x2 * x2, x1, x2, np.ones(m)])
    # SVD: smallest singular vector is last column of Vt^T
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    u = Vt[-1, :]

    a = u[0]
    b = u[1:3]
    c = u[3]
    if abs(a) < 1e-12:
        # Degenerate; fallback to mean center, large radius
        center = np.array([np.mean(x1), np.mean(x2)], dtype=float)
        radius = float(np.mean(np.sqrt((x1 - center[0]) ** 2 + (x2 - center[1]) ** 2)))
        residual = float(np.linalg.norm(np.sqrt((x1 - center[0]) ** 2 + (x2 - center[1]) ** 2) - radius))
        return center, radius, residual

    center = -b / (2.0 * a)
    rad_sq = (np.linalg.norm(b) / (2.0 * a)) ** 2 - c / a
    radius = float(np.sqrt(abs(rad_sq)))

    if not nonlinear:
        residual = float(np.linalg.norm(B @ u))
        return center, radius, residual

    # Nonlinear Gauss-Newton refinement minimizing geometric error
    u0 = np.array([center[0], center[1], radius], dtype=float)

    def sys(uvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cx, cy, r = float(uvec[0]), float(uvec[1]), float(uvec[2])
        dx = cx - x1
        dy = cy - x2
        denom = np.sqrt(dx * dx + dy * dy)
        # Avoid division by zero
        denom = np.where(denom < 1e-12, 1e-12, denom)
        f = denom - r
        J = np.column_stack([dx / denom, dy / denom, -np.ones_like(denom)])
        return f, J

    uvec = u0.copy()
    for _ in range(int(maxits)):
        f, J = sys(uvec)
        # Solve least squares J h = -f
        h, *_ = np.linalg.lstsq(J, -f, rcond=None)
        uvec = uvec + h
        delta = np.linalg.norm(h, ord=np.inf) / max(np.linalg.norm(uvec, ord=np.inf), 1e-12)
        if delta < tol:
            break

    cx, cy, r = float(uvec[0]), float(uvec[1]), float(uvec[2])
    f, _ = sys(uvec)
    residual = float(np.linalg.norm(f))
    return np.array([cx, cy], dtype=float), float(r), residual


def filter_npcs_min_locs_sigma_unc(
    npcs: Sequence[NPCRecord],
    min_n_locs: int,
    max_mean_sigma_nm: float,
    max_mean_uncertainty_nm: float,
) -> List[NPCRecord]:
    out: List[NPCRecord] = []
    for n in npcs:
        if n.npc_n_locs < int(min_n_locs):
            continue
        if np.isfinite(max_mean_sigma_nm) and n.npc_sigma_nm_mean >= float(max_mean_sigma_nm):
            continue
        if np.isfinite(max_mean_uncertainty_nm) and n.npc_uncertainty_mean >= float(max_mean_uncertainty_nm):
            continue
        out.append(n)
    return out


def filter_npcs_by_radius(
    npcs: Sequence[NPCRecord],
    min_radius_nm: float,
    max_radius_nm: float,
) -> List[NPCRecord]:
    out: List[NPCRecord] = []
    for n in npcs:
        if n.x_nm.size < 3:
            continue
        center, R, resid = fitcircle(np.column_stack([n.x_nm, n.y_nm]), nonlinear=True)
        n.center_fitcircle_step1_nm = (float(center[0]), float(center[1]))
        n.radius_fitcircle_step1_nm = float(R)
        n.residual_fitcircle_step1 = float(resid)

        if R < float(min_radius_nm):
            n.npc_status = 2
        elif R > float(max_radius_nm):
            n.npc_status = 3
        out.append(n)
    return out


def recenter_and_filter_distribution(
    npcs: Sequence[NPCRecord],
    fix_radius_nm: float,
    max_locs_too_close: float,
    max_locs_too_far: float,
    inner_radius_nm: float,
    outer_radius_nm: float,
) -> List[NPCRecord]:
    """Port of filter_NPC_by_loc_distribution.m (center refinement with fixed radius + QC)."""
    fixR = float(fix_radius_nm)
    inR = float(inner_radius_nm)
    outR = float(outer_radius_nm)

    out: List[NPCRecord] = []
    for n in npcs:
        if n.x_nm.size < 3:
            out.append(n)
            continue

        x = n.x_nm.astype(float)
        y = n.y_nm.astype(float)
        unc = n.uncertainty_nm.astype(float)
        idx_all = n.idx_all.astype(int)

        # initial center guess
        xc0 = float(np.mean(x))
        yc0 = float(np.mean(y))

        # Minimize f = ((x-xc)^2)/R^2 + ((y-yc)^2)/R^2 - 1
        def fun(p):
            xc, yc = float(p[0]), float(p[1])
            return ((x - xc) ** 2 + (y - yc) ** 2) / (fixR * fixR) - 1.0

        res = least_squares(fun, x0=np.array([xc0, yc0], dtype=float), method="trf")
        xc, yc = float(res.x[0]), float(res.x[1])

        n.center_fitcircle_step2_nm = (xc, yc)
        n.radius_fitcircle_step2_nm = fixR

        # distances to center
        dist = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        # classification using uncertainty allowance (as in MATLAB)
        array_too_close = (dist + 3.0 * unc) < inR
        array_too_far = (dist - 3.0 * unc) > outR

        nloc = max(1, int(len(dist)))
        if float(np.sum(array_too_close)) / nloc > float(max_locs_too_close):
            n.npc_status = 4
        elif float(np.sum(array_too_far)) / nloc > float(max_locs_too_far):
            n.npc_status = 5

        # define "bad" locs for removal and good-locs lists (without uncertainty allowance)
        bad_tooclose = dist < inR
        bad_toofar = dist > outR
        bad = bad_tooclose | bad_toofar

        n.idx_good = idx_all[~bad]
        n.idx_tooclose = idx_all[bad_tooclose]
        n.idx_toofar = idx_all[bad_toofar]

        n.n_locs_good = int(np.sum(~bad))
        n.n_locs_tooclose = int(np.sum(bad_tooclose))
        n.n_locs_toofar = int(np.sum(bad_toofar))

        out.append(n)
    return out


def remove_close_npcs(
    npcs: Sequence[NPCRecord],
    min_distance_nm: float,
) -> List[NPCRecord]:
    """Port of remove_close_NPCs.m.

    Strategy:
    - sort by npc_n_locs descending
    - remove later NPCs that are closer than min_distance_nm to any earlier one
    - center is taken from center_fitcircle_step2 if available, else step1, else npc_center_nm
    """
    if len(npcs) <= 2:
        return list(npcs)

    # Sort by #locs desc
    order = np.argsort([-n.npc_n_locs for n in npcs])
    sorted_npcs = [npcs[i] for i in order]

    centers = []
    for n in sorted_npcs:
        if n.center_fitcircle_step2_nm is not None:
            centers.append(n.center_fitcircle_step2_nm)
        elif n.center_fitcircle_step1_nm is not None:
            centers.append(n.center_fitcircle_step1_nm)
        else:
            centers.append(n.npc_center_nm)
    centers = np.asarray(centers, dtype=float)

    keep = [True] * len(sorted_npcs)
    for i in range(len(sorted_npcs) - 1, 0, -1):
        if not keep[i]:
            continue
        c1 = centers[i]
        for j in range(i - 1, -1, -1):
            if not keep[j]:
                continue
            c2 = centers[j]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            if abs(dx) < min_distance_nm and abs(dy) < min_distance_nm:
                d = float(np.hypot(dx, dy))
                if d < min_distance_nm:
                    keep[i] = False
                    break

    out = [n for n, k in zip(sorted_npcs, keep) if k]
    return out


def update_loc_table_remove_used(
    df,
    npcs: Sequence[NPCRecord],
    mode: str = "by_index",
):
    """Remove localizations attributed to NPCs from the localization table.

    MATLAB's update_loc_table.m removes:
      - x_locs_good
      - x_locs_tooclose
    and similarly for y, using an ismember() approach.

    For robustness, Labelis defaults to index-based removal.

    Parameters
    ----------
    mode:
        - 'by_index' (recommended): remove rows whose index is in npc.idx_good or npc.idx_tooclose
        - 'matlab_xy_intersection': mimic MATLAB logic using x/y membership (less robust)
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return df

    if mode == "by_index":
        idx_remove = []
        for n in npcs:
            if n.idx_good.size:
                idx_remove.append(n.idx_good)
            if n.idx_tooclose.size:
                idx_remove.append(n.idx_tooclose)
        if len(idx_remove) == 0:
            return df
        idx_remove = np.unique(np.concatenate(idx_remove).astype(int))
        return df.drop(index=idx_remove, errors="ignore")

    elif mode == "matlab_xy_intersection":
        # Gather x,y values from good + tooclose (as floats)
        NPC_x = []
        NPC_y = []
        for n in npcs:
            if n.idx_good.size:
                # take values from df (ensures consistent rounding)
                sub = df.loc[df.index.isin(n.idx_good)]
                NPC_x.append(sub["x_nm_"].to_numpy(dtype=float))
                NPC_y.append(sub["y_nm_"].to_numpy(dtype=float))
            if n.idx_tooclose.size:
                sub = df.loc[df.index.isin(n.idx_tooclose)]
                NPC_x.append(sub["x_nm_"].to_numpy(dtype=float))
                NPC_y.append(sub["y_nm_"].to_numpy(dtype=float))
        if len(NPC_x) == 0:
            return df
        NPC_x = np.concatenate(NPC_x)
        NPC_y = np.concatenate(NPC_y)

        xt = df["x_nm_"].to_numpy(dtype=float)
        yt = df["y_nm_"].to_numpy(dtype=float)

        # Keep indices where x NOT in NPC_x AND y NOT in NPC_y (intersection of the two sets)
        keep_mask = (~np.isin(xt, NPC_x)) & (~np.isin(yt, NPC_y))
        return df.loc[keep_mask]

    else:
        raise ValueError(f"Unknown update mode: {mode}")
