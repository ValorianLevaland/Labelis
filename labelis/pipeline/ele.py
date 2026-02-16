from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import math

import numpy as np
from scipy.optimize import least_squares


def fun_fit_ele(p_label: float, xdata: Sequence[int]) -> np.ndarray:
    """Port of fun_fit_ELE.m.

    IMPORTANT: This function intentionally reproduces the legacy MATLAB behavior,
    where the number of modeled corners is inferred from len(xdata)-1.

    Parameters
    ----------
    p_label : float
        Probability that a single label site is present.
    xdata : Sequence[int]
        Used only for its length (legacy behavior).

    Returns
    -------
    probs : ndarray
        Modeled probabilities (length == len(xdata))
    """
    p_label = float(p_label)

    # probability that a corner is dark: 0 successes out of 4 trials
    p_dark = (1.0 - p_label) ** 4
    p_bright = 1.0 - p_dark

    n = len(list(xdata)) - 1  # legacy
    n = int(max(n, 1))

    probs = np.zeros((n + 1,), dtype=float)
    for k in range(n + 1):
        # binomial coefficient
        coeff = float(math.comb(n, k))
        probs[k] = coeff * (p_bright ** k) * ((1.0 - p_bright) ** (n - k))
    return probs


def fit_ele_bootstrap(
    ele_list: Sequence[int],
    n_samples: int = 20,
    bins_to_fit: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
    seed: int = 0,
) -> Tuple[float, float]:
    """Port of fit_ELE_bootstrap.m (including its historical quirks)."""
    rng = np.random.default_rng(seed)

    ele_arr = np.asarray(ele_list, dtype=int)
    if ele_arr.size == 0:
        return float("nan"), float("nan")

    edges = np.arange(-0.5, 8.5 + 1e-9, 1.0)  # -0.5..8.5 inclusive

    def ele_hist_prob(arr: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=edges)
        counts = counts.astype(float)
        if counts.sum() > 0:
            counts /= counts.sum()
        return counts  # length 9 (bins 0..8)

    bins_to_fit = np.asarray(list(bins_to_fit), dtype=int)
    # crop to match MATLAB: count_ELE(min+1 : max+1) with +1 offset
    # For bins 1..8 => python indices 1..8 inclusive
    lo = int(np.min(bins_to_fit))
    hi = int(np.max(bins_to_fit))

    def ydata_from_counts(counts: np.ndarray) -> np.ndarray:
        return counts[lo:hi + 1]  # bins lo..hi

    xdata = bins_to_fit  # legacy: used only for length in fun_fit_ele

    def fit_one(counts: np.ndarray) -> float:
        y = ydata_from_counts(counts)

        def residual(p):
            p0 = float(p[0])
            model = fun_fit_ele(p0, xdata)
            # model length must match y length
            if model.shape[0] != y.shape[0]:
                # truncate or pad to be safe
                if model.shape[0] > y.shape[0]:
                    model = model[: y.shape[0]]
                else:
                    model = np.pad(model, (0, y.shape[0] - model.shape[0]), constant_values=0.0)
            return model - y

        res = least_squares(residual, x0=np.array([0.5], dtype=float), bounds=(0.0, 1.0), method="trf")
        return float(res.x[0])

    # Fit on all data
    p_all = fit_one(ele_hist_prob(ele_arr))

    # Bootstrap
    p_boot = np.zeros((int(n_samples),), dtype=float)
    for i in range(int(n_samples)):
        sample = rng.choice(ele_arr, size=ele_arr.size, replace=True)
        p_boot[i] = fit_one(ele_hist_prob(sample))

    err = float(np.std(p_boot, ddof=0))
    err = float(np.round(err, 2))
    return p_all, err
