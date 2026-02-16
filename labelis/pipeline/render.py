from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import math

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    njit = None  # type: ignore

from scipy.ndimage import gaussian_filter


def _ceil_div(x: float, d: float) -> int:
    # MATLAB ceil(x/d) with x>=0
    return int(math.ceil(x / d))


def render_bruteforce_legacy(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    sigma: np.ndarray,
    pixel_size_nm: float,
    grid_size_px: Optional[Tuple[int, int]] = None,
    signal_half_width_px: int = 10,
    sigma_is_variance: bool = True,
    compat_kernel_crop: bool = True,
    dtype=np.float32,
) -> np.ndarray:
    """Legacy brute-force renderer matching the MATLAB implementation.

    Important legacy conventions reproduced:
    - center pixel index = ceil(x/pixel_size)
    - kernel grid centered at (center_px*pixel_size) in nm
    - covariance diagonal uses `sigma` directly if sigma_is_variance=True (MATLAB mvnpdf behavior)
    - boundary truncation uses the *top-left* sub-kernel (MATLAB quirk)

    Parameters
    ----------
    grid_size_px : (ny, nx) or None
        If None, size is determined from the maximum ceil(x/pixel_size) and ceil(y/pixel_size),
        then made square by taking max(nx,ny).
    """
    x_nm = np.asarray(x_nm, dtype=np.float64)
    y_nm = np.asarray(y_nm, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 1:
        sigma = np.full_like(x_nm, float(sigma.ravel()[0]))

    if grid_size_px is None:
        max_x = int(np.max(np.ceil(x_nm / pixel_size_nm)))
        max_y = int(np.max(np.ceil(y_nm / pixel_size_nm)))
        n_pixels = int(max(max_x, max_y))
        ny = nx = n_pixels
    else:
        # MATLAB wrapper uses imSize_px(1) as max_x and imSize_px(2) as max_y but then makes square.
        max_x = int(grid_size_px[0])
        max_y = int(grid_size_px[1])
        n_pixels = int(max(max_x, max_y))
        ny = nx = n_pixels

    img = np.zeros((ny, nx), dtype=dtype)

    # Precompute kernel coordinate vectors in pixel offsets (0..L-1),
    # but note that in compat mode we always start from the "top-left" of the kernel.
    L = 2 * signal_half_width_px + 1

    # Evaluate per localization (slow but faithful)
    two_pi = 2.0 * math.pi
    for i in range(x_nm.size):
        x0 = float(x_nm[i])
        y0 = float(y_nm[i])
        sig_in = float(sigma[i])

        # MATLAB mvnpdf expects covariance, not std, when given a vector => treat as variance
        var = sig_in if sigma_is_variance else (sig_in * sig_in)
        if var <= 0:
            continue

        cx = int(math.ceil(x0 / pixel_size_nm))
        cy = int(math.ceil(y0 / pixel_size_nm))

        # build index ranges on the canvas (MATLAB 1-based, we convert to 0-based)
        x_start = max(1, cx - signal_half_width_px)
        x_end = min(cx + signal_half_width_px, nx)
        y_start = max(1, cy - signal_half_width_px)
        y_end = min(cy + signal_half_width_px, ny)

        lx = x_end - x_start + 1
        ly = y_end - y_start + 1

        # Compute the corresponding kernel patch.
        # In legacy compat mode, the sub-kernel always starts at the top-left corner (0,0),
        # irrespective of which part was clipped by the image boundary.
        # In "correct" mode, we offset into the kernel so the Gaussian remains centered.
        if compat_kernel_crop:
            kx0 = 0
            ky0 = 0
        else:
            # Offset into kernel based on how much we clipped on the left/top
            kx0 = (x_start - (cx - signal_half_width_px)) - 1  # convert to 0-based offset
            ky0 = (y_start - (cy - signal_half_width_px)) - 1

        # nm coordinate of the *kernel* top-left sample
        x_tl_nm = cx * pixel_size_nm - signal_half_width_px * pixel_size_nm
        y_tl_nm = cy * pixel_size_nm - signal_half_width_px * pixel_size_nm

        # Add values
        norm = 1.0 / (two_pi * var)
        inv2var = 1.0 / (2.0 * var)

        for jj in range(ly):
            y_nm_sample = y_tl_nm + (ky0 + jj) * pixel_size_nm
            dy = y_nm_sample - y0
            for ii in range(lx):
                x_nm_sample = x_tl_nm + (kx0 + ii) * pixel_size_nm
                dx = x_nm_sample - x0
                val = norm * math.exp(-(dx * dx + dy * dy) * inv2var)
                img[(y_start - 1) + jj, (x_start - 1) + ii] += val

    return img


if _HAVE_NUMBA:
    @njit(cache=True, fastmath=False)
    def _render_numba_legacy(
        x_nm: np.ndarray,
        y_nm: np.ndarray,
        sigma: np.ndarray,
        pixel_size_nm: float,
        ny: int,
        nx: int,
        signal_half_width_px: int,
        sigma_is_variance: bool,
        compat_kernel_crop: bool,
    ) -> np.ndarray:
        img = np.zeros((ny, nx), dtype=np.float32)
        two_pi = 2.0 * math.pi
        L = 2 * signal_half_width_px + 1

        for i in range(x_nm.size):
            x0 = float(x_nm[i])
            y0 = float(y_nm[i])
            sig_in = float(sigma[i])

            var = sig_in if sigma_is_variance else (sig_in * sig_in)
            if var <= 0:
                continue

            cx = int(math.ceil(x0 / pixel_size_nm))
            cy = int(math.ceil(y0 / pixel_size_nm))

            x_start = 1 if (cx - signal_half_width_px) < 1 else (cx - signal_half_width_px)
            x_end = nx if (cx + signal_half_width_px) > nx else (cx + signal_half_width_px)
            y_start = 1 if (cy - signal_half_width_px) < 1 else (cy - signal_half_width_px)
            y_end = ny if (cy + signal_half_width_px) > ny else (cy + signal_half_width_px)

            lx = x_end - x_start + 1
            ly = y_end - y_start + 1

            if compat_kernel_crop:
                kx0 = 0
                ky0 = 0
            else:
                kx0 = (x_start - (cx - signal_half_width_px)) - 1
                ky0 = (y_start - (cy - signal_half_width_px)) - 1

            x_tl_nm = cx * pixel_size_nm - signal_half_width_px * pixel_size_nm
            y_tl_nm = cy * pixel_size_nm - signal_half_width_px * pixel_size_nm

            norm = 1.0 / (two_pi * var)
            inv2var = 1.0 / (2.0 * var)

            for jj in range(ly):
                y_nm_sample = y_tl_nm + (ky0 + jj) * pixel_size_nm
                dy = y_nm_sample - y0
                for ii in range(lx):
                    x_nm_sample = x_tl_nm + (kx0 + ii) * pixel_size_nm
                    dx = x_nm_sample - x0
                    val = norm * math.exp(-(dx * dx + dy * dy) * inv2var)
                    img[(y_start - 1) + jj, (x_start - 1) + ii] += val

        return img


def render_numba_legacy(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    sigma: np.ndarray,
    pixel_size_nm: float,
    grid_size_px: Optional[Tuple[int, int]] = None,
    signal_half_width_px: int = 10,
    sigma_is_variance: bool = True,
    compat_kernel_crop: bool = True,
) -> np.ndarray:
    if not _HAVE_NUMBA:
        raise RuntimeError("Numba is not available. Install numba or use the reference_bruteforce engine.")
    x_nm = np.asarray(x_nm, dtype=np.float64)
    y_nm = np.asarray(y_nm, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.size == 1:
        sigma = np.full_like(x_nm, float(sigma.ravel()[0]))

    if grid_size_px is None:
        max_x = int(np.max(np.ceil(x_nm / pixel_size_nm)))
        max_y = int(np.max(np.ceil(y_nm / pixel_size_nm)))
        n_pixels = int(max(max_x, max_y))
        ny = nx = n_pixels
    else:
        max_x = int(grid_size_px[0])
        max_y = int(grid_size_px[1])
        n_pixels = int(max(max_x, max_y))
        ny = nx = n_pixels

    return _render_numba_legacy(
        x_nm=x_nm,
        y_nm=y_nm,
        sigma=sigma,
        pixel_size_nm=float(pixel_size_nm),
        ny=int(ny),
        nx=int(nx),
        signal_half_width_px=int(signal_half_width_px),
        sigma_is_variance=bool(sigma_is_variance),
        compat_kernel_crop=bool(compat_kernel_crop),
    )


def render_turbo_bin_blur(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    pixel_size_nm: float,
    grid_size_px: Optional[Tuple[int, int]] = None,
    blur_sigma_px: float = 1.2,
) -> np.ndarray:
    """Very fast renderer: bin localizations to pixels then blur.

    This is *not* strictly identical to brute-force Gaussian stamping.
    It is intended for interactive segmentation/preview or high-throughput processing.
    """
    x_nm = np.asarray(x_nm, dtype=np.float64)
    y_nm = np.asarray(y_nm, dtype=np.float64)

    if grid_size_px is None:
        max_x = int(np.max(np.ceil(x_nm / pixel_size_nm)))
        max_y = int(np.max(np.ceil(y_nm / pixel_size_nm)))
        n_pixels = int(max(max_x, max_y))
        ny = nx = n_pixels
    else:
        max_x = int(grid_size_px[0])
        max_y = int(grid_size_px[1])
        n_pixels = int(max(max_x, max_y))
        ny = nx = n_pixels

    # pixel index (0-based)
    ix = np.clip(np.ceil(x_nm / pixel_size_nm).astype(int) - 1, 0, nx - 1)
    iy = np.clip(np.ceil(y_nm / pixel_size_nm).astype(int) - 1, 0, ny - 1)

    img = np.zeros((ny, nx), dtype=np.float32)
    # accumulate counts
    for i in range(ix.size):
        img[iy[i], ix[i]] += 1.0

    if blur_sigma_px and blur_sigma_px > 0:
        img = gaussian_filter(img, sigma=float(blur_sigma_px))
    return img


def render_dispatch(
    engine: str,
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    sigma_nm: np.ndarray,
    pixel_size_nm: float,
    grid_size_px: Optional[Tuple[int, int]],
    signal_half_width_px: int,
    sigma_is_variance: bool,
    compat_kernel_crop: bool,
    turbo_blur_sigma_px: float,
) -> np.ndarray:
    if engine == "reference_bruteforce":
        return render_bruteforce_legacy(
            x_nm=x_nm,
            y_nm=y_nm,
            sigma=sigma_nm,
            pixel_size_nm=pixel_size_nm,
            grid_size_px=grid_size_px,
            signal_half_width_px=signal_half_width_px,
            sigma_is_variance=sigma_is_variance,
            compat_kernel_crop=compat_kernel_crop,
        )
    if engine == "numba":
        return render_numba_legacy(
            x_nm=x_nm,
            y_nm=y_nm,
            sigma=sigma_nm,
            pixel_size_nm=pixel_size_nm,
            grid_size_px=grid_size_px,
            signal_half_width_px=signal_half_width_px,
            sigma_is_variance=sigma_is_variance,
            compat_kernel_crop=compat_kernel_crop,
        )
    if engine == "turbo_bin_blur":
        return render_turbo_bin_blur(
            x_nm=x_nm,
            y_nm=y_nm,
            pixel_size_nm=pixel_size_nm,
            grid_size_px=grid_size_px,
            blur_sigma_px=turbo_blur_sigma_px,
        )
    raise ValueError(f"Unknown render engine: {engine}")


def have_numba() -> bool:
    return _HAVE_NUMBA
