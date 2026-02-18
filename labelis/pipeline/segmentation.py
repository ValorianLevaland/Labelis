from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from scipy.ndimage import convolve
from skimage import exposure, feature, transform, filters

from skimage.measure import regionprops

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAVE_CV2 = False


def disk_kernel(radius_px: float) -> np.ndarray:
    """Approximate MATLAB fspecial('disk', r) kernel.

    MATLAB's disk filter is a circular averaging filter with anti-aliasing.
    Here we create a binary disk and normalize it, which is usually close enough
    for segmentation.
    """
    r = float(radius_px)
    if r <= 0:
        return np.array([[1.0]], dtype=np.float32)
    rad = int(np.ceil(r))
    y, x = np.ogrid[-rad:rad+1, -rad:rad+1]
    mask = x*x + y*y <= r*r
    k = np.zeros((2*rad+1, 2*rad+1), dtype=np.float32)
    k[mask] = 1.0
    s = float(k.sum())
    if s > 0:
        k /= s
    return k


def imadjust_like(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Rough equivalent of MATLAB imadjust(I) for grayscale arrays."""
    img = np.asarray(img, dtype=np.float32)
    lo = np.percentile(img, low_pct)
    hi = np.percentile(img, high_pct)
    if hi <= lo:
        return np.clip(img, 0, None)
    return exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0.0, 1.0)).astype(np.float32)


def prepare_image_for_segmentation(
    image: np.ndarray,
    thres_clip: float,
    expected_radius_nm: float,
    pixel_size_nm: float,
) -> np.ndarray:
    # thresholding (clip high intensities)
    img = np.asarray(image, dtype=np.float32).copy()
    if np.isfinite(thres_clip):
        img[img > thres_clip] = thres_clip

    # disk filter (like fspecial('disk', expected_radius/pixel_size))
    r_px = float(expected_radius_nm) / float(pixel_size_nm)
    k = disk_kernel(r_px)
    img_f = convolve(img, k, mode="nearest")

    # contrast adjustment
    img_adj = imadjust_like(img_f, 1.0, 99.0)
    return img_adj


def remove_duplicate_circles(
    centers_px: np.ndarray,
    radii_px: np.ndarray,
    metric: np.ndarray,
    min_distance_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of remove_duplicate_circles.m.

    The MATLAB code assumes circles are ordered by descending metric and removes later ones.
    We explicitly sort by metric descending first.
    """
    if centers_px.size == 0:
        return centers_px, radii_px, metric
    centers = np.asarray(centers_px, dtype=np.float32)
    radii = np.asarray(radii_px, dtype=np.float32)
    metric = np.asarray(metric, dtype=np.float32)

    # sort by metric descending
    order = np.argsort(-metric)
    centers = centers[order]
    radii = radii[order]
    metric = metric[order]

    keep = [True] * centers.shape[0]
    for i in range(centers.shape[0]-1, 0, -1):
        if not keep[i]:
            continue
        c1 = centers[i]
        for j in range(i-1, -1, -1):
            if not keep[j]:
                continue
            c2 = centers[j]
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]
            if abs(dx) < min_distance_px and abs(dy) < min_distance_px:
                d = float(np.hypot(dx, dy))
                if d < min_distance_px:
                    keep[i] = False
                    break
    keep_idx = np.array([i for i, k in enumerate(keep) if k], dtype=int)
    return centers[keep_idx], radii[keep_idx], metric[keep_idx]


def locate_npcs_hough_cv2(
    image: np.ndarray,
    sensitivity: float,
    pixel_size_nm: float,
    min_distance_nm: float,
    radius_range_nm: Tuple[float, float] = (40.0, 60.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circle detection using OpenCV HoughCircles.

    This is not a perfect clone of MATLAB imfindcircles, but is fast and
    typically yields similar candidates on ring-like objects after smoothing.
    """
    if not _HAVE_CV2:
        raise RuntimeError("OpenCV is not available. Install opencv-python or use hough_skimage/blob_log.")

    img = np.asarray(image, dtype=np.float32)
    img_norm = img - img.min()
    if img_norm.max() > 0:
        img_norm /= img_norm.max()

    img8 = np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)

    # Some smoothing helps Hough
    img8_blur = cv2.GaussianBlur(img8, (0, 0), sigmaX=1.0)

    minDist = max(1.0, float(min_distance_nm) / float(pixel_size_nm))
    minR = int(np.floor(radius_range_nm[0] / pixel_size_nm))
    maxR = int(np.ceil(radius_range_nm[1] / pixel_size_nm))
    minR = max(1, minR)
    maxR = max(minR + 1, maxR)

    # Map "sensitivity" ~ [0.7..0.95] to accumulator threshold.
    # Higher sensitivity -> lower threshold -> more detections.
    sens = float(np.clip(sensitivity, 0.01, 0.99))
    param2 = int(max(5, round(30 * (1.0 - sens) + 5)))  # heuristic

    circles = cv2.HoughCircles(
        img8_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=minDist,
        param1=100,
        param2=param2,
        minRadius=minR,
        maxRadius=maxR,
    )

    if circles is None or len(circles) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    circles = np.squeeze(circles, axis=0)  # (N,3) => x,y,r
    centers = circles[:, :2].astype(np.float32)
    radii = circles[:, 2].astype(np.float32)

    # OpenCV doesn't provide a metric. Use image intensity at center as a proxy.
    cx = np.clip(np.round(centers[:, 0]).astype(int), 0, img.shape[1] - 1)
    cy = np.clip(np.round(centers[:, 1]).astype(int), 0, img.shape[0] - 1)
    metric = img[cy, cx].astype(np.float32)

    centers, radii, metric = remove_duplicate_circles(centers, radii, metric, min_distance_px=minDist)
    return centers, radii, metric


def locate_npcs_hough_skimage(
    image: np.ndarray,
    sensitivity: float,
    pixel_size_nm: float,
    min_distance_nm: float,
    radius_range_nm: Tuple[float, float] = (40.0, 60.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circle detection using scikit-image Hough transform."""
    img = np.asarray(image, dtype=np.float32)
    img_norm = img - img.min()
    if img_norm.max() > 0:
        img_norm /= img_norm.max()

    edges = feature.canny(img_norm, sigma=1.0)

    minR = int(np.floor(radius_range_nm[0] / pixel_size_nm))
    maxR = int(np.ceil(radius_range_nm[1] / pixel_size_nm))
    minR = max(1, minR)
    maxR = max(minR + 1, maxR)
    radii = np.arange(minR, maxR + 1, dtype=int)

    hspaces = transform.hough_circle(edges, radii)
    # heuristic threshold based on sensitivity
    # higher sens => accept lower peaks
    accums = hspaces.max(axis=(1, 2))
    base_thr = np.max(accums) * (1.0 - 0.7 * np.clip(sensitivity, 0.0, 1.0))
    base_thr = float(max(base_thr, 0.0))

    centers_y = []
    centers_x = []
    radii_out = []
    metric = []
    minDist = max(1.0, float(min_distance_nm) / float(pixel_size_nm))

    for r_idx, r in enumerate(radii):
        h = hspaces[r_idx]
        peaks = feature.peak_local_max(h, num_peaks=2000, threshold_abs=base_thr, min_distance=int(np.ceil(minDist)))
        for (py, px) in peaks:
            centers_y.append(py)
            centers_x.append(px)
            radii_out.append(r)
            metric.append(h[py, px])

    if len(centers_x) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    centers = np.stack([np.array(centers_x), np.array(centers_y)], axis=1).astype(np.float32)
    radii_out = np.array(radii_out, dtype=np.float32)
    metric = np.array(metric, dtype=np.float32)

    centers, radii_out, metric = remove_duplicate_circles(centers, radii_out, metric, min_distance_px=minDist)
    return centers, radii_out, metric


def locate_npcs_blob_log(
    image: np.ndarray,
    expected_radius_nm: float,
    pixel_size_nm: float,
    min_distance_nm: float,
    threshold_rel: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blob-based candidate detection (LoG peaks).

    Returns pseudo radii (constant) and a metric (LoG response).
    """
    img = np.asarray(image, dtype=np.float32)
    img_norm = img - img.min()
    if img_norm.max() > 0:
        img_norm /= img_norm.max()

    # LoG sigma in pixels is ~ radius / sqrt(2) for blobs; for rings it's approximate.
    sigma_px = max(1.0, (expected_radius_nm / pixel_size_nm) / np.sqrt(2.0))
    blobs = feature.blob_log(
        img_norm,
        min_sigma=max(0.5, sigma_px * 0.7),
        max_sigma=sigma_px * 1.3,
        num_sigma=10,
        threshold=threshold_rel * float(np.max(img_norm)),
        overlap=0.5,
    )
    if blobs.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # blob_log returns (y, x, sigma)
    centers = blobs[:, [1, 0]].astype(np.float32)
    radii = (expected_radius_nm / pixel_size_nm) * np.ones((centers.shape[0],), dtype=np.float32)
    metric = blobs[:, 2].astype(np.float32)

    # Deduplicate with same algorithm
    minDist = max(1.0, float(min_distance_nm) / float(pixel_size_nm))
    centers, radii, metric = remove_duplicate_circles(centers, radii, metric, min_distance_px=minDist)
    return centers, radii, metric

from typing import Optional
from .config import PipelineConfig

def locate_npcs_dispatch(
    engine: str,
    image_to_segment: np.ndarray,
    sensitivity: float,
    expected_radius_nm: float,
    pixel_size_nm: float,
    min_distance_nm: float,
    circle_radius_range_nm: Tuple[float, float] = (40.0, 60.0),
    cfg : Optional["PipelineConfig"] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if engine == "hough_cv2":
        return locate_npcs_hough_cv2(
            image=image_to_segment,
            sensitivity=sensitivity,
            pixel_size_nm=pixel_size_nm,
            min_distance_nm=min_distance_nm,
            radius_range_nm=circle_radius_range_nm,
        )
    if engine == "hough_skimage":
        return locate_npcs_hough_skimage(
            image=image_to_segment,
            sensitivity=sensitivity,
            pixel_size_nm=pixel_size_nm,
            min_distance_nm=min_distance_nm,
            radius_range_nm=circle_radius_range_nm,
        )
    if engine == "blob_log":
        return locate_npcs_blob_log(
            image=image_to_segment,
            expected_radius_nm=expected_radius_nm,
            pixel_size_nm=pixel_size_nm,
            min_distance_nm=min_distance_nm,
        )

    if engine == "cpsam":
        if cfg is None:
            raise ValueError("cfg must be provided when engine='cpsam'")
        return locate_npcs_cpsam(
            image=image_to_segment,
            pixel_size_nm=pixel_size_nm,
            min_distance_nm=min_distance_nm,
            radius_range_nm=circle_radius_range_nm,
            cfg=cfg,
        )

    raise ValueError(f"Unknown segmentation engine: {engine}")

def locate_npcs_cpsam(
    image,
    pixel_size_nm,
    min_distance_nm,
    radius_range_nm,
    cfg,
):
    """
    Use Cellpose-SAM to segment image and convert masks to circle candidates.
    """

    from napari_cellpose_sam.segmentation import segment_single_slice

    img = np.asarray(image, dtype=np.float32)

    masks, flows = segment_single_slice(
        img,
        model_type=cfg.cpsam_model,
        model_path=cfg.cpsam_custom_model_path or None,
        flow_threshold=cfg.cpsam_flow_threshold,
        cellprob_threshold=cfg.cpsam_cellprob_threshold,
        #gpu=cfg.cpsam_use_gpu,
    )

    if masks is None or np.max(masks) == 0:
        return (
            np.zeros((0, 2), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.float32),
        )

    centers = []
    radii = []
    metrics = []

    min_r_nm, max_r_nm = radius_range_nm

    for rp in regionprops(masks):

        if rp.area < cfg.cpsam_min_area_px:
            continue

        cy, cx = rp.centroid

        # robust outer-scale radius estimation
        coords = rp.coords
        d = np.sqrt((coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2)
        r_px = np.percentile(d, 95)

        r_nm = r_px * pixel_size_nm
        if r_nm < min_r_nm or r_nm > max_r_nm:
            continue

        # optional circularity filter
        if cfg.cpsam_min_circularity > 0:
            per = rp.perimeter if rp.perimeter else 0
            if per > 0:
                circ = 4 * np.pi * rp.area / (per * per)
                if circ < cfg.cpsam_min_circularity:
                    continue

        # metric
        vals = img[rp.coords[:, 0], rp.coords[:, 1]]
        metric = float(np.mean(vals)) if vals.size else 0.0

        centers.append((cx, cy))
        radii.append(r_px)
        metrics.append(metric)

    if not centers:
        return (
            np.zeros((0, 2), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.float32),
        )

    centers = np.asarray(centers, np.float32)
    radii = np.asarray(radii, np.float32)
    metrics = np.asarray(metrics, np.float32)

    minDist_px = max(1.0, min_distance_nm / pixel_size_nm)

    centers, radii, metrics = remove_duplicate_circles(
        centers,
        radii,
        metrics,
        min_distance_px=minDist_px,
    )

    return centers, radii, metrics

