from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional, Tuple
import json
import pathlib
import math

ComputeEngine = Literal["reference_bruteforce", "numba", "turbo_bin_blur"]
SegmentationEngine = Literal["hough_cv2", "hough_skimage", "blob_log"]
AlignmentMethod = Literal["template", "smap", "none"]


@dataclass
class PipelineConfig:
    # -------------------------
    # I/O
    # -------------------------
    input_path: str = ""
    output_dir: str = ""
    fig_dir: Optional[str] = None

    # -------------------------
    # Pre-filtering (ThunderSTORM / SMAP tables)
    # -------------------------
    min_uncertainty_nm: float = 0.0
    max_uncertainty_nm: float = 15.0
    min_sigma_nm: float = 75.0
    max_sigma_nm: float = 155.0
    min_frame: int = 0
    max_frame: int = 2_147_483_647  # acts like inf

    min_x_nm: float = 0.0
    max_x_nm: float = float("inf")
    min_y_nm: float = 0.0
    max_y_nm: float = float("inf")

    # ROI selection (GUI provides polygon/rectangle; pipeline receives polygon in nm)
    use_roi: bool = False
    roi_polygon_nm: Optional[list[tuple[float, float]]] = None  # list of (x_nm, y_nm)

    # -------------------------
    # Rendering
    # -------------------------
    render_px_size_nm: float = 10.0
    signal_half_width_px: int = 10  # +/-10 px kernel (legacy)
    sigma_is_variance: bool = True  # matches MATLAB mvnpdf usage (diag covariance = sigma)
    compat_kernel_crop: bool = True  # replicate MATLAB boundary cropping quirk

    compute_engine: ComputeEngine = "reference_bruteforce"

    # "Turbo" mode parameters
    turbo_blur_sigma_px: float = 1.2

    # -------------------------
    # NPC detection / segmentation
    # -------------------------
    expected_npc_radius_nm: float = 55.0
    sensitivity: float = 0.95
    thres_clip: float = 0.03  # clip rendered image before segmentation
    min_dist_between_npcs_nm: float = -1.0  # -1 => auto

    segmentation_engine: SegmentationEngine = "hough_cv2"

    # circle detection range for imfindcircles-equivalent
    circle_min_radius_nm: float = 40.0
    circle_max_radius_nm: float = 60.0

    # -------------------------
    # Extraction
    # -------------------------
    extract_box_radius_nm: float = 120.0
    extract_circle_radius_nm: float = 100.0  # initial; may be auto-tuned after first fit

    # -------------------------
    # Filtering
    # -------------------------
    min_n_locs: int = 20
    max_mean_sigma_nm: float = 150.0
    max_mean_uncertainty_nm: float = 20.0

    # radius filtering (auto if -1)
    min_radius_nm: float = -1.0
    max_radius_nm: float = -1.0
    min_max_radius_tolerance: float = 2.0

    # loc distribution filtering
    max_locs_too_close: float = 0.3
    max_locs_too_far: float = 0.7
    inner_radius_nm: float = -1.0
    outer_radius_nm: float = -1.0
    inner_outer_radius_tolerance: float = 3.0

    # -------------------------
    # Corner counting / ELE
    # -------------------------
    alignment_method: AlignmentMethod = "smap"
    min_locs_per_corner: int = 2

    # ELE fit / bootstrap
    ele_bootstrap_samples: int = 20
    ele_bins_to_fit: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

    # -------------------------
    # Iterative segmentation
    # -------------------------
    max_cycles: int = 1

    # -------------------------
    # Diagnostics
    # -------------------------
    save_intermediate_images: bool = True
    show_intermediate: bool = False  # GUI can visualize; pipeline should not block
    # Guided pipeline checkpoints (GUI only): the pipeline can pause and ask the
    # GUI to validate intermediate outputs and optionally adjust parameters.
    step_by_step: bool = False
    # Individual checkpoint toggles (only used when step_by_step=True)
    checkpoint_segmentation: bool = True
    checkpoint_thresholds: bool = True
    checkpoint_centering: bool = True
    checkpoint_cycle_qc: bool = True
    checkpoint_model_qc: bool = True

    # Per-NPC QC inspector (GUI) display parameters
    qc_template_radius_nm: float = 55.0
    qc_corner_tolerance_nm: float = 15.0
    qc_max_npcs_per_dialog: int = 0  # 0 => show all
    random_seed: int = 0

    def to_json(self, path: str | pathlib.Path) -> None:
        """Write config to JSON (scientific-friendly)."""
        from .utils import sanitize_for_json
        path = pathlib.Path(path)
        path.write_text(json.dumps(sanitize_for_json(asdict(self)), indent=2))

    @staticmethod
    def from_json(path: str | pathlib.Path) -> "PipelineConfig":
        d = json.loads(pathlib.Path(path).read_text())

        # Convert JSON-friendly nulls back to scientific defaults where appropriate
        if d.get("max_x_nm", None) is None:
            d["max_x_nm"] = float("inf")
        if d.get("max_y_nm", None) is None:
            d["max_y_nm"] = float("inf")
        if d.get("max_frame", None) is None:
            d["max_frame"] = 2_147_483_647

        return PipelineConfig(**d)


def preset_reference() -> PipelineConfig:
    """Closest behavior to the legacy MATLAB v7 pipeline."""
    return PipelineConfig(
        compute_engine="reference_bruteforce",
        segmentation_engine="hough_cv2",
        alignment_method="smap",
        max_cycles=1,
        render_px_size_nm=10.0,
        thres_clip=0.03,
        sensitivity=0.95,
        expected_npc_radius_nm=55.0,
        extract_box_radius_nm=120.0,
        extract_circle_radius_nm=100.0,
    )


def preset_balanced() -> PipelineConfig:
    """Recommended for routine use: robust + faster."""
    return PipelineConfig(
        compute_engine="numba",
        segmentation_engine="blob_log",
        alignment_method="smap",
        max_cycles=1,
        render_px_size_nm=10.0,
        thres_clip=0.03,
        sensitivity=0.90,
        expected_npc_radius_nm=55.0,
        extract_box_radius_nm=120.0,
        extract_circle_radius_nm=100.0,
    )


def preset_turbo() -> PipelineConfig:
    """High-throughput mode: very fast, potentially approximate."""
    return PipelineConfig(
        compute_engine="turbo_bin_blur",
        segmentation_engine="blob_log",
        alignment_method="smap",
        max_cycles=1,
        render_px_size_nm=10.0,
        thres_clip=0.03,
        sensitivity=0.85,
        expected_npc_radius_nm=55.0,
        extract_box_radius_nm=120.0,
        extract_circle_radius_nm=100.0,
    )
