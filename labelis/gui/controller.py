from __future__ import annotations

import traceback
import pathlib
from typing import Optional, Sequence, Tuple

import numpy as np

from PyQt5 import QtCore, QtWidgets

import napari

from .dock_widget import LabelisDockWidget
from .checkpoint_dialogs import (
    CenteringCheckpointDialog,
    SegmentationCheckpointDialog,
    ThresholdCheckpointDialog,
)
from .results_dialog import ResultsDialog
from .npc_inspector_dialog import CycleNPCQCDialog, FinalNPCQCDialog

from ..pipeline.config import PipelineConfig, preset_balanced
from ..pipeline.render import render_dispatch
from ..pipeline.io import load_localizations
from ..pipeline.pipeline import PipelineAborted, run_pipeline, PipelineResult


class _PipelineWorker(QtCore.QObject):
    log_line = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)  # PipelineResult
    finished_error = QtCore.pyqtSignal(str)

    checkpoint_requested = QtCore.pyqtSignal(str, object)  # (stage, payload)

    def __init__(self, cfg: PipelineConfig, df0=None):
        super().__init__()
        self._cfg = cfg
        self._df0 = df0

        # Checkpoint synchronization
        self._chk_mutex = QtCore.QMutex()
        self._chk_wait = QtCore.QWaitCondition()
        self._chk_response: Optional[object] = None

    def _checkpoint_cb(self, stage: str, payload: object) -> dict:
        """Blocking callback used by the pipeline to request GUI checkpoints."""
        self._chk_mutex.lock()
        try:
            self._chk_response = None
            self.checkpoint_requested.emit(str(stage), payload)
            while self._chk_response is None:
                self._chk_wait.wait(self._chk_mutex)
            resp = self._chk_response
        finally:
            self._chk_mutex.unlock()

        if isinstance(resp, dict):
            return resp
        return {"action": "continue"}

    @QtCore.pyqtSlot(object)
    def set_checkpoint_response(self, resp: object) -> None:
        self._chk_mutex.lock()
        try:
            self._chk_response = resp
            self._chk_wait.wakeAll()
        finally:
            self._chk_mutex.unlock()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            res = run_pipeline(
                self._cfg,
                log_cb=lambda s: self.log_line.emit(s),
                df0=self._df0,
                checkpoint_cb=self._checkpoint_cb,
            )
            self.finished_ok.emit(res)
        except PipelineAborted:
            # user cancelled via checkpoint UI
            self.finished_error.emit("Aborted by user.")
        except Exception:
            self.finished_error.emit(traceback.format_exc())


class LabelisController(QtCore.QObject):
    """
    Owns the Napari viewer + the dock widget + the analysis worker thread.

    Guarantees:
    - Analysis cannot start before ROI is confirmed.
    - If ROI is modified after confirmation, analysis locks again until reconfirmed.
    - Worker thread is kept alive correctly (no 'QThread destroyed while running').
    """

    def __init__(self, logger):
        super().__init__()
        self._logger = logger

        # App state
        self._df0 = None           # original loaded df (pandas)
        self._df_shifted = None    # shifted df (pandas)
        self._x_offset_nm = 0.0
        self._y_offset_nm = 0.0

        self._roi_polygon_nm: Optional[list[tuple[float, float]]] = None
        self._roi_confirmed: bool = False

        self._render_px_size_nm: float = 10.0
        self._compute_engine: str = "turbo_bin_blur"

        self._result: Optional[PipelineResult] = None

        # Worker thread
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_PipelineWorker] = None

        # Step-by-step checkpoint UI state
        self._checkpoint_dialog: Optional[QtWidgets.QDialog] = None
        self._checkpoint_stage: Optional[str] = None

        # Napari viewer
        self.viewer = napari.Viewer(title="Labelis")
        self.dock = LabelisDockWidget()
        self.viewer.window.add_dock_widget(self.dock, area="right", name="Labelis")

        # Connect UI signals
        self.dock.sig_load_render.connect(self.load_and_render)
        self.dock.sig_confirm_roi.connect(self.confirm_roi)
        self.dock.sig_run_analysis.connect(self.run_analysis)

        # Install close-event filter to prevent closing while computing
        self._qt_window = getattr(self.viewer.window, "_qt_window", None)
        if self._qt_window is not None:
            self._qt_window.installEventFilter(self)

        self._roi_layer = None
        self._connect_roi_events_if_present()

        self.log("[Labelis] Ready. Load+Render -> draw ROI -> Confirm ROI -> Run.")

    # -------------------------
    # Window management
    # -------------------------
    def show(self) -> None:
        if self._qt_window is not None:
            self._qt_window.show()

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if obj is self._qt_window and event.type() == QtCore.QEvent.Close:
            if self._thread is not None and self._thread.isRunning():
                QtWidgets.QMessageBox.warning(
                    self._qt_window,
                    "Labelis",
                    "Analysis is still running.\n\nPlease wait for it to finish.",
                )
                event.ignore()
                return True
        return super().eventFilter(obj, event)

    # -------------------------
    # Logging
    # -------------------------
    @QtCore.pyqtSlot(str)
    def log(self, msg: str) -> None:
        self._logger.info(msg)
        self.dock.append_log(msg)

    def _error(self, title: str, msg: str) -> None:
        self._logger.error(f"{title}: {msg}")
        QtWidgets.QMessageBox.critical(self._qt_window, title, msg)

    # -------------------------
    # ROI layer helpers
    # -------------------------
    def _ensure_roi_layer(self) -> None:
        if "ROI" in self.viewer.layers:
            self._roi_layer = self.viewer.layers["ROI"]
        else:
            self._roi_layer = self.viewer.add_shapes(
                name="ROI",
                shape_type="polygon",
            )
        # Make sure ROI aligns with image scale
        try:
            self._roi_layer.scale = (self._render_px_size_nm, self._render_px_size_nm)
        except Exception:
            pass

        # Put in polygon drawing mode for convenience
        try:
            self._roi_layer.mode = "add_polygon"
        except Exception:
            pass

        self._connect_roi_events_if_present()

    def _connect_roi_events_if_present(self) -> None:
        # If ROI exists, connect to changes so we can invalidate confirmation.
        try:
            if "ROI" in self.viewer.layers:
                layer = self.viewer.layers["ROI"]
                self._roi_layer = layer
                # Prevent duplicate connections by disconnecting first (best-effort)
                try:
                    layer.events.data.disconnect(self._on_roi_changed)
                except Exception:
                    pass
                layer.events.data.connect(self._on_roi_changed)
        except Exception:
            pass

    def _on_roi_changed(self, event=None) -> None:
        if self._roi_confirmed:
            self._roi_confirmed = False
            self._roi_polygon_nm = None
            self.dock.set_analysis_enabled(False)
            self.dock.set_status("ROI changed → please Confirm ROI again.")
            self.log("[Labelis] ROI changed. Analysis locked until ROI is confirmed again.")

    # -------------------------
    # Step 1: Load + Render
    # -------------------------
    @QtCore.pyqtSlot()
    def load_and_render(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._error("Labelis", "Analysis is running. Wait before reloading.")
            return

        in_path = self.dock.input_path()
        out_dir = self.dock.output_dir()
        if not in_path:
            self._error("Labelis", "Please select a localization CSV/TSV file.")
            return
        if not pathlib.Path(in_path).exists():
            self._error("Labelis", f"Input file does not exist:\n{in_path}")
            return
        if not out_dir:
            self._error("Labelis", "Please select an output directory.")
            return

        self._render_px_size_nm = float(self.dock.render_px_size_nm())
        self._compute_engine = str(self.dock.compute_engine())

        self.dock.set_status("Loading localizations…")
        self.log(f"[Labelis] Loading: {in_path}")

        try:
            df0 = load_localizations(in_path)
        except Exception as e:
            self._error("Labelis - Load failed", str(e))
            return

        # Shift to origin (MATLAB-like, consistent with pipeline)
        self._x_offset_nm = float(df0["x_nm_"].min())
        self._y_offset_nm = float(df0["y_nm_"].min())

        df = df0.copy()
        df["x_nm_"] = df["x_nm_"] - self._x_offset_nm
        df["y_nm_"] = df["y_nm_"] - self._y_offset_nm

        self._df0 = df0
        self._df_shifted = df

        self.log(f"[Labelis] XY offsets: ({self._x_offset_nm:.3f} nm, {self._y_offset_nm:.3f} nm)")
        self.log(f"[Labelis] Rendering (engine={self._compute_engine}, px={self._render_px_size_nm:.2f} nm)…")

        try:
            img = render_dispatch(
                engine=self._compute_engine,
                x_nm=df["x_nm_"].to_numpy(),
                y_nm=df["y_nm_"].to_numpy(),
                sigma_nm=df["sigma_nm_"].to_numpy(),
                pixel_size_nm=self._render_px_size_nm,
                grid_size_px=None,
                signal_half_width_px=10,
                sigma_is_variance=True,
                compat_kernel_crop=True,
                turbo_blur_sigma_px=1.2,
            )
        except Exception as e:
            self._error("Labelis - Render failed", str(e))
            return

        # Replace image layers
        for name in ["Rendered", "Segmentation input", "Candidates", "NPC OK", "NPC rejected", "Locs (downsampled)"]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])

        self.viewer.add_image(
            img,
            name="Rendered",
            scale=(self._render_px_size_nm, self._render_px_size_nm),
        )

        # Optional: show a downsampled point cloud for context (kept light to avoid freezes)
        try:
            pts = df[["y_nm_", "x_nm_"]].to_numpy(dtype=float) / float(self._render_px_size_nm)  # to pixel coords
            if pts.shape[0] > 200_000:
                idx = np.random.default_rng(0).choice(pts.shape[0], size=200_000, replace=False)
                pts = pts[idx]
            # Add as points in pixel coords but scaled to nm to align with image
            self.viewer.add_points(
                pts,
                name="Locs (downsampled)",
                size=1.0,
                scale=(self._render_px_size_nm, self._render_px_size_nm),
            )
        except Exception:
            # Never block main workflow on point overlay
            pass

        # ROI: always reset after a re-render
        if "ROI" in self.viewer.layers:
            try:
                self.viewer.layers.remove(self.viewer.layers["ROI"])
            except Exception:
                pass

        self._roi_confirmed = False
        self._roi_polygon_nm = None
        self.dock.set_analysis_enabled(False)

        self._ensure_roi_layer()

        self.dock.set_status("Rendered. Draw ROI in 'ROI' layer, then Confirm ROI.")
        self.log("[Labelis] Rendered image displayed. Draw ROI in 'ROI' layer, then confirm.")

    # -------------------------
    # Step 2: Confirm ROI
    # -------------------------
    @QtCore.pyqtSlot()
    def confirm_roi(self) -> None:
        if self._df_shifted is None:
            self._error("Labelis", "Load + Render first.")
            return

        self._ensure_roi_layer()
        roi = self._roi_layer
        if roi is None:
            self._error("Labelis", "ROI layer not available.")
            return

        if not getattr(roi, "data", None) or len(roi.data) == 0:
            self._error("Labelis", "ROI is empty. Draw a polygon in the 'ROI' layer first.")
            return

        # Use the first shape as ROI
        verts_yx = np.asarray(roi.data[0], dtype=float)  # (N,2) y,x in *data* coords
        if verts_yx.ndim != 2 or verts_yx.shape[0] < 3 or verts_yx.shape[1] != 2:
            self._error("Labelis", "ROI polygon is invalid.")
            return

        # Convert to nm (x,y) for pipeline
        px = float(self._render_px_size_nm)
        poly_nm = [(float(v[1]) * px, float(v[0]) * px) for v in verts_yx]

        self._roi_polygon_nm = poly_nm
        self._roi_confirmed = True

        # Convert to nm (x,y) for pipeline
        px = float(self._render_px_size_nm)
        poly_nm = [(float(v[1]) * px, float(v[0]) * px) for v in verts_yx]
        # Validate polygon (simple and non‑degenerate)
        try:
            import shapely.geometry as _shgeom  # optional dependency
            if not _shgeom.Polygon(poly_nm).is_valid or _shgeom.Polygon(poly_nm).area <= 0:
                raise ValueError
        except Exception:
            self._error("Labelis", "ROI polygon is invalid or self‑intersecting.")
            return

        self._roi_polygon_nm = poly_nm
        self._roi_confirmed = True
        

        self.dock.set_analysis_enabled(True)
        self.dock.set_status(f"ROI confirmed ({len(poly_nm)} vertices). Now set analysis inputs and Run.")

        self.log(f"[Labelis] ROI confirmed with {len(poly_nm)} vertices.")

    # -------------------------
    # Step 3: Run analysis
    # -------------------------
    @QtCore.pyqtSlot()
    def run_analysis(self) -> None:
        if self._df0 is None:
            self._error("Labelis", "Load + Render first.")
            return
        if not self._roi_confirmed or not self._roi_polygon_nm:
            self._error("Labelis", "ROI is not confirmed. Confirm ROI first.")
            return
        if self._thread is not None and self._thread.isRunning():
            self._error("Labelis", "Analysis is already running.")
            return

        in_path = self.dock.input_path()
        out_dir = self.dock.output_dir()
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        a = self.dock.analysis_settings()

        # Build config
        cfg = preset_balanced()
        cfg.input_path = in_path
        cfg.output_dir = out_dir
        cfg.fig_dir = out_dir

        cfg.render_px_size_nm = float(self._render_px_size_nm)
        cfg.compute_engine = str(self._compute_engine)

        cfg.use_roi = True
        cfg.roi_polygon_nm = list(self._roi_polygon_nm)

        # Pre-filtering
        cfg.min_uncertainty_nm = float(a.min_uncertainty_nm)
        cfg.max_uncertainty_nm = float(a.max_uncertainty_nm)
        cfg.min_sigma_nm = float(a.min_sigma_nm)
        cfg.max_sigma_nm = float(a.max_sigma_nm)
        cfg.min_frame = int(a.min_frame)
        cfg.max_frame = int(a.max_frame)

        # Segmentation
        cfg.segmentation_engine = str(a.segmentation_engine)
        cfg.sensitivity = float(a.sensitivity)
        cfg.expected_npc_radius_nm = float(a.expected_npc_radius_nm)
        cfg.circle_min_radius_nm = float(a.circle_min_radius_nm)
        cfg.circle_max_radius_nm = float(a.circle_max_radius_nm)
        cfg.min_dist_between_npcs_nm = float(a.min_dist_between_npcs_nm)
        cfg.thres_clip = float(getattr(a, "thres_clip", cfg.thres_clip))


        # Extraction / filtering
        cfg.extract_box_radius_nm = float(a.extract_box_radius_nm)
        cfg.extract_circle_radius_nm = float(a.extract_circle_radius_nm)
        cfg.min_n_locs = int(a.min_n_locs)
        cfg.max_mean_sigma_nm = float(a.max_mean_sigma_nm)
        cfg.max_mean_uncertainty_nm = float(a.max_mean_uncertainty_nm)

        # Alignment / ELE
        cfg.alignment_method = str(a.alignment_method)
        cfg.min_locs_per_corner = int(a.min_locs_per_corner)
        cfg.ele_bootstrap_samples = int(a.ele_bootstrap_samples)

        # Checkpoints
        cfg.step_by_step = bool(getattr(a, "step_by_step", False))
        # Individual checkpoint toggles (effective only when step_by_step=True)
        cfg.checkpoint_segmentation = bool(getattr(a, "checkpoint_segmentation", True))
        cfg.checkpoint_thresholds = bool(getattr(a, "checkpoint_thresholds", True))
        cfg.checkpoint_centering = bool(getattr(a, "checkpoint_centering", True))
        cfg.checkpoint_cycle_qc = bool(getattr(a, "checkpoint_cycle_qc", True))
        cfg.checkpoint_model_qc = bool(getattr(a, "checkpoint_model_qc", True))

        # Per-NPC inspector display parameters
        cfg.qc_template_radius_nm = float(getattr(a, "qc_template_radius_nm", getattr(cfg, "qc_template_radius_nm", 55.0)))
        cfg.qc_corner_tolerance_nm = float(getattr(a, "qc_corner_tolerance_nm", getattr(cfg, "qc_corner_tolerance_nm", 15.0)))
        cfg.qc_max_npcs_per_dialog = int(getattr(a, "qc_max_npcs_per_dialog", getattr(cfg, "qc_max_npcs_per_dialog", 0)))


        self.log("[Labelis] Starting analysis thread…")
        self.dock.set_status("Running analysis…")
        self.dock.set_busy(True)

        # Thread + worker (keep references!)
        self._thread = QtCore.QThread(self)
        self._worker = _PipelineWorker(cfg=cfg, df0=self._df0)

        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_line.connect(self.log)
        self._worker.checkpoint_requested.connect(self._on_checkpoint_requested)
        self._worker.finished_ok.connect(self._analysis_ok)
        self._worker.finished_error.connect(self._analysis_error)

        # Use queued connections to ensure UI methods run in the main thread
        self._thread.started.connect(self._worker.run, type=QtCore.Qt.QueuedConnection)
        self._worker.log_line.connect(self.log, type=QtCore.Qt.QueuedConnection)
        self._worker.checkpoint_requested.connect(self._on_checkpoint_requested, type=QtCore.Qt.QueuedConnection)
        self._worker.finished_ok.connect(self._analysis_ok, type=QtCore.Qt.QueuedConnection)
        self._worker.finished_error.connect(self._analysis_error, type=QtCore.Qt.QueuedConnection)

        # Cleanup wiring
        self._worker.finished_ok.connect(self._thread.quit)
        self._worker.finished_error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._cleanup_worker)

        self._thread.start()

    # -------------------------
    # Checkpoints (step-by-step)
    # -------------------------
    @QtCore.pyqtSlot(str, object)
    def _on_checkpoint_requested(self, stage: str, payload: object) -> None:
        """Handle a blocking checkpoint request from the pipeline worker.

        The pipeline worker is paused until we call worker.set_checkpoint_response().
        """
        stage = str(stage)
        p = payload if isinstance(payload, dict) else {}

        # Update napari overlays first (so user can inspect while dialog is open)
        try:
            if stage == "segmentation":
                self._checkpoint_update_layers_segmentation(p)
            elif stage == "centering":
                self._checkpoint_update_layers_centering(p)
            elif stage == "npc_qc":
                self._checkpoint_update_layers_npc_qc(p)

            elif stage == "model_qc":
                self._checkpoint_update_layers_model_qc(p)
        except Exception:
            # overlays are best-effort; do not break checkpoint logic
            pass

        # Choose dialog class
        dlg_cls = None
        if stage == "segmentation":
            dlg_cls = SegmentationCheckpointDialog
        elif stage == "thresholds":
            dlg_cls = ThresholdCheckpointDialog
        elif stage == "centering":
            dlg_cls = CenteringCheckpointDialog
        elif stage == "npc_qc":
            dlg_cls = CycleNPCQCDialog
        elif stage == "model_qc":
            dlg_cls = FinalNPCQCDialog

        if dlg_cls is None:
            # Unknown checkpoint stage: continue by default
            self.log(f"[Labelis] Unknown checkpoint stage '{stage}'. Continuing.")
            if self._worker is not None:
                self._worker.set_checkpoint_response({"action": "continue"})
            return

        # Reuse existing dialog when possible (for update/recompute loops)
        if (
            self._checkpoint_dialog is not None
            and self._checkpoint_stage == stage
            and isinstance(self._checkpoint_dialog, dlg_cls)
        ):
            try:
                self._checkpoint_dialog.update_payload(p)  # type: ignore[attr-defined]
            except Exception:
                pass
            return

        # Stage changed -> close previous dialog
        self._close_checkpoint_dialog()

        dlg = dlg_cls(parent=self._qt_window)
        self._checkpoint_dialog = dlg
        self._checkpoint_stage = stage
        dlg.actionRequested.connect(self._on_checkpoint_action)
        dlg.finished.connect(self._on_checkpoint_dialog_finished)
        try:
            dlg.update_payload(p)  # type: ignore[attr-defined]
        except Exception:
            pass
        dlg.show()

    @QtCore.pyqtSlot(object)
    def _on_checkpoint_action(self, action: object) -> None:
        """Forward a checkpoint action (continue/update/abort) to the worker."""
        if self._worker is None:
            return
        try:
            self._worker.set_checkpoint_response(action)
        except Exception:
            pass

    @QtCore.pyqtSlot(int)
    def _on_checkpoint_dialog_finished(self, _result_code: int) -> None:
        # Clear dialog references when it closes.
        self._checkpoint_dialog = None
        self._checkpoint_stage = None

    def _close_checkpoint_dialog(self) -> None:
        try:
            if self._checkpoint_dialog is not None:
                self._checkpoint_dialog.close()
        except Exception:
            pass
        self._checkpoint_dialog = None
        self._checkpoint_stage = None

    # -------------------------
    # Napari layer helpers (checkpoint overlays)
    # -------------------------
    def _layer_remove_if_exists(self, name: str) -> None:
        try:
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])
        except Exception:
            pass

    def _checkpoint_update_layers_segmentation(self, payload: dict) -> None:
        px = float(payload.get("pixel_size_nm", self._render_px_size_nm))
        img_r = payload.get("image_rendered", None)
        img_s = payload.get("image_segment", None)
        centers_px = np.asarray(payload.get("centers_px", np.zeros((0, 2))), dtype=float)
        radii_px = np.asarray(payload.get("radii_px", np.zeros((0,), dtype=float)), dtype=float)

        if isinstance(img_r, np.ndarray):
            self._layer_remove_if_exists("Rendered (analysis)")
            self.viewer.add_image(img_r, name="Rendered (analysis)", scale=(px, px), opacity=1.0)

        if isinstance(img_s, np.ndarray):
            self._layer_remove_if_exists("Segmentation input (analysis)")
            self.viewer.add_image(img_s, name="Segmentation input (analysis)", scale=(px, px), opacity=0.6)

        # Candidates overlay (points with per-candidate diameter)
        try:
            pts_yx = centers_px[:, [1, 0]] if centers_px.size else np.zeros((0, 2), dtype=float)
            self._layer_remove_if_exists("Candidates (preview)")
            size = 8.0
            if radii_px.size == pts_yx.shape[0]:
                size = np.clip(2.0 * radii_px, 2.0, 200.0)
            else:
                size = 6.0
            self.viewer.add_points(
                pts_yx,
                name="Candidates (preview)",
                size=size,
                scale=(px, px),
                face_color=[0, 0, 0, 0],
                edge_color="yellow",
                edge_width=1.0,
            )
        except Exception:
            # fallback: small points
            try:
                pts_yx = centers_px[:, [1, 0]] if centers_px.size else np.zeros((0, 2), dtype=float)
                self._layer_remove_if_exists("Candidates (preview)")
                self.viewer.add_points(
                    pts_yx,
                    name="Candidates (preview)",
                    size=6.0,
                    scale=(px, px),
                )
            except Exception:
                pass

    def _checkpoint_update_layers_centering(self, payload: dict) -> None:
        px = float(payload.get("pixel_size_nm", self._render_px_size_nm))
        det = np.asarray(payload.get("centers_detected_px", np.zeros((0, 2))), dtype=float)
        fit = np.asarray(payload.get("centers_fit_px", np.zeros((0, 2))), dtype=float)

        det_yx = det[:, [1, 0]] if det.size else np.zeros((0, 2), dtype=float)
        fit_yx = fit[:, [1, 0]] if fit.size else np.zeros((0, 2), dtype=float)

        self._layer_remove_if_exists("Centers detected")
        self._layer_remove_if_exists("Centers fit (step1)")

        if det_yx.size:
            self.viewer.add_points(det_yx, name="Centers detected", size=7.0, scale=(px, px), edge_color="magenta")
        if fit_yx.size:
            self.viewer.add_points(fit_yx, name="Centers fit (step1)", size=7.0, scale=(px, px), edge_color="cyan")

    def _checkpoint_update_layers_npc_qc(self, payload: dict) -> None:
        px = float(payload.get("pixel_size_nm", self._render_px_size_nm))
        npcs = payload.get("npcs", []) or []

        centers_nm: list = []
        status: list = []
        for n in npcs:
            try:
                if getattr(n, "center_fitcircle_step2_nm", None) is not None:
                    cx, cy = n.center_fitcircle_step2_nm
                elif getattr(n, "center_fitcircle_step1_nm", None) is not None:
                    cx, cy = n.center_fitcircle_step1_nm
                else:
                    cx, cy = n.npc_center_nm
                centers_nm.append((cx, cy))
                status.append(int(getattr(n, "npc_status", 1)))
            except Exception:
                pass

        if len(centers_nm) == 0:
            pts_yx = np.zeros((0, 2), dtype=float)
        else:
            c_nm = np.asarray(centers_nm, dtype=float)
            centers_px = c_nm / px
            pts_yx = centers_px[:, [1, 0]]

        self._layer_remove_if_exists("Cycle NPCs (QC)")

        # Try feature-based coloring; fall back to plain points if unsupported.
        try:
            import pandas as pd

            feat = pd.DataFrame({"status": np.asarray(status, dtype=int)}) if len(status) else pd.DataFrame({"status": []})
            self.viewer.add_points(
                pts_yx,
                name="Cycle NPCs (QC)",
                size=8.0,
                scale=(px, px),
                features=feat,
                face_color="status",
                edge_color="white",
            )
        except Exception:
            self.viewer.add_points(
                pts_yx,
                name="Cycle NPCs (QC)",
                size=8.0,
                scale=(px, px),
            )


    def _checkpoint_update_layers_model_qc(self, payload: dict) -> None:
        px = float(payload.get("pixel_size_nm", self._render_px_size_nm))
        centers_px = np.asarray(payload.get("centers_px", np.zeros((0, 2))), dtype=float)
        ele = np.asarray(payload.get("ele", np.zeros((0,), dtype=int)), dtype=int)
        pts_yx = centers_px[:, [1, 0]] if centers_px.size else np.zeros((0, 2), dtype=float)

        self._layer_remove_if_exists("NPC final (ELE)")

        # Try feature-based coloring; fall back to plain points if unsupported.
        try:
            import pandas as pd

            feat = pd.DataFrame({"ELE": ele.astype(int)}) if ele.size else pd.DataFrame({"ELE": []})
            self.viewer.add_points(
                pts_yx,
                name="NPC final (ELE)",
                size=8.0,
                scale=(px, px),
                features=feat,
                face_color="ELE",
                edge_color="white",
            )
        except Exception:
            self.viewer.add_points(
                pts_yx,
                name="NPC final (ELE)",
                size=8.0,
                scale=(px, px),
            )

    @QtCore.pyqtSlot()
    def _cleanup_worker(self) -> None:
        try:
            if self._worker is not None:
                self._worker.deleteLater()
        except Exception:
            pass
        self._worker = None
        self._thread = None

    @QtCore.pyqtSlot(object)
    def _analysis_ok(self, res_obj: object) -> None:
        # Close any open checkpoint dialogs
        self._close_checkpoint_dialog()
        res: PipelineResult = res_obj  # type: ignore
        self._result = res

        self.dock.set_busy(False)
        self.dock.set_status("Done.")

        self.log("[Labelis] Analysis finished.")

        # Update / add layers
        try:
            if res.image_segment is not None:
                if "Segmentation input" in self.viewer.layers:
                    self.viewer.layers.remove(self.viewer.layers["Segmentation input"])
                self.viewer.add_image(
                    res.image_segment,
                    name="Segmentation input",
                    scale=(self._render_px_size_nm, self._render_px_size_nm),
                    opacity=0.6,
                )
        except Exception:
            pass

        try:
            # candidates
            if res.all_centers_px is not None and res.all_centers_px.size:
                pts_yx = np.asarray(res.all_centers_px, dtype=float)[:, [1, 0]]
                if "Candidates" in self.viewer.layers:
                    self.viewer.layers.remove(self.viewer.layers["Candidates"])
                self.viewer.add_points(
                    pts_yx,
                    name="Candidates",
                    size=6.0,
                    scale=(self._render_px_size_nm, self._render_px_size_nm),
                )
        except Exception:
            pass

        self.viewer.add_image(res.image_segment, ...)
        if res.cpsam_instance_labels is not None:
            if "CPSAM labels" in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers["CPSAM labels"])
            self.viewer.add_labels(
                res.cpsam_instance_labels,
                name="CPSAM labels",
                scale=(self._render_px_size_nm, self._render_px_size_nm),
                opacity=0.6,
            )

        def _npc_centers_px(npcs):
            if not npcs:
                return np.zeros((0, 2), dtype=float)
            centers_nm = []
            for n in npcs:
                if n.center_fitcircle_step2_nm is not None:
                    c = n.center_fitcircle_step2_nm
                elif n.center_fitcircle_step1_nm is not None:
                    c = n.center_fitcircle_step1_nm
                else:
                    c = n.npc_center_nm
                centers_nm.append([float(c[0]), float(c[1])])
            centers_nm = np.asarray(centers_nm, dtype=float)
            return centers_nm / float(self._render_px_size_nm)

        try:
            ok_px = _npc_centers_px(res.npc_ok)
            if ok_px.size:
                pts = ok_px[:, [1, 0]]
                if "NPC OK" in self.viewer.layers:
                    self.viewer.layers.remove(self.viewer.layers["NPC OK"])
                self.viewer.add_points(
                    pts,
                    name="NPC OK",
                    size=8.0,
                    scale=(self._render_px_size_nm, self._render_px_size_nm),
                )
        except Exception:
            pass

        try:
            rej_px = _npc_centers_px(res.npc_rejected)
            if rej_px.size:
                pts = rej_px[:, [1, 0]]
                if "NPC rejected" in self.viewer.layers:
                    self.viewer.layers.remove(self.viewer.layers["NPC rejected"])
                self.viewer.add_points(
                    pts,
                    name="NPC rejected",
                    size=6.0,
                    scale=(self._render_px_size_nm, self._render_px_size_nm),
                )
        except Exception:
            pass

        # Results dialog
        try:
            df = res.df_npc_final if res.df_npc_final is not None else None
            if df is not None:
                s = res.summary or {}
                summary_text = (
                    f"Input: {s.get('input_path','')}\n"
                    f"Filtered localizations: {s.get('n_localizations_filtered','?')}\n"
                    f"OK NPCs: {s.get('n_npcs_ok','?')}\n"
                    f"Rejected NPCs: {s.get('n_npcs_rejected','?')}\n"
                    f"p_label: {s.get('p_label','?')} ± {s.get('p_label_error','?')}\n"
                    f"Alignment: {s.get('alignment_method','?')}\n"
                    f"Compute: {s.get('compute_engine','?')}\n"
                    f"Segmentation: {s.get('segmentation_engine','?')}\n"
                )
                dlg = ResultsDialog(df=df, summary_text=summary_text, parent=self._qt_window)
                dlg.setModal(False)
                dlg.show()
        except Exception:
            pass

    @QtCore.pyqtSlot(str)
    def _analysis_error(self, tb: str) -> None:
        # Close any open checkpoint dialogs
        self._close_checkpoint_dialog()
        self.dock.set_busy(False)
        self.dock.set_status("Error.")
        self.log("[Labelis] ERROR (see traceback below).")
        self.log(tb)
        QtWidgets.QMessageBox.critical(self._qt_window, "Labelis - Analysis error", tb)
