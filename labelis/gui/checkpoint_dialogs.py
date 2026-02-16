from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets


def _fig_to_pixmap(fig) -> QtGui.QPixmap:
    """Render a matplotlib figure to a QPixmap using an in-memory PNG."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    pm = QtGui.QPixmap()
    pm.loadFromData(buf.getvalue())
    return pm


def _hist_pixmap(
    data: np.ndarray,
    threshold: Optional[float],
    title: str,
    xlabel: str,
    bins: int = 50,
) -> QtGui.QPixmap:
    """Build a histogram pixmap (matplotlib Agg) for display in Qt."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]

    fig = plt.figure(figsize=(4.2, 2.6), dpi=150)
    ax = fig.add_subplot(111)
    if x.size:
        ax.hist(x, bins=bins)
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(float(threshold), linestyle="-", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()

    pm = _fig_to_pixmap(fig)
    plt.close(fig)
    return pm


def _ele_bar_pixmap(ele: np.ndarray, min_locs_per_corner: int, title: str = "ELE distribution") -> QtGui.QPixmap:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = np.asarray(ele, dtype=int)
    e = e[np.isfinite(e)] if e.size else e

    fig = plt.figure(figsize=(4.2, 2.6), dpi=150)
    ax = fig.add_subplot(111)
    bins = np.arange(0.5, 8.5 + 1e-9, 1.0)
    if e.size:
        ax.hist(e, bins=bins)
    ax.set_xticks(np.arange(1, 9))
    ax.set_xlabel("ELE (corners >= min locs)")
    ax.set_ylabel("NPC count")
    ax.set_title(f"{title} (min/corner={int(min_locs_per_corner)})")
    fig.tight_layout()
    pm = _fig_to_pixmap(fig)
    plt.close(fig)
    return pm


class _BaseCheckpointDialog(QtWidgets.QDialog):
    """Base class for non-modal checkpoint dialogs.

    The dialog emits a dict-based action to the controller, which forwards it
    to the pipeline worker.
    """

    actionRequested = QtCore.pyqtSignal(object)  # dict

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        try:
            self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        except Exception:
            pass

        self._waiting = False
        self._terminal_action_sent = False

    def set_waiting(self, waiting: bool) -> None:
        self._waiting = bool(waiting)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt API)
        # If the user closes the dialog window, default to abort to avoid
        # leaving the pipeline blocked on a checkpoint.
        if not self._terminal_action_sent:
            try:
                self.actionRequested.emit({"action": "abort"})
            except Exception:
                pass
            self._terminal_action_sent = True
        super().closeEvent(event)


class SegmentationCheckpointDialog(_BaseCheckpointDialog):
    """Checkpoint dialog for candidate detection / centering preview."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Labelis - Checkpoint: Candidate detection")
        self.resize(520, 260)

        layout = QtWidgets.QVBoxLayout(self)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.combo_engine = QtWidgets.QComboBox()
        self.combo_engine.addItems(["blob_log", "hough_cv2", "hough_skimage"])
        form.addRow("Segmentation engine:", self.combo_engine)

        self.spin_sens = QtWidgets.QDoubleSpinBox()
        self.spin_sens.setRange(0.01, 0.99)
        self.spin_sens.setDecimals(3)
        self.spin_sens.setSingleStep(0.01)
        form.addRow("Sensitivity:", self.spin_sens)

        self.spin_expected_r = QtWidgets.QDoubleSpinBox()
        self.spin_expected_r.setRange(1.0, 1000.0)
        self.spin_expected_r.setDecimals(2)
        form.addRow("Expected radius (nm):", self.spin_expected_r)

        self.spin_rmin = QtWidgets.QDoubleSpinBox()
        self.spin_rmin.setRange(1.0, 2000.0)
        self.spin_rmin.setDecimals(2)
        self.spin_rmax = QtWidgets.QDoubleSpinBox()
        self.spin_rmax.setRange(1.0, 2000.0)
        self.spin_rmax.setDecimals(2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.spin_rmin)
        row.addWidget(QtWidgets.QLabel("to"))
        row.addWidget(self.spin_rmax)
        form.addRow("Circle radius range (nm):", row)

        self.spin_min_dist = QtWidgets.QDoubleSpinBox()
        self.spin_min_dist.setRange(-1.0, 1e6)
        self.spin_min_dist.setDecimals(2)
        form.addRow("Min dist between NPCs (nm):", self.spin_min_dist)

        self.spin_thres_clip = QtWidgets.QDoubleSpinBox()
        self.spin_thres_clip.setRange(0.0, 1.0)
        self.spin_thres_clip.setDecimals(4)
        self.spin_thres_clip.setSingleStep(0.005)
        form.addRow("Clip threshold (thres_clip):", self.spin_thres_clip)

        btn_row = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_row)

        self.btn_rerun = QtWidgets.QPushButton("Re-run detection")
        self.btn_continue = QtWidgets.QPushButton("Continue")
        self.btn_abort = QtWidgets.QPushButton("Abort")

        btn_row.addWidget(self.btn_rerun)
        btn_row.addWidget(self.btn_continue)
        btn_row.addWidget(self.btn_abort)

        self.btn_rerun.clicked.connect(self._on_rerun)
        self.btn_continue.clicked.connect(self._on_continue)
        self.btn_abort.clicked.connect(self._on_abort)

    def update_payload(self, payload: dict) -> None:
        cycle = payload.get("cycle", "?")
        n = payload.get("n_candidates", "?")
        px = payload.get("pixel_size_nm", float("nan"))
        eff_min_dist = payload.get("effective_min_dist_nm", None)
        if eff_min_dist is None:
            eff_min_dist = payload.get("min_distance_nm", None)

        self.lbl_info.setText(
            f"Cycle {cycle}: {n} candidate NPCs detected.\n"
            f"Inspect the overlay in Napari (Rendered + Segmentation input + Candidates).\n"
            f"Pixel size: {px} nm/px. "
            + (f"Effective minDist: {float(eff_min_dist):.1f} nm." if eff_min_dist is not None else "")
        )

        # Current parameters
        self.combo_engine.setCurrentText(str(payload.get("segmentation_engine", self.combo_engine.currentText())))
        self.spin_sens.setValue(float(payload.get("sensitivity", self.spin_sens.value())))
        self.spin_expected_r.setValue(float(payload.get("expected_npc_radius_nm", self.spin_expected_r.value())))
        self.spin_rmin.setValue(float(payload.get("circle_min_radius_nm", self.spin_rmin.value())))
        self.spin_rmax.setValue(float(payload.get("circle_max_radius_nm", self.spin_rmax.value())))
        self.spin_min_dist.setValue(float(payload.get("min_dist_between_npcs_nm", self.spin_min_dist.value())))
        self.spin_thres_clip.setValue(float(payload.get("thres_clip", self.spin_thres_clip.value())))

        # Re-enable controls if we were waiting for a recompute.
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.combo_engine.setEnabled(enabled)
        self.spin_sens.setEnabled(enabled)
        self.spin_expected_r.setEnabled(enabled)
        self.spin_rmin.setEnabled(enabled)
        self.spin_rmax.setEnabled(enabled)
        self.spin_min_dist.setEnabled(enabled)
        self.spin_thres_clip.setEnabled(enabled)
        self.btn_rerun.setEnabled(enabled)
        self.btn_continue.setEnabled(enabled)
        self.btn_abort.setEnabled(True)

    def _on_rerun(self) -> None:
        self._set_controls_enabled(False)
        updates = {
            "segmentation_engine": str(self.combo_engine.currentText()),
            "sensitivity": float(self.spin_sens.value()),
            "expected_npc_radius_nm": float(self.spin_expected_r.value()),
            "circle_min_radius_nm": float(self.spin_rmin.value()),
            "circle_max_radius_nm": float(self.spin_rmax.value()),
            "min_dist_between_npcs_nm": float(self.spin_min_dist.value()),
            "thres_clip": float(self.spin_thres_clip.value()),
        }
        self.actionRequested.emit({"action": "update", "updates": updates})

    def _on_continue(self) -> None:
        self._terminal_action_sent = True
        self.actionRequested.emit({"action": "continue"})
        self.accept()

    def _on_abort(self) -> None:
        self._terminal_action_sent = True
        self.actionRequested.emit({"action": "abort"})
        self.reject()


class ThresholdCheckpointDialog(_BaseCheckpointDialog):
    """Checkpoint dialog for tuning max mean sigma / max mean uncertainty thresholds."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Labelis - Checkpoint: Threshold tuning")
        self.resize(860, 520)

        layout = QtWidgets.QVBoxLayout(self)
        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        # plots
        plots = QtWidgets.QHBoxLayout()
        layout.addLayout(plots)
        self.lbl_hist_sigma = QtWidgets.QLabel()
        self.lbl_hist_unc = QtWidgets.QLabel()
        self.lbl_hist_sigma.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_hist_unc.setAlignment(QtCore.Qt.AlignCenter)
        plots.addWidget(self.lbl_hist_sigma, 1)
        plots.addWidget(self.lbl_hist_unc, 1)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.spin_min_locs = QtWidgets.QSpinBox()
        self.spin_min_locs.setRange(0, 1_000_000)
        form.addRow("Min locs / NPC:", self.spin_min_locs)

        self.spin_max_mean_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_max_mean_sigma.setRange(0.0, 1e6)
        self.spin_max_mean_sigma.setDecimals(2)
        form.addRow("Max mean sigma (nm):", self.spin_max_mean_sigma)

        self.spin_max_mean_unc = QtWidgets.QDoubleSpinBox()
        self.spin_max_mean_unc.setRange(0.0, 1e6)
        self.spin_max_mean_unc.setDecimals(2)
        form.addRow("Max mean uncertainty (nm):", self.spin_max_mean_unc)

        btn_row = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_row)

        self.btn_apply = QtWidgets.QPushButton("Apply thresholds")
        self.btn_continue = QtWidgets.QPushButton("Continue")
        self.btn_abort = QtWidgets.QPushButton("Abort")
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_continue)
        btn_row.addWidget(self.btn_abort)

        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_continue.clicked.connect(self._on_continue)
        self.btn_abort.clicked.connect(self._on_abort)

        # cached arrays (for display only)
        self._sigma_means = np.zeros((0,), dtype=float)
        self._unc_means = np.zeros((0,), dtype=float)

    def update_payload(self, payload: dict) -> None:
        cycle = payload.get("cycle", "?")
        n_total = int(payload.get("n_total", 0))
        n_pass = int(payload.get("n_pass", 0))

        self.lbl_info.setText(
            f"Cycle {cycle}: threshold tuning on extracted candidate NPC ROIs.\n"
            f"Would pass current thresholds: {n_pass} / {n_total}.\n"
            "Adjust thresholds and click 'Apply thresholds' to recompute."  # noqa: E501
        )

        self._sigma_means = np.asarray(payload.get("sigma_means", np.zeros((0,), dtype=float)), dtype=float)
        self._unc_means = np.asarray(payload.get("unc_means", np.zeros((0,), dtype=float)), dtype=float)

        self.spin_min_locs.setValue(int(payload.get("min_n_locs", self.spin_min_locs.value())))
        self.spin_max_mean_sigma.setValue(float(payload.get("max_mean_sigma_nm", self.spin_max_mean_sigma.value())))
        self.spin_max_mean_unc.setValue(float(payload.get("max_mean_uncertainty_nm", self.spin_max_mean_unc.value())))

        # plots
        thr_sigma = float(self.spin_max_mean_sigma.value())
        thr_unc = float(self.spin_max_mean_unc.value())

        self.lbl_hist_sigma.setPixmap(
            _hist_pixmap(self._sigma_means, thr_sigma, title="Mean sigma (per NPC)", xlabel="nm")
        )
        self.lbl_hist_unc.setPixmap(
            _hist_pixmap(self._unc_means, thr_unc, title="Mean uncertainty (per NPC)", xlabel="nm")
        )

        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.spin_min_locs.setEnabled(enabled)
        self.spin_max_mean_sigma.setEnabled(enabled)
        self.spin_max_mean_unc.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled)
        self.btn_continue.setEnabled(enabled)
        self.btn_abort.setEnabled(True)

    def _on_apply(self) -> None:
        self._set_controls_enabled(False)
        updates = {
            "min_n_locs": int(self.spin_min_locs.value()),
            "max_mean_sigma_nm": float(self.spin_max_mean_sigma.value()),
            "max_mean_uncertainty_nm": float(self.spin_max_mean_unc.value()),
        }
        self.actionRequested.emit({"action": "update", "updates": updates})

    def _on_continue(self) -> None:
        self._terminal_action_sent = True
        self.actionRequested.emit({"action": "continue"})
        self.accept()

    def _on_abort(self) -> None:
        self._terminal_action_sent = True
        self.actionRequested.emit({"action": "abort"})
        self.reject()


class CenteringCheckpointDialog(_BaseCheckpointDialog):
    """Checkpoint dialog to validate centering after step-1 circle fit."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Labelis - Checkpoint: Centering QC")
        self.resize(640, 420)

        layout = QtWidgets.QVBoxLayout(self)
        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        self.lbl_hist = QtWidgets.QLabel()
        self.lbl_hist.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_hist, 1)

        btn_row = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_row)
        self.btn_continue = QtWidgets.QPushButton("Continue")
        self.btn_abort = QtWidgets.QPushButton("Abort")
        btn_row.addWidget(self.btn_continue)
        btn_row.addWidget(self.btn_abort)

        self.btn_continue.clicked.connect(self._on_continue)
        self.btn_abort.clicked.connect(self._on_abort)

    def update_payload(self, payload: dict) -> None:
        cycle = payload.get("cycle", "?")
        n = int(payload.get("n", 0))
        med = payload.get("median_shift_nm", float("nan"))
        p95 = payload.get("p95_shift_nm", float("nan"))
        px = payload.get("pixel_size_nm", float("nan"))

        self.lbl_info.setText(
            f"Cycle {cycle}: centering QC after step-1 circle fit (free-radius).\n"
            f"NPC ROIs with valid fit: {n}.\n"
            f"Median |Δcenter| = {float(med):.2f} nm, 95th pct = {float(p95):.2f} nm.\n"
            f"Pixel size: {float(px):.2f} nm/px.\n\n"
            "Napari layers: 'Centers detected', 'Centers fit (step1)'."
        )

        shift_nm = np.asarray(payload.get("shift_nm", np.zeros((0,), dtype=float)), dtype=float)
        self.lbl_hist.setPixmap(_hist_pixmap(shift_nm, threshold=None, title="|Δcenter| after fitcircle step1", xlabel="nm"))

    def _on_continue(self) -> None:
        self._terminal_action_sent = True
        self.actionRequested.emit({"action": "continue"})
        self.accept()

    def _on_abort(self) -> None:
        self._terminal_action_sent = True
        self.actionRequested.emit({"action": "abort"})
        self.reject()


class ModelQCCheckpointDialog(_BaseCheckpointDialog):
    """Checkpoint dialog for validating alignment + ELE model application."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Labelis - Checkpoint: Model / alignment QC")
        self.resize(820, 520)

        layout = QtWidgets.QVBoxLayout(self)
        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        self.lbl_ele = QtWidgets.QLabel()
        self.lbl_ele.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_ele, 1)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.combo_align = QtWidgets.QComboBox()
        self.combo_align.addItems(["smap", "template", "none"])
        form.addRow("Alignment method:", self.combo_align)

        self.spin_min_corner = QtWidgets.QSpinBox()
        self.spin_min_corner.setRange(1, 1000)
        form.addRow("Min locs / corner:", self.spin_min_corner)

        btn_row = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_row)

        self.btn_recompute = QtWidgets.QPushButton("Recompute alignment")
        self.btn_continue = QtWidgets.QPushButton("Continue")
        self.btn_abort = QtWidgets.QPushButton("Abort")
        btn_row.addWidget(self.btn_recompute)
        btn_row.addWidget(self.btn_continue)
        btn_row.addWidget(self.btn_abort)

        self.btn_recompute.clicked.connect(self._on_recompute)
        self.btn_continue.clicked.connect(self._on_continue)
        self.btn_abort.clicked.connect(self._on_abort)

        self._ele = np.zeros((0,), dtype=int)

    def update_payload(self, payload: dict) -> None:
        cycle = payload.get("cycle", "?")
        n = int(payload.get("n_npcs", 0))
        self._ele = np.asarray(payload.get("ele", np.zeros((0,), dtype=int)), dtype=int)

        self.combo_align.setCurrentText(str(payload.get("alignment_method", self.combo_align.currentText())))
        self.spin_min_corner.setValue(int(payload.get("min_locs_per_corner", self.spin_min_corner.value())))

        # Basic quality metrics
        ele_mean = float(np.nanmean(self._ele)) if self._ele.size else float("nan")
        ele_med = float(np.nanmedian(self._ele)) if self._ele.size else float("nan")

        self.lbl_info.setText(
            f"Cycle {cycle}: model/alignment QC on {n} NPCs (status=OK).\n"
            f"ELE median={ele_med:.2f}, mean={ele_mean:.2f}.\n"
            "Inspect the overlay in Napari ('NPC final (ELE)').\n"
            "You may change alignment method/min corner threshold and recompute."  # noqa: E501
        )

        self.lbl_ele.setPixmap(
            _ele_bar_pixmap(self._ele, min_locs_per_corner=int(self.spin_min_corner.value()))
        )

        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.combo_align.setEnabled(enabled)
        self.spin_min_corner.setEnabled(enabled)
        self.btn_recompute.setEnabled(enabled)
        self.btn_continue.setEnabled(enabled)
        self.btn_abort.setEnabled(True)

    def _on_recompute(self) -> None:
        self._set_controls_enabled(False)
        updates = {
            "alignment_method": str(self.combo_align.currentText()),
            "min_locs_per_corner": int(self.spin_min_corner.value()),
        }
        self.actionRequested.emit({"action": "update", "updates": updates})

    def _on_continue(self) -> None:
        self._terminal_action_sent = True
        self.actionRequested.emit({"action": "continue"})
        self.accept()

    def _on_abort(self) -> None:
        self._terminal_action_sent = True
        self.actionRequested.emit({"action": "abort"})
        self.reject()
