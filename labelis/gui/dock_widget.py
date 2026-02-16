from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt5 import QtCore, QtWidgets


@dataclass
class RenderSettings:
    render_px_size_nm: float
    compute_engine: str


@dataclass
class AnalysisSettings:
    # Pre-filtering
    min_uncertainty_nm: float
    max_uncertainty_nm: float
    min_sigma_nm: float
    max_sigma_nm: float
    min_frame: int
    max_frame: int

    # Segmentation
    segmentation_engine: str
    sensitivity: float
    expected_npc_radius_nm: float
    circle_min_radius_nm: float
    circle_max_radius_nm: float
    min_dist_between_npcs_nm: float
    thres_clip: float

    # Extraction / filtering
    extract_box_radius_nm: float
    extract_circle_radius_nm: float

    min_n_locs: int
    max_mean_sigma_nm: float
    max_mean_uncertainty_nm: float

    # Alignment / ELE
    alignment_method: str
    min_locs_per_corner: int
    ele_bootstrap_samples: int

    # Guided checkpoints
    step_by_step: bool
    checkpoint_segmentation: bool
    checkpoint_thresholds: bool
    checkpoint_centering: bool
    checkpoint_cycle_qc: bool
    checkpoint_model_qc: bool

    # Per-NPC QC inspector (display parameters)
    qc_template_radius_nm: float
    qc_corner_tolerance_nm: float
    qc_max_npcs_per_dialog: int


class LabelisDockWidget(QtWidgets.QWidget):
    """Napari dock widget implementing a strict workflow.

    Workflow
    --------
    1) Load+Render (render settings chosen here)
    2) Draw ROI in Napari Shapes layer named "ROI"
    3) Confirm ROI (unlocks analysis inputs)
    4) Run analysis
    """

    sig_load_render = QtCore.pyqtSignal()
    sig_confirm_roi = QtCore.pyqtSignal()
    sig_run_analysis = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.setMinimumWidth(440)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        #--------------------------------
        # Scroll area container
        #--------------------------------
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        #-----------------------------
        # Internal container widget
        #-----------------------------
        container = QtWidgets.QWidget()
        scroll.setWidget(container)

        main_layout = QtWidgets.QVBoxLayout(container)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # -------------------------
        # Group: Input + Render
        # -------------------------
        self.grp_input = QtWidgets.QGroupBox("1) Input + Render (before ROI)")
        main_layout.addWidget(self.grp_input)
        g = QtWidgets.QFormLayout(self.grp_input)
        g.setLabelAlignment(QtCore.Qt.AlignLeft)
        g.setFormAlignment(QtCore.Qt.AlignTop)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)

        # Input file
        self.edit_input = QtWidgets.QLineEdit()
        self.btn_browse_input = QtWidgets.QPushButton("Browse…")
        self.btn_browse_input.clicked.connect(self._browse_input)

        row_in = QtWidgets.QHBoxLayout()
        row_in.addWidget(self.edit_input, 1)
        row_in.addWidget(self.btn_browse_input, 0)
        g.addRow("Localization CSV/TSV:", row_in)

        # Output dir
        self.edit_out = QtWidgets.QLineEdit()
        self.btn_browse_out = QtWidgets.QPushButton("Browse…")
        self.btn_browse_out.clicked.connect(self._browse_out)

        row_out = QtWidgets.QHBoxLayout()
        row_out.addWidget(self.edit_out, 1)
        row_out.addWidget(self.btn_browse_out, 0)
        g.addRow("Output directory:", row_out)

        # Render pixel size
        self.spin_px = QtWidgets.QDoubleSpinBox()
        self.spin_px.setDecimals(2)
        self.spin_px.setRange(1.0, 100.0)
        self.spin_px.setSingleStep(1.0)
        self.spin_px.setValue(10.0)
        g.addRow("Render pixel size (nm):", self.spin_px)

        # Compute engine (render)
        self.combo_engine = QtWidgets.QComboBox()
        self.combo_engine.addItems(["turbo_bin_blur", "numba", "reference_bruteforce"])
        self.combo_engine.setCurrentText("turbo_bin_blur")
        g.addRow("Compute engine:", self.combo_engine)

        self.btn_load_render = QtWidgets.QPushButton("Load + Render")
        self.btn_load_render.clicked.connect(self.sig_load_render.emit)
        g.addRow(self.btn_load_render)

        # -------------------------
        # Group: ROI
        # -------------------------
        self.grp_roi = QtWidgets.QGroupBox("2) ROI selection (Napari)")
        main_layout.addWidget(self.grp_roi)
        gr = QtWidgets.QFormLayout(self.grp_roi)
        gr.setLabelAlignment(QtCore.Qt.AlignLeft)
        gr.setFormAlignment(QtCore.Qt.AlignTop)
        gr.setHorizontalSpacing(8)
        gr.setVerticalSpacing(6)

        self.lbl_roi = QtWidgets.QLabel(
            "Draw a polygon in a Shapes layer named <b>ROI</b> in Napari, then click <b>Confirm ROI</b>."
        )
        self.lbl_roi.setWordWrap(True)
        gr.addRow(self.lbl_roi)

        self.btn_confirm_roi = QtWidgets.QPushButton("Confirm ROI")
        self.btn_confirm_roi.clicked.connect(self.sig_confirm_roi.emit)
        gr.addRow(self.btn_confirm_roi)

        # -------------------------
        # Group: Analysis
        # -------------------------
        self.grp_analysis = QtWidgets.QGroupBox("3) Analysis")
        main_layout.addWidget(self.grp_analysis)
        ga = QtWidgets.QFormLayout(self.grp_analysis)
        ga.setLabelAlignment(QtCore.Qt.AlignLeft)
        ga.setFormAlignment(QtCore.Qt.AlignTop)
        ga.setHorizontalSpacing(8)
        ga.setVerticalSpacing(6)

        # Pre-filtering
        self.spin_min_unc = QtWidgets.QDoubleSpinBox(); self.spin_min_unc.setRange(0.0, 1e6); self.spin_min_unc.setDecimals(2); self.spin_min_unc.setValue(0.0)
        self.spin_max_unc = QtWidgets.QDoubleSpinBox(); self.spin_max_unc.setRange(0.0, 1e6); self.spin_max_unc.setDecimals(2); self.spin_max_unc.setValue(15.0)
        ga.addRow("Min uncertainty (nm):", self.spin_min_unc)
        ga.addRow("Max uncertainty (nm):", self.spin_max_unc)

        self.spin_min_sigma = QtWidgets.QDoubleSpinBox(); self.spin_min_sigma.setRange(0.0, 1e6); self.spin_min_sigma.setDecimals(2); self.spin_min_sigma.setValue(75.0)
        self.spin_max_sigma = QtWidgets.QDoubleSpinBox(); self.spin_max_sigma.setRange(0.0, 1e6); self.spin_max_sigma.setDecimals(2); self.spin_max_sigma.setValue(155.0)
        ga.addRow("Min sigma (nm):", self.spin_min_sigma)
        ga.addRow("Max sigma (nm):", self.spin_max_sigma)

        self.spin_min_frame = QtWidgets.QSpinBox(); self.spin_min_frame.setRange(0, 2_147_483_647); self.spin_min_frame.setValue(0)
        self.spin_max_frame = QtWidgets.QSpinBox(); self.spin_max_frame.setRange(0, 2_147_483_647); self.spin_max_frame.setValue(2_147_483_647)
        ga.addRow("Min frame:", self.spin_min_frame)
        ga.addRow("Max frame:", self.spin_max_frame)

        # Segmentation
        self.combo_seg = QtWidgets.QComboBox()
        self.combo_seg.addItems(["blob_log", "hough_cv2", "hough_skimage"])
        self.combo_seg.setCurrentText("blob_log")
        ga.addRow("Segmentation engine:", self.combo_seg)

        self.spin_sens = QtWidgets.QDoubleSpinBox(); self.spin_sens.setRange(0.01, 0.99); self.spin_sens.setDecimals(3); self.spin_sens.setValue(0.95)
        ga.addRow("Sensitivity:", self.spin_sens)

        self.spin_thres_clip = QtWidgets.QDoubleSpinBox(); self.spin_thres_clip.setRange(0.0, 1.0); self.spin_thres_clip.setDecimals(4); self.spin_thres_clip.setValue(0.03)
        ga.addRow("Clip threshold (thres_clip):", self.spin_thres_clip)

        self.spin_expected_r = QtWidgets.QDoubleSpinBox(); self.spin_expected_r.setRange(1.0, 1000.0); self.spin_expected_r.setDecimals(2); self.spin_expected_r.setValue(55.0)
        ga.addRow("Expected NPC radius (nm):", self.spin_expected_r)

        self.spin_rmin = QtWidgets.QDoubleSpinBox(); self.spin_rmin.setRange(1.0, 2000.0); self.spin_rmin.setDecimals(2); self.spin_rmin.setValue(40.0)
        self.spin_rmax = QtWidgets.QDoubleSpinBox(); self.spin_rmax.setRange(1.0, 2000.0); self.spin_rmax.setDecimals(2); self.spin_rmax.setValue(60.0)
        row_rr = QtWidgets.QHBoxLayout()
        row_rr.addWidget(self.spin_rmin)
        row_rr.addWidget(QtWidgets.QLabel("to"))
        row_rr.addWidget(self.spin_rmax)
        ga.addRow("Circle radius range (nm):", row_rr)

        self.spin_min_dist = QtWidgets.QDoubleSpinBox(); self.spin_min_dist.setRange(-1.0, 1e9); self.spin_min_dist.setDecimals(2); self.spin_min_dist.setValue(-1.0)
        ga.addRow("Min dist between NPCs (nm):", self.spin_min_dist)

        # Extraction
        self.spin_box_r = QtWidgets.QDoubleSpinBox(); self.spin_box_r.setRange(1.0, 2000.0); self.spin_box_r.setDecimals(2); self.spin_box_r.setValue(120.0)
        ga.addRow("Extract box radius (nm):", self.spin_box_r)

        self.spin_circle_r = QtWidgets.QDoubleSpinBox(); self.spin_circle_r.setRange(1.0, 2000.0); self.spin_circle_r.setDecimals(2); self.spin_circle_r.setValue(100.0)
        ga.addRow("Extract circle radius (nm):", self.spin_circle_r)

        # Filtering thresholds
        self.spin_min_locs = QtWidgets.QSpinBox(); self.spin_min_locs.setRange(0, 1_000_000); self.spin_min_locs.setValue(20)
        ga.addRow("Min locs / NPC:", self.spin_min_locs)

        self.spin_max_mean_sigma = QtWidgets.QDoubleSpinBox(); self.spin_max_mean_sigma.setRange(0.0, 1e6); self.spin_max_mean_sigma.setDecimals(2); self.spin_max_mean_sigma.setValue(150.0)
        self.spin_max_mean_unc = QtWidgets.QDoubleSpinBox(); self.spin_max_mean_unc.setRange(0.0, 1e6); self.spin_max_mean_unc.setDecimals(2); self.spin_max_mean_unc.setValue(20.0)
        ga.addRow("Max mean sigma (nm):", self.spin_max_mean_sigma)
        ga.addRow("Max mean uncertainty (nm):", self.spin_max_mean_unc)

        # Alignment / ELE
        self.combo_align = QtWidgets.QComboBox()
        self.combo_align.addItems(["smap", "template", "none"])
        self.combo_align.setCurrentText("smap")
        ga.addRow("Alignment:", self.combo_align)

        self.spin_min_corner = QtWidgets.QSpinBox(); self.spin_min_corner.setRange(1, 1000); self.spin_min_corner.setValue(2)
        ga.addRow("Min locs / corner:", self.spin_min_corner)

        self.spin_boot = QtWidgets.QSpinBox(); self.spin_boot.setRange(1, 10000); self.spin_boot.setValue(20)
        ga.addRow("ELE bootstrap samples:", self.spin_boot)

        # -------------------------
        # Guided pipeline mode / checkpoints
        # -------------------------
        self.chk_step = QtWidgets.QCheckBox("Step-by-step checkpoints (pause + QC)")
        self.chk_step.setChecked(False)
        self.chk_step.toggled.connect(self._update_checkpoint_controls)
        ga.addRow(self.chk_step)

        self.chk_cp_seg = QtWidgets.QCheckBox("Segmentation + detection checkpoint")
        self.chk_cp_thr = QtWidgets.QCheckBox("Threshold tuning checkpoint (sigma/unc)")
        self.chk_cp_center = QtWidgets.QCheckBox("Centering QC checkpoint (global stats)")
        self.chk_cp_cycle = QtWidgets.QCheckBox("Per-cycle NPC inspector (centering + structure)")
        self.chk_cp_model = QtWidgets.QCheckBox("Final NPC model inspector (8-corner mapping)")

        for cb in (self.chk_cp_seg, self.chk_cp_thr, self.chk_cp_center, self.chk_cp_cycle, self.chk_cp_model):
            cb.setChecked(True)

        ga.addRow("Enabled checkpoints:", self._vbox([self.chk_cp_seg, self.chk_cp_thr, self.chk_cp_center, self.chk_cp_cycle, self.chk_cp_model]))

        self.spin_qc_template_r = QtWidgets.QDoubleSpinBox(); self.spin_qc_template_r.setRange(1.0, 200.0); self.spin_qc_template_r.setDecimals(2); self.spin_qc_template_r.setValue(55.0)
        self.spin_qc_corner_tol = QtWidgets.QDoubleSpinBox(); self.spin_qc_corner_tol.setRange(0.1, 100.0); self.spin_qc_corner_tol.setDecimals(2); self.spin_qc_corner_tol.setValue(15.0)
        self.spin_qc_max_npcs = QtWidgets.QSpinBox(); self.spin_qc_max_npcs.setRange(0, 10000); self.spin_qc_max_npcs.setValue(0)
        ga.addRow("QC template radius (nm):", self.spin_qc_template_r)
        ga.addRow("QC corner tolerance (nm):", self.spin_qc_corner_tol)
        ga.addRow("Max NPCs per QC dialog (0=all):", self.spin_qc_max_npcs)

        self._update_checkpoint_controls(bool(self.chk_step.isChecked()))

        # Run
        self.btn_run = QtWidgets.QPushButton("Run analysis")
        self.btn_run.clicked.connect(self.sig_run_analysis.emit)
        ga.addRow(self.btn_run)

        # -------------------------
        # Log box
        # -------------------------
        self.grp_log = QtWidgets.QGroupBox("Log")
        main_layout.addWidget(self.grp_log, 1)
        vlog = QtWidgets.QVBoxLayout(self.grp_log)
        self.text_log = QtWidgets.QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        vlog.addWidget(self.text_log, 1)

        self.lbl_status = QtWidgets.QLabel("Idle.")
        main_layout.addWidget(self.lbl_status)

    @staticmethod
    def _vbox(widgets: list[QtWidgets.QWidget]) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)
        for x in widgets:
            l.addWidget(x)
        l.addStretch(1)
        return w

    def _update_checkpoint_controls(self, enabled: bool) -> None:
        enabled = bool(enabled)
        for cb in (self.chk_cp_seg, self.chk_cp_thr, self.chk_cp_center, self.chk_cp_cycle, self.chk_cp_model):
            cb.setEnabled(enabled)
        for spin in (self.spin_qc_template_r, self.spin_qc_corner_tol, self.spin_qc_max_npcs):
            spin.setEnabled(enabled)

    # -------------------------
    # Public helpers
    # -------------------------
    def append_log(self, line: str) -> None:
        self.text_log.append(line)
        sb = self.text_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_status(self, s: str) -> None:
        self.lbl_status.setText(s)

    def set_analysis_enabled(self, enabled: bool) -> None:
        self.grp_analysis.setEnabled(bool(enabled))

    def set_input_enabled(self, enabled: bool) -> None:
        self.grp_input.setEnabled(bool(enabled))

    def set_roi_enabled(self, enabled: bool) -> None:
        self.grp_roi.setEnabled(bool(enabled))

    def set_busy(self, busy: bool) -> None:
        # When busy: lock everything except log
        self.grp_input.setEnabled(not busy)
        self.grp_roi.setEnabled(not busy)
        # Analysis group stays enabled state but run button is locked
        self.btn_run.setEnabled((not busy) and self.grp_analysis.isEnabled())
        self.btn_load_render.setEnabled(not busy)
        self.btn_confirm_roi.setEnabled(not busy)

    def input_path(self) -> str:
        return self.edit_input.text().strip()

    def output_dir(self) -> str:
        return self.edit_out.text().strip()

    def render_px_size_nm(self) -> float:
        return float(self.spin_px.value())

    def compute_engine(self) -> str:
        return str(self.combo_engine.currentText())

    def analysis_settings(self) -> AnalysisSettings:
        return AnalysisSettings(
            min_uncertainty_nm=float(self.spin_min_unc.value()),
            max_uncertainty_nm=float(self.spin_max_unc.value()),
            min_sigma_nm=float(self.spin_min_sigma.value()),
            max_sigma_nm=float(self.spin_max_sigma.value()),
            min_frame=int(self.spin_min_frame.value()),
            max_frame=int(self.spin_max_frame.value()),
            segmentation_engine=str(self.combo_seg.currentText()),
            sensitivity=float(self.spin_sens.value()),
            expected_npc_radius_nm=float(self.spin_expected_r.value()),
            circle_min_radius_nm=float(self.spin_rmin.value()),
            circle_max_radius_nm=float(self.spin_rmax.value()),
            min_dist_between_npcs_nm=float(self.spin_min_dist.value()),
            thres_clip=float(self.spin_thres_clip.value()),
            extract_box_radius_nm=float(self.spin_box_r.value()),
            extract_circle_radius_nm=float(self.spin_circle_r.value()),
            min_n_locs=int(self.spin_min_locs.value()),
            max_mean_sigma_nm=float(self.spin_max_mean_sigma.value()),
            max_mean_uncertainty_nm=float(self.spin_max_mean_unc.value()),
            alignment_method=str(self.combo_align.currentText()),
            min_locs_per_corner=int(self.spin_min_corner.value()),
            ele_bootstrap_samples=int(self.spin_boot.value()),
            step_by_step=bool(self.chk_step.isChecked()),
            checkpoint_segmentation=bool(self.chk_cp_seg.isChecked()),
            checkpoint_thresholds=bool(self.chk_cp_thr.isChecked()),
            checkpoint_centering=bool(self.chk_cp_center.isChecked()),
            checkpoint_cycle_qc=bool(self.chk_cp_cycle.isChecked()),
            checkpoint_model_qc=bool(self.chk_cp_model.isChecked()),
            qc_template_radius_nm=float(self.spin_qc_template_r.value()),
            qc_corner_tolerance_nm=float(self.spin_qc_corner_tol.value()),
            qc_max_npcs_per_dialog=int(self.spin_qc_max_npcs.value()),
        )

    # -------------------------
    # Browsing
    # -------------------------
    def _browse_input(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select localization table",
            "",
            "Tables (*.csv *.tsv *.txt);;All files (*)",
        )
        if p:
            self.edit_input.setText(p)

    def _browse_out(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.edit_out.setText(d)
