from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from PyQt5 import QtCore, QtWidgets

import napari

from pipeline.io import load_localizations
from pipeline.render import render_dispatch


@dataclass
class RenderResult:
    input_path: str
    image: np.ndarray
    pixel_size_nm: float
    engine: str
    n_localizations: int
    x_offset_nm: float
    y_offset_nm: float
    shift_to_origin: bool


def save_render_tiff(
    out_path: str | Path,
    img: np.ndarray,
    *,
    pixel_size_nm: float,
    meta: dict,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # OME physical size in µm (most microscopy tools understand this).
    px_um = float(pixel_size_nm) / 1000.0
    ome_meta = {
        "axes": "YX",
        "PhysicalSizeX": px_um,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": px_um,
        "PhysicalSizeYUnit": "µm",
    }

    # Classic TIFF resolution tags (pixels per cm)
    px_cm = float(pixel_size_nm) * 1e-7
    ppcm = (1.0 / px_cm) if px_cm > 0 else 1.0

    tifffile.imwrite(
        out_path,
        np.asarray(img, dtype=np.float32),
        ome=True,
        metadata=ome_meta,
        description=json.dumps(meta, indent=2),
        resolution=(ppcm, ppcm),
        resolutionunit="CENTIMETER",
    )

    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class _RenderWorker(QtCore.QObject):
    log_line = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)  # RenderResult
    finished_error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        input_path: str,
        pixel_size_nm: float,
        engine: str,
        signal_half_width_px: int,
        sigma_is_variance: bool,
        compat_kernel_crop: bool,
        turbo_blur_sigma_px: float,
        shift_to_origin: bool,
    ):
        super().__init__()
        self.input_path = input_path
        self.pixel_size_nm = float(pixel_size_nm)
        self.engine = str(engine)
        self.signal_half_width_px = int(signal_half_width_px)
        self.sigma_is_variance = bool(sigma_is_variance)
        self.compat_kernel_crop = bool(compat_kernel_crop)
        self.turbo_blur_sigma_px = float(turbo_blur_sigma_px)
        self.shift_to_origin = bool(shift_to_origin)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            self.log_line.emit(f"Loading localizations: {self.input_path}")
            df = load_localizations(self.input_path)

            x_offset_nm = float(df['x_nm_'].min())
            y_offset_nm = float(df['y_nm_'].min())

            if self.shift_to_origin:
                df = df.copy()
                df["x_nm_"] = df["x_nm_"] - x_offset_nm
                df["y_nm_"] = df["y_nm_"] - y_offset_nm
            else:
                x_offset_nm = 0.0
                y_offset_nm = 0.0

            self.log_line.emit(f"Localizations loaded: {len(df)} rows")
            self.log_line.emit(
                f"Rendering: engine={self.engine}, px={self.pixel_size_nm:g} nm, "
                f"halfWidth={self.signal_half_width_px}px"
            )

            img = render_dispatch(
                engine=self.engine,
                x_nm=df["x_nm_"].to_numpy(),
                y_nm=df["y_nm_"].to_numpy(),
                sigma_nm=df["sigma_nm_"].to_numpy(),
                pixel_size_nm=self.pixel_size_nm,
                grid_size_px=None,
                signal_half_width_px=self.signal_half_width_px,
                sigma_is_variance=self.sigma_is_variance,
                compat_kernel_crop=self.compat_kernel_crop,
                turbo_blur_sigma_px=self.turbo_blur_sigma_px,
            )

            res = RenderResult(
                input_path=self.input_path,
                image=np.asarray(img),
                pixel_size_nm=self.pixel_size_nm,
                engine=self.engine,
                n_localizations=int(len(df)),
                x_offset_nm=float(x_offset_nm),
                y_offset_nm=float(y_offset_nm),
                shift_to_origin=bool(self.shift_to_origin),
            )
            self.finished_ok.emit(res)

        except Exception:
            self.finished_error.emit(traceback.format_exc())


class RenderOnlyDockWidget(QtWidgets.QWidget):
    sig_render = QtCore.pyqtSignal()
    sig_save = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(420)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Input
        grp_in = QtWidgets.QGroupBox("Input")
        root.addWidget(grp_in)
        g = QtWidgets.QFormLayout(grp_in)

        self.edit_input = QtWidgets.QLineEdit()
        self.btn_browse_input = QtWidgets.QPushButton("Browse…")
        self.btn_browse_input.clicked.connect(self._browse_input)

        row_in = QtWidgets.QHBoxLayout()
        row_in.addWidget(self.edit_input, 1)
        row_in.addWidget(self.btn_browse_input, 0)
        g.addRow("Localization CSV/TSV:", row_in)

        self.edit_out = QtWidgets.QLineEdit()
        self.btn_browse_out = QtWidgets.QPushButton("Browse…")
        self.btn_browse_out.clicked.connect(self._browse_out)

        row_out = QtWidgets.QHBoxLayout()
        row_out.addWidget(self.edit_out, 1)
        row_out.addWidget(self.btn_browse_out, 0)
        g.addRow("Output TIFF:", row_out)

        # Rendering params
        grp_r = QtWidgets.QGroupBox("Rendering")
        root.addWidget(grp_r)
        gr = QtWidgets.QFormLayout(grp_r)

        self.spin_px = QtWidgets.QDoubleSpinBox()
        self.spin_px.setDecimals(2)
        self.spin_px.setRange(0.5, 200.0)
        self.spin_px.setSingleStep(0.5)
        self.spin_px.setValue(10.0)
        gr.addRow("Pixel size (nm):", self.spin_px)

        self.combo_engine = QtWidgets.QComboBox()
        self.combo_engine.addItems(["numba", "turbo_bin_blur", "reference_bruteforce"])
        self.combo_engine.setCurrentText("numba")
        gr.addRow("Engine:", self.combo_engine)

        self.spin_halfwidth = QtWidgets.QSpinBox()
        self.spin_halfwidth.setRange(1, 200)
        self.spin_halfwidth.setValue(10)
        gr.addRow("Kernel half-width (px):", self.spin_halfwidth)

        self.chk_sigma_is_variance = QtWidgets.QCheckBox("Sigma column is variance (Labelis/MATLAB-compatible)")
        self.chk_sigma_is_variance.setChecked(True)
        gr.addRow(self.chk_sigma_is_variance)

        self.chk_compat_crop = QtWidgets.QCheckBox("Compat kernel crop (MATLAB boundary behavior)")
        self.chk_compat_crop.setChecked(True)
        gr.addRow(self.chk_compat_crop)

        self.spin_turbo_blur = QtWidgets.QDoubleSpinBox()
        self.spin_turbo_blur.setDecimals(2)
        self.spin_turbo_blur.setRange(0.0, 50.0)
        self.spin_turbo_blur.setSingleStep(0.1)
        self.spin_turbo_blur.setValue(1.2)
        gr.addRow("Turbo blur sigma (px):", self.spin_turbo_blur)

        self.chk_shift = QtWidgets.QCheckBox("Shift XY to origin (recommended)")
        self.chk_shift.setChecked(True)
        gr.addRow(self.chk_shift)

        # Actions
        row_btn = QtWidgets.QHBoxLayout()
        self.btn_render = QtWidgets.QPushButton("Render (preview)")
        self.btn_render.clicked.connect(self.sig_render.emit)
        row_btn.addWidget(self.btn_render, 1)

        self.btn_save = QtWidgets.QPushButton("Save TIFF")
        self.btn_save.clicked.connect(self.sig_save.emit)
        self.btn_save.setEnabled(False)
        row_btn.addWidget(self.btn_save, 1)
        root.addLayout(row_btn)

        # Log
        grp_log = QtWidgets.QGroupBox("Log")
        root.addWidget(grp_log, 1)
        vlog = QtWidgets.QVBoxLayout(grp_log)
        self.text_log = QtWidgets.QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        vlog.addWidget(self.text_log, 1)

        self.lbl_status = QtWidgets.QLabel("Idle.")
        root.addWidget(self.lbl_status)

    def _browse_input(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select localization table", "", "Tables (*.csv *.tsv *.txt);;All files (*)"
        )
        if p:
            self.edit_input.setText(p)
            # auto output name next to input if empty
            if not self.edit_out.text().strip():
                stem = Path(p).with_suffix("").name
                out = Path(p).with_suffix("")  # remove extension
                out = out.with_name(stem + "_reconstruction.tif")
                self.edit_out.setText(str(out))

    def _browse_out(self) -> None:
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save reconstruction TIFF", "", "TIFF (*.tif *.tiff);;All files (*)"
        )
        if p:
            self.edit_out.setText(p)

    def input_path(self) -> str:
        return self.edit_input.text().strip()

    def output_path(self) -> str:
        return self.edit_out.text().strip()

    def render_px_size_nm(self) -> float:
        return float(self.spin_px.value())

    def engine(self) -> str:
        return str(self.combo_engine.currentText())

    def halfwidth_px(self) -> int:
        return int(self.spin_halfwidth.value())

    def sigma_is_variance(self) -> bool:
        return bool(self.chk_sigma_is_variance.isChecked())

    def compat_kernel_crop(self) -> bool:
        return bool(self.chk_compat_crop.isChecked())

    def turbo_blur_sigma_px(self) -> float:
        return float(self.spin_turbo_blur.value())

    def shift_to_origin(self) -> bool:
        return bool(self.chk_shift.isChecked())

    def append_log(self, line: str) -> None:
        self.text_log.append(line)
        sb = self.text_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_status(self, s: str) -> None:
        self.lbl_status.setText(s)

    def set_busy(self, busy: bool) -> None:
        self.btn_render.setEnabled(not busy)
        self.btn_save.setEnabled((not busy) and self.btn_save.isEnabled())

    def set_can_save(self, can_save: bool) -> None:
        self.btn_save.setEnabled(bool(can_save))


class RenderOnlyController(QtCore.QObject):
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.logger = logger

        self.viewer = napari.Viewer(title="Labelis – Render Only")
        self.dock = RenderOnlyDockWidget()
        self.viewer.window.add_dock_widget(self.dock, area="right", name="Render Only")

        self.dock.sig_render.connect(self.render_preview)
        self.dock.sig_save.connect(self.save_tiff)

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_RenderWorker] = None

        self._last_result: Optional[RenderResult] = None

        # Nice defaults
        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = "nm"
        except Exception:
            pass

        self.log("Ready. Select CSV -> Render -> Save TIFF.")

    def show(self) -> None:
        qt_window = getattr(self.viewer.window, "_qt_window", None)
        if qt_window is not None:
            qt_window.show()

    @QtCore.pyqtSlot(str)
    def log(self, msg: str) -> None:
        self.logger.info(msg)
        self.dock.append_log(msg)

    def _error(self, title: str, msg: str) -> None:
        self.logger.error(f"{title}: {msg}")
        QtWidgets.QMessageBox.critical(getattr(self.viewer.window, "_qt_window", None), title, msg)

    @QtCore.pyqtSlot()
    def render_preview(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._error("Render Only", "A render job is already running.")
            return

        in_path = self.dock.input_path()
        if not in_path:
            self._error("Render Only", "Select an input CSV/TSV first.")
            return
        if not Path(in_path).exists():
            self._error("Render Only", f"Input file does not exist:\n{in_path}")
            return

        self.dock.set_status("Rendering…")
        self.dock.set_busy(True)
        self.dock.set_can_save(False)

        # Worker
        self._thread = QtCore.QThread(self)
        self._worker = _RenderWorker(
            input_path=in_path,
            pixel_size_nm=self.dock.render_px_size_nm(),
            engine=self.dock.engine(),
            signal_half_width_px=self.dock.halfwidth_px(),
            sigma_is_variance=self.dock.sigma_is_variance(),
            compat_kernel_crop=self.dock.compat_kernel_crop(),
            turbo_blur_sigma_px=self.dock.turbo_blur_sigma_px(),
            shift_to_origin=self.dock.shift_to_origin(),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_line.connect(self.log)
        self._worker.finished_ok.connect(self._render_ok)
        self._worker.finished_error.connect(self._render_error)

        # Cleanup
        self._worker.finished_ok.connect(self._thread.quit)
        self._worker.finished_error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._cleanup_worker)

        self._thread.start()

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
    def _render_ok(self, res_obj: object) -> None:
        res: RenderResult = res_obj  # type: ignore
        self._last_result = res

        # Replace existing image layer
        if "Reconstruction" in self.viewer.layers:
            try:
                self.viewer.layers.remove(self.viewer.layers["Reconstruction"])
            except Exception:
                pass

        self.viewer.add_image(
            res.image,
            name="Reconstruction",
            scale=(res.pixel_size_nm, res.pixel_size_nm),
        )

        # Optional: show scale bar unit
        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = "nm"
        except Exception:
            pass

        self.dock.set_busy(False)
        self.dock.set_can_save(True)
        self.dock.set_status(f"Rendered {res.image.shape[1]}×{res.image.shape[0]} px.")

        self.log(
            f"Render done. Shape={res.image.shape}, px={res.pixel_size_nm:g} nm, "
            f"engine={res.engine}, locs={res.n_localizations}"
        )

    @QtCore.pyqtSlot(str)
    def _render_error(self, tb: str) -> None:
        self.dock.set_busy(False)
        self.dock.set_status("Error.")
        self.log("ERROR:\n" + tb)
        self._error("Render Only – Render error", tb)

    @QtCore.pyqtSlot()
    def save_tiff(self) -> None:
        if self._last_result is None:
            self._error("Render Only", "Nothing to save yet. Render first.")
            return

        out_path = self.dock.output_path()
        if not out_path:
            self._error("Render Only", "Select an output TIFF path first.")
            return

        res = self._last_result
        meta = {
            "input": str(Path(res.input_path).resolve()),
            "output": str(Path(out_path).resolve()),
            "n_localizations": int(res.n_localizations),
            "render_px_size_nm": float(res.pixel_size_nm),
            "engine": str(res.engine),
            "shift_to_origin": bool(res.shift_to_origin),
            "x_offset_nm": float(res.x_offset_nm),
            "y_offset_nm": float(res.y_offset_nm),
            "image_shape": [int(res.image.shape[0]), int(res.image.shape[1])],
        }

        try:
            save_render_tiff(out_path, res.image, pixel_size_nm=res.pixel_size_nm, meta=meta)
        except Exception as e:
            self._error("Render Only – Save failed", str(e))
            return

        self.dock.set_status("Saved.")
        self.log(f"Saved TIFF: {out_path}")
        self.log(f"Saved sidecar JSON: {out_path}.json")


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("labelis_render_only")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
    return logger


def main() -> int:
    logger = _setup_logging()
    logger.info("Starting Labelis Render-Only…")

    try:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)

        ctrl = RenderOnlyController(logger=logger)
        ctrl.show()

        return int(app.exec_())

    except Exception:
        tb = traceback.format_exc()
        logger.error("Fatal error:\n" + tb)
        try:
            QtWidgets.QMessageBox.critical(None, "Render Only – Fatal error", tb)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
