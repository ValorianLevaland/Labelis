from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from ..pipeline.npc import NPCRecord

# We intentionally use the Agg renderer and convert figures to QPixmaps.
from matplotlib.figure import Figure


def _npc_center_nm(n: NPCRecord) -> Tuple[float, float]:
    if n.center_fitcircle_step2_nm is not None:
        return float(n.center_fitcircle_step2_nm[0]), float(n.center_fitcircle_step2_nm[1])
    if n.center_fitcircle_step1_nm is not None:
        return float(n.center_fitcircle_step1_nm[0]), float(n.center_fitcircle_step1_nm[1])
    return float(n.npc_center_nm[0]), float(n.npc_center_nm[1])


def _mask_good_locs(n: NPCRecord) -> np.ndarray:
    """Return a boolean mask over n.x_nm/n.y_nm selecting 'good' locs.

    n.idx_good stores indices in the original localization table. n.idx_all stores
    the same for the extracted ROI. We therefore need a set-membership mapping.
    """
    try:
        idx_all = np.asarray(n.idx_all, dtype=int)
    except Exception:
        idx_all = np.arange(len(n.x_nm), dtype=int)

    try:
        idx_good = np.asarray(n.idx_good, dtype=int)
    except Exception:
        idx_good = np.asarray([], dtype=int)

    if idx_good.size == 0:
        return np.ones(idx_all.shape[0], dtype=bool)

    # np.in1d is fine here; arrays are small.
    return np.in1d(idx_all, idx_good, assume_unique=False)


def _polar_from_xy(dx: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx) + np.pi  # 0..2pi
    theta = np.mod(theta, 2.0 * np.pi)
    return rho, theta


def _aligned_theta(n: NPCRecord, theta_raw: np.ndarray) -> np.ndarray:
    th = getattr(n, "coords_polar_theta_alligned_all", None)
    if th is None:
        # Fallback: if a rotation was stored, we attempt the most common convention
        # used in the pipeline (aligned = raw + rotation) for template, and
        # aligned = raw - rotation for SMAP. Without knowing the method, we keep raw.
        rot = getattr(n, "rotation_rad", None)
        if rot is None:
            return theta_raw
        return np.mod(theta_raw + float(rot), 2.0 * np.pi)

    th = np.asarray(th, dtype=float)
    if th.shape[0] != theta_raw.shape[0]:
        return theta_raw
    return np.mod(th, 2.0 * np.pi)


def _fig_to_pixmap(fig: Figure, dpi: int = 150) -> QtGui.QPixmap:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    pm = QtGui.QPixmap()
    pm.loadFromData(buf.getvalue(), "PNG")
    return pm


def _add_template_circles_polar(
    ax,
    *,
    template_radius_nm: float,
    corner_tol_nm: float,
    n_corners: int = 8,
    color: str = "tab:blue",
    lw: float = 1.0,
    alpha: float = 0.8,
) -> None:
    """Draw tolerance circles around the 8 template corners on a polar plot.

    The circle is drawn in XY (nm) and mapped into (theta,rho).
    """
    R = float(template_radius_nm)
    tol = float(corner_tol_nm)
    tt = np.linspace(0.0, 2.0 * np.pi, 128)

    for k in range(int(n_corners)):
        theta_corner = (np.pi / 8.0) + k * (2.0 * np.pi / float(n_corners))
        phi = theta_corner - np.pi  # invert the +pi used in theta definition
        x0 = R * np.cos(phi)
        y0 = R * np.sin(phi)
        x = x0 + tol * np.cos(tt)
        y = y0 + tol * np.sin(tt)
        rho = np.sqrt(x * x + y * y)
        th = np.mod(np.arctan2(y, x) + np.pi, 2.0 * np.pi)
        ax.plot(th, rho, color=color, lw=lw, alpha=alpha)


def _add_corner_sector_lines(ax, *, r_max: float, n_corners: int = 8, color: str = "0.6", lw: float = 0.8, alpha: float = 0.5):
    for k in range(int(n_corners) + 1):
        th = k * (2.0 * np.pi / float(n_corners))
        ax.plot([th, th], [0.0, float(r_max)], color=color, lw=lw, alpha=alpha)


def _plot_polar_raw_aligned(
    n: NPCRecord,
    *,
    template_radius_nm: float,
    corner_tol_nm: float,
    title: str,
) -> Figure:
    cx, cy = _npc_center_nm(n)
    x = np.asarray(n.x_nm, dtype=float)
    y = np.asarray(n.y_nm, dtype=float)
    dx = x - cx
    dy = y - cy

    rho, theta_raw = _polar_from_xy(dx, dy)
    theta_aligned = _aligned_theta(n, theta_raw)

    good_mask = _mask_good_locs(n)

    fig = Figure(figsize=(9.2, 4.2))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    for ax in (ax1, ax2):
        ax.set_theta_zero_location("W")
        ax.set_ylim(0.0, max(float(np.nanmax(rho)) if rho.size else 1.0, template_radius_nm + 2.0 * corner_tol_nm))
        _add_template_circles_polar(
            ax,
            template_radius_nm=template_radius_nm,
            corner_tol_nm=corner_tol_nm,
        )
        _add_corner_sector_lines(ax, r_max=ax.get_ylim()[1])
        ax.grid(True, alpha=0.25)

    # Raw
    ax1.set_title("NPC raw")
    if rho.size:
        ax1.scatter(theta_raw[~good_mask], rho[~good_mask], s=10, alpha=0.25)
        ax1.scatter(theta_raw[good_mask], rho[good_mask], s=12, alpha=0.9)

    # Aligned
    ax2.set_title("NPC aligned")
    if rho.size:
        ax2.scatter(theta_aligned[~good_mask], rho[~good_mask], s=10, alpha=0.25)
        ax2.scatter(theta_aligned[good_mask], rho[good_mask], s=12, alpha=0.9)

    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    return fig


def _plot_xy_raw_aligned(
    n: NPCRecord,
    *,
    template_radius_nm: float,
    corner_tol_nm: float,
    title: str,
) -> Figure:
    cx, cy = _npc_center_nm(n)
    x = np.asarray(n.x_nm, dtype=float)
    y = np.asarray(n.y_nm, dtype=float)
    dx = x - cx
    dy = y - cy

    rho, theta_raw = _polar_from_xy(dx, dy)
    theta_aligned = _aligned_theta(n, theta_raw)

    dx_aligned = rho * np.cos(theta_aligned - np.pi)
    dy_aligned = rho * np.sin(theta_aligned - np.pi)

    good_mask = _mask_good_locs(n)

    lim = max(float(np.nanmax(rho)) if rho.size else 1.0, template_radius_nm + 2.0 * corner_tol_nm)

    fig = Figure(figsize=(9.2, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for ax in (ax1, ax2):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.25)
        # template radius
        circ = np.linspace(0.0, 2.0 * np.pi, 256)
        ax.plot(template_radius_nm * np.cos(circ), template_radius_nm * np.sin(circ), lw=1.0, alpha=0.4)
        # template corners
        for k in range(8):
            th = (np.pi / 8.0) + k * (np.pi / 4.0)
            phi = th - np.pi
            x0 = template_radius_nm * np.cos(phi)
            y0 = template_radius_nm * np.sin(phi)
            ax.plot([x0], [y0], marker="o", markersize=4)

    ax1.set_title("XY raw (centered)")
    if dx.size:
        ax1.scatter(dx[~good_mask], dy[~good_mask], s=10, alpha=0.25)
        ax1.scatter(dx[good_mask], dy[good_mask], s=12, alpha=0.9)
    ax1.plot([0.0], [0.0], marker="+", markersize=10)

    ax2.set_title("XY aligned")
    if dx_aligned.size:
        ax2.scatter(dx_aligned[~good_mask], dy_aligned[~good_mask], s=10, alpha=0.25)
        ax2.scatter(dx_aligned[good_mask], dy_aligned[good_mask], s=12, alpha=0.9)
    ax2.plot([0.0], [0.0], marker="+", markersize=10)

    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    return fig


def _plot_corner_counts(
    n: NPCRecord,
    *,
    min_locs_per_corner: int,
    title: str,
) -> Figure:
    counts = getattr(n, "locs_per_corner", None)
    if counts is None:
        counts = [0] * 8
    counts = list(counts)
    if len(counts) != 8:
        counts = (counts + [0] * 8)[:8]

    fig = Figure(figsize=(6.0, 3.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.arange(1, 9), counts)
    ax.axhline(float(min_locs_per_corner), lw=1.2)
    ax.set_xlabel("Corner")
    ax.set_ylabel("# locs")
    ax.set_title("Locs per corner")
    ax.set_xticks(np.arange(1, 9))
    ele = getattr(n, "ele", None)
    if ele is not None:
        ax.text(0.02, 0.95, f"ELE={int(ele)}", transform=ax.transAxes, va="top")
    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])
    return fig


@dataclass
class _NPCRow:
    npc_id: int
    npc_status: int
    n_locs: int
    n_good: int
    radius_step1: float
    radius_step2: float
    shift_det_step2: float
    shift_step1_step2: float
    ele: int
    sum_distance_fit_nm: float


class NPCQCTableModel(QtCore.QAbstractTableModel):
    """Table model with a checkable 'Keep' column."""

    HEADERS = [
        "Keep",
        "NPC ID",
        "Status",
        "n_locs",
        "n_good",
        "r_step1 (nm)",
        "r_step2 (nm)",
        "shift(det→step2) (nm)",
        "shift(step1→step2) (nm)",
        "ELE",
        "match (nm)",
    ]

    def __init__(self, rows: List[_NPCRow], keep_mask: List[bool]):
        super().__init__()
        self._rows = rows
        self._keep = keep_mask

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole):  # noqa: N802
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            if 0 <= section < len(self.HEADERS):
                return self.HEADERS[section]
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:  # noqa: N802
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        base = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if index.column() == 0:
            base |= QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEditable
        return base

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None

        row = self._rows[index.row()]
        col = index.column()

        if col == 0:
            if role == QtCore.Qt.CheckStateRole:
                return QtCore.Qt.Checked if self._keep[index.row()] else QtCore.Qt.Unchecked
            if role == QtCore.Qt.DisplayRole:
                return ""
            return None

        if role not in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole):
            return None

        def _fmt_float(v: float) -> str:
            if v is None or (isinstance(v, float) and (math.isnan(v) or not math.isfinite(v))):
                return ""
            return f"{float(v):.2f}"

        mapping = {
            1: str(row.npc_id),
            2: str(row.npc_status),
            3: str(row.n_locs),
            4: str(row.n_good),
            5: _fmt_float(row.radius_step1),
            6: _fmt_float(row.radius_step2),
            7: _fmt_float(row.shift_det_step2),
            8: _fmt_float(row.shift_step1_step2),
            9: "" if row.ele < 0 else str(row.ele),
            10: _fmt_float(row.sum_distance_fit_nm),
        }
        return mapping.get(col, "")

    def setData(self, index: QtCore.QModelIndex, value: Any, role: int = QtCore.Qt.EditRole) -> bool:  # noqa: N802
        if not index.isValid() or index.column() != 0:
            return False
        if role == QtCore.Qt.CheckStateRole:
            self._keep[index.row()] = value == QtCore.Qt.Checked
            self.dataChanged.emit(index, index, [QtCore.Qt.CheckStateRole])
            return True
        return False

    def sort(self, column: int, order: QtCore.Qt.SortOrder = QtCore.Qt.AscendingOrder) -> None:  # noqa: N802
        if column == 0:
            # Sort by keep flag
            key = lambda i: (not self._keep[i], self._rows[i].npc_id)
        elif column == 1:
            key = lambda i: self._rows[i].npc_id
        elif column == 2:
            key = lambda i: self._rows[i].npc_status
        elif column == 3:
            key = lambda i: self._rows[i].n_locs
        elif column == 4:
            key = lambda i: self._rows[i].n_good
        elif column == 5:
            key = lambda i: self._rows[i].radius_step1
        elif column == 6:
            key = lambda i: self._rows[i].radius_step2
        elif column == 7:
            key = lambda i: self._rows[i].shift_det_step2
        elif column == 8:
            key = lambda i: self._rows[i].shift_step1_step2
        elif column == 9:
            key = lambda i: self._rows[i].ele
        elif column == 10:
            key = lambda i: self._rows[i].sum_distance_fit_nm
        else:
            return

        reverse = order == QtCore.Qt.DescendingOrder
        idx = list(range(len(self._rows)))
        idx.sort(key=key, reverse=reverse)

        self.beginResetModel()
        self._rows = [self._rows[i] for i in idx]
        self._keep = [self._keep[i] for i in idx]
        self.endResetModel()

    def npc_id_at(self, row: int) -> Optional[int]:
        if row < 0 or row >= len(self._rows):
            return None
        return int(self._rows[row].npc_id)

    def keep_ids(self) -> List[int]:
        return [int(r.npc_id) for r, keep in zip(self._rows, self._keep) if keep]


class _NPCInspectorBase(QtWidgets.QDialog):
    """Shared UI for per-NPC inspector checkpoints."""

    actionRequested = QtCore.pyqtSignal(object)

    def __init__(self, *, parent=None, title: str = "NPC QC"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        # Internal guard to avoid emitting an Abort action when the dialog is
        # closed programmatically after a Continue/Update click.
        self._closing_action_sent = False


        # Data
        self._npcs: List[NPCRecord] = []
        self._npc_by_id: Dict[int, NPCRecord] = {}
        self._table_model: Optional[NPCQCTableModel] = None

        # Parameters (set in update_payload)
        self._template_radius_nm = 55.0
        self._corner_tol_nm = 15.0
        self._min_locs_per_corner = 10
        self._alignment_method = "smap"
        self._cycle = 1

        # ----- Layout -----
        layout = QtWidgets.QVBoxLayout(self)

        # Summary header
        self.lbl_summary = QtWidgets.QLabel("")
        self.lbl_summary.setWordWrap(True)
        layout.addWidget(self.lbl_summary)

        # Split view: table (left) and plots (right)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left: table
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        self.table = QtWidgets.QTableView()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        left_layout.addWidget(self.table, stretch=1)

        # quick actions
        qa = QtWidgets.QHBoxLayout()
        self.btn_keep_all = QtWidgets.QPushButton("Keep all")
        self.btn_reject_all = QtWidgets.QPushButton("Reject all")
        qa.addWidget(self.btn_keep_all)
        qa.addWidget(self.btn_reject_all)
        qa.addStretch(1)
        left_layout.addLayout(qa)

        splitter.addWidget(left)

        # Right: tabs with plots
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        self.tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.tabs, stretch=1)

        self.lbl_polar = QtWidgets.QLabel("Select an NPC")
        self.lbl_polar.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_polar.setScaledContents(True)
        self.tabs.addTab(self.lbl_polar, "Polar")

        self.lbl_xy = QtWidgets.QLabel("Select an NPC")
        self.lbl_xy.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_xy.setScaledContents(True)
        self.tabs.addTab(self.lbl_xy, "XY")

        self.lbl_counts = QtWidgets.QLabel("Select an NPC")
        self.lbl_counts.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_counts.setScaledContents(True)
        self.tabs.addTab(self.lbl_counts, "Corners")

        self.txt_meta = QtWidgets.QPlainTextEdit()
        self.txt_meta.setReadOnly(True)
        self.tabs.addTab(self.txt_meta, "Details")

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # Footer buttons
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.btn_abort = QtWidgets.QPushButton("Abort")
        self.btn_continue = QtWidgets.QPushButton("Continue")
        footer.addWidget(self.btn_abort)
        footer.addWidget(self.btn_continue)
        layout.addLayout(footer)

        # ----- Signals -----
        self.btn_abort.clicked.connect(self._emit_abort)
        self.btn_continue.clicked.connect(self._emit_continue)
        self.btn_keep_all.clicked.connect(self._keep_all)
        self.btn_reject_all.clicked.connect(self._reject_all)


    # --------------------------
    # Payload handling
    # --------------------------
    def update_payload(self, payload: dict) -> None:
        """Called by the controller when the pipeline pauses at a QC checkpoint."""
        self._cycle = int(payload.get("cycle", 1))
        self._alignment_method = str(payload.get("alignment_method", "smap"))
        self._min_locs_per_corner = int(payload.get("min_locs_per_corner", 10))
        self._template_radius_nm = float(payload.get("qc_template_radius_nm", 55.0))
        self._corner_tol_nm = float(payload.get("qc_corner_tolerance_nm", 15.0))

        npcs = payload.get("npcs", [])
        if not isinstance(npcs, (list, tuple)):
            npcs = []

        self._npcs = list(npcs)
        self._npc_by_id = {int(n.npc_id): n for n in self._npcs if hasattr(n, "npc_id")}

        # Which NPCs are currently kept? Pipeline can send keep ids to preserve state.
        keep_ids_payload = payload.get("keep_npc_ids", None)
        keep_ids: Optional[set[int]] = None
        if isinstance(keep_ids_payload, (list, tuple, np.ndarray)):
            try:
                keep_ids = set(int(x) for x in keep_ids_payload)
            except Exception:
                keep_ids = None

        rows: List[_NPCRow] = []
        keep_mask: List[bool] = []

        for n in self._npcs:
            nid = int(getattr(n, "npc_id", -1))
            status = int(getattr(n, "npc_status", 1))
            n_locs = int(getattr(n, "npc_n_locs", len(getattr(n, "x_nm", []))))
            n_good = int(getattr(n, "n_locs_good", n_locs))

            r1 = float(getattr(n, "radius_fitcircle_step1_nm", float("nan")) or float("nan"))
            r2 = float(getattr(n, "radius_fitcircle_step2_nm", float("nan")) or float("nan"))

            # Center shifts
            det = np.array(getattr(n, "npc_center_nm", (float("nan"), float("nan"))), dtype=float)
            c1 = np.array(getattr(n, "center_fitcircle_step1_nm", (float("nan"), float("nan"))), dtype=float)
            c2 = np.array(getattr(n, "center_fitcircle_step2_nm", (float("nan"), float("nan"))), dtype=float)

            shift_det2 = float(np.hypot(*(c2 - det))) if np.all(np.isfinite(c2)) and np.all(np.isfinite(det)) else float("nan")
            shift_12 = float(np.hypot(*(c2 - c1))) if np.all(np.isfinite(c2)) and np.all(np.isfinite(c1)) else float("nan")

            ele = int(getattr(n, "ele", -1)) if getattr(n, "ele", None) is not None else -1
            sdist = float(getattr(n, "sum_distance_fit_nm", float("nan")) or float("nan"))

            rows.append(
                _NPCRow(
                    npc_id=nid,
                    npc_status=status,
                    n_locs=n_locs,
                    n_good=n_good,
                    radius_step1=r1,
                    radius_step2=r2,
                    shift_det_step2=shift_det2,
                    shift_step1_step2=shift_12,
                    ele=ele,
                    sum_distance_fit_nm=sdist,
                )
            )

            if keep_ids is not None:
                keep_mask.append(nid in keep_ids)
            else:
                keep_mask.append(status == 1)

        self._table_model = NPCQCTableModel(rows, keep_mask)
        self.table.setModel(self._table_model)
        # (Re)connect selection handler: the selection model is recreated when the model changes.
        sel = self.table.selectionModel()
        if sel is not None:
            try:
                sel.currentRowChanged.disconnect(self._on_row_changed)
            except Exception:
                pass
            sel.currentRowChanged.connect(self._on_row_changed)


        # Select first row for immediate preview
        if rows:
            self.table.selectRow(0)
            self._update_views(self._npcs[0])
        else:
            self.lbl_polar.setText("No NPCs")
            self.lbl_xy.setText("No NPCs")
            self.lbl_counts.setText("No NPCs")
            self.txt_meta.setPlainText("")

        n_total = len(rows)
        n_keep = int(sum(keep_mask))
        self.lbl_summary.setText(
            f"Cycle {self._cycle} | NPCs in checkpoint: {n_total} (kept: {n_keep}) | "
            f"Alignment preview: {self._alignment_method}, min locs/corner: {self._min_locs_per_corner} | "
            f"Template R={self._template_radius_nm:.1f} nm, tol={self._corner_tol_nm:.1f} nm"
        )

    # --------------------------
    # Selection / rendering
    # --------------------------
    def _on_row_changed(self, current: QtCore.QModelIndex, _prev: QtCore.QModelIndex) -> None:
        if self._table_model is None:
            return
        nid = self._table_model.npc_id_at(current.row())
        if nid is None:
            return
        n = self._npc_by_id.get(int(nid), None)
        if n is None:
            return
        self._update_views(n)

    def _update_views(self, n: NPCRecord) -> None:
        title = f"NPC {int(n.npc_id)}"

        try:
            fig = _plot_polar_raw_aligned(n, template_radius_nm=self._template_radius_nm, corner_tol_nm=self._corner_tol_nm, title=title)
            pm = _fig_to_pixmap(fig)
            self.lbl_polar.setPixmap(pm)
        except Exception as e:
            self.lbl_polar.setText(f"Polar plot error: {e}")

        try:
            fig = _plot_xy_raw_aligned(n, template_radius_nm=self._template_radius_nm, corner_tol_nm=self._corner_tol_nm, title=title)
            pm = _fig_to_pixmap(fig)
            self.lbl_xy.setPixmap(pm)
        except Exception as e:
            self.lbl_xy.setText(f"XY plot error: {e}")

        try:
            fig = _plot_corner_counts(n, min_locs_per_corner=self._min_locs_per_corner, title=title)
            pm = _fig_to_pixmap(fig)
            self.lbl_counts.setPixmap(pm)
        except Exception as e:
            self.lbl_counts.setText(f"Corner count plot error: {e}")

        # Text details
        lines: List[str] = []
        lines.append(f"npc_id: {int(getattr(n, 'npc_id', -1))}")
        lines.append(f"status: {int(getattr(n, 'npc_status', -1))}")
        lines.append(f"n_locs: {int(getattr(n, 'npc_n_locs', 0))}")
        if getattr(n, "n_locs_good", None) is not None:
            lines.append(f"n_good: {int(getattr(n, 'n_locs_good', 0))}")
        if getattr(n, "center_fitcircle_step1_nm", None) is not None:
            lines.append(f"center step1 (nm): {n.center_fitcircle_step1_nm}")
        if getattr(n, "center_fitcircle_step2_nm", None) is not None:
            lines.append(f"center step2 (nm): {n.center_fitcircle_step2_nm}")
        if getattr(n, "radius_fitcircle_step1_nm", None) is not None:
            lines.append(f"radius step1 (nm): {float(n.radius_fitcircle_step1_nm):.2f}")
        if getattr(n, "radius_fitcircle_step2_nm", None) is not None:
            lines.append(f"radius step2 (nm): {float(n.radius_fitcircle_step2_nm):.2f}")
        if getattr(n, "rotation_rad", None) is not None:
            lines.append(f"rotation (rad): {float(n.rotation_rad):.3f}")
        if getattr(n, "sum_distance_fit_nm", None) is not None and np.isfinite(float(n.sum_distance_fit_nm)):
            lines.append(f"template match distance (nm): {float(n.sum_distance_fit_nm):.2f}")
        if getattr(n, "locs_per_corner", None) is not None:
            lines.append(f"locs_per_corner: {list(getattr(n, 'locs_per_corner'))}")
        if getattr(n, "ele", None) is not None:
            lines.append(f"ELE: {int(getattr(n, 'ele'))}")

        self.txt_meta.setPlainText("\n".join(lines))

    # --------------------------
    # Mass actions
    # --------------------------
    def _keep_all(self) -> None:
        if self._table_model is None:
            return
        self._table_model.beginResetModel()
        self._table_model._keep = [True] * len(self._table_model._keep)
        self._table_model.endResetModel()
        self.lbl_summary.setText(self.lbl_summary.text().replace("kept:", "kept:"))

    def _reject_all(self) -> None:
        if self._table_model is None:
            return
        self._table_model.beginResetModel()
        self._table_model._keep = [False] * len(self._table_model._keep)
        self._table_model.endResetModel()

    # --------------------------
    # Emit actions
    # --------------------------
    def _emit_abort(self) -> None:
        self._closing_action_sent = True

        self.actionRequested.emit({"action": "abort"})
        self.close()

    def _emit_continue(self) -> None:
        self._closing_action_sent = True

        if self._table_model is None:
            keep_ids: List[int] = []
        else:
            keep_ids = self._table_model.keep_ids()
        self.actionRequested.emit({"action": "continue", "keep_npc_ids": keep_ids})
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Closing via the window manager (X) is treated as Abort.
        # When the dialog is closed programmatically after a Continue/Update click,
        # we already emitted an action and must not emit Abort again.
        if not getattr(self, "_closing_action_sent", False):
            self.actionRequested.emit({"action": "abort"})
        super().closeEvent(event)



class CycleNPCQCDialog(_NPCInspectorBase):
    def __init__(self, parent=None):
        super().__init__(parent=parent, title="Cycle NPC QC (centering + model preview)")


class FinalNPCQCDialog(_NPCInspectorBase):
    """Final model QC dialog.

    Extends the base dialog with alignment parameter controls and a 'Recompute'
    (update) action.
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent, title="Final NPC QC (model mapping)")

        # Insert alignment controls above the tabs.
        top = QtWidgets.QHBoxLayout()

        self.cmb_alignment = QtWidgets.QComboBox()
        self.cmb_alignment.addItems(["smap", "template", "none"])

        self.spin_min_locs_corner = QtWidgets.QSpinBox()
        self.spin_min_locs_corner.setRange(1, 10_000)
        self.spin_min_locs_corner.setValue(10)

        self.btn_recompute = QtWidgets.QPushButton("Recompute alignment")

        top.addWidget(QtWidgets.QLabel("Alignment method:"))
        top.addWidget(self.cmb_alignment)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("min locs/corner:"))
        top.addWidget(self.spin_min_locs_corner)
        top.addSpacing(12)
        top.addWidget(self.btn_recompute)
        top.addStretch(1)

        # We need to insert this layout in the parent layout (QVBoxLayout):
        layout = self.layout()
        if isinstance(layout, QtWidgets.QVBoxLayout):
            layout.insertLayout(1, top)

        self.btn_recompute.clicked.connect(self._emit_update)

    def update_payload(self, payload: dict) -> None:
        super().update_payload(payload)
        # sync controls
        try:
            method = str(payload.get("alignment_method", "smap"))
            i = self.cmb_alignment.findText(method)
            if i >= 0:
                self.cmb_alignment.setCurrentIndex(i)
        except Exception:
            pass
        try:
            self.spin_min_locs_corner.setValue(int(payload.get("min_locs_per_corner", 10)))
        except Exception:
            pass

    def _emit_update(self) -> None:
        self._closing_action_sent = True

        # Close the dialog to avoid concurrent mutation of NPCRecord objects while the
        # pipeline recomputes alignment in the worker thread.
        keep_ids: List[int] = []
        if self._table_model is not None:
            keep_ids = self._table_model.keep_ids()

        updates = {
            "alignment_method": str(self.cmb_alignment.currentText()),
            "min_locs_per_corner": int(self.spin_min_locs_corner.value()),
        }
        self.actionRequested.emit({"action": "update", "updates": updates, "keep_npc_ids": keep_ids})
        self.close()
