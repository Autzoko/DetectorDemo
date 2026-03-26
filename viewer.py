"""
3D ABUS Viewer — Lesion Detection Visualization

Interactive viewer for 3D breast ultrasound data with detection overlay.
Supports standard (DWBC) and oracle (GT-guided) post-processing.

Usage:
    python viewer.py                                    # empty viewer
    python viewer.py --input scan.nii.gz                # single file
    python viewer.py --input_dir /path/to/nii_files     # batch mode
    python viewer.py --pred_dir test_predictions        # with predictions

Requirements: PyQt5, matplotlib, nibabel, numpy
"""

import sys
import os
import json
import pickle
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import nibabel as nib

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QSplitter, QSlider, QLabel, QPushButton, QComboBox,
    QFileDialog, QListWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QDoubleSpinBox, QSpinBox, QStatusBar,
    QMessageBox, QProgressDialog, QFrame, QSizePolicy, QCheckBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# ---------------------------------------------------------------------------
# Optional imports from the inference project
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

try:
    from postprocess import (
        load_predictions,
        density_wbc_filter,
        DWBC_DEFAULTS,
        _calibrated_filter,
        _load_spatial_anchors,
        load_case_mapping,
        load_case_ids_from_pred_dir,
        BIRADS_CLASS_NAMES,
    )
    HAS_PP = True
except ImportError:
    HAS_PP = False
    DWBC_DEFAULTS = {
        "min_score": 0.12, "density_radius": 45,
        "density_power": 0.1, "cluster_iou": 0.2, "top_k": 0,
    }
    BIRADS_CLASS_NAMES = {0: "BI-RADS 2", 1: "BI-RADS 3", 2: "BI-RADS 4"}

# =====================================================================
# Constants & Axis Conventions
# =====================================================================
# Data array from nibabel: shape = (X, Y, Z)  (ax0=X, ax1=Y, ax2=Z)
# Box format: [z_min, y_min, z_max, y_max, x_min, x_max]

PRED_COLOR = "#00ff41"
GT_COLOR = "#ffd700"
CROSS_COLOR = "#ff4444"

# View definitions: slice axis, box index ranges, axis labels
VIEW_DEFS = [
    {   # View 0 — Axial: slice along Z (axis 2)
        "name": "Axial (Z)", "slice_axis": 2,
        "slice_box": (0, 2),        # Z range in box
        "rect_x_box": (1, 3),       # Y range → horizontal
        "rect_y_box": (4, 5),       # X range → vertical
        "cross_h_axis": 1,          # horizontal crosshair = Y
        "cross_v_axis": 0,          # vertical crosshair = X
    },
    {   # View 1 — Coronal: slice along Y (axis 1)
        "name": "Coronal (Y)", "slice_axis": 1,
        "slice_box": (1, 3),
        "rect_x_box": (0, 2),       # Z range → horizontal
        "rect_y_box": (4, 5),       # X range → vertical
        "cross_h_axis": 2,
        "cross_v_axis": 0,
    },
    {   # View 2 — Sagittal: slice along X (axis 0)
        "name": "Sagittal (X)", "slice_axis": 0,
        "slice_box": (4, 5),
        "rect_x_box": (0, 2),       # Z range → horizontal
        "rect_y_box": (1, 3),       # Y range → vertical
        "cross_h_axis": 2,
        "cross_v_axis": 1,
    },
]

# axis → which view slices that axis
AXIS_TO_VIEW = {2: 0, 1: 1, 0: 2}


# =====================================================================
# Helpers
# =====================================================================
def box_rect_on_slice(box, view_idx, slice_idx):
    """Return (x, y, w, h) rectangle for a box on the given 2D slice,
    or None if the box doesn't intersect the slice."""
    vd = VIEW_DEFS[view_idx]
    smin, smax = vd["slice_box"]
    if not (box[smin] <= slice_idx <= box[smax]):
        return None
    xmin_i, xmax_i = vd["rect_x_box"]
    ymin_i, ymax_i = vd["rect_y_box"]
    return (box[xmin_i], box[ymin_i],
            box[xmax_i] - box[xmin_i], box[ymax_i] - box[ymin_i])


def extract_boxes_from_mask(mask_data):
    """Extract bounding boxes from an instance-segmentation NIfTI mask.
    Returns list of dicts compatible with _calibrated_filter anchors."""
    boxes = []
    for rid in np.unique(mask_data):
        if rid == 0:
            continue
        coords = np.argwhere(mask_data == int(rid))
        if len(coords) == 0:
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        # data shape (X,Y,Z) → box [z_min, y_min, z_max, y_max, x_min, x_max]
        boxes.append({
            "box": [float(mins[2]), float(mins[1]), float(maxs[2]),
                    float(maxs[1]), float(mins[0]), float(maxs[0])],
            "class": 0,
            "region_id": int(rid),
        })
    return boxes


def load_pkl_raw(pkl_path):
    """Load raw predictions from a _boxes.pkl file without postprocess import."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    boxes = data["pred_boxes"]
    scores = data["pred_scores"]
    labels = data.get("pred_labels", np.zeros(len(scores), dtype=int))
    birads_probs = data.get("pred_birads_probs", None)
    birads_label = data.get("pred_birads_label", None)

    preds = []
    for i in range(len(scores)):
        p = {"box": [float(v) for v in boxes[i]],
             "score": float(scores[i]),
             "label": int(labels[i])}
        if birads_probs is not None:
            p["birads_probs"] = birads_probs
            p["birads_label"] = int(birads_label)
            p["birads_name"] = BIRADS_CLASS_NAMES.get(int(birads_label),
                                                       f"class_{birads_label}")
        preds.append(p)
    preds.sort(key=lambda x: x["score"], reverse=True)
    for idx, p in enumerate(preds):
        p["instance"] = idx
    return preds


# =====================================================================
# Dark theme
# =====================================================================
DARK_STYLE = """
QMainWindow, QWidget {background:#2b2b2b; color:#ccc;}
QLabel {color:#ccc;}
QPushButton {background:#3c3f41; color:#ccc; border:1px solid #555;
             padding:4px 10px; border-radius:3px;}
QPushButton:hover {background:#4c5052;}
QPushButton:pressed {background:#2d2d2d;}
QComboBox {background:#3c3f41; color:#ccc; border:1px solid #555; padding:3px;}
QListWidget {background:#1e1e1e; color:#ccc; border:1px solid #555;}
QListWidget::item:selected {background:#264f78;}
QTableWidget {background:#1e1e1e; color:#ccc; border:1px solid #555;
              gridline-color:#444;}
QTableWidget::item:selected {background:#264f78;}
QHeaderView::section {background:#3c3f41; color:#ccc; border:1px solid #555;
                      padding:3px;}
QSlider::groove:horizontal {height:6px; background:#555; border-radius:3px;}
QSlider::handle:horizontal {width:14px; margin:-4px 0; background:#888;
                            border-radius:7px;}
QGroupBox {border:1px solid #555; border-radius:5px; margin-top:10px;
           padding-top:10px;}
QGroupBox::title {subcontrol-origin:margin; left:10px; padding:0 5px; color:#aaa;}
QStatusBar {background:#1e1e1e; color:#888;}
QDoubleSpinBox, QSpinBox {background:#3c3f41; color:#ccc; border:1px solid #555;}
QSplitter::handle {background:#444;}
QCheckBox {color:#ccc;}
QProgressDialog {background:#2b2b2b; color:#ccc;}
"""


# =====================================================================
# Data Model
# =====================================================================
class CaseData:
    """All data for a single case."""
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.filename = ""
        self.nii_path: Optional[str] = None
        self.volume: Optional[np.ndarray] = None   # shape (X, Y, Z)
        self.spacing = (1.0, 1.0, 1.0)
        self.vmin = 0.0
        self.vmax = 1.0
        self.raw_preds: List[dict] = []
        self.filtered_preds: List[dict] = []
        self.gt_boxes: List[dict] = []
        self.has_gt = False


# =====================================================================
# SliceView
# =====================================================================
class SliceView(QWidget):
    """Orthogonal 2D slice view with box overlay."""

    slice_changed = pyqtSignal(int, int)          # view_idx, new_slice
    position_clicked = pyqtSignal(int, float, float)  # view_idx, h, v

    def __init__(self, view_idx: int, parent=None):
        super().__init__(parent)
        self.view_idx = view_idx
        self.vd = VIEW_DEFS[view_idx]
        self.slice_axis = self.vd["slice_axis"]

        self.volume: Optional[np.ndarray] = None
        self.current_slice = 0
        self.max_slice = 0
        self.pred_boxes: List[dict] = []
        self.gt_boxes: List[dict] = []
        self.vmin = 0.0
        self.vmax = 1.0
        self.crosshair_h: Optional[float] = None
        self.crosshair_v: Optional[float] = None

        # Overlay objects (for efficient update)
        self._patches: list = []
        self._lines: list = []
        self._texts: list = []

        self._build_ui()

    # ---- UI setup ------------------------------------------------
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)

        self.title_label = QLabel(self.vd["name"])
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("", 9, QFont.Bold))
        lay.addWidget(self.title_label)

        self.figure = Figure(facecolor="#1e1e1e")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_axes([0.02, 0.02, 0.96, 0.96])
        self.ax.set_facecolor("#1e1e1e")
        self.ax.axis("off")
        self.im = None
        lay.addWidget(self.canvas, stretch=1)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        self.slice_label = QLabel("0 / 0")
        self.slice_label.setMinimumWidth(55)
        self.slice_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.slider, stretch=1)
        row.addWidget(self.slice_label)
        lay.addLayout(row)

    # ---- Public API ------------------------------------------------
    def set_volume(self, vol: np.ndarray, vmin: float, vmax: float):
        self.volume = vol
        self.vmin, self.vmax = vmin, vmax
        self.max_slice = vol.shape[self.slice_axis] - 1
        self.current_slice = self.max_slice // 2
        self.slider.setMaximum(self.max_slice)
        self.slider.setValue(self.current_slice)
        self.im = None  # force recreate
        self._refresh()

    def set_boxes(self, preds, gts):
        self.pred_boxes = preds
        self.gt_boxes = gts
        self._refresh()

    def set_slice(self, idx: int):
        idx = max(0, min(idx, self.max_slice))
        if idx != self.current_slice:
            self.current_slice = idx
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
            self._refresh()

    def set_crosshair(self, h, v):
        self.crosshair_h, self.crosshair_v = h, v
        self._refresh()

    # ---- Internal --------------------------------------------------
    def _get_slice(self):
        s = self.current_slice
        if self.slice_axis == 0:
            return self.volume[s, :, :]
        elif self.slice_axis == 1:
            return self.volume[:, s, :]
        else:
            return self.volume[:, :, s]

    def _refresh(self):
        if self.volume is None:
            return
        data = self._get_slice()

        # Remove old overlays
        for obj in self._patches + self._lines + self._texts:
            try:
                obj.remove()
            except Exception:
                pass
        self._patches.clear()
        self._lines.clear()
        self._texts.clear()

        if self.im is None:
            self.im = self.ax.imshow(
                data, cmap="gray", origin="lower", aspect="equal",
                interpolation="bilinear", vmin=self.vmin, vmax=self.vmax,
            )
        else:
            self.im.set_data(data)
            self.im.set_clim(self.vmin, self.vmax)
            self.im.set_extent((-0.5, data.shape[1] - 0.5,
                                -0.5, data.shape[0] - 0.5))

        # Prediction boxes
        for p in self.pred_boxes:
            rp = box_rect_on_slice(p["box"], self.view_idx, self.current_slice)
            if rp is None:
                continue
            x, y, w, h = rp
            sc = p.get("score", 1.0)
            alpha = max(0.45, min(1.0, sc))
            r = Rectangle((x, y), w, h, lw=2, ec=PRED_COLOR, fc="none",
                           alpha=alpha)
            self.ax.add_patch(r)
            self._patches.append(r)
            lbl = f"{sc:.2f}"
            if "birads_name" in p:
                lbl = f"{p['birads_name']} {lbl}"
            t = self.ax.text(x, y + h + 1, lbl, color=PRED_COLOR,
                             fontsize=7, alpha=alpha,
                             clip_on=True)
            self._texts.append(t)

        # GT boxes
        for g in self.gt_boxes:
            rp = box_rect_on_slice(g["box"], self.view_idx, self.current_slice)
            if rp is None:
                continue
            x, y, w, h = rp
            r = Rectangle((x, y), w, h, lw=2, ec=GT_COLOR, fc="none",
                           ls="--", alpha=0.7)
            self.ax.add_patch(r)
            self._patches.append(r)

        # Crosshair
        if self.crosshair_h is not None:
            ln = self.ax.axvline(self.crosshair_h, color=CROSS_COLOR,
                                 lw=0.6, alpha=0.45)
            self._lines.append(ln)
        if self.crosshair_v is not None:
            ln = self.ax.axhline(self.crosshair_v, color=CROSS_COLOR,
                                 lw=0.6, alpha=0.45)
            self._lines.append(ln)

        self.slice_label.setText(f"{self.current_slice} / {self.max_slice}")
        self.canvas.draw_idle()

    # ---- Events ----------------------------------------------------
    def _on_slider(self, val):
        self.current_slice = val
        self._refresh()
        self.slice_changed.emit(self.view_idx, val)

    def _on_scroll(self, event):
        delta = 1 if event.button == "up" else -1
        new = max(0, min(self.current_slice + delta, self.max_slice))
        if new != self.current_slice:
            self.set_slice(new)
            self.slice_changed.emit(self.view_idx, new)

    def _on_click(self, event):
        if event.inaxes and event.xdata is not None and self.volume is not None:
            self.position_clicked.emit(self.view_idx, event.xdata, event.ydata)


# =====================================================================
# 3D Wireframe View
# =====================================================================
class View3D(QWidget):
    """3D bounding-box wireframe visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vol_shape = None
        self.pred_boxes: list = []
        self.gt_boxes: list = []
        self.slices = [0, 0, 0]  # X, Y, Z
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)

        title = QLabel("3D View")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("", 9, QFont.Bold))
        lay.addWidget(title)

        self.figure = Figure(facecolor="#1e1e1e")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d",
                                          facecolor="#1e1e1e")
        lay.addWidget(self.canvas, stretch=1)

        btn = QPushButton("Reset View")
        btn.setMaximumWidth(110)
        btn.clicked.connect(self._reset)
        lay.addWidget(btn, alignment=Qt.AlignCenter)

    # ---- Public API ------------------------------------------------
    def update_data(self, shape, preds, gts, slices):
        self.vol_shape = shape
        self.pred_boxes = preds
        self.gt_boxes = gts
        self.slices = slices
        self._draw()

    def update_slices(self, slices):
        self.slices = slices
        self._draw()

    # ---- Drawing ---------------------------------------------------
    def _draw_box(self, box, color, lw=1.5, ls="-", alpha=0.8):
        """Draw wireframe box.  3D axes: X=ax0, Y=ax1, Z=ax2."""
        z0, y0, z1, y1, x0, x1 = box[:6]
        c = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        edges = [[c[i], c[j]] for i, j in
                 [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                  (0,4),(1,5),(2,6),(3,7)]]
        lc = Line3DCollection(edges, colors=color, linewidths=lw,
                              alpha=alpha, linestyles="--" if ls == "--" else "-")
        self.ax.add_collection3d(lc)

    def _draw(self):
        if self.vol_shape is None:
            return
        self.ax.clear()
        X, Y, Z = self.vol_shape

        # Volume bbox
        self._draw_box([0, 0, Z, Y, 0, X], "#666666", lw=0.7, alpha=0.2)

        # Predictions
        for p in self.pred_boxes:
            sc = p.get("score", 1.0) if isinstance(p, dict) else 1.0
            self._draw_box(p["box"], PRED_COLOR, lw=2, alpha=max(0.35, sc * 0.9))

        # GT
        for g in self.gt_boxes:
            self._draw_box(g["box"], GT_COLOR, lw=2, ls="--", alpha=0.55)

        # Slice planes
        sx, sy, sz = self.slices
        pa = 0.06
        self.ax.add_collection3d(Poly3DCollection(
            [[(0, 0, sz), (X, 0, sz), (X, Y, sz), (0, Y, sz)]],
            alpha=pa, facecolors="#ff4444", edgecolors="#ff4444", linewidths=0.5))
        self.ax.add_collection3d(Poly3DCollection(
            [[(0, sy, 0), (X, sy, 0), (X, sy, Z), (0, sy, Z)]],
            alpha=pa, facecolors="#4488ff", edgecolors="#4488ff", linewidths=0.5))
        self.ax.add_collection3d(Poly3DCollection(
            [[(sx, 0, 0), (sx, Y, 0), (sx, Y, Z), (sx, 0, Z)]],
            alpha=pa, facecolors="#44ff44", edgecolors="#44ff44", linewidths=0.5))

        self.ax.set_xlim(0, X)
        self.ax.set_ylim(0, Y)
        self.ax.set_zlim(0, Z)
        self.ax.set_xlabel("X", color="white", fontsize=7)
        self.ax.set_ylabel("Y", color="white", fontsize=7)
        self.ax.set_zlabel("Z", color="white", fontsize=7)
        self.ax.tick_params(colors="white", labelsize=5)
        for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#444444")

        self.canvas.draw_idle()

    def _reset(self):
        self.ax.view_init(elev=25, azim=-60)
        self.canvas.draw_idle()


# =====================================================================
# Detection Table
# =====================================================================
class DetectionTable(QWidget):
    detection_clicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.detections: list = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Detections")
        lbl.setFont(QFont("", 9, QFont.Bold))
        lay.addWidget(lbl)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["#", "Score", "Agg", "Size", "Center (X,Y,Z)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellClicked.connect(self._on_click)
        lay.addWidget(self.table)

    def set_detections(self, dets: list):
        self.detections = dets
        self.table.setRowCount(len(dets))
        for i, d in enumerate(dets):
            box = d["box"]
            cx = (box[4] + box[5]) / 2
            cy = (box[1] + box[3]) / 2
            cz = (box[0] + box[2]) / 2

            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))

            sc = d.get("score", 0)
            item = QTableWidgetItem(f"{sc:.3f}")
            if sc >= 0.5:
                item.setForeground(QColor(PRED_COLOR))
            elif sc >= 0.3:
                item.setForeground(QColor("#ffff00"))
            else:
                item.setForeground(QColor("#ff6666"))
            self.table.setItem(i, 1, item)

            agg = d.get("agg_score", sc)
            self.table.setItem(i, 2, QTableWidgetItem(f"{agg:.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(
                str(d.get("cluster_size", 1))))
            self.table.setItem(i, 4, QTableWidgetItem(
                f"({cx:.0f}, {cy:.0f}, {cz:.0f})"))

    def _on_click(self, row, _col):
        if 0 <= row < len(self.detections):
            self.detection_clicked.emit(self.detections[row])


# =====================================================================
# Inference Worker (QThread)
# =====================================================================
class InferenceWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, cmd, cwd, parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.cwd = cwd

    def run(self):
        try:
            r = subprocess.run(self.cmd, capture_output=True, text=True,
                               cwd=self.cwd, timeout=3600)
            ok = r.returncode == 0
            msg = (r.stdout[-800:] if r.stdout else "") + \
                  (r.stderr[-400:] if r.stderr else "")
            self.finished.emit(ok, msg.strip() or ("Done" if ok else "Failed"))
        except Exception as e:
            self.finished.emit(False, str(e))


# =====================================================================
# Main Window
# =====================================================================
class MainWindow(QMainWindow):

    def __init__(self, args=None):
        super().__init__()
        self.setWindowTitle("3D ABUS Viewer — Lesion Detection")
        self.resize(1500, 950)

        self.cases: Dict[str, CaseData] = {}
        self.current_case: Optional[CaseData] = None
        self.pred_dir = str(SCRIPT_DIR / "test_predictions")
        self.filter_mode = "standard"
        self._worker: Optional[InferenceWorker] = None

        self._load_config()
        self._build_ui()
        self._build_toolbar()

        # Handle CLI args
        if args:
            if args.pred_dir:
                self.pred_dir = str(Path(args.pred_dir).resolve())
            QTimer.singleShot(100, lambda: self._handle_args(args))

    # ----------------------------------------------------------------
    # Config
    # ----------------------------------------------------------------
    def _load_config(self):
        cfg_path = SCRIPT_DIR / "config.json"
        self.config: dict = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                self.config = json.load(f)
        self.dwbc_params = dict(DWBC_DEFAULTS)
        dwbc_cfg = self.config.get("density_wbc", {})
        for k in self.dwbc_params:
            if k in dwbc_cfg:
                self.dwbc_params[k] = dwbc_cfg[k]

    # ----------------------------------------------------------------
    # Toolbar
    # ----------------------------------------------------------------
    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        tb.addAction("Open File", self._open_file)
        tb.addAction("Open Directory", self._open_dir)
        tb.addSeparator()
        tb.addAction("Load Predictions", self._load_pred_dir)
        tb.addAction("Load GT Mask", self._load_gt_file)
        tb.addSeparator()
        tb.addAction("Run Inference", self._run_inference)
        tb.addSeparator()

        lbl = QLabel("  Mode: ")
        tb.addWidget(lbl)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Standard (DWBC)", "Oracle (GT-guided)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        tb.addWidget(self.mode_combo)

        tb.addSeparator()
        tb.addAction("Re-process", self._rerun_pp)

    # ----------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(4, 4, 4, 4)
        main_lay.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)
        main_lay.addWidget(splitter)

        # -- Left panel --
        left = QWidget()
        left.setMinimumWidth(210)
        left.setMaximumWidth(300)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        # Case list
        grp = QGroupBox("Cases")
        gl = QVBoxLayout(grp)
        self.case_list = QListWidget()
        self.case_list.currentRowChanged.connect(self._on_case_selected)
        gl.addWidget(self.case_list)
        left_lay.addWidget(grp)

        # Detection table
        self.det_table = DetectionTable()
        self.det_table.detection_clicked.connect(self._navigate_to_det)
        left_lay.addWidget(self.det_table)

        # Parameters
        pgrp = QGroupBox("DWBC Parameters")
        pgl = QGridLayout(pgrp)
        self.param_widgets: Dict[str, QWidget] = {}
        defs = [
            ("min_score",       "Min Score",     0.0, 1.0, 0.01, float),
            ("density_radius",  "Density Radius", 1.0, 200.0, 1.0, float),
            ("density_power",   "Density Power", 0.0, 2.0, 0.01, float),
            ("cluster_iou",     "Cluster IoU",   0.0, 1.0, 0.01, float),
            ("top_k",           "Top K (0=auto)", 0, 50, 1, int),
        ]
        for row, (key, label, lo, hi, step, tp) in enumerate(defs):
            pgl.addWidget(QLabel(label), row, 0)
            if tp == float:
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setSingleStep(step)
                w.setDecimals(2)
                w.setValue(float(self.dwbc_params.get(key, lo)))
            else:
                w = QSpinBox()
                w.setRange(lo, hi)
                w.setSingleStep(step)
                w.setValue(int(self.dwbc_params.get(key, lo)))
            self.param_widgets[key] = w
            pgl.addWidget(w, row, 1)
        left_lay.addWidget(pgrp)

        # Show raw checkbox
        self.show_raw_cb = QCheckBox("Show raw predictions")
        self.show_raw_cb.stateChanged.connect(self._update_views)
        left_lay.addWidget(self.show_raw_cb)

        left_lay.addStretch()
        splitter.addWidget(left)

        # -- Right: 2x2 views --
        view_w = QWidget()
        vlay = QGridLayout(view_w)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(2)

        self.slice_views: List[SliceView] = []
        for i in range(3):
            sv = SliceView(i, self)
            sv.slice_changed.connect(self._on_slice_changed)
            sv.position_clicked.connect(self._on_position_clicked)
            self.slice_views.append(sv)

        self.view_3d = View3D(self)

        # Layout: Axial(0,0)  Sagittal(0,1)  Coronal(1,0)  3D(1,1)
        vlay.addWidget(self.slice_views[0], 0, 0)
        vlay.addWidget(self.slice_views[2], 0, 1)
        vlay.addWidget(self.slice_views[1], 1, 0)
        vlay.addWidget(self.view_3d, 1, 1)

        splitter.addWidget(view_w)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.statusBar().showMessage(
            "Ready — open a NIfTI file or directory to begin.")

    # ----------------------------------------------------------------
    # File operations
    # ----------------------------------------------------------------
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open NIfTI File", "",
            "NIfTI (*.nii *.nii.gz);;All (*)")
        if not path:
            return
        self._load_nifti(path)

    def _open_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Open NIfTI Directory")
        if not d:
            return
        nii_files = sorted(Path(d).glob("*.nii.gz")) + \
                    sorted(Path(d).glob("*.nii"))
        if not nii_files:
            QMessageBox.warning(self, "No files",
                                f"No .nii/.nii.gz files in {d}")
            return
        for f in nii_files:
            self._load_nifti(str(f), select=False)
        if self.case_list.count() > 0:
            self.case_list.setCurrentRow(0)
        self.statusBar().showMessage(f"Loaded {len(nii_files)} files from {d}")

    def _load_pred_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Predictions Directory")
        if d:
            self.pred_dir = d
            # Reload predictions for current case
            if self.current_case:
                self._load_preds_for_case(self.current_case)
                self._apply_pp(self.current_case)
                self._update_views()
            self.statusBar().showMessage(f"Predictions dir: {d}")

    def _load_gt_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GT Mask", "",
            "NIfTI (*.nii *.nii.gz);;All (*)")
        if not path or not self.current_case:
            return
        try:
            mask = nib.load(path).get_fdata()
            boxes = extract_boxes_from_mask(mask)
            self.current_case.gt_boxes = boxes
            self.current_case.has_gt = bool(boxes)
            self._apply_pp(self.current_case)
            self._update_views()
            self.statusBar().showMessage(
                f"Loaded GT: {len(boxes)} boxes from {Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load GT:\n{e}")

    # ----------------------------------------------------------------
    # NIfTI loading
    # ----------------------------------------------------------------
    def _load_nifti(self, path: str, select=True):
        try:
            case_id = self._path_to_case_id(path)
            if case_id in self.cases:
                # Already loaded — just select
                for i in range(self.case_list.count()):
                    if self.case_list.item(i).data(Qt.UserRole) == case_id:
                        self.case_list.setCurrentRow(i)
                        return
                return

            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            if data.ndim == 4:
                data = data[..., 0]

            case = CaseData(case_id)
            case.filename = Path(path).name
            case.nii_path = path
            case.volume = data
            case.spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
            case.vmin = float(np.percentile(data, 1))
            case.vmax = float(np.percentile(data, 99))

            # Try load predictions
            self._load_preds_for_case(case)

            # Try load GT from .cache/refs/
            self._load_gt_auto(case)

            # Apply post-processing
            self._apply_pp(case)

            self.cases[case_id] = case

            item_text = f"{case_id}"
            if case.filename and case.filename != case_id:
                item_text += f"  ({case.filename})"
            from PyQt5.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, case_id)
            self.case_list.addItem(item)

            if select:
                self.case_list.setCurrentRow(self.case_list.count() - 1)

            self.statusBar().showMessage(
                f"Loaded {case.filename}  shape={data.shape}  "
                f"spacing={tuple(round(s,2) for s in case.spacing)}  "
                f"preds={len(case.raw_preds)}  gt={len(case.gt_boxes)}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _path_to_case_id(self, path: str) -> str:
        name = Path(path).name
        for ext in (".nii.gz", ".nii"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        if "_0000" in name:
            name = name.replace("_0000", "")
        return name

    def _load_preds_for_case(self, case: CaseData):
        pkl_path = Path(self.pred_dir) / f"{case.case_id}_boxes.pkl"
        if not pkl_path.exists():
            case.raw_preds = []
            return
        if HAS_PP:
            case.raw_preds = load_predictions(self.pred_dir, case.case_id)
        else:
            case.raw_preds = load_pkl_raw(str(pkl_path))

    def _load_gt_auto(self, case: CaseData):
        """Try to auto-load GT from .cache/refs/."""
        cache_dir = SCRIPT_DIR / ".cache" / "refs"
        json_path = cache_dir / f"{case.case_id}.json"
        nii_path = cache_dir / f"{case.case_id}.nii.gz"
        if json_path.exists() and nii_path.exists():
            try:
                mask = nib.load(str(nii_path)).get_fdata()
                case.gt_boxes = extract_boxes_from_mask(mask)
                case.has_gt = bool(case.gt_boxes)
            except Exception:
                pass

    # ----------------------------------------------------------------
    # Post-processing
    # ----------------------------------------------------------------
    def _get_params(self) -> dict:
        return {k: w.value() for k, w in self.param_widgets.items()}

    def _apply_pp(self, case: CaseData):
        """Apply post-processing to a case."""
        if not case.raw_preds:
            case.filtered_preds = []
            return

        if self.filter_mode == "oracle" and case.gt_boxes and HAS_PP:
            iou_t = self.config.get("postprocess", {}).get("iou_thresh", 0.1)
            results = _calibrated_filter(
                case.raw_preds, case.gt_boxes, 0.9, 0.25, iou_t)
            kept = []
            for r in results:
                if r["status"] == "keep" and r["pred"] is not None:
                    p = dict(r["pred"])
                    p["agg_score"] = p["score"]
                    p["cluster_size"] = 1
                    kept.append(p)
            case.filtered_preds = kept
        elif HAS_PP:
            params = self._get_params()
            case.filtered_preds = density_wbc_filter(
                case.raw_preds, **params)
        else:
            # Fallback: simple score thresholding
            params = self._get_params()
            case.filtered_preds = [
                p for p in case.raw_preds
                if p["score"] >= params["min_score"]
            ]

    def _rerun_pp(self):
        if self.current_case is None:
            return
        self._apply_pp(self.current_case)
        self._update_views()
        n = len(self.current_case.filtered_preds)
        self.statusBar().showMessage(
            f"Post-processing: {n} detections  "
            f"(mode={self.filter_mode})")

    def _on_mode_change(self, idx):
        self.filter_mode = "oracle" if idx == 1 else "standard"
        if self.current_case:
            if self.filter_mode == "oracle" and not self.current_case.has_gt:
                self.statusBar().showMessage(
                    "No GT available — falling back to standard mode")
            self._apply_pp(self.current_case)
            self._update_views()

    # ----------------------------------------------------------------
    # Case selection
    # ----------------------------------------------------------------
    def _on_case_selected(self, row):
        if row < 0:
            return
        item = self.case_list.item(row)
        case_id = item.data(Qt.UserRole)
        case = self.cases.get(case_id)
        if case is None:
            return
        self.current_case = case

        vmin, vmax = case.vmin, case.vmax
        for sv in self.slice_views:
            sv.set_volume(case.volume, vmin, vmax)
        self._update_views()
        self.statusBar().showMessage(
            f"{case.case_id}  shape={case.volume.shape}  "
            f"preds={len(case.raw_preds)}  "
            f"filtered={len(case.filtered_preds)}  "
            f"gt={len(case.gt_boxes)}")

    # ----------------------------------------------------------------
    # View updates
    # ----------------------------------------------------------------
    def _update_views(self, _=None):
        if self.current_case is None:
            return
        case = self.current_case

        show_raw = self.show_raw_cb.isChecked()
        display_preds = case.raw_preds if show_raw else case.filtered_preds

        for sv in self.slice_views:
            sv.set_boxes(display_preds, case.gt_boxes)
        self._update_crosshairs()
        self._update_3d()
        self.det_table.set_detections(display_preds)

    def _update_crosshairs(self):
        for vi, sv in enumerate(self.slice_views):
            vd = VIEW_DEFS[vi]
            h_view = AXIS_TO_VIEW[vd["cross_h_axis"]]
            v_view = AXIS_TO_VIEW[vd["cross_v_axis"]]
            sv.set_crosshair(
                self.slice_views[h_view].current_slice,
                self.slice_views[v_view].current_slice,
            )

    def _update_3d(self):
        if self.current_case is None or self.current_case.volume is None:
            return
        case = self.current_case
        show_raw = self.show_raw_cb.isChecked()
        display_preds = case.raw_preds if show_raw else case.filtered_preds
        slices = [
            self.slice_views[2].current_slice,  # X (sagittal)
            self.slice_views[1].current_slice,  # Y (coronal)
            self.slice_views[0].current_slice,  # Z (axial)
        ]
        self.view_3d.update_data(
            case.volume.shape, display_preds, case.gt_boxes, slices)

    # ----------------------------------------------------------------
    # Navigation
    # ----------------------------------------------------------------
    def _on_slice_changed(self, view_idx, _val):
        self._update_crosshairs()
        if self.current_case and self.current_case.volume is not None:
            slices = [
                self.slice_views[2].current_slice,
                self.slice_views[1].current_slice,
                self.slice_views[0].current_slice,
            ]
            self.view_3d.update_slices(slices)

    def _on_position_clicked(self, view_idx, h_pos, v_pos):
        vd = VIEW_DEFS[view_idx]
        h_view = AXIS_TO_VIEW[vd["cross_h_axis"]]
        v_view = AXIS_TO_VIEW[vd["cross_v_axis"]]
        self.slice_views[h_view].set_slice(int(round(h_pos)))
        self.slice_views[v_view].set_slice(int(round(v_pos)))
        self._update_crosshairs()
        if self.current_case:
            slices = [
                self.slice_views[2].current_slice,
                self.slice_views[1].current_slice,
                self.slice_views[0].current_slice,
            ]
            self.view_3d.update_slices(slices)

    def _navigate_to_det(self, det: dict):
        box = det["box"]
        z_c = int((box[0] + box[2]) / 2)
        y_c = int((box[1] + box[3]) / 2)
        x_c = int((box[4] + box[5]) / 2)
        self.slice_views[0].set_slice(z_c)  # Axial → Z
        self.slice_views[1].set_slice(y_c)  # Coronal → Y
        self.slice_views[2].set_slice(x_c)  # Sagittal → X
        self._update_crosshairs()
        if self.current_case:
            self.view_3d.update_slices([x_c, y_c, z_c])
        self.statusBar().showMessage(
            f"Detection: score={det.get('score',0):.3f}  "
            f"center=({x_c},{y_c},{z_c})")

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    def _run_inference(self):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "Inference already running.")
            return

        # Build command
        cmd = [sys.executable, str(SCRIPT_DIR / "predict.py"),
               "--config", str(SCRIPT_DIR / "config.json")]

        # Check for --no_tta option
        reply = QMessageBox.question(
            self, "Test-Time Augmentation",
            "Disable TTA? (8x faster, slightly less accurate)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            cmd.append("--no_tta")

        # Single case?
        if self.current_case:
            reply2 = QMessageBox.question(
                self, "Scope",
                f"Run only on current case ({self.current_case.case_id})?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply2 == QMessageBox.Yes:
                cmd.extend(["--case_id", self.current_case.case_id])

        self.statusBar().showMessage(f"Running inference: {' '.join(cmd)}")

        self._worker = InferenceWorker(cmd, str(SCRIPT_DIR), self)
        self._worker.finished.connect(self._on_inference_done)
        self._worker.start()

    def _on_inference_done(self, ok, msg):
        self._worker = None
        if ok:
            self.statusBar().showMessage("Inference complete — reloading...")
            # Reload predictions for all loaded cases
            for case in self.cases.values():
                self._load_preds_for_case(case)
                self._apply_pp(case)
            self._update_views()
            self.statusBar().showMessage(
                f"Inference done. Predictions reloaded.")
        else:
            QMessageBox.warning(self, "Inference Failed", msg[:1000])
            self.statusBar().showMessage("Inference failed.")

    # ----------------------------------------------------------------
    # CLI args handler
    # ----------------------------------------------------------------
    def _handle_args(self, args):
        if args.input:
            self._load_nifti(args.input)
        if args.input_dir:
            nii_files = sorted(Path(args.input_dir).glob("*.nii.gz")) + \
                        sorted(Path(args.input_dir).glob("*.nii"))
            for f in nii_files:
                self._load_nifti(str(f), select=False)
            if self.case_list.count() > 0:
                self.case_list.setCurrentRow(0)


# =====================================================================
# Entry point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="3D ABUS Viewer — Lesion Detection Visualization")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Single NIfTI file to open")
    parser.add_argument("--input_dir", "-d", type=str, default=None,
                        help="Directory of NIfTI files (batch mode)")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Predictions directory (default: test_predictions/)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    app.setStyle("Fusion")

    window = MainWindow(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
