"""
=============================================================================
main_app.py  —  Multi-Camera Pose Estimation Benchmarker
=============================================================================

Flux utilisateur :
    1. Charger vidéo + calibration pour chaque caméra (4 slots individuels).
    2. La Ground Truth 2D est auto-détectée depuis le chemin vidéo
       (structure Fit3D : s0X/joints3d_25/<exercice>.json) puis calculée
       dès que la calibration est chargée.
    3. Une preview de la 1ère frame s'affiche dans la grille.
    4. L'évaluation lance 3 modèles × 4 caméras = 12 flux simultanés.

Layout :
    ┌───────────────────────────────────────────────────────────────┐
    │  SLOTS CAMÉRAS (4 colonnes)                                   │
    │  [Cam1: vidéo+calib+GT]  ...  [Cam4: vidéo+calib+GT]  [▶]   │
    ├──────────────────────────────────────────┬────────────────────┤
    │  GRILLE 3×4 (BlazePose / YOLO / OpenPose)│  GRAPHIQUES        │
    ├──────────────────────────────────────────┤                    │
    │  TABLEAU DE MÉTRIQUES                    │                    │
    └──────────────────────────────────────────┴────────────────────┘
=============================================================================
"""

import sys
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QProgressBar, QFrame, QSizePolicy, QHeaderView,
    QGroupBox, QMessageBox, QGridLayout, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pose_estimators import BlazePoseEstimator, YoloPoseEstimator, OpenPoseEstimator
from ground_truth import GroundTruthLoader, CameraCalibration
from metrics import MetricsAccumulator
from export_utils import save_csv, generate_pdf


# =============================================================================
# CONSTANTES
# =============================================================================

N_CAMS        = 1
N_MODELS      = 3
MODEL_NAMES   = ["BlazePose", "YOLO-Pose", "OpenPose"]
MODEL_COLORS  = ["#00C853",   "#2196F3",   "#F44336"]
CAM_LABELS    = ["Cam 1", "Cam 2", "Cam 3", "Cam 4"]


# =============================================================================
# AUTO-DÉTECTION DE LA GROUND TRUTH (structure Fit3D)
# =============================================================================

def auto_detect_gt(video_path: str) -> str | None:
    """
    Déduit le chemin joints3d_25 depuis le chemin d'une vidéo Fit3D.
    Structure : <subject>/videos/<cam_id>/<exercice>.mp4
             → <subject>/joints3d_25/<exercice>.json
    """
    try:
        exercise    = os.path.splitext(os.path.basename(video_path))[0]
        cam_dir     = os.path.dirname(video_path)
        videos_dir  = os.path.dirname(cam_dir)
        subject_dir = os.path.dirname(videos_dir)
        gt_path = os.path.join(subject_dir, "joints3d_25", f"{exercise}.json")
        if os.path.isfile(gt_path):
            return gt_path
    except Exception:
        pass
    return None


def extract_first_frame(video_path: str) -> np.ndarray | None:
    """Extrait la première frame d'une vidéo pour la prévisualisation."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# =============================================================================
# WORKER THREAD — Évaluation multi-caméras
# =============================================================================

class MultiCamEvaluationWorker(QThread):
    """
    Thread d'évaluation : 4 vidéos × 3 modèles = 12 flux annotés.

    Signal frames_ready : list[12 frames BGR np.ndarray], frame_idx
        Ordre : [model0_cam0, model0_cam1, …, model0_cam3,
                 model1_cam0, …, model2_cam3]
    """

class SingleModelWorker(QThread):
    frame_ready    = pyqtSignal(str, np.ndarray, int)     # model_name, frame, frame_idx
    metrics_update = pyqtSignal(str, object, float, object)   # model_name, summary, fps, history
    finished_eval  = pyqtSignal(str, object, float, object)   # model_name, summary, fps, history
    progress       = pyqtSignal(str, int, int)            # model_name, current, total
    error_signal   = pyqtSignal(str, str)                 # model_name, error_msg

    UPDATE_INTERVAL = 30

    def __init__(self, video_path: str, estimator, gt_loader, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.estimator  = estimator
        self.gt_loader  = gt_loader
        self.model_name = estimator.model_name
        self._running   = True

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_signal.emit(self.model_name, f"Impossible d'ouvrir la vidéo: {self.video_path}")
            return

        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        accumulator = MetricsAccumulator(self.model_name)

        frame_idx = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                keypoints = self.estimator.detect(frame)
            except Exception as e:
                print(f"Erreur inférence {self.model_name}: {e}")
                keypoints = {}

            # Passer les keypoints à get_gt_2d pour alignement centroïdal
            gt_2d = self.gt_loader.get_gt_2d(frame_idx, pred_keypoints=keypoints)
            bbox_size = self.gt_loader.compute_bbox_from_gt(gt_2d)

            accumulator.update(keypoints, gt_2d, bbox_size)
            annotated = self.estimator.draw_skeleton(frame, keypoints, gt_2d)
            
            self.frame_ready.emit(self.model_name, annotated, frame_idx)

            if frame_idx % self.UPDATE_INTERVAL == 0:
                self.metrics_update.emit(
                    self.model_name,
                    accumulator.get_summary(),
                    self.estimator.get_fps(),
                    accumulator.get_mpjpe_history()
                )

            self.progress.emit(self.model_name, frame_idx + 1, total_frames)
            
            # x3 speed : skip 2 frames after each processed frame
            cap.read()  # skip
            cap.read()  # skip
            frame_idx += 3

        cap.release()
        
        self.finished_eval.emit(
            self.model_name,
            accumulator.get_summary(),
            self.estimator.get_fps(),
            accumulator.get_mpjpe_history()
        )


# =============================================================================
# CANVAS MATPLOTLIB
# =============================================================================

class LiveChartCanvas(FigureCanvas):
    COLORS = {"BlazePose": "#00C853", "YOLO-Pose": "#2196F3", "OpenPose": "#F44336"}

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4), dpi=85, facecolor='#1E1E2E')
        super().__init__(self.fig)
        self.setParent(parent)
        gs = self.fig.add_gridspec(1, 2, wspace=0.3, top=0.85,
                                   bottom=0.2, left=0.1, right=0.95)
        self.ax_curve = self.fig.add_subplot(gs[0])
        self.ax_bars  = self.fig.add_subplot(gs[1])
        self._style()
        self.fig.suptitle("Métriques en temps réel", fontsize=10,
                          color='white', fontweight='bold')

    def _style(self):
        for ax in [self.ax_curve, self.ax_bars]:
            ax.set_facecolor('#2A2A3E')
            ax.tick_params(colors='#AAAACC', labelsize=6)
            ax.spines[:].set_color('#444466')

    def update_charts(self, summaries, fps_dict, histories):
        self.ax_curve.cla()
        self.ax_bars.cla()
        models = list(summaries.keys())
        colors = [self.COLORS.get(m, '#AAAAAA') for m in models]

        # Graphe gauche : évolution MPJPE par frame
        for m, c in zip(models, colors):
            hist = histories.get(m, [])
            if hist:
                self.ax_curve.plot(hist, color=c, label=m, linewidth=1.2)
        self.ax_curve.set_title("MPJPE / frame", color='#CCCCEE', fontsize=7, pad=3)
        self.ax_curve.set_xlabel("Frame", color='#AAAACC', fontsize=6)
        self.ax_curve.set_ylabel("px",    color='#AAAACC', fontsize=6)
        self.ax_curve.legend(fontsize=5, facecolor='#2A2A3E',
                             labelcolor='white', framealpha=0.5)
        self.ax_curve.set_facecolor('#2A2A3E')
        self.ax_curve.tick_params(colors='#AAAACC', labelsize=5)
        self.ax_curve.spines[:].set_color('#444466')

        # Graphe droite : MPJPE moyen par modèle
        vals = [summaries[m].get("mpjpe_mean", 0) for m in models]
        bars = self.ax_bars.bar(models, vals, color=colors, edgecolor='#1E1E2E', width=0.5)
        for bar, val in zip(bars, vals):
            self.ax_bars.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                              f"{val:.1f}", ha='center', color='white',
                              fontsize=6, fontweight='bold')
        self.ax_bars.set_title("MPJPE moyen (px)", color='#CCCCEE', fontsize=7, pad=3)
        self.ax_bars.set_facecolor('#2A2A3E')
        self.ax_bars.tick_params(colors='#AAAACC', labelsize=6)
        self.ax_bars.spines[:].set_color('#444466')
        self.fig.canvas.draw_idle()




# =============================================================================
# WIDGET SLOT CAMÉRA
# =============================================================================

class CameraSlotWidget(QGroupBox):
    """
    Widget représentant un slot caméra dans la barre de contrôle.
    Contient : bouton vidéo, bouton calibration, indicateurs de statut GT.
    """

    def __init__(self, cam_idx: int, on_video_loaded, on_calib_loaded, parent=None):
        super().__init__(f"📷 Cam {cam_idx + 1}", parent)
        self.cam_idx        = cam_idx
        self.on_video_loaded = on_video_loaded
        self.on_calib_loaded = on_calib_loaded

        self.video_path = None
        self.calib_path = None
        self.gt_path    = None

        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(3)
        layout.setContentsMargins(4, 12, 4, 4)

        # Boutons vidéo + calibration côte-à-côte
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self.btn_video = QPushButton("🎬 Vidéo")
        self.btn_video.setObjectName("btn_slot")
        self.btn_video.setFixedHeight(26)
        self.btn_video.clicked.connect(self._pick_video)
        btn_row.addWidget(self.btn_video)

        self.btn_calib = QPushButton("📐 Calib")
        self.btn_calib.setObjectName("btn_slot")
        self.btn_calib.setFixedHeight(26)
        self.btn_calib.clicked.connect(self._pick_calib)
        btn_row.addWidget(self.btn_calib)
        layout.addLayout(btn_row)

        # Label vidéo
        self.lbl_video = QLabel("Vidéo : —")
        self.lbl_video.setObjectName("lbl_slot")
        self.lbl_video.setWordWrap(True)
        self.lbl_video.setMaximumHeight(32)
        layout.addWidget(self.lbl_video)

        # Label calibration
        self.lbl_calib = QLabel("Calib : —")
        self.lbl_calib.setObjectName("lbl_slot")
        self.lbl_calib.setWordWrap(True)
        self.lbl_calib.setMaximumHeight(32)
        layout.addWidget(self.lbl_calib)

        # GT status
        self.lbl_gt = QLabel("GT : —")
        self.lbl_gt.setObjectName("lbl_gt_pending")
        layout.addWidget(self.lbl_gt)

    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Vidéo — Cam {self.cam_idx + 1}", "",
            "Vidéos (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        self.video_path = path
        name = os.path.basename(path)
        self.lbl_video.setText(f"✅ {name}")
        self.lbl_video.setObjectName("lbl_slot_ok")

        # Auto-détection GT
        self.gt_path = auto_detect_gt(path)
        if self.gt_path:
            self.lbl_gt.setText("GT : ⏳ Calibration requise")
            self.lbl_gt.setObjectName("lbl_gt_pending")
        else:
            self.lbl_gt.setText("GT : ❌ Fichier joints3d introuvable")
            self.lbl_gt.setObjectName("lbl_gt_error")

        self._refresh_style()
        self.on_video_loaded(self.cam_idx, path)

    def _pick_calib(self):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Calibration — Cam {self.cam_idx + 1}", "",
            "JSON (*.json)"
        )
        if not path:
            return
        self.calib_path = path
        name = os.path.basename(path)
        self.lbl_calib.setText(f"✅ {name}")
        self.lbl_calib.setObjectName("lbl_slot_ok")

        # GT est calculable si joints3d trouvé
        if self.gt_path:
            self.lbl_gt.setText("GT : ✅ Prête (calculée par calibration)")
            self.lbl_gt.setObjectName("lbl_gt_ok")
        else:
            self.lbl_gt.setText("GT : ❌ Charger une vidéo d'abord")
            self.lbl_gt.setObjectName("lbl_gt_error")

        self._refresh_style()
        self.on_calib_loaded(self.cam_idx, path)

    def is_ready(self) -> bool:
        return all([self.video_path, self.calib_path, self.gt_path])

    def _refresh_style(self):
        self.style().unpolish(self)
        self.style().polish(self)


# =============================================================================
# FENÊTRE PRINCIPALE
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.workers    = []
        self.estimators = []
        self._current_summaries = {}
        self._current_fps       = {}
        self._current_histories = {}
        self._finished_workers  = 0

        self._setup_ui()
        self._apply_dark_theme()

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    def _setup_ui(self):
        self.setWindowTitle(
            "Comparaison Multi-Caméra — BlazePose / YOLO-Pose / OpenPose"
        )
        self.setMinimumSize(1500, 920)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        root.addWidget(self._build_control_bar())

        # Création de la zone défilante (scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)

        scroll_layout.addWidget(self._build_grid_panel())
        # Tableau de métriques supprimé de l'interface (trop encombrant)

        
        charts_panel = self._build_charts_panel()
        charts_panel.setMinimumHeight(350)
        scroll_layout.addWidget(charts_panel)

        scroll_area.setWidget(scroll_content)
        root.addWidget(scroll_area, stretch=1)

        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(340)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage(
            "Prêt — Chargez une vidéo et une calibration pour chaque caméra."
        )

        # Auto-chargement par défaut (squat) appelé une fois que l'UI est prête
        self._auto_load_default_squat()

    def _auto_load_default_squat(self):
        default_vid = "/Users/HP/Desktop/ikram/M2/PFE/models_comparaison/s03/videos/65906101LF/squat.mp4"
        default_calib = "/Users/HP/Desktop/ikram/M2/PFE/models_comparaison/s03/camera_parameters/65906101LF/squat.json"
        
        if os.path.exists(default_vid) and os.path.exists(default_calib):
            # On charge sur le 1er slot (index 0)
            slot = self.cam_slots[0]
            slot.video_path = default_vid
            slot.calib_path = default_calib
            slot.gt_path = "/Users/HP/Desktop/ikram/M2/PFE/models_comparaison/s03/joints3d_25/squat.json"
            
            slot.lbl_video.setText(f"✅ squat.mp4")
            slot.lbl_video.setObjectName("lbl_slot_ok")
            slot.lbl_calib.setText(f"✅ squat.json")
            slot.lbl_calib.setObjectName("lbl_slot_ok")
            slot.lbl_gt.setText("GT : ✅ Auto-chargé")
            slot.lbl_gt.setObjectName("lbl_gt_ok")
            slot._refresh_style()
            
            # Émet les events de chargement pour mettre à jour la logique interne
            self._on_slot_video_loaded(0, slot.video_path)
            self._on_slot_calib_loaded(0, slot.calib_path)

    # --- Barre de contrôle : 4 slots + boutons lancer/arrêter ---------------

    def _build_control_bar(self) -> QWidget:
        outer = QGroupBox("Configuration de la vidéo")
        layout = QHBoxLayout(outer)
        layout.setSpacing(10)
        
        # Slot caméra (on n'en a qu'un pour l'instant)
        self.cam_slots: list[CameraSlotWidget] = []
        for i in range(N_CAMS):
            slot = CameraSlotWidget(
                i,
                on_video_loaded=self._on_slot_video_loaded,
                on_calib_loaded=self._on_slot_calib_loaded,
            )
            slot.setMinimumWidth(200)
            layout.addWidget(slot, stretch=1)
            self.cam_slots.append(slot)

        layout.addWidget(self._vsep())

        # Boutons lancer / arrêter
        btn_col = QVBoxLayout()
        btn_col.setSpacing(8)

        self.btn_run = QPushButton("▶  Lancer\nl'évaluation")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.clicked.connect(self._start_evaluation)
        self.btn_run.setEnabled(False)
        self.btn_run.setMinimumHeight(60)
        btn_col.addWidget(self.btn_run)

        self.btn_stop = QPushButton("⏹  Arrêter")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self._stop_evaluation)
        self.btn_stop.setEnabled(False)
        btn_col.addWidget(self.btn_stop)

        layout.addLayout(btn_col)
        return outer

    # --- Grille 3 × 4 -------------------------------------------------------

    def _build_grid_panel(self) -> QWidget:
        container = QGroupBox("Visualisation — Vidéo (3 Modèles)")
        outer = QVBoxLayout(container)

        grid = QGridLayout()
        grid.setSpacing(4)

        # self.panels[model_idx][cam_idx]
        self.panels: list[list[QLabel]] = []
        for model_idx, (name, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
            model_lbl = QLabel(name)
            model_lbl.setAlignment(Qt.AlignCenter)
            model_lbl.setStyleSheet(
                f"color:{color};font-weight:bold;font-size:14px;"
            )
            grid.addWidget(model_lbl, 0, model_idx)

            row_panels = []
            for cam_idx in range(N_CAMS):
                p = self._make_video_label(f"{name}", color)
                grid.addWidget(p, 1, model_idx)
                row_panels.append(p)
            self.panels.append(row_panels)

        outer.addLayout(grid)
        return container

    def _make_video_label(self, title: str, color: str) -> QLabel:
        lbl = QLabel(title)
        lbl.setObjectName("video_panel")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl.setMinimumSize(560, 420)  # Panneaux vidéo agrandis
        lbl.setStyleSheet(
            f"QLabel#video_panel{{"
            f"background:#0D1117;border:2px solid {color};"
            f"border-radius:5px;color:{color};"
            f"font-weight:bold;font-size:10px;}}"
        )
        return lbl

    # --- Tableau de métriques ------------------------------------------------

    def _build_table_panel(self) -> QWidget:
        container = QGroupBox("Tableau de métriques (mis à jour toutes les 30 frames)")
        layout = QVBoxLayout(container)

        self.metrics_table = QTableWidget(N_MODELS, 5)
        self.metrics_table.setHorizontalHeaderLabels([
            "Modèle", "FPS", "MPJPE (px)",
            "Meilleur joint", "Pire joint"
        ])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setSelectionMode(QTableWidget.NoSelection)
        self.metrics_table.setMaximumHeight(120)

        row_colors = ["#003D1A", "#002050", "#3D0000"]
        for row, (name, clr) in enumerate(zip(MODEL_NAMES, row_colors)):
            item = QTableWidgetItem(name)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFont(QFont("Arial", 10, QFont.Bold))
            self.metrics_table.setItem(row, 0, item)
            for col in range(1, 6):
                cell = QTableWidgetItem("—")
                cell.setTextAlignment(Qt.AlignCenter)
                self.metrics_table.setItem(row, col, cell)
            for col in range(6):
                self.metrics_table.item(row, col).setBackground(QColor(clr))

        layout.addWidget(self.metrics_table)
        return container

    # --- Graphiques ----------------------------------------------------------

    def _build_charts_panel(self) -> QWidget:
        container = QGroupBox("Graphiques dynamiques")
        layout = QVBoxLayout(container)
        container.setMinimumWidth(280)
        self.chart_canvas = LiveChartCanvas(parent=container)
        layout.addWidget(self.chart_canvas)
        return container

    # -------------------------------------------------------------------------
    # CALLBACKS DES SLOTS CAMÉRA
    # -------------------------------------------------------------------------

    def _on_slot_video_loaded(self, cam_idx: int, video_path: str):
        """Appelé quand l'utilisateur charge une vidéo pour un slot."""
        # Afficher la première frame dans les 3 panneaux de la colonne
        frame = extract_first_frame(video_path)
        if frame is not None:
            for model_idx in range(N_MODELS):
                self._show_frame(self.panels[model_idx][cam_idx], frame)
        self._check_ready()

    def _on_slot_calib_loaded(self, cam_idx: int, calib_path: str):
        """Appelé quand l'utilisateur charge une calibration pour un slot."""
        self._check_ready()

    def _check_ready(self):
        """Active le bouton Lancer seulement si les 4 slots sont complets."""
        all_ready = all(slot.is_ready() for slot in self.cam_slots)
        self.btn_run.setEnabled(all_ready)

    # -------------------------------------------------------------------------
    # LANCEMENT / ARRÊT
    # -------------------------------------------------------------------------

    def _start_evaluation(self):
        if hasattr(self, 'workers'):
            for w in self.workers:
                if w.isRunning():
                    w.stop()
                    w.wait()

        self.status_bar.showMessage("Initialisation des modèles et calibrations...")
        QApplication.processEvents()

        # Chargement des GT loaders (1 par caméra, projection via calibration)
        gt_loaders = []
        for i, slot in enumerate(self.cam_slots):
            try:
                # Récupérer les dimensions de la vidéo pour le filtre GT
                cap = cv2.VideoCapture(slot.video_path)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                calib     = CameraCalibration(slot.calib_path)
                gt_loader = GroundTruthLoader(slot.gt_path, calib, img_width=w, img_height=h)
                gt_loaders.append(gt_loader)
            except Exception as e:
                QMessageBox.critical(
                    self, f"Erreur Cam {i+1}",
                    f"Impossible de charger calibration/GT pour Cam {i+1}:\n{e}"
                )
                return

        # Initialisation des estimateurs
        self.estimators = []
        for name, cls in [("BlazePose", BlazePoseEstimator),
                           ("YOLO-Pose", YoloPoseEstimator),
                           ("OpenPose",  OpenPoseEstimator)]:
            try:
                self.status_bar.showMessage(f"Chargement {name}...")
                QApplication.processEvents()
                self.estimators.append(cls())
            except Exception as e:
                QMessageBox.warning(self, name, f"Erreur init {name}:\n{e}")

        if not self.estimators:
            QMessageBox.critical(self, "Erreur",
                                 "Aucun estimateur n'a pu être initialisé.")
            return

        video_path = self.cam_slots[0].video_path
        gt_loader  = gt_loaders[0]
        
        self.workers = []
        self._finished_workers = 0
        for est in self.estimators:
            w = SingleModelWorker(video_path, est, gt_loader)
            w.frame_ready.connect(self._on_model_frame_ready)
            w.metrics_update.connect(self._on_model_metrics_update)
            w.finished_eval.connect(self._on_model_finished)
            w.progress.connect(self._on_model_progress)
            w.error_signal.connect(self._on_error)
            self.workers.append(w)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        for w in self.workers:
            w.start()
        self.status_bar.showMessage("Évaluation asynchrone des modèles en cours...")

    def _stop_evaluation(self):
        for w in self.workers:
            w.stop()
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.status_bar.showMessage("Évaluation arrêtée.")

    # -------------------------------------------------------------------------
    # SLOTS
    # -------------------------------------------------------------------------

    def _on_model_frame_ready(self, model_name: str, frame: np.ndarray, frame_idx: int):
        """Affiche la frame annotée dans le bon panneau."""
        if model_name in MODEL_NAMES:
            model_idx = MODEL_NAMES.index(model_name)
            # En mode 1 caméra, la grille est Panels[model_idx][0]
            self._show_frame(self.panels[model_idx][0], frame)

    def _show_frame(self, label: QLabel, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # La copie est obligatoire pour éviter que Python ne libère la mémoire de l'image
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix  = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    def _on_model_metrics_update(self, model_name: str, summary: dict, fps: float, history: list):
        self._current_summaries[model_name] = summary
        self._current_fps[model_name]       = fps
        self._current_histories[model_name] = history
        self._update_table(self._current_summaries, self._current_fps)
        self.chart_canvas.update_charts(self._current_summaries, self._current_fps, self._current_histories)

    def _update_table(self, summaries, fps_dict):
        if not hasattr(self, 'metrics_table'):
            return   # tableau supprimé de l'interface
        for row, model_name in enumerate(MODEL_NAMES):
            if model_name not in summaries:
                continue
            s   = summaries[model_name]
            fps = fps_dict.get(model_name, 0.0)
            data = [
                model_name,
                f"{fps:.1f}",
                f"{s.get('mpjpe_mean', 0):.2f}",
                s.get("best_joint",  "N/A").replace("_", " ").title(),
                s.get("worst_joint", "N/A").replace("_", " ").title(),
            ]
            for col, val in enumerate(data):
                item = self.metrics_table.item(row, col)
                if item:
                    item.setText(val)


    def _on_model_finished(self, model_name: str, summary: dict, fps: float, history: list):
        self._on_model_metrics_update(model_name, summary, fps, history)
        self._finished_workers += 1

        if self._finished_workers >= len(self.workers):
            self.btn_stop.setEnabled(False)
            self.btn_run.setEnabled(True)
            self.progress_bar.setVisible(False)

        out_dir = "/Users/HP/Desktop/ikram/M2/PFE/models_comparaison/output_result"
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "results.csv")
        pdf_path = os.path.join(out_dir, "report.pdf")
        try:
            save_csv(self._current_summaries, self._current_fps, csv_path)
            generate_pdf(self._current_summaries, self._current_fps, self._current_histories, pdf_path)
            self.status_bar.showMessage(
                f"✅ Terminé ! CSV : {csv_path} | PDF : {pdf_path}"
            )
            QMessageBox.information(
                self, "Évaluation terminée",
                f"Résultats exportés :\n\n📊 CSV : {csv_path}\n📄 PDF : {pdf_path}"
            )
        except Exception as e:
            self.status_bar.showMessage(f"Terminé. Erreur export : {e}")
            QMessageBox.warning(self, "Erreur export", str(e))

    def _on_model_progress(self, model_name: str, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            # Met à jour la barre de progression uniquement avec le modèle le plus avancé
            if pct > self.progress_bar.value():
                self.progress_bar.setValue(pct)
            self.status_bar.showMessage(
                f"Évaluation [{model_name}] : frame {current}/{total}"
            )

    def _on_error(self, model_name: str, msg: str):
        QMessageBox.critical(self, f"Erreur {model_name}", msg)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # -------------------------------------------------------------------------
    # UTILITAIRES
    # -------------------------------------------------------------------------

    @staticmethod
    def _vsep() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #444466;")
        return sep

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0D1117;
                color: #E0E0FF;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                background-color: #161B27;
                border: 1px solid #2A2D4A;
                border-radius: 8px;
                margin-top: 12px;
                padding: 8px;
                font-weight: bold;
                color: #8B9BFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 2px 8px;
                background-color: #0D1117;
                border-radius: 4px;
            }

            /* Slots caméra */
            QPushButton#btn_slot {
                background-color: #1A2340;
                border: 1px solid #2A3A6A;
                border-radius: 6px;
                padding: 5px 8px;
                color: #8B9BFF;
                font-size: 11px;
            }
            QPushButton#btn_slot:hover {
                background-color: #253060;
                border-color: #4A6ACA;
            }

            QLabel#lbl_slot      { color: #5A6A8A; font-size: 10px; font-style: italic; }
            QLabel#lbl_slot_ok   { color: #00C853; font-size: 10px; }
            QLabel#lbl_gt_pending{ color: #FF9800; font-size: 10px; font-style: italic; }
            QLabel#lbl_gt_ok     { color: #00C853; font-size: 10px; font-weight: bold; }
            QLabel#lbl_gt_error  { color: #F44336; font-size: 10px; }

            /* Grille */
            QLabel#grid_header {
                color: #8B9BFF;
                font-weight: bold;
                font-size: 11px;
                background: #161B27;
                border-radius: 4px;
                padding: 2px;
            }

            /* Bouton Lancer */
            QPushButton#btn_run {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1B5E20, stop:1 #2E7D32);
                border: none;
                border-radius: 8px;
                padding: 8px 18px;
                color: white;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton#btn_run:hover    { background-color: #388E3C; }
            QPushButton#btn_run:disabled { background-color: #1A2A1A; color: #444; }

            /* Bouton Arrêter */
            QPushButton#btn_stop {
                background-color: #3D1010;
                border: 1px solid #7A3030;
                border-radius: 8px;
                padding: 8px 18px;
                color: #FF8080;
                font-weight: bold;
            }
            QPushButton#btn_stop:hover { background-color: #5A1515; }

            /* Tableau */
            QTableWidget {
                background-color: #111827;
                gridline-color: #2A2D4A;
                color: #E0E0FF;
                border: 1px solid #2A2D4A;
                border-radius: 4px;
            }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section {
                background-color: #1A237E;
                color: white;
                padding: 5px;
                border: none;
                font-weight: bold;
                font-size: 10px;
            }
            QTableWidget::item:alternate { background-color: #161B27; }

            QStatusBar  { color: #8B9BFF; font-size: 11px; background: #0D1117; }
            QProgressBar {
                border: 1px solid #2A2D4A; border-radius: 4px;
                background: #161B27; text-align: center; color: white;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1B5E20, stop:1 #2196F3);
                border-radius: 3px;
            }
            QScrollBar:vertical {
                background: #161B27; width: 8px; border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #3A4A7A; border-radius: 4px;
            }
        """)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Multi-Camera Pose Benchmarker")
    app.setOrganizationName("PFE Master 2")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
