"""
=============================================================================
main_app.py
=============================================================================
Application principale PyQt5 pour la comparaison visuelle des 3 estimateurs
de pose 2D sur le dataset Fit3D.

Layout de la fenêtre :
    ┌─────────────────────────────────────────────────────┐
    │                 BARRE DE CONTRÔLE                   │
    │  [Vidéo] [Ground Truth] [Calibration] [▶ Lancer]   │
    ├────────────────────────────────────────┬────────────┤
    │  SPLIT-SCREEN (3 panneaux côte à côte) │  GRAPHIQUES│
    │  [BlazePose] [YOLO-Pose] [OpenPose]    │  - Courbe  │
    │                                        │  - Barres  │
    ├────────────────────────────────────────┤            │
    │  TABLEAU DE MÉTRIQUES (mis à jour/30f) │            │
    └────────────────────────────────────────┴────────────┘

Threading :
    - L'inférence tourne dans un QThread séparé (EvaluationWorker)
      pour ne pas bloquer l'UI.
    - Les résultats sont envoyés via des signaux Qt (thread-safe).

=============================================================================
"""

import sys
import os
import time
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QProgressBar, QSplitter, QFrame, QSizePolicy, QHeaderView,
    QStatusBar, QGroupBox, QMessageBox
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Modules locaux
from pose_estimators import (
    BlazePoseEstimator, YoloPoseEstimator, OpenPoseEstimator,
    SKELETON_CONNECTIONS
)
from ground_truth import GroundTruthLoader, CameraCalibration
from metrics import MetricsAccumulator
from export_utils import save_csv, generate_pdf


# =============================================================================
# WORKER THREAD — Évaluation en arrière-plan
# =============================================================================

class EvaluationWorker(QThread):
    """
    Thread d'évaluation indépendant de l'UI.

    Lit la vidéo frame par frame, applique les 3 estimateurs,
    compare avec la GT, accumule les métriques, et émet des signaux
    pour mettre à jour l'interface utilisateur.

    Signaux émis :
        frame_ready   : (frame_bp, frame_yolo, frame_op, frame_idx)
                         — 3 images BGR annotées + index de frame
        metrics_update: (summaries_dict, fps_dict, mpjpe_histories_dict)
                         — mis à jour toutes les UPDATE_INTERVAL frames
        finished_eval : (summaries_dict, fps_dict, mpjpe_histories_dict)
                         — émis une fois l'évaluation terminée
        progress      : (int, int) — frame_actuelle, total_frames
        error_signal  : (str) — message d'erreur
    """

    frame_ready    = pyqtSignal(object, object, object, int)
    metrics_update = pyqtSignal(dict, dict, dict)
    finished_eval  = pyqtSignal(dict, dict, dict)
    progress       = pyqtSignal(int, int)
    error_signal   = pyqtSignal(str)

    UPDATE_INTERVAL = 30   # Mise à jour métriques toutes les N frames

    def __init__(self, video_path: str, estimators: list,
                 gt_loader: GroundTruthLoader, parent=None):
        """
        Paramètres :
            video_path  : chemin vers la vidéo .mp4
            estimators  : liste des 3 PoseEstimator instanciés
            gt_loader   : instance GroundTruthLoader déjà chargée
        """
        super().__init__(parent)
        self.video_path  = video_path
        self.estimators  = estimators
        self.gt_loader   = gt_loader
        self._running    = True

    def stop(self):
        """Arrêt propre du thread depuis l'UI principale."""
        self._running = False

    def run(self):
        """Boucle principale d'évaluation."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_signal.emit(f"Impossible d'ouvrir la vidéo : {self.video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Accumulateurs de métriques (un par modèle)
        accumulators = {est.model_name: MetricsAccumulator(est.model_name)
                        for est in self.estimators}

        frame_idx = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Ground Truth 2D pour cette frame ---
            gt_2d     = self.gt_loader.get_gt_2d(frame_idx)
            bbox_size = self.gt_loader.compute_bbox_from_gt(gt_2d)

            # --- Inférence des 3 modèles ---
            annotated_frames = []
            for est in self.estimators:
                keypoints = est.detect(frame)
                accumulators[est.model_name].update(keypoints, gt_2d, bbox_size)

                # Dessin du squelette + GT sur une copie de la frame
                annotated = est.draw_skeleton(frame, keypoints, gt_2d)
                annotated_frames.append(annotated)

            # Émission des frames annotées vers l'UI
            self.frame_ready.emit(
                annotated_frames[0],   # BlazePose
                annotated_frames[1],   # YOLO-Pose
                annotated_frames[2],   # OpenPose
                frame_idx
            )

            # --- Mise à jour métriques toutes les UPDATE_INTERVAL frames ---
            if frame_idx % self.UPDATE_INTERVAL == 0:
                summaries = {m: accumulators[m].get_summary()
                             for m in accumulators}
                fps_dict  = {est.model_name: est.get_fps()
                             for est in self.estimators}
                histories = {m: accumulators[m].get_mpjpe_history()
                             for m in accumulators}
                self.metrics_update.emit(summaries, fps_dict, histories)

            self.progress.emit(frame_idx + 1, total_frames)
            frame_idx += 1

        cap.release()

        # --- Résultats finaux ---
        summaries = {m: accumulators[m].get_summary() for m in accumulators}
        fps_dict  = {est.model_name: est.get_fps() for est in self.estimators}
        histories = {m: accumulators[m].get_mpjpe_history() for m in accumulators}
        self.finished_eval.emit(summaries, fps_dict, histories)


# =============================================================================
# CANVAS MATPLOTLIB EMBARQUÉ
# =============================================================================

class LiveChartCanvas(FigureCanvas):
    """
    Widget matplotlib intégré dans PyQt5 pour les graphiques en temps réel.

    Contient 2 sous-graphiques :
        - Courbe MPJPE par frame (en haut)
        - Barres MPJPE par modèle (en bas)
    """

    COLORS = {
        "BlazePose": "#00C853",
        "YOLO-Pose": "#2196F3",
        "OpenPose":  "#F44336",
    }

    def __init__(self, parent=None, width=5, height=8, dpi=90):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor='#1E1E2E')
        super().__init__(self.fig)
        self.setParent(parent)

        # Layout 2 lignes
        gs = self.fig.add_gridspec(2, 1, hspace=0.45, top=0.93,
                                   bottom=0.1, left=0.15, right=0.95)
        self.ax_curve = self.fig.add_subplot(gs[0])
        self.ax_bars  = self.fig.add_subplot(gs[1])

        self._style_axes()

        self.fig.suptitle("Métriques en temps réel", fontsize=10,
                          color='white', fontweight='bold')

    def _style_axes(self):
        """Applique un thème sombre aux axes."""
        for ax in [self.ax_curve, self.ax_bars]:
            ax.set_facecolor('#2A2A3E')
            ax.tick_params(colors='#AAAACC', labelsize=7)
            ax.spines[:].set_color('#444466')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color('#AAAACC')

    def update_charts(self, summaries: dict, fps_dict: dict,
                      mpjpe_histories: dict):
        """
        Met à jour les deux graphiques avec les données les plus récentes.

        Paramètres :
            summaries       : dict {model_name: summary}
            fps_dict        : dict {model_name: fps}
            mpjpe_histories : dict {model_name: list[float]}
        """
        self.ax_curve.cla()
        self.ax_bars.cla()

        model_names = list(summaries.keys())
        colors      = [self.COLORS.get(m, '#AAAAAA') for m in model_names]

        # --- Courbe MPJPE par frame ---
        for m, c in zip(model_names, colors):
            hist = mpjpe_histories.get(m, [])
            if hist:
                self.ax_curve.plot(hist, color=c, label=m,
                                   linewidth=1.2, alpha=0.9)

        self.ax_curve.set_title("MPJPE par frame", color='#CCCCEE',
                                fontsize=8, pad=4)
        self.ax_curve.set_xlabel("Frame", color='#AAAACC', fontsize=7)
        self.ax_curve.set_ylabel("px", color='#AAAACC', fontsize=7)
        self.ax_curve.legend(fontsize=6, facecolor='#2A2A3E',
                             labelcolor='white', framealpha=0.5)
        self.ax_curve.set_facecolor('#2A2A3E')
        self.ax_curve.tick_params(colors='#AAAACC', labelsize=6)
        self.ax_curve.spines[:].set_color('#444466')

        # --- Barres MPJPE par modèle ---
        mpjpe_vals = [summaries[m].get("mpjpe_mean", 0) for m in model_names]
        bars = self.ax_bars.bar(model_names, mpjpe_vals, color=colors,
                                edgecolor='#1E1E2E', width=0.5)

        for bar, val in zip(bars, mpjpe_vals):
            self.ax_bars.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha='center', color='white', fontsize=7, fontweight='bold'
            )

        self.ax_bars.set_title("MPJPE moyen (px)", color='#CCCCEE',
                               fontsize=8, pad=4)
        self.ax_bars.set_facecolor('#2A2A3E')
        self.ax_bars.tick_params(colors='#AAAACC', labelsize=7)
        self.ax_bars.spines[:].set_color('#444466')

        self.fig.canvas.draw_idle()


# =============================================================================
# FENÊTRE PRINCIPALE
# =============================================================================

class MainWindow(QMainWindow):
    """
    Fenêtre principale de l'application de comparaison de pose 2D.

    Gère :
    - La barre de contrôle (chargement des fichiers + lancement)
    - Le split-screen (3 panels vidéo annotés)
    - Le tableau de métriques
    - Les graphiques temps réel
    - L'export CSV + PDF en fin d'évaluation
    """

    def __init__(self):
        super().__init__()

        # --- Chemins des fichiers chargés ---
        self.video_path   = None
        self.gt_path      = None
        self.calib_path   = None

        # --- Thread de travail ---
        self.worker = None

        # --- Estimateurs (initialisés au lancement) ---
        self.estimators = []

        # --- Données métriques courantes ---
        self._current_summaries  = {}
        self._current_fps        = {}
        self._current_histories  = {}

        self._setup_ui()
        self._apply_dark_theme()

    # -------------------------------------------------------------------------
    # INITIALISATION DE L'UI
    # -------------------------------------------------------------------------

    def _setup_ui(self):
        """Construit tous les widgets et les dispose dans la fenêtre."""
        self.setWindowTitle("Comparaison de Modèles de Pose 2D — Fit3D Evaluator")
        self.setMinimumSize(1400, 850)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # --- Barre de contrôle ---
        main_layout.addWidget(self._build_control_bar())

        # --- Corps principal : split-screen + graphiques ---
        body_layout = QHBoxLayout()
        body_layout.setSpacing(8)

        # Colonne gauche : vidéo + tableau
        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        left_col.addWidget(self._build_video_panel(), stretch=4)
        left_col.addWidget(self._build_table_panel(), stretch=1)

        left_widget = QWidget()
        left_widget.setLayout(left_col)
        body_layout.addWidget(left_widget, stretch=4)

        # Colonne droite : graphiques
        body_layout.addWidget(self._build_charts_panel(), stretch=1)

        main_layout.addLayout(body_layout, stretch=1)

        # --- Barre de statut / progression ---
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Prêt — Chargez une vidéo, la Ground Truth et la calibration caméra.")

    def _build_control_bar(self) -> QWidget:
        """Construit la barre de contrôle en haut avec les 4 boutons."""
        container = QGroupBox("Contrôles")
        layout    = QHBoxLayout(container)
        layout.setSpacing(10)

        # Bouton : Charger vidéo
        self.btn_video = QPushButton("🎬 Charger vidéo (.mp4)")
        self.btn_video.clicked.connect(self._load_video)
        self.btn_video.setObjectName("btn_load")
        layout.addWidget(self.btn_video)

        # Affichage du nom de fichier vidéo
        self.lbl_video = QLabel("Aucune vidéo")
        self.lbl_video.setObjectName("lbl_file")
        layout.addWidget(self.lbl_video)

        layout.addWidget(self._separator())

        # Bouton : Charger Ground Truth
        self.btn_gt = QPushButton("📐 Charger Ground Truth (.json)")
        self.btn_gt.clicked.connect(self._load_gt)
        self.btn_gt.setObjectName("btn_load")
        layout.addWidget(self.btn_gt)

        self.lbl_gt = QLabel("Aucune GT")
        self.lbl_gt.setObjectName("lbl_file")
        layout.addWidget(self.lbl_gt)

        layout.addWidget(self._separator())

        # Bouton : Charger Calibration
        self.btn_calib = QPushButton("📷 Charger Calibration (.json)")
        self.btn_calib.clicked.connect(self._load_calib)
        self.btn_calib.setObjectName("btn_load")
        layout.addWidget(self.btn_calib)

        self.lbl_calib = QLabel("Aucune calibration")
        self.lbl_calib.setObjectName("lbl_file")
        layout.addWidget(self.lbl_calib)

        layout.addStretch()

        # Bouton : Lancer l'évaluation
        self.btn_run = QPushButton("▶  Lancer l'évaluation")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.clicked.connect(self._start_evaluation)
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)

        # Bouton : Arrêter
        self.btn_stop = QPushButton("⏹  Arrêter")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self._stop_evaluation)
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)

        return container

    def _build_video_panel(self) -> QWidget:
        """Construit le split-screen avec 3 panels vidéo."""
        container = QGroupBox("Visualisation Pose 2D")
        layout    = QHBoxLayout(container)
        layout.setSpacing(4)

        # Panel BlazePose (vert)
        self.panel_bp   = self._make_video_label("BlazePose", "#00C853")
        # Panel YOLO-Pose (bleu)
        self.panel_yolo = self._make_video_label("YOLO-Pose", "#2196F3")
        # Panel OpenPose (rouge)
        self.panel_op   = self._make_video_label("OpenPose",  "#F44336")

        layout.addWidget(self.panel_bp)
        layout.addWidget(self.panel_yolo)
        layout.addWidget(self.panel_op)

        return container

    def _make_video_label(self, title: str, color: str) -> QLabel:
        """
        Crée un QLabel stylisé pour afficher une frame vidéo annotée.

        Paramètres :
            title : nom du modèle (affiché si pas de vidéo)
            color : couleur hex de l'encadrement
        """
        lbl = QLabel(f"{title}\n(en attente...)")
        lbl.setObjectName("video_panel")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl.setMinimumSize(350, 280)
        lbl.setStyleSheet(
            f"QLabel#video_panel {{"
            f"  background: #0D1117;"
            f"  border: 2px solid {color};"
            f"  border-radius: 6px;"
            f"  color: {color};"
            f"  font-weight: bold;"
            f"  font-size: 14px;"
            f"}}"
        )
        return lbl

    def _build_table_panel(self) -> QWidget:
        """Construit le tableau de métriques."""
        container = QGroupBox("Tableau de métriques (mis à jour toutes les 30 frames)")
        layout    = QVBoxLayout(container)

        self.metrics_table = QTableWidget(3, 6)
        self.metrics_table.setHorizontalHeaderLabels([
            "Modèle", "FPS", "MPJPE (px)", "PCK@0.1 (%)",
            "Meilleur joint", "Pire joint"
        ])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setSelectionMode(QTableWidget.NoSelection)
        self.metrics_table.setMaximumHeight(130)

        # Préremplir les noms de modèles
        model_names  = ["BlazePose", "YOLO-Pose", "OpenPose"]
        row_colors   = ["#003D1A", "#002050", "#3D0000"]
        for row, (name, clr) in enumerate(zip(model_names, row_colors)):
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

    def _build_charts_panel(self) -> QWidget:
        """Construit le panneau de graphiques matplotlib."""
        container = QGroupBox("Graphiques dynamiques")
        layout    = QVBoxLayout(container)
        container.setMinimumWidth(320)

        self.chart_canvas = LiveChartCanvas(parent=container,
                                            width=4, height=8, dpi=90)
        layout.addWidget(self.chart_canvas)
        return container

    # -------------------------------------------------------------------------
    # GESTION DES FICHIERS
    # -------------------------------------------------------------------------

    def _load_video(self):
        """Ouvre un sélecteur de fichier pour la vidéo .mp4."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une vidéo Fit3D", "",
            "Vidéos (*.mp4 *.avi *.mov)"
        )
        if path:
            self.video_path = path
            self.lbl_video.setText(os.path.basename(path))
            self._check_ready()

    def _load_gt(self):
        """Ouvre un sélecteur pour le fichier joints3d_25 .json."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger la Ground Truth (joints3d_25)", "",
            "JSON (*.json)"
        )
        if path:
            self.gt_path = path
            self.lbl_gt.setText(os.path.basename(path))
            self._check_ready()

    def _load_calib(self):
        """Ouvre un sélecteur pour le fichier de calibration caméra .json."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger la calibration caméra", "",
            "JSON (*.json)"
        )
        if path:
            self.calib_path = path
            self.lbl_calib.setText(os.path.basename(path))
            self._check_ready()

    def _check_ready(self):
        """Active le bouton Lancer si les 3 fichiers sont chargés."""
        ready = all([self.video_path, self.gt_path, self.calib_path])
        self.btn_run.setEnabled(ready)

    # -------------------------------------------------------------------------
    # LANCEMENT / ARRÊT DE L'ÉVALUATION
    # -------------------------------------------------------------------------

    def _start_evaluation(self):
        """
        Initialise les estimateurs, charge la GT, et lance le QThread.
        """
        # Nettoyage d'un éventuel thread précédent
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        self.status_bar.showMessage("Initialisation des modèles...")
        QApplication.processEvents()

        # --- Chargement calibration + GT ---
        try:
            calib     = CameraCalibration(self.calib_path)
            gt_loader = GroundTruthLoader(self.gt_path, calib)
        except Exception as e:
            QMessageBox.critical(self, "Erreur de chargement",
                                 f"Impossible de charger GT/Calibration :\n{e}")
            return

        # --- Initialisation des estimateurs ---
        self.estimators = []
        try:
            self.status_bar.showMessage("Chargement BlazePose...")
            QApplication.processEvents()
            self.estimators.append(BlazePoseEstimator())
        except Exception as e:
            QMessageBox.warning(self, "BlazePose", f"Erreur init BlazePose:\n{e}")

        try:
            self.status_bar.showMessage("Chargement YOLO-Pose...")
            QApplication.processEvents()
            self.estimators.append(YoloPoseEstimator())
        except Exception as e:
            QMessageBox.warning(self, "YOLO-Pose", f"Erreur init YOLO:\n{e}")

        try:
            self.status_bar.showMessage("Chargement OpenPose...")
            QApplication.processEvents()
            self.estimators.append(OpenPoseEstimator())
        except Exception as e:
            QMessageBox.warning(self, "OpenPose", f"Erreur init OpenPose:\n{e}")

        if not self.estimators:
            QMessageBox.critical(self, "Erreur",
                                 "Aucun estimateur n'a pu être initialisé.")
            return

        # --- Démarrage du worker ---
        self.worker = EvaluationWorker(self.video_path, self.estimators,
                                       gt_loader)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.metrics_update.connect(self._on_metrics_update)
        self.worker.finished_eval.connect(self._on_finished)
        self.worker.progress.connect(self._on_progress)
        self.worker.error_signal.connect(self._on_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker.start()
        self.status_bar.showMessage("Évaluation en cours...")

    def _stop_evaluation(self):
        """Arrête proprement le thread d'évaluation."""
        if self.worker:
            self.worker.stop()
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.status_bar.showMessage("Évaluation arrêtée.")

    # -------------------------------------------------------------------------
    # SLOTS (réponses aux signaux du worker)
    # -------------------------------------------------------------------------

    def _on_frame_ready(self, frame_bp, frame_yolo, frame_op, frame_idx):
        """
        Reçoit les 3 frames annotées et les affiche dans les panels.
        Convertit les images OpenCV (BGR numpy) en QPixmap.
        """
        self._show_frame(self.panel_bp,   frame_bp)
        self._show_frame(self.panel_yolo, frame_yolo)
        self._show_frame(self.panel_op,   frame_op)

    def _show_frame(self, label: QLabel, frame: np.ndarray):
        """
        Convertit une image OpenCV BGR en QPixmap et l'affiche dans un QLabel.

        Étapes :
        1. Conversion BGR→RGB
        2. Création d'un QImage depuis les données numpy
        3. Redimensionnement proportionnel au QLabel
        4. Affichage via setPixmap
        """
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg  = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix   = QPixmap.fromImage(qimg)

        # Redimensionnement proportionnel sans déformer
        pix_scaled = pix.scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix_scaled)

    def _on_metrics_update(self, summaries: dict, fps_dict: dict,
                           histories: dict):
        """Met à jour le tableau et les graphiques toutes les 30 frames."""
        self._current_summaries = summaries
        self._current_fps       = fps_dict
        self._current_histories = histories

        self._update_table(summaries, fps_dict)
        self.chart_canvas.update_charts(summaries, fps_dict, histories)

    def _update_table(self, summaries: dict, fps_dict: dict):
        """Met à jour le tableau de métriques."""
        model_order = ["BlazePose", "YOLO-Pose", "OpenPose"]

        for row, model_name in enumerate(model_order):
            # Chercher le bon estimateur (peut être absent si init a échoué)
            matching = [m for m in summaries if m == model_name]
            if not matching:
                continue

            s   = summaries[model_name]
            fps = fps_dict.get(model_name, 0.0)

            data = [
                model_name,
                f"{fps:.1f}",
                f"{s.get('mpjpe_mean', 0):.2f}",
                f"{s.get('pck_mean', 0):.1f}%",
                s.get("best_joint",  "N/A").replace("_", " ").title(),
                s.get("worst_joint", "N/A").replace("_", " ").title(),
            ]

            for col, val in enumerate(data):
                item = self.metrics_table.item(row, col)
                if item:
                    item.setText(val)

    def _on_finished(self, summaries: dict, fps_dict: dict, histories: dict):
        """
        Appelé quand la vidéo est entièrement traitée.
        Exporte les résultats et affiche un résumé.
        """
        self.btn_stop.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)

        self._update_table(summaries, fps_dict)
        self.chart_canvas.update_charts(summaries, fps_dict, histories)

        # Export CSV + PDF dans le même dossier que la vidéo
        out_dir = os.path.dirname(self.video_path)
        csv_path = os.path.join(out_dir, "results.csv")
        pdf_path = os.path.join(out_dir, "report.pdf")

        try:
            save_csv(summaries, fps_dict, csv_path)
            generate_pdf(summaries, fps_dict, histories, pdf_path)

            self.status_bar.showMessage(
                f"✅ Évaluation terminée ! CSV : {csv_path} | PDF : {pdf_path}"
            )
            QMessageBox.information(
                self, "Évaluation terminée",
                f"Les résultats ont été exportés :\n\n"
                f"📊 CSV : {csv_path}\n"
                f"📄 PDF : {pdf_path}"
            )
        except Exception as e:
            self.status_bar.showMessage(f"Évaluation terminée. Erreur export : {e}")
            QMessageBox.warning(self, "Erreur export", str(e))

    def _on_progress(self, current: int, total: int):
        """Met à jour la barre de progression."""
        if total > 0:
            pct = int(current / total * 100)
            self.progress_bar.setValue(pct)
            self.status_bar.showMessage(
                f"Évaluation : frame {current}/{total} ({pct}%)"
            )

    def _on_error(self, msg: str):
        """Affiche un message d'erreur."""
        QMessageBox.critical(self, "Erreur", msg)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # -------------------------------------------------------------------------
    # UTILITAIRES
    # -------------------------------------------------------------------------

    @staticmethod
    def _separator() -> QFrame:
        """Crée un séparateur vertical."""
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #444466;")
        return sep

    def _apply_dark_theme(self):
        """Applique un thème sombre complet à l'application."""
        self.setStyleSheet("""
            /* Fenêtre principale */
            QMainWindow, QWidget {
                background-color: #0D1117;
                color: #E0E0FF;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }

            /* GroupBox */
            QGroupBox {
                background-color: #161B27;
                border: 1px solid #2A2D4A;
                border-radius: 8px;
                margin-top: 10px;
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

            /* Boutons de chargement */
            QPushButton#btn_load {
                background-color: #1E2A45;
                border: 1px solid #3A4A7A;
                border-radius: 6px;
                padding: 6px 12px;
                color: #A0B0FF;
                font-weight: 500;
            }
            QPushButton#btn_load:hover {
                background-color: #2A3A60;
                border-color: #5A7ADA;
            }

            /* Bouton Lancer */
            QPushButton#btn_run {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1B5E20, stop:1 #2E7D32);
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                color: white;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton#btn_run:hover  { background-color: #388E3C; }
            QPushButton#btn_run:disabled { background-color: #1A2A1A; color: #444; }

            /* Bouton Arrêter */
            QPushButton#btn_stop {
                background-color: #3D1010;
                border: 1px solid #7A3030;
                border-radius: 8px;
                padding: 8px 20px;
                color: #FF8080;
                font-weight: bold;
            }
            QPushButton#btn_stop:hover { background-color: #5A1515; }

            /* Labels de noms de fichiers */
            QLabel#lbl_file {
                color: #7080A0;
                font-style: italic;
                font-size: 11px;
            }

            /* Tableau */
            QTableWidget {
                background-color: #111827;
                gridline-color: #2A2D4A;
                color: #E0E0FF;
                border: 1px solid #2A2D4A;
                border-radius: 4px;
            }
            QTableWidget::item { padding: 4px; }
            QTableWidget::item:selected { background-color: #2A3A60; }
            QHeaderView::section {
                background-color: #1A237E;
                color: white;
                padding: 6px;
                border: none;
                font-weight: bold;
                font-size: 11px;
            }
            QTableWidget::item:alternate { background-color: #161B27; }

            /* Barre de statut */
            QStatusBar { color: #8B9BFF; font-size: 11px; background: #0D1117; }

            /* Barre de progression */
            QProgressBar {
                border: 1px solid #2A2D4A;
                border-radius: 4px;
                background: #161B27;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1B5E20, stop:1 #2196F3);
                border-radius: 3px;
            }

            /* Scrollbar */
            QScrollBar:vertical {
                background: #161B27;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #3A4A7A;
                border-radius: 4px;
            }
        """)

    def closeEvent(self, event):
        """Arrête proprement le thread au fermeture de la fenêtre."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    """Lance l'application PyQt5."""
    app = QApplication(sys.argv)
    app.setApplicationName("Pose Model Comparator")
    app.setOrganizationName("PFE Master 2")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
