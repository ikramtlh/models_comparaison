"""
=============================================================================
ground_truth.py
=============================================================================
Ce module gère la génération de la Ground Truth 2D à partir des données Fit3D.

Fit3D fournit :
    1. joints3d_25/<exercice>.json     → positions 3D des joints (SMPL-X, 25 pts)
    2. camera_parameters/<cam>/<exercice>.json → calibration caméra

Pipeline de projection 3D → 2D :
    ┌─────────────────┐
    │ Coordonnées 3D  │  (X, Y, Z) en mètres, repère monde
    │ (Fit3D joints)  │
    └────────┬────────┘
             │  P = K @ [R | t]
             ↓
    ┌─────────────────┐
    │ Coordonnées 2D  │  (px, py) en pixels sur l'image caméra
    │ (Ground Truth)  │
    └─────────────────┘

Format des fichiers caméra Fit3D :
    {
      "extrinsics": {
        "R": [[...3x3...]],    # Matrice de rotation monde→caméra
        "T": [[tx, ty, tz]]   # Translation monde→caméra (en mètres)
      },
      "intrinsics_wo_distortion": {
        "f": [fx, fy],         # Distances focales en pixels
        "c": [cx, cy]          # Point principal en pixels
      }
    }

Format joints3d_25 :
    {
      "joints3d_25": [
        [  [X0,Y0,Z0], [X1,Y1,Z1], ..., [X24,Y24,Z24]  ],  ← frame 0
        ...
      ]
    }

Mapping joints BODY_25 (Fit3D) → 14 joints communs :
    La numérotation BODY_25 d'OpenPose est utilisée dans Fit3D joints3d_25.
    Index utilisés pour les 12 joints communs :
        right_shoulder → 2,  right_elbow  → 3,  right_wrist  → 4
        left_shoulder  → 5,  left_elbow   → 6,  left_wrist   → 7
        right_hip      → 9,  right_knee   → 10, right_ankle  → 11
        left_hip       → 12, left_knee    → 13, left_ankle   → 14

=============================================================================
"""

import json
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Mapping : index BODY_25 (Fit3D joints3d_25) → nom joint commun
# ---------------------------------------------------------------------------
FIT3D_JOINT_MAPPING = {
    2:  "right_shoulder",
    3:  "right_elbow",
    4:  "right_wrist",
    5:  "left_shoulder",
    6:  "left_elbow",
    7:  "left_wrist",
    9:  "right_hip",
    10: "right_knee",
    11: "right_ankle",
    12: "left_hip",
    13: "left_knee",
    14: "left_ankle",
}


class CameraCalibration:
    """
    Encapsule les paramètres de calibration d'une caméra Fit3D.

    Construit la matrice de projection P = K @ [R | t]
    permettant de projeter des points 3D (monde) en 2D (image).

    Attributs :
        K (np.ndarray, 3×3) : matrice intrinsèque
        R (np.ndarray, 3×3) : matrice de rotation (monde→caméra)
        t (np.ndarray, 3×1) : vecteur de translation (monde→caméra)
        P (np.ndarray, 3×4) : matrice de projection complète
    """

    def __init__(self, json_path: str):
        """
        Charge et parse le fichier JSON de calibration caméra Fit3D.

        Paramètres :
            json_path : chemin absolu vers le fichier .json de calibration

        Raises :
            FileNotFoundError : si le fichier n'existe pas
            KeyError          : si le format JSON est inattendu
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # --- Paramètres intrinsèques ---
        intr = data["intrinsics_wo_distortion"]
        fx, fy = intr["f"]   # Distances focales (pixels)
        cx, cy = intr["c"]   # Point principal (pixels)

        # Matrice intrinsèque K (3×3)
        # ┌ fx  0  cx ┐
        # │  0 fy  cy │
        # └  0  0   1 ┘
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

        # --- Paramètres extrinsèques ---
        extr = data["extrinsics"]

        # Matrice de rotation R (3×3) — monde vers caméra
        self.R = np.array(extr["R"], dtype=np.float64)  # Déjà une liste 3×3

        # Vecteur de translation T (3,) — monde vers caméra
        T_flat = np.array(extr["T"], dtype=np.float64).flatten()
        self.t = T_flat.reshape(3, 1)  # Colonne (3×1)

        # --- Matrice de projection P = K @ [R | t] ---
        # [R | t] est la matrice 3×4 qui transforme des coordonnées homogènes
        # monde [X, Y, Z, 1] en coordonnées caméra
        Rt = np.hstack([self.R, self.t])   # 3×4
        self.P = self.K @ Rt               # 3×4

    def project(self, X: float, Y: float, Z: float) -> tuple:
        """
        Projette un point 3D (repère monde) en 2D (pixels image).

        Formule :
            pt_h = P @ [X, Y, Z, 1]ᵀ        (coordonnées homogènes)
            px   = pt_h[0] / pt_h[2]
            py   = pt_h[1] / pt_h[2]

        Paramètres :
            X, Y, Z : coordonnées 3D en mètres (repère monde Fit3D)

        Retourne :
            (px, py) : tuple float — coordonnées en pixels sur l'image
        """
        # Point homogène 4D
        pt_world = np.array([X, Y, Z, 1.0], dtype=np.float64)

        # Projection → coordonnées homogènes 3D
        pt_h = self.P @ pt_world

        # Division perspective (division par la profondeur z)
        if abs(pt_h[2]) < 1e-8:
            return (0.0, 0.0)  # Évite la division par zéro

        px = pt_h[0] / pt_h[2]
        py = pt_h[1] / pt_h[2]
        return (float(px), float(py))

    def project_batch(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Projette un tableau de points 3D en batch (plus efficace).

        Paramètres :
            points_3d (np.ndarray, N×3) : N points [X, Y, Z]

        Retourne :
            points_2d (np.ndarray, N×2) : N points [px, py]
        """
        N = len(points_3d)
        # Ajout de la coordonnée homogène
        ones   = np.ones((N, 1), dtype=np.float64)
        pts_h  = np.hstack([points_3d, ones])      # N×4
        proj   = (self.P @ pts_h.T).T              # N×3

        # Division perspective
        depths = proj[:, 2:3]
        depths = np.where(np.abs(depths) < 1e-8, 1e-8, depths)
        pts_2d = proj[:, :2] / depths              # N×2

        return pts_2d


class GroundTruthLoader:
    """
    Charge et gère les joints 3D Fit3D pour un exercice donné.

    Utilise CameraCalibration pour projeter les joints 3D en 2D
    frame par frame.

    Attributs :
        frames_3d  (list)            : liste des frames (chaque frame = 25 joints × [X,Y,Z])
        n_frames   (int)             : nombre total de frames dans l'exercice
        calibration (CameraCalibration) : calibration caméra associée
    """

    def __init__(self, joints3d_path: str, calibration: CameraCalibration):
        """
        Paramètres :
            joints3d_path : chemin vers le fichier joints3d_25/<exercice>.json
            calibration   : instance CameraCalibration déjà chargée
        """
        self.calibration = calibration

        # Chargement du fichier JSON
        with open(joints3d_path, 'r') as f:
            data = json.load(f)

        # frames_3d : liste de N frames, chaque frame est une liste de 25 joints
        # Chaque joint est [X, Y, Z] en mètres
        self.frames_3d = data["joints3d_25"]
        self.n_frames  = len(self.frames_3d)

    def get_gt_2d(self, frame_idx: int) -> dict:
        """
        Génère la Ground Truth 2D pour une frame donnée.

        Pipeline :
        1. Récupère les 25 joints 3D de la frame
        2. Filtre les 12 joints communs (via FIT3D_JOINT_MAPPING)
        3. Projette chaque joint 3D → 2D via la calibration caméra

        Paramètres :
            frame_idx : index de la frame (0-based, clampé si hors limites)

        Retourne :
            dict {joint_name: (px, py)} — coordonnées pixels des joints GT
        """
        # Clamp de l'index pour éviter les erreurs hors-limites
        idx = min(max(0, frame_idx), self.n_frames - 1)

        joints_3d = self.frames_3d[idx]   # Liste de 25 joints [X, Y, Z]
        gt_2d = {}

        for joint_idx, joint_name in FIT3D_JOINT_MAPPING.items():
            if joint_idx < len(joints_3d):
                X, Y, Z = joints_3d[joint_idx]
                px, py  = self.calibration.project(X, Y, Z)
                gt_2d[joint_name] = (px, py)

        return gt_2d

    def get_all_gt_2d(self) -> list:
        """
        Pré-calcule la GT 2D pour TOUTES les frames (batch).
        Utile pour accélérer l'évaluation si la vidéo est longue.

        Retourne :
            list de dicts — un dict par frame
        """
        return [self.get_gt_2d(i) for i in range(self.n_frames)]

    @staticmethod
    def compute_bbox_from_gt(gt_2d: dict) -> Optional[float]:
        """
        Estime la taille de la bounding box à partir des joints GT.
        Utilisé pour le calcul du seuil PCK@0.1.

        Retourne :
            Diagonale de la bounding box en pixels, ou None si < 2 joints.
        """
        if len(gt_2d) < 2:
            return None

        pts = np.array(list(gt_2d.values()))
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        # Vecteur diagonale de la bounding box
        diagonal = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        return float(diagonal) if diagonal > 0 else None
