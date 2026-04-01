"""
=============================================================================
pose_estimators.py
=============================================================================
Ce module définit l'architecture polymorphe des estimateurs de pose 2D.

Architecture :
    PoseEstimator (classe abstraite)
    ├── BlazePoseEstimator   → MediaPipe Pose
    ├── YoloPoseEstimator    → Ultralytics YOLOv8n-pose
    └── OpenPoseEstimator    → OpenCV DNN (BODY_25 Caffe model)

Chaque estimateur implémente :
    detect(frame) → dict[str, tuple[float, float]]
        Entrée  : frame BGR (numpy array H×W×3)
        Sortie  : dictionnaire {nom_joint: (x_pixels, y_pixels)}
                  Seuls les 14 joints communs sont renvoyés.
                  Un joint absent/non-détecté est omis du dictionnaire.

Mapping des 14 joints communs (standardisation inter-modèles) :
    Les 3 modèles utilisent des conventions d'indexation différentes.
    Ce module effectue la traduction vers des noms sémantiques universels.

=============================================================================
"""

import abc
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Noms des 14 joints communs utilisés pour la comparaison
# ---------------------------------------------------------------------------
# Ces noms servent de clés communes entre tous les estimateurs et la GT.
COMMON_JOINTS = [
    "left_shoulder",   # Épaule gauche
    "right_shoulder",  # Épaule droite
    "left_elbow",      # Coude gauche
    "right_elbow",     # Coude droit
    "left_wrist",      # Poignet gauche
    "right_wrist",     # Poignet droit
    "left_hip",        # Hanche gauche
    "right_hip",       # Hanche droite
    "left_knee",       # Genou gauche
    "right_knee",      # Genou droit
    "left_ankle",      # Cheville gauche
    "right_ankle",     # Cheville droite
]

# ---------------------------------------------------------------------------
# Couleurs des connexions squelette pour chaque modèle (BGR)
# ---------------------------------------------------------------------------
COLOR_BLAZEPOSE = (0, 255, 0)    # Vert
COLOR_YOLO     = (255, 0, 0)    # Bleu
COLOR_OPENPOSE = (0, 0, 255)    # Rouge

# Connexions du squelette pour l'affichage (paires de noms de joints)
SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
]


# =============================================================================
# CLASSE ABSTRAITE
# =============================================================================

class PoseEstimator(abc.ABC):
    """
    Classe de base abstraite pour tous les estimateurs de pose 2D.

    Chaque sous-classe doit :
    1. Appeler super().__init__() pour initialiser les compteurs de FPS
    2. Implémenter la méthode detect(frame)
    3. Implémenter la propriété model_name

    Attributs :
        _fps_history (list[float]) : historique des FPS des 30 dernières frames
        color (tuple)              : couleur BGR pour le dessin du squelette
    """

    def __init__(self):
        import typing
        self._fps_history = []   # Historique glissant des FPS
        self._last_time   = None
        self.color: tuple[int, int, int] = (255, 255, 255)  # Blanc par défaut

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Retourne le nom lisible du modèle."""
        ...

    @abc.abstractmethod
    def _detect_impl(self, frame: np.ndarray) -> dict:
        """
        Implémentation interne de la détection.
        Entrée  : frame BGR numpy
        Sortie  : dict {joint_name: (x, y)} — coordonnées en pixels
        """
        ...

    def detect(self, frame: np.ndarray) -> dict:
        """
        Méthode publique : lance la détection ET mesure le temps d'inférence.
        Retourne aussi le FPS instantané via self.get_fps().
        """
        t0 = time.perf_counter()
        result = self._detect_impl(frame)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # Fenêtre glissante de 60 mesures pour lisser les FPS
        self._fps_history.append(fps)
        if len(self._fps_history) > 60:
            self._fps_history.pop(0)

        return result

    def get_fps(self) -> float:
        """Retourne le FPS moyen sur la fenêtre glissante."""
        if not self._fps_history:
            return 0.0
        return float(np.mean(self._fps_history))

    def draw_skeleton(self, frame: np.ndarray, keypoints: dict,
                      gt_points: dict = None) -> np.ndarray:
        """
        Dessine le squelette prédit sur la frame.
        Optionnellement superpose les points GT en blanc.

        Paramètres :
            frame      : image BGR (ne sera PAS modifiée en place — copie faite)
            keypoints  : dict {joint_name: (x, y)}
            gt_points  : dict {joint_name: (x, y)} ou None

        Retourne : image BGR avec squelette dessiné
        """
        out = frame.copy()

        # --- Connexions ---
        for (j1, j2) in SKELETON_CONNECTIONS:
            if j1 in keypoints and j2 in keypoints:
                p1 = (int(keypoints[j1][0]), int(keypoints[j1][1]))
                p2 = (int(keypoints[j2][0]), int(keypoints[j2][1]))
                cv2.line(out, p1, p2, self.color, 2, cv2.LINE_AA)

        # --- Points articulaires prédits ---
        for name, (x, y) in keypoints.items():
            cv2.circle(out, (int(x), int(y)), 5, self.color, -1, cv2.LINE_AA)

        # --- Points Ground Truth (cercles blancs) — désactivés (mapping Fit3D non documenté) ---
        # if gt_points:
        #     for name, (x, y) in gt_points.items():
        #         cv2.circle(out, (int(x), int(y)), 6, (255, 255, 255), 2, cv2.LINE_AA)

        # --- FPS en haut à gauche (lisible sur n'importe quel fond) ---
        fps_text = f"{self.model_name} | {self.get_fps():.1f} FPS"
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.75
        thickness  = 2
        (tw, th), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
        # Fond noir semi-transparent
        cv2.rectangle(out, (5, 5), (tw + 16, th + 16), (0, 0, 0), -1)
        cv2.putText(out, fps_text, (10, th + 10),
                    font, font_scale, self.color, thickness, cv2.LINE_AA)

        return out


# =============================================================================
# ESTIMATEUR 1 : BLAZEPOSE (MediaPipe)
# =============================================================================

import mediapipe as mp

class BlazePoseEstimator(PoseEstimator):
    BLAZEPOSE_MAPPING = {
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
    }

    def __init__(self):
        super().__init__()
        self.color = COLOR_BLAZEPOSE

        import os
        model_path = "pose_landmarker_lite.task"
        if not os.path.exists(model_path):
            import urllib.request
            print(f"[BlazePose] Téléchargement du modèle {model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)
            print("[BlazePose] Téléchargement terminé.")

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    @property
    def model_name(self) -> str:
        return "BlazePose"

    def _detect_impl(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        import mediapipe as mp
        
        # Le landmarker requiert une image RGB (MediaPipe Frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        results = self._landmarker.detect(mp_image)

        keypoints = {}
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]
            for idx, name in self.BLAZEPOSE_MAPPING.items():
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    # Visibilité suffisante
                    if lm.visibility >= 0.5:
                        keypoints[name] = (lm.x * w, lm.y * h)

        return keypoints

    def __del__(self):
        if hasattr(self, '_landmarker'):
            self._landmarker.close()


# =============================================================================
# ESTIMATEUR 2 : YOLO-POSE (Ultralytics YOLOv8)
# =============================================================================

class YoloPoseEstimator(PoseEstimator):
    """
    Estimateur basé sur YOLOv8n-pose (Ultralytics).

    YOLOv8-pose utilise le format COCO (17 keypoints).
    On extrait 12 joints via le mapping ci-dessous.

    Mapping COCO → joints communs :
        left_shoulder  → keypoint 5
        right_shoulder → keypoint 6
        left_elbow     → keypoint 7
        right_elbow    → keypoint 8
        left_wrist     → keypoint 9
        right_wrist    → keypoint 10
        left_hip       → keypoint 11
        right_hip      → keypoint 12
        left_knee      → keypoint 13
        right_knee     → keypoint 14
        left_ankle     → keypoint 15
        right_ankle    → keypoint 16

    YOLOv8-pose renvoie des coordonnées directement en pixels.
    """

    # Mapping index COCO → nom joint commun
    YOLO_MAPPING = {
        5:  "left_shoulder",
        6:  "right_shoulder",
        7:  "left_elbow",
        8:  "right_elbow",
        9:  "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
    }

    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        """
        Paramètres :
            model_path : chemin vers le fichier .pt YOLOv8-pose
                         Si non trouvé localement, téléchargé automatiquement.
        """
        super().__init__()
        self.color = COLOR_YOLO

        from ultralytics import YOLO
        # verbose=False supprime les logs de progression à chaque frame
        self._model = YOLO(model_path)

    @property
    def model_name(self) -> str:
        return "YOLO-Pose"

    def _detect_impl(self, frame: np.ndarray) -> dict:
        """
        Détecte les keypoints avec YOLOv8.
        On prend uniquement la personne avec la plus grande bounding box
        (hypothèse : une seule personne dans le champ — valide pour Fit3D).
        """
        # verbose=False pour éviter de spammer la console
        results = self._model(frame, verbose=False)

        keypoints = {}
        best_area = -1
        best_kps  = None

        import typing
        for r_raw in results:
            r = typing.cast(typing.Any, r_raw)
            if r.keypoints is None:
                continue
            # Itération sur chaque personne détectée
            for i, kps in enumerate(r.keypoints.data):
                # Calcul de la surface de la bounding box pour choisir la principale
                if r.boxes is not None and i < len(r.boxes.xyxy):
                    box = r.boxes.xyxy[i].cpu().numpy()
                    area = (box[2] - box[0]) * (box[3] - box[1])
                else:
                    area = 0

                if area > best_area:
                    best_area = area
                    best_kps  = kps.cpu().numpy()  # Shape: (17, 3) → x, y, conf

        if best_kps is not None:
            for idx, name in self.YOLO_MAPPING.items():
                x, y, conf = best_kps[idx]
                # Confiance < 0.5 → keypoint non fiable
                if conf >= 0.5:
                    keypoints[name] = (float(x), float(y))

        return keypoints


# =============================================================================
# ESTIMATEUR 3 : OPENPOSE (OpenCV DNN BODY_25)
# =============================================================================

class OpenPoseEstimator(PoseEstimator):
    """
    Estimateur basé sur OpenPose via OpenCV DNN (modèle BODY_25 Caffe).

    OpenPose BODY_25 détecte 25 joints.
    On extrait 12 joints via le mapping ci-dessous.

    Mapping BODY_25 → joints communs :
        right_shoulder → index 2
        right_elbow    → index 3
        right_wrist    → index 4
        left_shoulder  → index 5
        left_elbow     → index 6
        left_wrist     → index 7
        right_hip      → index 9
        right_knee     → index 10
        right_ankle    → index 11
        left_hip       → index 12
        left_knee      → index 13
        left_ankle     → index 14

    Prérequis : fichiers modèle Caffe dans ./openpose_models/
        - pose_iter_440000.caffemodel
        - pose_deploy_linevec.prototxt

    Si les fichiers sont absents, le modèle retourne toujours un dict vide
    et affiche un avertissement. L'application reste fonctionnelle pour
    les 2 autres modèles.
    """

    # Mapping index COCO → nom joint commun
    OPENPOSE_MAPPING = {
        2:  "right_shoulder",
        3:  "right_elbow",
        4:  "right_wrist",
        5:  "left_shoulder",
        6:  "left_elbow",
        7:  "left_wrist",
        8:  "right_hip",
        9:  "right_knee",
        10: "right_ankle",
        11: "left_hip",
        12: "left_knee",
        13: "left_ankle",
    }

    # Résolution d'entrée du réseau (recommandée pour COCO)
    NET_INPUT_WIDTH  = 368
    NET_INPUT_HEIGHT = 368
    N_PARTS          = 18   # Nombre de joints dans COCO

    def __init__(self,
                 proto_path: str = "openpose_models/pose_deploy_linevec.prototxt",
                 model_path: str = "openpose_models/pose_iter_440000.caffemodel"):
        """
        Paramètres :
            proto_path : chemin vers le fichier .prototxt OpenPose BODY_25
            model_path : chemin vers le fichier .caffemodel OpenPose
        """
        super().__init__()
        self.color       = COLOR_OPENPOSE
        self._available  = False
        import typing
        self._net: typing.Any = None

        try:
            import os
            if not (os.path.exists(proto_path) and os.path.exists(model_path)):
                print(
                    f"[OpenPose] Fichiers introuvables :\n"
                    f"  Proto : {proto_path}\n"
                    f"  Model : {model_path}\n"
                    f"Téléchargez-les depuis https://github.com/CMU-Perceptual-Computing-Lab/openpose "
                    f"et placez-les dans ./openpose_models/"
                )
                self._available = False
                return

            # Chargement du réseau via OpenCV DNN (CPU)
            self._net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._available = True
            print("[OpenPose] Modèle chargé avec succès.")

        except Exception as e:
            print(f"[OpenPose] AVERTISSEMENT : {e}")
            self._available = False

    @property
    def model_name(self) -> str:
        return "OpenPose"

    def _detect_impl(self, frame: np.ndarray) -> dict:
        """
        Détecte les joints avec OpenPose via OpenCV DNN.

        Pipeline :
        1. Créer un blob (resize + normalisation) depuis la frame
        2. Passer dans le réseau
        3. Extraire les heat maps de sortie (une par joint)
        4. Trouver le maximum dans chaque heat map → position du joint
        5. Mettre à l'échelle vers les coordonnées originales
        """
        if not self._available:
            return {}

        h, w = frame.shape[:2]

        # Création du blob d'entrée (normalisé, 368×368)
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.NET_INPUT_WIDTH, self.NET_INPUT_HEIGHT),
            mean=(0, 0, 0),           # Pas de soustraction de moyenne pour BODY_25
            swapRB=False,             # Déjà en BGR
            crop=False
        )
        self._net.setInput(blob)

        # Inférence → shape sortie: (1, N_PARTS, H_map, W_map)
        output = self._net.forward()

        out_h, out_w = output.shape[2], output.shape[3]

        # Facteurs d'échelle heat map → pixels originaux
        scale_x = w / out_w
        scale_y = h / out_h

        keypoints = {}
        for idx, name in self.OPENPOSE_MAPPING.items():
            # Heat map du joint idx
            prob_map = output[0, idx, :, :]

            # Trouver la position du maximum de probabilité
            _, prob, _, point = cv2.minMaxLoc(prob_map)

            # Seuil de confiance (la valeur max de la heat map)
            if prob >= 0.1:
                x = (point[0] + 0.5) * scale_x
                y = (point[1] + 0.5) * scale_y
                keypoints[name] = (float(x), float(y))

        return keypoints
