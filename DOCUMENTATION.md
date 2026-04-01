# Documentation Complète — Système de Comparaison de Modèles de Pose 2D

**Projet** : PFE Master 2 — Comparaison automatique de BlazePose, YOLO-Pose et OpenPose sur le dataset Fit3D  
**Auteur** : Ikram  
**Date** : Avril 2026

---

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture du code](#2-architecture-du-code)
3. [Module `pose_estimators.py`](#3-module-pose_estimatorspy)
4. [Module `ground_truth.py`](#4-module-ground_truthpy)
5. [Module `metrics.py`](#5-module-metricspy)
6. [Module `export_utils.py`](#6-module-export_utilspy)
7. [Module `main_app.py`](#7-module-main_apppy)
8. [Dataset Fit3D — Format et structure](#8-dataset-fit3d--format-et-structure)
9. [Pipeline complet frame par frame](#9-pipeline-complet-frame-par-frame)
10. [Comment utiliser l'application](#10-comment-utiliser-lapplication)
11. [Métriques d'évaluation](#11-métriques-dévaluation)
12. [Standardisation des squelettes](#12-standardisation-des-squelettes)
13. [Résultats et fichiers de sortie](#13-résultats-et-fichiers-de-sortie)
14. [Dépendances et installation](#14-dépendances-et-installation)
15. [Limites et perspectives](#15-limites-et-perspectives)

---

## 1. Vue d'ensemble du projet

### Objectif

Ce programme Python compare trois modèles d'estimation de pose 2D sur le dataset Fit3D, afin de justifier le choix technologique dans un système de comparaison de mouvement humain 3D. Les modèles évalués sont :

| Modèle | Bibliothèque | Couleur affichage |
|--------|-------------|-------------------|
| **BlazePose** | `mediapipe` | 🟢 Vert |
| **YOLO-Pose** | `ultralytics` (YOLOv8n-pose) | 🔵 Bleu |
| **OpenPose** | `opencv-dnn` (modèle Caffe BODY_25) | 🔴 Rouge |

### Pourquoi comparer ces 3 modèles ?

- **BlazePose** est léger, optimisé temps réel, très utilisé sur mobile et dans les applications de fitness. Il est intégré directement dans MediaPipe.
- **YOLO-Pose** est la version pose de la famille YOLO, reconnue pour sa vitesse et sa précision sur des objets multiples.
- **OpenPose** est le modèle de référence académique (CMU), souvent cité dans les papiers de recherche. Il est plus lourd mais très documenté.

### Contexte système

Le projet global reconstruit la pose 3D d'une personne par triangulation multi-vues. Pour évaluer la qualité de la détection 2D, une Ground Truth 2D est générée automatiquement en projetant les joints 3D connus (Fit3D) sur le plan image de chaque caméra via la matrice de projection.

---

## 2. Architecture du code

```
models_comparaison/
├── main_app.py           ← Application PyQt5 (point d'entrée)
├── pose_estimators.py    ← Estimateurs de pose (BlazePose, YOLO, OpenPose)
├── ground_truth.py       ← Chargement GT 3D et projection 2D
├── metrics.py            ← MPJPE, PCK@0.1, accumulation
├── export_utils.py       ← Export CSV + rapport PDF
├── requirements.txt      ← Dépendances Python
├── DOCUMENTATION.md      ← Ce fichier
└── openpose_models/      ← (optionnel) Fichiers modèle OpenPose Caffe
    ├── pose_deploy_linevec.prototxt
    └── pose_iter_440000.caffemodel
```

### Diagramme de dépendances

```
main_app.py
├── pose_estimators.py  (BlazePoseEstimator, YoloPoseEstimator, OpenPoseEstimator)
├── ground_truth.py     (CameraCalibration, GroundTruthLoader)
├── metrics.py          (MetricsAccumulator, compute_mpjpe, compute_pck)
└── export_utils.py     (save_csv, generate_pdf)
```

### Flux de données

```
Fichier vidéo .mp4
    ↓ (cv2.VideoCapture)
Frame BGR (numpy H×W×3)
    ↓
    ├──→ BlazePoseEstimator.detect() → dict {joint: (x,y)} ──→ draw_skeleton()
    ├──→ YoloPoseEstimator.detect()  → dict {joint: (x,y)} ──→ draw_skeleton()
    └──→ OpenPoseEstimator.detect()  → dict {joint: (x,y)} ──→ draw_skeleton()
                                                              ↑
Ground Truth 2D (GroundTruthLoader.get_gt_2d(frame_idx)) ───┘
    (générée par projection K @ [R|t] @ [X,Y,Z,1])

3 images annotées → affichées dans split-screen PyQt5
Métriques accumulées → MetricsAccumulator → tableau + graphiques
En fin de vidéo → export CSV + PDF
```

---

## 3. Module `pose_estimators.py`

### Classe abstraite `PoseEstimator`

```
PoseEstimator (ABC)
    ├── detect(frame) → dict            ← méthode publique avec mesure FPS
    ├── _detect_impl(frame) → dict      ← méthode abstraite à implémenter
    ├── get_fps() → float               ← FPS moyen sur fenêtre glissante 60 frames
    └── draw_skeleton(frame, kps, gt)   ← dessine squelette + GT sur image
```

La mesure du temps d'inférence est faite dans `detect()` via `time.perf_counter()`, qui est la fonction de mesure de temps la plus précise disponible en Python (résolution sub-microseconde). Elle encapsule l'appel à `_detect_impl()`.

Le FPS est calculé comme la moyenne glissante sur les 60 dernières frames pour lisser les pics dus au garbage collector ou à d'autres processus système.

### `BlazePoseEstimator`

**Technologie** : MediaPipe Pose (modèle BlazePose Full)

**Paramètres de configuration** :
- `model_complexity=1` : compromis entre précision et vitesse (0=rapide, 2=précis)
- `smooth_landmarks=True` : filtre temporel — réduit le bruit entre frames
- `min_detection_confidence=0.5` : seuil de détection initiale d'une personne
- `min_tracking_confidence=0.5` : seuil pour le suivi entre frames (évite re-détection)

**Conversion** : MediaPipe retourne des coordonnées normalisées [0..1]. La conversion en pixels se fait par `x × largeur_image` et `y × hauteur_image`.

**Filtrage** : Les joints avec `visibility < 0.5` sont ignorés (non inclus dans le dictionnaire retourné). La visibilité est une valeur entre 0 et 1 que MediaPipe estime en interne.

### `YoloPoseEstimator`

**Technologie** : YOLOv8n-pose (Ultralytics)

**Comment YOLO détecte la pose** : YOLOv8-pose est un modèle à plusieurs têtes — une tête de détection d'objets classique (bounding boxes) et une tête de régression de keypoints pour chaque personne détectée.

**Sélection de la personne principale** : Pour les vidéos Fit3D (une seule personne), on sélectionne la détection avec la plus grande bounding box (= la personne la plus proche/principale). Cela est robuste aux détections parasites en fond.

**Filtrage** : Les keypoints avec `confidence < 0.5` sont exclus.

**Format de sortie Ultralytics** : `results[0].keypoints.data` est un tensor de shape `(N_personnes, 17, 3)` où chaque keypoint est `[x, y, confidence]`.

### `OpenPoseEstimator`

**Technologie** : OpenCV DNN avec les poids pré-entraînés OpenPose BODY_25 (format Caffe)

**Pourquoi via OpenCV DNN ?** : Le package OpenPose officiel nécessite une compilation complexe avec CUDA. Utiliser OpenCV DNN permet de charger le même modèle Caffe sans installation complexe, au prix d'une exécution CPU uniquement.

**Pipeline de détection** :
1. Redimensionner l'image en 368×368 pixels (résolution d'entrée optimale du réseau)
2. Créer un blob normalisé (valeurs dans [0, 1])
3. Passer dans le réseau : sortie de shape `(1, 25, H_map, W_map)` (une heat map par joint)
4. Pour chaque heat map, trouver le maximum (`cv2.minMaxLoc`)
5. Mettre à l'échelle les coordonnées de la heat map vers les pixels originaux

**Fallback gracieux** : Si les fichiers `.caffemodel` et `.prototxt` sont absents, l'estimateur s'initialise en mode inactif et retourne toujours `{}`. L'application reste fonctionnelle pour les 2 autres modèles.

**Téléchargement des modèles OpenPose** : Voir https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/getModels.bat

### Constantes partagées

```python
COMMON_JOINTS = [
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
]

SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),  # Barre d'épaules
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),       # Barre de hanches
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
]
```

---

## 4. Module `ground_truth.py`

### Classe `CameraCalibration`

**Rôle** : Charge les paramètres intrinsèques et extrinsèques d'une caméra Fit3D et construit la matrice de projection.

#### Format du fichier JSON Fit3D

```json
{
  "extrinsics": {
    "R": [[-0.371, -0.927, 0.040], [...]],   // Matrice 3×3
    "T": [[-4.003, 1.770, 1.552]]            // Vecteur 1×3
  },
  "intrinsics_wo_distortion": {
    "f": [1093.67, 1087.31],                 // [fx, fy] en pixels
    "c": [470.20, 443.12]                    // [cx, cy] en pixels
  }
}
```

#### Construction de la matrice de projection

**Étape 1 — Matrice intrinsèque K** :

```
K = | fx   0  cx |
    |  0  fy  cy |
    |  0   0   1 |
```

Où `fx = f[0]`, `fy = f[1]`, `cx = c[0]`, `cy = c[1]`.

**Étape 2 — Matrice extrinsèque [R|t]** :

```
[R|t] = | R[0,0]  R[0,1]  R[0,2]  t[0] |   (3×4)
        | R[1,0]  R[1,1]  R[1,2]  t[1] |
        | R[2,0]  R[2,1]  R[2,2]  t[2] |
```

Note : Dans Fit3D, `T` est déjà le vecteur de translation en coordonnées caméra (≠ centre de caméra C). On utilise `t = T.reshape(3,1)`.

**Étape 3 — Matrice de projection P** :

```
P = K @ [R|t]        (3×3 @ 3×4 = 3×4)
```

#### Projection d'un point 3D → 2D

```python
pt_h = P @ [X, Y, Z, 1]    # Coordonnées homogènes (3,)
px   = pt_h[0] / pt_h[2]   # Division perspective
py   = pt_h[1] / pt_h[2]
```

La division par `pt_h[2]` est la division perspective — elle tient compte de la profondeur du point par rapport à la caméra (perspective non linéaire).

### Classe `GroundTruthLoader`

**Rôle** : Charge le fichier `joints3d_25/<exercice>.json` et fournit la GT 2D frame par frame.

#### Format du fichier JSON joints3d_25

```json
{
  "joints3d_25": [
    [                           // Frame 0
      [-0.075, 0.101, 0.990],   // Joint 0 (pelvis)
      [-0.090, 0.246, 0.974],   // Joint 1 (L_hip)
      ...                       // 25 joints au total
    ],
    ...                         // N frames
  ]
}
```

**Unités** : Les coordonnées 3D sont en **mètres** dans le repère monde.

**Méthode `get_gt_2d(frame_idx)`** :
1. Récupère la liste des 25 joints 3D pour la frame demandée
2. Sélectionne uniquement les 12 joints contenus dans `FIT3D_JOINT_MAPPING`
3. Projette chacun via `CameraCalibration.project(X, Y, Z)`
4. Retourne `dict {joint_name: (px, py)}`

**Méthode `compute_bbox_from_gt(gt_2d)`** :
Calcule la diagonale de la bounding box englobant tous les joints GT. Cette valeur est utilisée comme référence pour le calcul du seuil PCK@0.1.

```
bbox_size = sqrt((x_max - x_min)² + (y_max - y_min)²)
seuil_PCK = 0.1 × bbox_size
```

---

## 5. Module `metrics.py`

### MPJPE (Mean Per-Joint Position Error)

**Définition** : Erreur euclidienne moyenne en pixels entre les joints prédits et les joints GT.

```
MPJPE = (1/N) × Σᵢ ||pred_i - gt_i||₂
```

Où `N` est le nombre de joints communs (présents à la fois dans la prédiction et la GT).

**Interprétation** :
- Une valeur de 5 pixels signifie que, en moyenne, chaque joint est détecté à 5 pixels de sa vraie position.
- Plus la valeur est petite, meilleur est le modèle.
- Une erreur < 10 px est généralement considérée bonne pour des images HD (1920×1080).

**Note** : Si un joint n'est pas détecté (absent du dictionnaire `pred`), il est ignoré dans le calcul. Cela peut biaiser le MPJPE si un modèle rate systématiquement les joints difficiles (ex. mains derrière le dos).

### PCK@0.1 (Percentage of Correct Keypoints)

**Définition** : Pourcentage de joints dont l'erreur est inférieure à 10% de la taille de la bounding box.

```
PCK@0.1 = (nombre de joints avec dist < 0.1 × bbox_size) / (total joints communs) × 100
```

**Interprétation** :
- 100% = tous les joints sont détectés correctement
- 80% = 80% des joints sont dans le seuil de tolérance
- Avantage par rapport au MPJPE : invariant à la distance caméra-sujet (normalisé par la taille du corps)

**Seuil 10%** : C'est la valeur standard dans la littérature académique (articles PCK, PCKh). Pour des images de résolution typique (720p), cela correspond à environ 20-40 pixels — suffisamment strict pour filtrer les détections grossières.

### `MetricsAccumulator`

**Rôle** : Accumule les métriques de toutes les frames pour produire des statistiques globales.

**Méthode `update(pred, gt, bbox_size)`** :
Appelée à chaque frame — ajoute le MPJPE et le PCK de cette frame aux listes respectives.

**Méthode `get_summary()`** :
Calcule et retourne un dictionnaire avec :
- `mpjpe_mean` : MPJPE moyen sur toutes les frames
- `mpjpe_std` : écart-type (mesure de la stabilité)
- `pck_mean` : PCK@0.1 moyen en %
- `best_joint` : joint avec la plus petite erreur moyenne (modèle le plus fiable pour ce joint)
- `worst_joint` : joint avec la plus grande erreur moyenne (point faible du modèle)
- `per_joint_mpjpe` : dictionnaire `{joint_name: erreur_moyenne_px}` pour l'analyse fine

---

## 6. Module `export_utils.py`

### Export CSV (`save_csv`)

Le fichier `results.csv` contient deux sections :

**Section 1** — Résumé par modèle :
```
Modèle,FPS,MPJPE (px),MPJPE std,PCK@0.1 (%),Meilleur joint,Pire joint,Frames évaluées
BlazePose,28.3,12.45,3.21,82.5%,left_shoulder,right_wrist,500
YOLO-Pose,35.7,14.23,4.12,78.3%,left_hip,right_ankle,500
OpenPose,8.2,16.78,5.44,71.2%,right_shoulder,left_wrist,500
```

**Section 2** — Erreur par articulation :
```
Joint,BlazePose,YOLO-Pose,OpenPose
left_ankle,15.23,17.45,21.34
left_elbow,9.12,11.23,14.56
...
```

### Rapport PDF (`generate_pdf`)

Le rapport est un PDF multi-pages généré avec `matplotlib.PdfPages` :

| Page | Contenu |
|------|---------|
| 1 | Titre + tableau récapitulatif avec couleurs par modèle |
| 2 | Barres horizontales MPJPE avec barres d'erreur (std) |
| 3 | Barres verticales FPS |
| 4 | Scatter plot précision (PCK@0.1) × vitesse (FPS) — idéalement haut-droite |
| 5 | Barres groupées : erreur par articulation pour chaque modèle |
| 6 | Courbe MPJPE frame par frame + lignes de moyenne pointillées |

---

## 7. Module `main_app.py`

### Architecture PyQt5

```
QMainWindow (MainWindow)
├── QWidget (central)
│   ├── QGroupBox (contrôles)
│   │   ├── QPushButton × 3 (chargement fichiers)
│   │   └── QPushButton × 2 (lancer/arrêter)
│   ├── QHBoxLayout (corps)
│   │   ├── QVBoxLayout (colonne gauche)
│   │   │   ├── QGroupBox (split-screen)
│   │   │   │   └── QLabel × 3 (panels vidéo)
│   │   │   └── QGroupBox (tableau)
│   │   │       └── QTableWidget (3 lignes, 6 colonnes)
│   │   └── QGroupBox (graphiques)
│   │       └── LiveChartCanvas (FigureCanvasQTAgg)
└── QStatusBar + QProgressBar
```

### Threading : `EvaluationWorker` (QThread)

**Pourquoi un thread séparé ?** : L'inférence des modèles (surtout OpenPose) prend plusieurs millisecondes par frame. Exécuter cela dans le thread principal bloquerait l'interface utilisateur, rendant les boutons non réactifs et l'affichage gelé.

**Communication entre threads** : PyQt5 utilise un système de signaux/slots thread-safe :

```python
# Déclaration des signaux dans EvaluationWorker
frame_ready    = pyqtSignal(object, object, object, int)
metrics_update = pyqtSignal(dict, dict, dict)
finished_eval  = pyqtSignal(dict, dict, dict)
progress       = pyqtSignal(int, int)
error_signal   = pyqtSignal(str)
```

Les signaux sont émis depuis le thread worker et reçus dans le thread principal (via `connect()`). Qt garantit la sécurité de ces échanges.

### Conversion OpenCV → Qt

La conversion image est une opération critique faite à chaque frame :

```python
rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # BGR → RGB
qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
pix  = QPixmap.fromImage(qimg)
# Redimensionnement proportionnel
pix_scaled = pix.scaled(label.width(), label.height(),
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
label.setPixmap(pix_scaled)
```

OpenCV stocke les images en BGR (Blue-Green-Red), mais Qt attend du RGB. La conversion est indispensable pour éviter des couleurs inversées.

### `LiveChartCanvas`

Widget matplotlib intégré dans Qt via `FigureCanvasQTAgg`. Contient 2 sous-graphiques actualisés toutes les 30 frames :
1. **Courbe MPJPE par frame** : permet de voir la stabilité temporelle de chaque modèle
2. **Barres MPJPE moyen** : vue synthétique instantanée

---

## 8. Dataset Fit3D — Format et structure

### Structure des dossiers

```
models_comparaison/
└── s03/
    ├── videos/
    │   ├── 50591643Lb/         ← Caméra 1 (Left Back)
    │   │   └── squat.mp4
    │   ├── 58860488RB/         ← Caméra 2 (Right Back)
    │   ├── 60457274RF/         ← Caméra 3 (Right Front)
    │   └── 65906101LF/         ← Caméra 4 (Left Front)
    ├── camera_parameters/
    │   ├── 50591643Lb/
    │   │   └── squat.json      ← Calibration de la caméra 1 pour l'exercice squat
    │   └── ...
    └── joints3d_25/
        └── squat.json          ← Joints 3D pour squat (même pour toutes les caméras)
```

### Correspondance fichiers

Pour évaluer un modèle, il faut choisir **une caméra** et charger :
1. La **vidéo** de cette caméra : `videos/<camera_id>/<exercice>.mp4`
2. La **calibration** de cette caméra : `camera_parameters/<camera_id>/<exercice>.json`
3. Les **joints 3D** de l'exercice : `joints3d_25/<exercice>.json` (commun à toutes les caméras)

### Joints BODY_25 de Fit3D

Le fichier `joints3d_25` contient les 25 joints du modèle BODY_25 d'OpenPose :

| Index | Joint |
|-------|-------|
| 0 | Pelvis (racine) |
| 1 | L_Hip (hanche gauche) |
| 2 | R_Hip (hanche droite) |
| 3 | Spine1 |
| 4 | L_Knee |
| 5 | R_Knee |
| ... | ... |

*Note* : La numérotation exacte peut légèrement varier selon la version de SMPL-X utilisée. Ce programme utilise les 12 joints les plus fiables et les plus visibles.

---

## 9. Pipeline complet frame par frame

Voici ce qui se passe pour chaque frame de la vidéo :

```
1. cap.read() → frame BGR (ex: 1280×720×3)

2. gt_loader.get_gt_2d(frame_idx)
   └── Récupère les 25 joints 3D de la frame
   └── Projette les 12 joints d'intérêt : K @ [R|t] @ [X,Y,Z,1]
   └── Retourne : {"left_shoulder": (245.3, 180.1), ...}

3. Pour chaque estimateur :
   a. estimator.detect(frame)
      ├── t0 = time.perf_counter()
      ├── Prétraitement (RGB/blob/resize selon modèle)
      ├── Inférence du réseau de neurones
      ├── Post-traitement (filtrage confiance, mapping indices)
      ├── t1 = time.perf_counter() → fps = 1/(t1-t0)
      └── Retourne : {"left_shoulder": (241.5, 183.2), ...}

   b. accumulator.update(pred, gt_2d, bbox_size)
      ├── MPJPE = mean(||pred_i - gt_i||₂ pour joints communs)
      ├── PCK@0.1 = % joints avec dist < 0.1 × bbox_diag
      └── Stocke erreur par joint

   c. estimator.draw_skeleton(frame, pred, gt_2d)
      ├── Dessine les connexions en couleur du modèle
      ├── Dessine les joints prédits (cercles colorés)
      ├── Dessine les joints GT (cercles blancs)
      └── Ajoute le FPS en haut à gauche

4. Émission du signal frame_ready → mise à jour de l'UI

5. Toutes les 30 frames : émission metrics_update → actualisation tableau + graphiques

6. Fin de vidéo : émission finished_eval → export CSV + PDF
```

---

## 10. Comment utiliser l'application

### Installation

```bash
# 1. Installer les dépendances
cd /Users/HP/Desktop/ikram/M2/PFE/models_comparaison
pip install -r requirements.txt

# 2. (Optionnel) Installer les modèles OpenPose
mkdir openpose_models
# Télécharger pose_iter_440000.caffemodel et pose_deploy_linevec.prototxt
# depuis le repo officiel CMU et les placer dans openpose_models/

# 3. Lancer l'application
python3 main_app.py
```

### Utilisation pas à pas

**Étape 1** : Cliquer sur **🎬 Charger vidéo (.mp4)**
- Naviguer dans `s03/videos/50591643Lb/`
- Sélectionner `squat.mp4` (ou tout autre exercice)

**Étape 2** : Cliquer sur **📐 Charger Ground Truth (.json)**
- Naviguer dans `s03/joints3d_25/`
- Sélectionner `squat.json` ← **même exercice que la vidéo**

**Étape 3** : Cliquer sur **📷 Charger Calibration (.json)**
- Naviguer dans `s03/camera_parameters/50591643Lb/`
- Sélectionner `squat.json` ← **même caméra que la vidéo**

> ⚠️ **Important** : Le nom de l'exercice doit être identique pour les 3 fichiers. La calibration doit correspondre à la caméra de la vidéo choisie.

**Étape 4** : Cliquer sur **▶ Lancer l'évaluation**
- L'initialisation des modèles peut prendre 5-15 secondes
- Le split-screen s'animera avec les 3 modèles en parallèle
- Le tableau et les graphiques se mettent à jour toutes les 30 frames

**Étape 5** : Fin automatique
- Quand la vidéo est entièrement traitée, un dialogue apparaît avec les chemins
- `results.csv` et `report.pdf` sont créés **dans le même dossier que la vidéo**

### Lecture des résultats

**Dans le tableau** :
- **FPS** : vitesse réelle d'inférence. Préférer le modèle le plus rapide si les FPS sont similaires.
- **MPJPE** : erreur en pixels. Préférer le plus petit.
- **PCK@0.1** : % de joints corrects. Préférer le plus grand.
- **Meilleur joint** : pour quel joint le modèle est le plus précis.
- **Pire joint** : point faible du modèle (à surveiller).

**Dans les graphiques** :
- La courbe MPJPE montre la stabilité frame par frame. Un modèle avec beaucoup de pics est moins stable.
- Les barres permettent la comparaison directe des moyennes.

---

## 11. Métriques d'évaluation

### Pourquoi ces métriques ?

Ces métriques sont les standards de la littérature académique pour l'évaluation de l'estimation de pose 2D :

| Métrique | Référence | Avantage |
|----------|-----------|----------|
| MPJPE 2D | Standard HPE (Human Pose Estimation) | Simple, interprétable en pixels |
| PCK@0.1 | Yang & Ramanan (2013) | Robusse à l'échelle, invariant à la résolution |
| FPS | Standard industry | Mesure la viabilité temps réel |

### MPJPE — références de la littérature

Pour des images à résolution typique (720p à 1080p) :
- **Excellent** : < 5 px
- **Bon** : 5-15 px
- **Acceptable** : 15-30 px
- **Mauvais** : > 30 px

### PCK@0.1 — références

- **Excellent** : > 90%
- **Bon** : 80-90%
- **Acceptable** : 70-80%
- **Mauvais** : < 70%

---

## 12. Standardisation des squelettes

Le principal défi de la comparaison inter-modèles est l'**hétérogénéité des conventions de squelette**. Chaque modèle numérote ses joints différemment et en détecte un nombre différent.

### Mapping complet

| Joint commun | BlazePose | YOLO (COCO) | OpenPose BODY_25 | Fit3D joints3d_25 |
|-------------|-----------|-------------|-----------------|-------------------|
| left_shoulder | 11 | 5 | 5 | 5 |
| right_shoulder | 12 | 6 | 2 | 2 |
| left_elbow | 13 | 7 | 6 | 6 |
| right_elbow | 14 | 8 | 3 | 3 |
| left_wrist | 15 | 9 | 7 | 7 |
| right_wrist | 16 | 10 | 4 | 4 |
| left_hip | 23 | 11 | 12 | 12 |
| right_hip | 24 | 12 | 9 | 9 |
| left_knee | 25 | 13 | 13 | 13 |
| right_knee | 26 | 14 | 10 | 10 |
| left_ankle | 27 | 15 | 14 | 14 |
| right_ankle | 28 | 16 | 11 | 11 |

### Convention gauche/droite

Dans Fit3D (comme dans OpenPose BODY_25) :
- **Gauche** = gauche du sujet (côté gauche de la personne filmée)
- **Droite** = droite du sujet

Cette convention est cohérente avec la vue de face (caméra face au sujet). Pour les caméras latérales, la visibilité des membres dépend de la caméra choisie.

---

## 13. Résultats et fichiers de sortie

### `results.csv`

Créé automatiquement dans le même dossier que la vidéo à l'issue de l'évaluation.

**Usage** : Import dans Excel, pandas, ou tout outil d'analyse pour générer d'autres visualisations.

```python
import pandas as pd
df = pd.read_csv("results.csv")
```

### `report.pdf`

Rapport complet 6 pages créé avec matplotlib PdfPages. Prêt à être intégré dans un mémoire de master ou un rapport de stage.

**Pages du rapport** :
1. Page de titre + tableau résumé coloré
2. Comparaison MPJPE (barres horizontales + barres d'erreur)
3. Comparaison FPS (barres verticales)
4. Scatter plot précision × vitesse (identification du meilleur compromis)
5. Analyse par articulation (barres groupées)
6. Évolution temporelle du MPJPE frame par frame

---

## 14. Dépendances et installation

### Dépendances Python

| Package | Usage | Version min |
|---------|-------|-------------|
| `mediapipe` | BlazePose | 0.10.0 |
| `ultralytics` | YOLOv8-Pose | 8.0.0 |
| `opencv-python` | Lecture vidéo, OpenPose DNN, dessin | 4.8.0 |
| `numpy` | Calculs numériques, matrices | 1.24.0 |
| `PyQt5` | Interface graphique | 5.15.0 |
| `matplotlib` | Graphiques live + export PDF | 3.7.0 |
| `pandas` | (optionnel) Analyse CSV | 2.0.0 |

### Installation

```bash
pip install -r requirements.txt
```

### Fichiers modèle OpenPose (optionnel)

Si vous souhaitez utiliser OpenPose :

```bash
mkdir openpose_models && cd openpose_models

# Télécharger le prototxt (petit fichier texte)
wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt -O pose_deploy_linevec.prototxt

# Télécharger le modèle Caffe (428 MB)
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel -O pose_iter_440000.caffemodel
```

---

## 15. Limites et perspectives

### Limites actuelles

1. **OpenPose CPU uniquement** : L'implémentation via OpenCV DNN tourne sur CPU, ce qui le rend beaucoup plus lent que si on utilisait le package OpenPose officiel avec GPU CUDA.

2. **Une seule personne supposée** : L'application suppose qu'il n'y a qu'une personne dans le champ (valide pour Fit3D). Pour des scènes multi-personnes, la sélection de la bonne personne serait plus complexe.

3. **12 joints sur 25** : On n'évalue que 12 des 25 joints disponibles. Le visage, les pieds et les doigts sont exclus de la comparaison.

4. **Synchronisation frame-GT** : On suppose que la frame N de la vidéo correspond exactement au joint N dans le fichier JSON. Si la vidéo a été recoupée ou rééchantillonnée, cette correspondance pourrait être décalée.

### Améliorations possibles

- **GPU support** : Utiliser CUDA via OpenCV DNN ou torch pour accélérer OpenPose
- **Evaluation multi-exercices** : Batchifier l'évaluation sur plusieurs exercices automatiquement
- **Analyse temporelle** : Calculer le taux de frames où un joint est perdu (occlusion)
- **Export vidéo annotée** : Exporter les 3 panneaux annotés en vidéo `.mp4`
- **Interface de comparaison** : Permettre la sélection d'une articulation spécifique et zoomer dessus
