"""
=============================================================================
metrics.py
=============================================================================
Calcul des métriques d'évaluation de la pose 2D.

Métriques implémentées :
    1. MPJPE (Mean Per Joint Position Error)
       Mesure l'erreur de localisation en pixels.
       Plus petit = meilleur.

    2. PCK@0.1 (Percentage of Correct Keypoints)
       % de joints dont l'erreur est < 10% de la taille de la bounding box.
       Plus grand = meilleur (max 100%).

    3. FPS — géré dans pose_estimators.py (via time.perf_counter)

Classes :
    MetricsAccumulator : accumule les résultats frame par frame
                         et calcule les agrégats (MPJPE moyen, PCK moyen)

=============================================================================
"""

import numpy as np
from typing import Optional
from collections import defaultdict


# =============================================================================
# FONCTIONS DE BASE
# =============================================================================

def compute_mpjpe(pred: dict, gt: dict) -> Optional[float]:
    """
    Calcule le MPJPE (Mean Per Joint Position Error) pour une frame.

    Formule :
        MPJPE = (1/N) × Σ ||pred_i - gt_i||₂

    Seuls les joints présents à la fois dans pred et gt sont utilisés.

    Paramètres :
        pred (dict) : {joint_name: (x, y)} — prédiction du modèle
        gt   (dict) : {joint_name: (x, y)} — ground truth

    Retourne :
        MPJPE en pixels (float), ou None si aucun joint commun.
    """
    errors = []
    for joint_name in gt:
        if joint_name in pred:
            px_pred, py_pred = pred[joint_name]
            px_gt,   py_gt   = gt[joint_name]

            # Distance euclidienne en pixels
            dist = np.sqrt((px_pred - px_gt) ** 2 + (py_pred - py_gt) ** 2)
            errors.append(dist)

    if not errors:
        return None

    return float(np.mean(errors))


def compute_pck(pred: dict, gt: dict, bbox_size: Optional[float],
                threshold: float = 0.1) -> Optional[float]:
    """
    Calcule le PCK@threshold (Percentage of Correct Keypoints).

    Un joint est dit "correct" si sa distance à la GT est inférieure à
    threshold × bbox_size.

    Paramètre PCK@0.1 standard :
        threshold = 0.1 (10% de la bounding box)

    Paramètres :
        pred      : {joint_name: (x, y)} — prédiction
        gt        : {joint_name: (x, y)} — ground truth
        bbox_size : taille de référence (diagonale bounding box) en pixels
        threshold : seuil en fraction de bbox_size (défaut 0.1 → PCK@0.1)

    Retourne :
        PCK en % [0..100] (float), ou None si non calculable.
    """
    if bbox_size is None or bbox_size <= 0:
        return None

    # Seuil absolu en pixels
    threshold_px = threshold * bbox_size

    correct = 0
    total   = 0

    for joint_name in gt:
        if joint_name in pred:
            px_pred, py_pred = pred[joint_name]
            px_gt,   py_gt   = gt[joint_name]

            dist = np.sqrt((px_pred - px_gt) ** 2 + (py_pred - py_gt) ** 2)

            if dist <= threshold_px:
                correct += 1
            total += 1

    if total == 0:
        return None

    return float(correct / total * 100.0)


def compute_per_joint_error(pred: dict, gt: dict) -> dict:
    """
    Calcule l'erreur de distance pour chaque joint individuellement.

    Paramètres :
        pred : {joint_name: (x, y)}
        gt   : {joint_name: (x, y)}

    Retourne :
        dict {joint_name: erreur_pixels}
        Seuls les joints communs sont inclus.
    """
    errors = {}
    for joint_name in gt:
        if joint_name in pred:
            px_pred, py_pred = pred[joint_name]
            px_gt,   py_gt   = gt[joint_name]
            dist = np.sqrt((px_pred - px_gt) ** 2 + (py_pred - py_gt) ** 2)
            errors[joint_name] = float(dist)
    return errors


# =============================================================================
# ACCUMULATEUR DE MÉTRIQUES
# =============================================================================

class MetricsAccumulator:
    """
    Accumule les métriques au fil des frames pour un modèle donné.

    Permet de :
    - Ajouter les résultats d'une nouvelle frame (update)
    - Récupérer les métriques agrégées actuelles (get_summary)
    - Savoir quel joint est le meilleur/pire globalement

    Attributs :
        model_name   (str)          : nom du modèle
        mpjpe_per_frame (list)      : MPJPE pour chaque frame traitée
        pck_per_frame   (list)      : PCK@0.1 pour chaque frame traitée
        per_joint_errors (defaultdict) : {joint_name: [erreurs par frame]}
        n_frames     (int)          : nombre de frames accumulées
    """

    def __init__(self, model_name: str):
        self.model_name         = model_name
        self.mpjpe_per_frame    = []
        self.pck_per_frame      = []
        self.per_joint_errors   = defaultdict(list)  # joint → liste d'erreurs
        self.n_frames           = 0

    def update(self, pred: dict, gt: dict, bbox_size: Optional[float]):
        """
        Ajoute les résultats d'une frame à l'accumulateur.

        Paramètres :
            pred      : prédiction du modèle pour cette frame
            gt        : ground truth pour cette frame
            bbox_size : taille de la bounding box (pour PCK)
        """
        # MPJPE de la frame
        mpjpe = compute_mpjpe(pred, gt)
        if mpjpe is not None:
            self.mpjpe_per_frame.append(mpjpe)

        # PCK@0.1 de la frame
        pck = compute_pck(pred, gt, bbox_size, threshold=0.1)
        if pck is not None:
            self.pck_per_frame.append(pck)

        # Erreur par joint
        joint_errors = compute_per_joint_error(pred, gt)
        for joint_name, err in joint_errors.items():
            self.per_joint_errors[joint_name].append(err)

        self.n_frames += 1

    def get_summary(self) -> dict:
        """
        Calcule et retourne un résumé agrégé de toutes les frames.

        Retourne un dictionnaire contenant :
            model_name   (str)   : nom du modèle
            mpjpe_mean   (float) : MPJPE moyen en pixels
            mpjpe_std    (float) : écart-type du MPJPE
            pck_mean     (float) : PCK@0.1 moyen en %
            best_joint   (str)   : joint avec la plus petite erreur moyenne
            worst_joint  (str)   : joint avec la plus grande erreur moyenne
            per_joint_mpjpe (dict) : erreur moyenne par joint
            n_frames     (int)   : nombre de frames traitées
        """
        summary = {
            "model_name":     self.model_name,
            "mpjpe_mean":     0.0,
            "mpjpe_std":      0.0,
            "pck_mean":       0.0,
            "best_joint":     "N/A",
            "worst_joint":    "N/A",
            "per_joint_mpjpe": {},
            "n_frames":       self.n_frames,
        }

        # MPJPE agrégé
        if self.mpjpe_per_frame:
            summary["mpjpe_mean"] = float(np.mean(self.mpjpe_per_frame))
            summary["mpjpe_std"]  = float(np.std(self.mpjpe_per_frame))

        # PCK agrégé
        if self.pck_per_frame:
            summary["pck_mean"] = float(np.mean(self.pck_per_frame))

        # Erreur moyenne par joint
        per_joint_mean = {}
        for joint_name, errors in self.per_joint_errors.items():
            if errors:
                per_joint_mean[joint_name] = float(np.mean(errors))
        summary["per_joint_mpjpe"] = per_joint_mean

        # Meilleur et pire joint
        if per_joint_mean:
            summary["best_joint"]  = min(per_joint_mean, key=per_joint_mean.get)
            summary["worst_joint"] = max(per_joint_mean, key=per_joint_mean.get)

        return summary

    def get_mpjpe_history(self) -> list:
        """Retourne la liste des MPJPE par frame (pour le graphique en courbe)."""
        return list(self.mpjpe_per_frame)

    def reset(self):
        """Réinitialise l'accumulateur (utile pour relancer une évaluation)."""
        self.mpjpe_per_frame    = []
        self.pck_per_frame      = []
        self.per_joint_errors   = defaultdict(list)
        self.n_frames           = 0
