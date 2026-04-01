"""
=============================================================================
export_utils.py
=============================================================================
Fonctions d'export des résultats d'évaluation.

Deux types d'export :
    1. CSV   — tableau tabulaire via pandas
    2. PDF   — rapport complet via matplotlib (multi-pages)

Structure du rapport PDF :
    Page 1 : Titre + résumé textuel
    Page 2 : Comparaison MPJPE (barres horizontales)
    Page 3 : Comparaison FPS (barres verticales)
    Page 4 : Scatter plot Précision × Vitesse
    Page 5 : Détail par articulation (barres groupées)
    Page 6 : Courbe MPJPE frame par frame

=============================================================================
"""

import csv
import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Mode sans affichage (pour génération PDF headless)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


# Palette de couleurs pour les 3 modèles (cohérente avec la GUI)
MODEL_COLORS = {
    "BlazePose":  "#00C853",   # Vert vif
    "YOLO-Pose":  "#2196F3",   # Bleu
    "OpenPose":   "#F44336",   # Rouge
}


def save_csv(results: dict, fps_data: dict,
             output_path: str = "results.csv") -> str:
    """
    Exporte les métriques dans un fichier CSV.

    Paramètres :
        results     : dict {model_name: summary_dict} depuis MetricsAccumulator
        fps_data    : dict {model_name: fps_value}
        output_path : chemin du fichier CSV à créer

    Retourne :
        Le chemin absolu du fichier créé.

    Format CSV :
        Modèle,FPS,MPJPE (px),Meilleur joint,Pire joint
    """
    abs_path = os.path.abspath(output_path)

    with open(abs_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # En-tête
        writer.writerow([
            "Modèle", "FPS", "MPJPE (px)", "MPJPE std",
            "Meilleur joint", "Pire joint", "Frames évaluées"
        ])

        # Une ligne par modèle
        for model_name, summary in results.items():
            fps   = fps_data.get(model_name, 0.0)
            writer.writerow([
                model_name,
                f"{fps:.2f}",
                f"{summary.get('mpjpe_mean', 0):.2f}",
                f"{summary.get('mpjpe_std', 0):.2f}",
                summary.get("best_joint", "N/A"),
                summary.get("worst_joint", "N/A"),
                summary.get("n_frames", 0),
            ])

        # Détail par joint (section séparée dans le même CSV)
        writer.writerow([])
        writer.writerow(["=== Erreur par articulation (MPJPE moyen en pixels) ==="])
        writer.writerow(["Joint"] + list(results.keys()))

        # Collecte de tous les joints
        all_joints = set()
        for summary in results.values():
            all_joints.update(summary.get("per_joint_mpjpe", {}).keys())

        for joint in sorted(all_joints):
            row = [joint]
            for model_name in results:
                err = results[model_name].get("per_joint_mpjpe", {}).get(joint, "")
                row.append(f"{err:.2f}" if isinstance(err, float) else "")
            writer.writerow(row)

    print(f"[Export] CSV sauvegardé : {abs_path}")
    return abs_path


def generate_pdf(results: dict, fps_data: dict, mpjpe_histories: dict,
                 output_path: str = "report.pdf") -> str:
    """
    Génère un rapport PDF complet avec tous les graphiques d'analyse.

    Paramètres :
        results         : dict {model_name: summary_dict}
        fps_data        : dict {model_name: fps_float}
        mpjpe_histories : dict {model_name: list[float]} — MPJPE par frame
        output_path     : chemin du fichier PDF à créer

    Retourne :
        Le chemin absolu du fichier créé.
    """
    abs_path = os.path.abspath(output_path)
    model_names = list(results.keys())
    colors      = [MODEL_COLORS.get(m, "#888888") for m in model_names]

    with PdfPages(abs_path) as pdf:

        # -----------------------------------------------------------------
        # PAGE 1 : Titre + résumé textuel
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27))   # A4 paysage
        ax.axis('off')

        # Titre principal
        fig.text(0.5, 0.92, "Rapport d'Évaluation des Modèles de Pose 2D",
                 fontsize=20, fontweight='bold', ha='center', va='top',
                 color='#1A237E')
        fig.text(0.5, 0.87,
                 "Dataset Fit3D | Métriques : MPJPE, FPS",
                 fontsize=12, ha='center', va='top', color='#555555')
        fig.text(0.5, 0.83,
                 f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
                 fontsize=10, ha='center', va='top', color='#777777',
                 style='italic')

        # Tableau récapitulatif
        table_data = [["Modèle", "FPS", "MPJPE (px)",
                        "Meilleur joint", "Pire joint"]]
        for m in model_names:
            s   = results[m]
            fps = fps_data.get(m, 0.0)
            table_data.append([
                m,
                f"{fps:.1f}",
                f"{s.get('mpjpe_mean',0):.2f} ± {s.get('mpjpe_std',0):.2f}",
                s.get("best_joint",  "N/A").replace("_", " ").title(),
                s.get("worst_joint", "N/A").replace("_", " ").title(),
            ])

        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            cellLoc='center',
            loc='center',
            bbox=[0.0, 0.3, 1.0, 0.45],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(table_data[0]))))

        # Colorier l'en-tête
        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor('#1A237E')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Colorier les lignes de données alternées
        for i, m in enumerate(model_names):
            clr = MODEL_COLORS.get(m, "#BBBBBB") + "44"  # Alpha hex
            for j in range(len(table_data[0])):
                table[(i+1, j)].set_facecolor(clr)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # -----------------------------------------------------------------
        # PAGE 2 : Comparaison MPJPE (barres horizontales)
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        mpjpe_vals = [results[m].get("mpjpe_mean", 0) for m in model_names]
        mpjpe_stds = [results[m].get("mpjpe_std", 0)  for m in model_names]

        bars = ax.barh(model_names, mpjpe_vals, xerr=mpjpe_stds,
                       color=colors, edgecolor='white', height=0.5,
                       capsize=6, error_kw=dict(ecolor='gray', lw=2))

        # Valeurs sur les barres
        for bar, val in zip(bars, mpjpe_vals):
            ax.text(bar.get_width() + max(mpjpe_stds)*0.1,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.2f} px", va='center', fontsize=11, fontweight='bold')

        ax.set_xlabel("MPJPE moyen (pixels) — plus petit = meilleur", fontsize=12)
        ax.set_title("Erreur de Localisation de Pose (MPJPE)", fontsize=16,
                     fontweight='bold', pad=15)
        ax.set_facecolor('#F8F8F8')
        ax.grid(axis='x', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        _add_footnote(fig, "MPJPE : Mean Per-Joint Position Error en pixels")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # -----------------------------------------------------------------
        # PAGE 3 : Comparaison FPS
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        fps_vals = [fps_data.get(m, 0) for m in model_names]

        bars = ax.bar(model_names, fps_vals, color=colors,
                      edgecolor='white', width=0.5)

        for bar, val in zip(bars, fps_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha='center', fontsize=12, fontweight='bold')

        ax.set_ylabel("FPS (Frames par seconde) — plus grand = meilleur", fontsize=12)
        ax.set_title("Vitesse d'Inférence (FPS)", fontsize=16,
                     fontweight='bold', pad=15)
        ax.set_facecolor('#F8F8F8')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        _add_footnote(fig, "FPS mesuré via time.perf_counter() sur toutes les frames évaluées")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # -----------------------------------------------------------------
        # PAGE 4 : Scatter plot Erreur × Vitesse
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        mpjpe_vals_scatter = [results[m].get("mpjpe_mean", 0) for m in model_names]

        for m, fps_v, mpjpe_v, c in zip(model_names, fps_vals, mpjpe_vals_scatter, colors):
            ax.scatter(fps_v, mpjpe_v, s=300, color=c, zorder=5, label=m)
            ax.annotate(m, (fps_v, mpjpe_v),
                        textcoords='offset points', xytext=(10, 5),
                        fontsize=12, fontweight='bold', color=c)

        ax.set_xlabel("Vitesse (FPS)", fontsize=13)
        ax.set_ylabel("Erreur (MPJPE en pixels)", fontsize=13)
        ax.set_title("Trade-off Erreur / Vitesse", fontsize=16,
                     fontweight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.set_facecolor('#F8F8F8')
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

        # Zone idéale (bas-droite)
        ax.annotate("Zone idéale\n(rapide + précis)", xy=(0.85, 0.15),
                    xycoords='axes fraction', fontsize=10,
                    color='#4CAF50', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9'))

        _add_footnote(fig, "Le modèle idéal est en bas à droite : faible erreur MPJPE ET haute vitesse FPS")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # -----------------------------------------------------------------
        # PAGE 5 : Détail par articulation (barres groupées)
        # -----------------------------------------------------------------
        all_joints = sorted({
            j for m in model_names
            for j in results[m].get("per_joint_mpjpe", {}).keys()
        })

        if all_joints:
            fig, ax = plt.subplots(figsize=(14, 8.27))
            n_joints = len(all_joints)
            n_models = len(model_names)
            x        = np.arange(n_joints)
            width    = 0.8 / n_models   # Largeur de chaque barre

            for i, (m, c) in enumerate(zip(model_names, colors)):
                per_joint = results[m].get("per_joint_mpjpe", {})
                vals = [per_joint.get(j, 0) for j in all_joints]
                ax.bar(x + i * width - (n_models - 1) * width / 2,
                       vals, width, label=m, color=c, edgecolor='white')

            joint_labels = [j.replace("_", " ").title() for j in all_joints]
            ax.set_xticks(x)
            ax.set_xticklabels(joint_labels, rotation=35, ha='right', fontsize=10)
            ax.set_ylabel("MPJPE moyen (pixels)", fontsize=12)
            ax.set_title("Erreur par Articulation", fontsize=16,
                         fontweight='bold', pad=15)
            ax.legend(fontsize=11)
            ax.set_facecolor('#F8F8F8')
            ax.grid(axis='y', alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # -----------------------------------------------------------------
        # PAGE 6 : Courbe MPJPE frame par frame
        # -----------------------------------------------------------------
        if any(mpjpe_histories.get(m) for m in model_names):
            fig, ax = plt.subplots(figsize=(14, 7))

            for m, c in zip(model_names, colors):
                hist = mpjpe_histories.get(m, [])
                if hist:
                    frames = list(range(len(hist)))
                    ax.plot(frames, hist, color=c, label=m,
                            linewidth=1.5, alpha=0.85)
                    # Ligne de moyenne
                    ax.axhline(np.mean(hist), color=c, linestyle='--',
                               alpha=0.6, linewidth=1)

            ax.set_xlabel("Frame", fontsize=12)
            ax.set_ylabel("MPJPE (pixels)", fontsize=12)
            ax.set_title("Évolution du MPJPE Frame par Frame", fontsize=16,
                         fontweight='bold', pad=15)
            ax.legend(fontsize=11)
            ax.set_facecolor('#F8F8F8')
            ax.grid(alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
            _add_footnote(fig, "Les lignes pointillées représentent le MPJPE moyen de chaque modèle")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # -----------------------------------------------------------------
        # PAGE 7 : Classement final + Recommandation
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')

        fig.text(0.5, 0.96, "Classement Final & Recommandation",
                 fontsize=18, fontweight='bold', ha='center', va='top',
                 color='#1A237E')
        fig.text(0.5, 0.91,
                 "Synthèse comparative — BlazePose · YOLO-Pose · OpenPose",
                 fontsize=11, ha='center', va='top', color='#555555',
                 style='italic')

        medals = ["🥇", "🥈", "🥉"]

        def ranking_block(ax_fig, title, ranked, unit, note, y_top):
            """Dessine un bloc classement centré."""
            ax_fig.text(0.5, y_top, title, fontsize=13, fontweight='bold',
                        ha='center', va='top', color='#1A237E')
            ax_fig.text(0.5, y_top - 0.04, note, fontsize=8,
                        ha='center', va='top', color='#777777', style='italic')
            for rank, (model, val) in enumerate(ranked):
                color = MODEL_COLORS.get(model, "#888888")
                line = f"{medals[rank]}  {model}  —  {val:.2f} {unit}"
                ax_fig.text(0.5, y_top - 0.10 - rank * 0.07, line,
                            fontsize=12, ha='center', va='top',
                            color=color, fontweight='bold')

        # --- Classement par FPS (plus grand = meilleur) ---
        fps_sorted = sorted(model_names, key=lambda m: fps_data.get(m, 0), reverse=True)
        fps_ranked = [(m, fps_data.get(m, 0)) for m in fps_sorted]
        ranking_block(fig, "🚀  Classement Vitesse (FPS)", fps_ranked,
                      "fps", "Plus élevé = meilleur", 0.78)

        # --- Classement par MPJPE (plus petit = meilleur) ---
        mpjpe_sorted = sorted(model_names, key=lambda m: results[m].get("mpjpe_mean", 9999))
        mpjpe_ranked = [(m, results[m].get("mpjpe_mean", 0)) for m in mpjpe_sorted]
        ranking_block(fig, "🎯  Classement Précision (MPJPE)", mpjpe_ranked,
                      "px", "Plus faible = meilleur", 0.44)

        # --- Score pondéré et recommandation ---
        # Normalisation : rang FPS + rang MPJPE
        n = len(model_names)
        scores = {m: 0 for m in model_names}
        for rank, (m, _) in enumerate(fps_ranked):
            scores[m] += (n - rank)        # FPS : poids 1
        for rank, (m, _) in enumerate(mpjpe_ranked):
            scores[m] += (n - rank)        # MPJPE : poids 1
        best = max(scores, key=scores.get)
        best_color = MODEL_COLORS.get(best, "#1A237E")

        fig.text(0.5, 0.08,
                 f"✅  Modèle recommandé pour cette application :  {best}",
                 fontsize=15, fontweight='bold', ha='center', va='bottom',
                 color='white',
                 bbox=dict(boxstyle='round,pad=0.6',
                           facecolor=best_color, alpha=0.9))
        score_txt = "   |   ".join(
            [f"{m} : {scores[m]} pt{'s' if scores[m] > 1 else ''}" for m in model_names]
        )
        fig.text(0.5, 0.02, f"Score pondéré (FPS + MPJPE) : {score_txt}",
                 fontsize=8, ha='center', va='bottom',
                 color='#666666', style='italic')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


        # Métadonnées du PDF
        d = pdf.infodict()
        d['Title']   = "Rapport d'Évaluation des Modèles de Pose 2D"
        d['Author']  = "Système de Comparaison de Pose — PFE Master 2"
        d['Subject'] = "Comparaison BlazePose, YOLO-Pose, OpenPose sur Fit3D"

    print(f"[Export] Rapport PDF sauvegardé : {abs_path}")
    return abs_path


def _add_footnote(fig, text: str):
    """Ajoute une note de bas de page discrète à une figure matplotlib."""
    fig.text(0.5, 0.01, text,
             ha='center', va='bottom', fontsize=8,
             color='#888888', style='italic')
