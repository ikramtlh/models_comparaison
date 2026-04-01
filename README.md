# 🏋️ Pose Estimation Models Comparison — Fit3D Benchmark

A desktop application for benchmarking and comparing three 2D human pose estimation models on the [Fit3D dataset](https://fit3d.imar.ro/):

| Model | Library | Skeleton colour |
|-------|---------|----------------|
| **BlazePose** | MediaPipe | 🟢 Green |
| **YOLO-Pose** | Ultralytics YOLOv8 | 🔵 Blue |
| **OpenPose** | OpenCV DNN (BODY_25) | 🔴 Red |

---

## ✨ Features

- **Split-screen visualisation** — 3 annotated video panels side by side
- **Ground Truth overlay** — 2D keypoints projected from 3D Fit3D joints using camera calibration
- **Real-time metrics** — MPJPE (px), PCK@0.1 (%), FPS, best/worst joint
- **Live charts** — MPJPE curve per frame + bar chart per model (matplotlib embedded in Qt)
- **Export** — `results.csv` (pandas) + `report.pdf` (matplotlib multi-page)

---

## 🗂️ Project Structure

```
models_comparaison/
├── main_app.py          # PyQt5 main window + QThread worker
├── pose_estimators.py   # Abstract PoseEstimator + 3 subclasses
├── ground_truth.py      # Fit3D loader + 3D→2D camera projection
├── metrics.py           # MPJPE, PCK@0.1, FPS accumulator
├── export_utils.py      # CSV + PDF report generation
├── requirements.txt     # Python dependencies
├── DOCUMENTATION.md     # Full technical documentation (FR)
└── s03/                 # ⚠️  Dataset — NOT included (see below)
```

---

## ⚙️ Installation

```bash
# 1 — Clone the repository
git clone https://github.com/<your-username>/models_comparaison.git
cd models_comparaison

# 2 — (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# 3 — Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset Setup (Fit3D)

The Fit3D dataset is **not included** in this repository. Download it from [fit3d.imar.ro](https://fit3d.imar.ro/) and place it as follows:

```
models_comparaison/
└── s03/
    ├── videos/
    │   └── <camera_id>/
    │       └── squat.mp4
    ├── joints3d_25/
    │   └── squat.json
    └── camera_parameters/
        └── <camera_id>/
            └── squat.json
```

---

## 🚀 Running the Application

```bash
python3 main_app.py
```

In the GUI:
1. **🎬 Charger vidéo** → select `s03/videos/<camera_id>/squat.mp4`
2. **📐 Charger Ground Truth** → select `s03/joints3d_25/squat.json`
3. **📷 Charger Calibration** → select `s03/camera_parameters/<camera_id>/squat.json`
4. Click **▶ Lancer l'évaluation**

Results (`results.csv` + `report.pdf`) are saved next to the video file.

---

## 📊 Metrics

| Metric | Description |
|--------|-------------|
| **MPJPE** | Mean Per Joint Position Error (pixels) — lower is better |
| **PCK@0.1** | % of joints within 10% of bounding box size — higher is better |
| **FPS** | Inference frames per second — higher is better |

---

## 📝 Documentation

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for full technical documentation in French, including architecture details, module descriptions, and interpretation of results.

---

## 🎓 Academic Context

This project is developed as part of a Master's thesis (M2) PFE — comparing pose estimation models for human motion analysis in exercise rehabilitation scenarios.

---

## 📄 License

MIT
