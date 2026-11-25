# AI Cricket Shot Detection System

This repository contains a complete, production-ready AI-based **Cricket Shot Detection and Analysis System** built using Python, Computer Vision, and Machine Learning. The application utilizes a professional-grade Streamlit web dashboard to stream webcams or analyze uploaded video footage, rendering real-time biomechanics telemetry HUD overlays.

---

## 🚀 Key System Features

1. **Precision 3D Pose Tracking:**
   Uses Google MediaPipe Pose models to map 33 key skeleton vertices in 3D coordinate space.
2. **Advanced Feature Engineering:**
   Extracts translation-invariant physical traits including:
   - Elbow and Knee flexion angles.
   - Hip rotation and Torso tilt (spine posture).
   - Foot stride width ratio.
   - Temporal wrist velocity and ankle acceleration curves.
3. **Multi-Model Support:**
   Supports comparative classification using multiple model types:
   - **Random Forest & Support Vector Machine (SVM)** (Frame-level ML baselines)
   - **Multi-layer Perceptron (MLP)** (Frame-level Deep Learning)
   - **LSTM (Long Short-Term Memory)** (Temporal sequence-level sequence classification)
4. **Intelligent HUD & Telemetry Overlay:**
   Draws skeletons, angle readings, and a prediction speedometer directly on video feeds.
5. **AI Coaching Audit Engine:**
   Compares user joint metrics against professional batting profile thresholds, outputting constructive recommendations (e.g. *"Raise your elbow"* or *"Step forward more"*).
6. **Automatic Highlights & Metrics Reporting:**
   Tracks execution power, timings (stance, backswing, strike, follow-through), and models validation metrics.

---

## 📂 Project Architecture

```
cricket_shot_detection/
├── app.py                      # Streamlit dashboard app
├── requirements.txt            # Package dependencies
├── README.md                   # System documentation
├── src/
│   ├── __init__.py
│   ├── pose_estimator.py       # Landmark extraction & angle formulas
│   ├── data_collector.py       # Video processing & synthetic generator
│   ├── models.py               # PyTorch MLP & LSTM architectures
│   ├── train.py                # ML/DL training and evaluation scripts
│   ├── inference.py            # Streamlit-to-Model inference wrappers
│   └── utils.py                # HUD, timing, power, and coaching rule calculations
├── data/
│   ├── raw_videos/             # Directory for raw video clips (by class name)
│   └── processed/              # Compiled CSV/PKL datasets
└── models/                     # Saved model checkpoints and scaler scalars
```

---

## 🛠️ Installation & Getting Started

### 1. Set Workspace and Install Requirements
We recommend opening the `cricket_shot_detection` folder as your active workspace, then installing dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
There are two ways to get datasets:
- **Automatic Simulation Mode (Recommended for testing):** Run the train script or open the web app. It will automatically generate high-fidelity synthetic biomechanical profiles for all 8 cricket shots to let you test immediately.
- **Custom Video Dataset Mode:** Create directories under `data/raw_videos` for each cricket shot class name (e.g., `cover_drive/`, `sweep_shot/`, etc.). Place your `.mp4`/`.mov`/`.avi` clips in the folders, then run:

```bash
python -m src.data_collector
```

### 3. Model Training
Run the training script to train and save the Random Forest, SVM, MLP, and LSTM architectures:

```bash
python -m src.train
```
This generates saved model checkoints in `models/` along with `metrics.json`.

### 4. Launch the Web Application
Start the interactive Streamlit telemetry dashboard:

```bash
streamlit run app.py
```

---

## 🧠 Feature Extraction Mathematics

### Joint Flexion Angles
Calculated using the cosine similarity dot product formula for three joint vertices $A$, $B$ (vertex), and $C$:
$$\vec{u} = A - B, \quad \vec{v} = C - B$$
$$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right) \times \frac{180}{\pi}$$

### Stride Ratio
We divide Euclidean distance between ankles by the shoulder width to ensure player height scale-invariance:
$$\text{Stride Ratio} = \frac{\|\text{Ankle}_L - \text{Ankle}_R\|_2}{\|\text{Shoulder}_L - \text{Shoulder}_R\|_2}$$
