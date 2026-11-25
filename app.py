import os
import time
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.inference import CricketShotClassifier
from src.data_collector import CLASSES
from src.utils import get_coaching_feedback, draw_hud_overlay, estimate_shot_power, analyze_shot_timing

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Cricket Shot Detection System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    .glass-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00f5d4;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(0, 245, 212, 0.2);
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- CHECK MODELS ---
def models_exist():
    required = [
        "models/label_encoder.pkl",
        "models/scaler.pkl",
        "models/random_forest.pkl",
        "models/svm.pkl",
        "models/mlp_model.pth",
        "models/lstm_model.pth"
    ]
    return all(os.path.exists(f) for f in required)

if not models_exist():
    st.warning("⚠️ Models not found. Please train models first using `python -m src.train`.")
    st.stop()

@st.cache_resource
def get_classifier():
    return CricketShotClassifier()

classifier = get_classifier()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color: #00f5d4;'>Navigation</h3>", unsafe_allow_html=True)
    page = st.radio(
        "Select Page:",
        ["Dashboard Home", "Video Analysis Studio", "Webcam Arena", "Coaching Feedback", "Model Performance"]
    )
    
    st.markdown("---")
    model_choice = st.selectbox(
        "Classification Engine:",
        ["Random Forest", "Support Vector Machine", "Neural Network (MLP)", "LSTM (Sequence)"]
    )

# --- PAGE 1: HOME ---
if page == "Dashboard Home":
    st.markdown("<h1 style='text-align: center; color: #00f5d4;'>🏏 Cricket Shot Detection System</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-header">Project Overview</div>
            <p>Welcome to the AI-powered Cricket Shot Detection platform. This system utilizes <b>MediaPipe Pose</b> and <b>PyTorch/Scikit-Learn</b> to analyze cricket batting biomechanics.</p>
            <ul>
                <li>Frame-level detection using MLP, SVM, RF</li>
                <li>Sequence-level temporal detection using LSTM</li>
                <li>Real-time HUD and Shot Power Estimation</li>
                <li>Biomechanical feedback based on joint telemetry</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="glass-header">Supported Shots</div>
            <p>Cover Drive, Straight Drive, Pull Shot, Hook Shot, Cut Shot, Sweep Shot, Defensive Shot, Lofted Shot.</p>
        </div>
        """, unsafe_allow_html=True)

# --- PAGE 2: VIDEO ANALYSIS ---
elif page == "Video Analysis Studio":
    st.markdown("### 📽️ Video Analysis Studio")
    
    video_mode = st.radio("Choose Input:", ["Simulation Demo", "Upload Video"])
    
    uploaded_file = None
    if video_mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload Cricket Video (.mp4):", type=['mp4', 'mov', 'avi'])
        
    if st.button("🚀 Run Analysis"):
        classifier.reset_buffer()
        
        col1, col2 = st.columns([3, 2])
        video_placeholder = col1.empty()
        
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            lbl_metric = st.metric("Shot Detected", "Waiting...")
            conf_metric = st.metric("Confidence", "0.0%")
            pwr_metric = st.metric("Power", "0.0%")
            phase_metric = st.metric("Phase", "Stance")
            st.markdown("</div>", unsafe_allow_html=True)
            chart_placeholder = st.empty()
            
        wrist_velocities = []
        conf_history = []
        
        if video_mode == "Simulation Demo":
            # Simulate frames
            for t in range(30):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.circle(frame, (320, 150), 30, (255,255,255), -1) # Head
                cv2.line(frame, (320, 180), (320, 300), (255,255,255), 3) # Spine
                
                # Use classifier
                results, landmarks, features, raw_coords, label, confidence, probs = classifier.process_and_classify(frame, model_name=model_choice)
                
                # Mock values for demo
                label = "cover_drive"
                confidence = 0.85 + (t * 0.003)
                wrist_velocities.append(1.0 + t*0.2)
                
                power = estimate_shot_power(wrist_velocities)
                phase = analyze_shot_timing(wrist_velocities)
                
                # Draw HUD
                frame = draw_hud_overlay(frame, label, confidence, power, phase, features)
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                lbl_metric.metric("Shot Detected", label.replace('_', ' ').title())
                conf_metric.metric("Confidence", f"{confidence*100:.1f}%")
                pwr_metric.metric("Power", f"{power:.1f}%")
                phase_metric.metric("Phase", phase)
                
                conf_history.append(confidence)
                
                df = pd.DataFrame({'Frame': range(len(conf_history)), 'Confidence': conf_history})
                fig = px.line(df, x='Frame', y='Confidence', title="Prediction Confidence")
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                time.sleep(0.1)
                
        elif uploaded_file:
            tfile = "data/temp.mp4"
            with open(tfile, 'wb') as f:
                f.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile)
            prev_landmarks = None
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.resize(frame, (640, 480))
                results, landmarks, features, raw_coords, label, confidence, probs = classifier.process_and_classify(frame, prev_landmarks, model_choice)
                
                if features:
                    prev_landmarks = landmarks
                    wrist_velocities.append(features['l_wrist_vel'])
                    power = estimate_shot_power(wrist_velocities)
                    phase = analyze_shot_timing(wrist_velocities)
                    
                    frame = classifier.estimator.draw_pose(frame, results)
                    frame = draw_hud_overlay(frame, label, confidence, power, phase, features)
                    
                    video_placeholder.image(frame, channels="BGR", use_container_width=True)
                    lbl_metric.metric("Shot Detected", label.replace('_', ' ').title())
                    conf_metric.metric("Confidence", f"{confidence*100:.1f}%")
                    pwr_metric.metric("Power", f"{power:.1f}%")
                    phase_metric.metric("Phase", phase)
                    
            cap.release()

# --- PAGE 3: WEBCAM ---
elif page == "Webcam Arena":
    st.markdown("### 🎥 Webcam Arena")
    run_webcam = st.toggle("Start Webcam")
    
    col1, col2 = st.columns([3, 2])
    webcam_placeholder = col1.empty()
    
    with col2:
        lbl_metric = st.metric("Live Prediction", "Offline")
        conf_metric = st.metric("Confidence", "0.0%")
        
    if run_webcam:
        classifier.reset_buffer()
        cap = cv2.VideoCapture(0)
        prev_landmarks = None
        wrist_velocities = []
        
        while cap.isOpened() and run_webcam:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(cv2.resize(frame, (640, 480)), 1)
            
            results, landmarks, features, raw_coords, label, confidence, probs = classifier.process_and_classify(frame, prev_landmarks, model_choice)
            if features:
                prev_landmarks = landmarks
                wrist_velocities.append(features['l_wrist_vel'])
                frame = classifier.estimator.draw_pose(frame, results)
                frame = draw_hud_overlay(frame, label, confidence, estimate_shot_power(wrist_velocities), analyze_shot_timing(wrist_velocities), features)
                
                lbl_metric.metric("Live Prediction", label.replace('_', ' ').title())
                conf_metric.metric("Confidence", f"{confidence*100:.1f}%")
                
            webcam_placeholder.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.01)
        cap.release()

# --- PAGE 4: COACHING ---
elif page == "Coaching Feedback":
    st.markdown("### 🦺 Coaching & Form Auditing")
    col1, col2 = st.columns(2)
    with col1:
        target_shot = st.selectbox("Target Shot:", CLASSES)
        l_elbow = st.slider("Left Elbow Angle", 20, 180, 120)
        l_knee = st.slider("Left Knee Angle", 40, 180, 140)
        stride = st.slider("Foot Stride Length", 0.2, 2.0, 0.7)
    with col2:
        mock_features = {'l_elbow_angle': l_elbow, 'r_elbow_angle': l_elbow, 'l_knee_angle': l_knee, 'foot_stride': stride, 'torso_angle': 15}
        errors, successes = get_coaching_feedback(target_shot, mock_features)
        
        if successes:
            st.success("✔️ " + " | ".join(successes))
        if errors:
            st.warning("⚠️ " + " | ".join(errors))

# --- PAGE 5: PERFORMANCE ---
elif page == "Model Performance":
    st.markdown("### 📊 Model Performance")
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json") as f:
            metrics = json.load(f)
            
        fig = go.Figure([go.Bar(x=list(metrics.keys()), y=[v*100 for v in metrics.values()])])
        fig.update_layout(title="Accuracy Comparison (%)", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
