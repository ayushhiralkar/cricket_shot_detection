import os
import joblib
import torch
import numpy as np
from collections import deque
from src.pose_estimator import CricketPoseEstimator
from src.models import CricketMLP, CricketLSTM
from src.data_collector import FEATURE_COLUMNS, CLASSES

class CricketShotClassifier:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.estimator = CricketPoseEstimator()
        
        # Load encoders
        self.le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        
        # Load sklearn models
        self.rf = joblib.load(os.path.join(model_dir, "random_forest.pkl"))
        self.svm = joblib.load(os.path.join(model_dir, "svm.pkl"))
        
        # Load PyTorch MLP
        self.mlp = CricketMLP(input_dim=len(FEATURE_COLUMNS), num_classes=len(CLASSES))
        self.mlp.load_state_dict(torch.load(os.path.join(model_dir, "mlp_model.pth")))
        self.mlp.eval()
        
        # Load PyTorch LSTM
        self.lstm = CricketLSTM(input_dim=len(FEATURE_COLUMNS), num_classes=len(CLASSES))
        self.lstm.load_state_dict(torch.load(os.path.join(model_dir, "lstm_model.pth")))
        self.lstm.eval()
        
        # Sequence queue for LSTM
        self.sequence_buffer = deque(maxlen=30)
        
    def reset_buffer(self):
        self.sequence_buffer.clear()
        
    def predict_frame_ml(self, features, model_name="Random Forest"):
        """
        Predict shot class using frame-level machine learning models.
        """
        # Feature array: shape [1, num_features]
        feat_vector = np.array([[features[col] for col in FEATURE_COLUMNS]])
        scaled_feat = self.scaler.transform(feat_vector)
        
        if model_name == "Random Forest":
            probs = self.rf.predict_proba(scaled_feat)[0]
            pred_idx = np.argmax(probs)
            label = self.le.inverse_transform([pred_idx])[0]
            confidence = probs[pred_idx]
        elif model_name == "Support Vector Machine":
            probs = self.svm.predict_proba(scaled_feat)[0]
            pred_idx = np.argmax(probs)
            label = self.le.inverse_transform([pred_idx])[0]
            confidence = probs[pred_idx]
        elif model_name == "Neural Network (MLP)":
            with torch.no_grad():
                tensor_feat = torch.tensor(scaled_feat, dtype=torch.float32)
                logits = self.mlp(tensor_feat)
                probs = torch.softmax(logits, dim=1).numpy()[0]
                pred_idx = np.argmax(probs)
                label = self.le.inverse_transform([pred_idx])[0]
                confidence = probs[pred_idx]
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
        return label, confidence, probs

    def predict_sequence_lstm(self):
        """
        Predict shot class using temporal LSTM model.
        Requires the sequence buffer to be full (30 frames).
        """
        if len(self.sequence_buffer) < 30:
            return "Collecting Frames...", 0.0, None
            
        # Standardize the sequence of features
        seq_array = np.array(self.sequence_buffer) # Shape: [30, 17]
        scaled_seq = self.scaler.transform(seq_array) # Scale frame by frame
        
        # Shape: [1, 30, 17]
        tensor_seq = torch.tensor([scaled_seq], dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.lstm(tensor_seq)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = np.argmax(probs)
            label = self.le.inverse_transform([pred_idx])[0]
            confidence = probs[pred_idx]
            
        return label, confidence, probs

    def process_and_classify(self, frame, prev_landmarks=None, model_name="Random Forest"):
        """
        Run the complete pipeline for a single frame.
        """
        results, landmarks, features, raw_coords = self.estimator.process_frame(frame, prev_landmarks)
        
        label = "No Pose Detected"
        confidence = 0.0
        probabilities = None
        
        if features:
            # 1. Update sequence buffer for LSTM
            feat_vector = [features[col] for col in FEATURE_COLUMNS]
            self.sequence_buffer.append(feat_vector)
            
            # 2. Run prediction
            if model_name == "LSTM (Sequence)":
                label, confidence, probabilities = self.predict_sequence_lstm()
            else:
                label, confidence, probabilities = self.predict_frame_ml(features, model_name)
                
        return results, landmarks, features, raw_coords, label, confidence, probabilities
