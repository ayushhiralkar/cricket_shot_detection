import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models import CricketMLP, CricketLSTM
from src.data_collector import generate_synthetic_dataset, CLASSES, FEATURE_COLUMNS

# PyTorch Dataset wrappers
class FrameDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_sklearn_models(X_train, X_test, y_train, y_test, model_dir):
    """
    Train and save Random Forest & SVM models.
    """
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    
    print("\n--- Training Support Vector Machine ---")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    print(f"SVM Accuracy: {svm_acc*100:.2f}%")
    
    # Save sklearn models
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))
    joblib.dump(svm, os.path.join(model_dir, "svm.pkl"))
    
    return rf, svm, rf_acc, svm_acc

def train_pytorch_mlp(X_train, X_test, y_train, y_test, num_classes, model_dir, epochs=30, batch_size=32):
    """
    Train PyTorch MLP model on frame-level keypoint features.
    """
    print("\n--- Training PyTorch MLP ---")
    train_dataset = FrameDataset(X_train, y_train)
    test_dataset = FrameDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CricketMLP(input_dim=X_train.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    # Evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            
    mlp_acc = accuracy_score(y_test, all_preds)
    print(f"PyTorch MLP Accuracy: {mlp_acc*100:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(model_dir, "mlp_model.pth"))
    return model, mlp_acc

def train_pytorch_lstm(sequence_data_path, label_encoder, num_classes, model_dir, epochs=40, batch_size=16):
    """
    Train PyTorch LSTM sequence classification model.
    """
    print("\n--- Training PyTorch LSTM Sequence Model ---")
    
    with open(sequence_data_path, 'rb') as f:
        data = pickle.load(f)
        
    sequences = np.array([item['features'] for item in data])
    labels = np.array([item['label'] for item in data])
    
    # Encode labels
    encoded_labels = label_encoder.transform(labels)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
    )
    
    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CricketLSTM(input_dim=sequences.shape[2], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    # Evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            
    lstm_acc = accuracy_score(y_test, all_preds)
    print(f"PyTorch LSTM Accuracy: {lstm_acc*100:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(model_dir, "lstm_model.pth"))
    return model, lstm_acc

def main():
    data_dir = "data/processed"
    model_dir = "models"
    
    csv_path = os.path.join(data_dir, "pose_dataset.csv")
    pkl_path = os.path.join(data_dir, "sequence_dataset.pkl")
    
    # Check if dataset exists, if not generate synthetic data
    if not os.path.exists(csv_path) or not os.path.exists(pkl_path):
        generate_synthetic_dataset(csv_path, pkl_path)
        
    # Load frame-level features
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS].values
    y_raw = df['label'].values
    
    # Label encode target classes
    le = LabelEncoder()
    le.fit(CLASSES) # Force consistent order
    y = le.transform(y_raw)
    
    # Save label encoder
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    # Split frame-level data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 1. Train Sklearn Models (RF, SVM)
    rf, svm, rf_acc, svm_acc = train_sklearn_models(X_train, X_test, y_train, y_test, model_dir)
    
    # 2. Train PyTorch MLP
    mlp, mlp_acc = train_pytorch_mlp(X_train, X_test, y_train, y_test, len(CLASSES), model_dir)
    
    # 3. Train PyTorch LSTM
    lstm, lstm_acc = train_pytorch_lstm(pkl_path, le, len(CLASSES), model_dir)
    
    # Save performance summary metrics to render in Streamlit dashboard
    metrics = {
        'Random Forest': float(rf_acc),
        'Support Vector Machine': float(svm_acc),
        'Neural Network (MLP)': float(mlp_acc),
        'LSTM (Sequence)': float(lstm_acc)
    }
    
    with open(os.path.join(model_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("\nTraining completed successfully! Saved all models and encoder artifacts to directory: 'models/'.")

if __name__ == "__main__":
    main()
