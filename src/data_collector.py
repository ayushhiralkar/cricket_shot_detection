import os
import cv2
import pickle
import numpy as np
import pandas as pd
from src.pose_estimator import CricketPoseEstimator

# Define the classes
CLASSES = [
    'cover_drive',
    'straight_drive',
    'pull_shot',
    'hook_shot',
    'cut_shot',
    'sweep_shot',
    'defensive_shot',
    'lofted_shot'
]

FEATURE_COLUMNS = [
    'l_elbow_angle', 'r_elbow_angle',
    'l_knee_angle', 'r_knee_angle',
    'l_hip_angle', 'r_hip_angle',
    'l_shoulder_angle', 'r_shoulder_angle',
    'torso_angle',
    'l_wr_to_hip', 'r_wr_to_hip',
    'hand_dist', 'foot_stride',
    'l_wrist_vel', 'r_wrist_vel',
    'l_ankle_vel', 'r_ankle_vel'
]

def collect_from_videos(video_dir, output_csv_path, output_pkl_path, seq_len=30):
    """
    Process videos in subfolders and save features.
    video_dir: Root directory of videos (e.g. data/raw_videos)
    """
    print(f"Starting dataset collection from {video_dir}...")
    estimator = CricketPoseEstimator()
    
    frame_records = []
    sequence_records = []
    
    # Check if directories exist
    if not os.path.exists(video_dir):
        print(f"Error: Directory {video_dir} not found. Please place your videos there.")
        return False
        
    for label in CLASSES:
        label_dir = os.path.join(video_dir, label)
        if not os.path.isdir(label_dir):
            print(f"Warning: Directory {label_dir} does not exist. Skipping.")
            continue
            
        video_files = [f for f in os.listdir(label_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"Processing {len(video_files)} videos for class '{label}'...")
        
        for video_file in video_files:
            video_path = os.path.join(label_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            video_sequence = []
            prev_landmarks = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                _, landmarks, features, _ = estimator.process_frame(frame, prev_landmarks)
                
                if features:
                    # Save frame records
                    record = {col: features[col] for col in FEATURE_COLUMNS}
                    record['label'] = label
                    record['video_source'] = video_file
                    frame_records.append(record)
                    
                    # Track landmarks to compute velocity next frame
                    prev_landmarks = landmarks
                    
                    # Save for sequence
                    feature_vector = [features[col] for col in FEATURE_COLUMNS]
                    video_sequence.append(feature_vector)
                else:
                    prev_landmarks = None
                    
            cap.release()
            
            # Extract sliding windows for sequence classification (LSTM)
            if len(video_sequence) >= seq_len:
                for i in range(0, len(video_sequence) - seq_len + 1, 5): # Stride of 5
                    window = video_sequence[i : i + seq_len]
                    sequence_records.append({
                        'features': np.array(window),
                        'label': label
                    })
                    
    # Save frame-level features
    if frame_records:
        df = pd.DataFrame(frame_records)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved {len(df)} frame records to {output_csv_path}")
    else:
        print("No frame records collected.")
        
    # Save sequence-level features
    if sequence_records:
        os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(sequence_records, f)
        print(f"Successfully saved {len(sequence_records)} sequence records to {output_pkl_path}")
        
    return True

def generate_synthetic_dataset(output_csv_path, output_pkl_path, num_sequences_per_class=40, seq_len=30):
    """
    Generate synthetic dataset mimicking actual cricket shots for testing and bootstrapping.
    """
    print("Generating high-quality synthetic dataset mimicking cricket shot biomechanics...")
    
    frame_records = []
    sequence_records = []
    
    for label in CLASSES:
        for seq_idx in range(num_sequences_per_class):
            sequence_features = []
            
            # We model each shot as a time series of angles/velocities over 30 frames (1 second at 30 fps)
            # Shot timeline: Preparation (0-10 frames), Swing/Strike (10-20 frames), Follow-through (20-30 frames)
            for t in range(seq_len):
                # Baseline posture (standing, slightly bent knees)
                base_features = {
                    'l_elbow_angle': 140.0 + np.random.normal(0, 3),
                    'r_elbow_angle': 140.0 + np.random.normal(0, 3),
                    'l_knee_angle': 160.0 + np.random.normal(0, 2),
                    'r_knee_angle': 160.0 + np.random.normal(0, 2),
                    'l_hip_angle': 150.0 + np.random.normal(0, 3),
                    'r_hip_angle': 150.0 + np.random.normal(0, 3),
                    'l_shoulder_angle': 45.0 + np.random.normal(0, 2),
                    'r_shoulder_angle': 45.0 + np.random.normal(0, 2),
                    'torso_angle': 15.0 + np.random.normal(0, 1),
                    'l_wr_to_hip': 0.8 + np.random.normal(0, 0.05),
                    'r_wr_to_hip': 0.8 + np.random.normal(0, 0.05),
                    'hand_dist': 0.15 + np.random.normal(0, 0.01),
                    'foot_stride': 0.6 + np.random.normal(0, 0.05),
                    'l_wrist_vel': 0.1 + np.random.normal(0, 0.02),
                    'r_wrist_vel': 0.1 + np.random.normal(0, 0.02),
                    'l_ankle_vel': 0.05 + np.random.normal(0, 0.01),
                    'r_ankle_vel': 0.05 + np.random.normal(0, 0.01),
                }
                
                # Apply class-specific biomechanics curves
                if label == 'cover_drive':
                    # Front-foot step, high elbow, downward/forward swing
                    if 5 <= t < 15:
                        base_features['foot_stride'] += (t - 5) * 0.08  # Stride length increases
                        base_features['l_knee_angle'] -= (t - 5) * 3.5  # Knee bending (front foot)
                        base_features['l_elbow_angle'] -= (t - 5) * 2.0  # Elbow bending (cocking bat)
                        base_features['l_wrist_vel'] += (t - 5) * 0.15
                        base_features['r_wrist_vel'] += (t - 5) * 0.15
                    elif 15 <= t < 25:
                        base_features['foot_stride'] += 0.8
                        base_features['l_knee_angle'] -= 35.0
                        base_features['l_elbow_angle'] += (t - 15) * 4.0  # Elbow release/extension
                        base_features['l_wrist_vel'] += (25 - t) * 0.12 + 1.0
                        base_features['r_wrist_vel'] += (25 - t) * 0.12 + 1.0
                        
                elif label == 'straight_drive':
                    # Straight forward drive, balanced knees, vertical bat swing
                    if 5 <= t < 15:
                        base_features['foot_stride'] += (t - 5) * 0.06
                        base_features['l_knee_angle'] -= (t - 5) * 2.5
                        base_features['l_elbow_angle'] -= (t - 5) * 1.5
                        base_features['l_wrist_vel'] += (t - 5) * 0.12
                        base_features['r_wrist_vel'] += (t - 5) * 0.12
                    elif 15 <= t < 25:
                        base_features['foot_stride'] += 0.6
                        base_features['l_knee_angle'] -= 25.0
                        base_features['l_elbow_angle'] += (t - 15) * 3.5
                        base_features['l_wrist_vel'] += (25 - t) * 0.10 + 0.8
                        base_features['r_wrist_vel'] += (25 - t) * 0.10 + 0.8
                        
                elif label == 'pull_shot':
                    # Backfoot weight transfer, wide horizontal bat swing
                    if 5 <= t < 15:
                        base_features['foot_stride'] += (t - 5) * 0.03
                        base_features['r_knee_angle'] -= (t - 5) * 2.0  # Weight back
                        base_features['l_elbow_angle'] -= (t - 5) * 3.0
                        base_features['r_elbow_angle'] -= (t - 5) * 3.0
                        base_features['torso_angle'] += (t - 5) * 1.5  # Lean back
                    elif 15 <= t < 25:
                        base_features['foot_stride'] += 0.3
                        base_features['r_knee_angle'] -= 20.0
                        base_features['l_elbow_angle'] += (t - 15) * 5.0  # Strong arms extension
                        base_features['r_elbow_angle'] += (t - 15) * 5.0
                        base_features['l_wrist_vel'] += (25 - t) * 0.2 + 1.5
                        base_features['r_wrist_vel'] += (25 - t) * 0.2 + 1.5
                        base_features['torso_angle'] += 15.0
                        
                elif label == 'hook_shot':
                    # High bouncer, hands high above shoulders, torso bent back
                    if 5 <= t < 15:
                        base_features['l_elbow_angle'] -= (t - 5) * 3.0
                        base_features['r_elbow_angle'] -= (t - 5) * 3.0
                        base_features['torso_angle'] += (t - 5) * 2.5
                        base_features['l_shoulder_angle'] += (t - 5) * 4.0  # Raising hands
                        base_features['r_shoulder_angle'] += (t - 5) * 4.0
                    elif 15 <= t < 25:
                        base_features['l_elbow_angle'] += (t - 15) * 4.0
                        base_features['r_elbow_angle'] += (t - 15) * 4.0
                        base_features['torso_angle'] += 25.0
                        base_features['l_shoulder_angle'] += 40.0
                        base_features['r_shoulder_angle'] += 40.0
                        base_features['l_wrist_vel'] += (25 - t) * 0.2 + 1.8
                        base_features['r_wrist_vel'] += (25 - t) * 0.2 + 1.8
                        
                elif label == 'cut_shot':
                    # Lateral swing, weight on back foot, hand extend outward
                    if 5 <= t < 15:
                        base_features['r_knee_angle'] -= (t - 5) * 3.0
                        base_features['l_elbow_angle'] -= (t - 5) * 2.0
                        base_features['r_elbow_angle'] -= (t - 5) * 2.0
                        base_features['l_wr_to_hip'] += (t - 5) * 0.05
                        base_features['r_wr_to_hip'] += (t - 5) * 0.05
                    elif 15 <= t < 25:
                        base_features['r_knee_angle'] -= 30.0
                        base_features['l_elbow_angle'] += (t - 15) * 4.5
                        base_features['r_elbow_angle'] += (t - 15) * 4.5
                        base_features['l_wr_to_hip'] += 0.5
                        base_features['r_wr_to_hip'] += 0.5
                        base_features['l_wrist_vel'] += (25 - t) * 0.18 + 1.4
                        base_features['r_wrist_vel'] += (25 - t) * 0.18 + 1.4
                        
                elif label == 'sweep_shot':
                    # Drop down completely on knee, horizontal sweep swing
                    if 5 <= t < 15:
                        base_features['l_knee_angle'] -= (t - 5) * 6.0  # Drop down low
                        base_features['r_knee_angle'] -= (t - 5) * 5.0
                        base_features['l_hip_angle'] -= (t - 5) * 4.5
                        base_features['torso_angle'] += (t - 5) * 1.5
                    elif 15 <= t < 25:
                        base_features['l_knee_angle'] = 40.0 + np.random.normal(0, 2)  # Fully knelt
                        base_features['r_knee_angle'] = 60.0 + np.random.normal(0, 2)
                        base_features['l_hip_angle'] = 70.0 + np.random.normal(0, 3)
                        base_features['l_elbow_angle'] += (t - 15) * 3.5
                        base_features['r_elbow_angle'] += (t - 15) * 3.5
                        base_features['l_wrist_vel'] += (25 - t) * 0.15 + 1.0
                        base_features['r_wrist_vel'] += (25 - t) * 0.15 + 1.0
                        
                elif label == 'defensive_shot':
                    # Minimal body movement, hands close to body, soft block
                    if 5 <= t < 15:
                        base_features['foot_stride'] += (t - 5) * 0.02
                        base_features['l_knee_angle'] -= (t - 5) * 1.5
                        base_features['l_elbow_angle'] -= (t - 5) * 1.0
                        base_features['l_wr_to_hip'] -= (t - 5) * 0.02 # Hands close to body
                        base_features['r_wr_to_hip'] -= (t - 5) * 0.02
                        base_features['l_wrist_vel'] += (t - 5) * 0.02
                        base_features['r_wrist_vel'] += (t - 5) * 0.02
                    elif 15 <= t < 25:
                        base_features['foot_stride'] += 0.2
                        base_features['l_knee_angle'] -= 15.0
                        base_features['l_wr_to_hip'] = 0.4 + np.random.normal(0, 0.02)
                        base_features['r_wr_to_hip'] = 0.4 + np.random.normal(0, 0.02)
                        base_features['l_wrist_vel'] = 0.15 + np.random.normal(0, 0.02)
                        base_features['r_wrist_vel'] = 0.15 + np.random.normal(0, 0.02)
                        
                elif label == 'lofted_shot':
                    # Big stance step, hip extension, hand upward swing
                    if 5 <= t < 15:
                        base_features['foot_stride'] += (t - 5) * 0.06
                        base_features['l_knee_angle'] -= (t - 5) * 2.0
                        base_features['l_elbow_angle'] -= (t - 5) * 2.5
                        base_features['l_shoulder_angle'] += (t - 5) * 3.5 # Swing up
                        base_features['r_shoulder_angle'] += (t - 5) * 3.5
                    elif 15 <= t < 25:
                        base_features['foot_stride'] += 0.6
                        base_features['l_knee_angle'] -= 20.0
                        base_features['l_elbow_angle'] += (t - 15) * 4.0
                        base_features['l_shoulder_angle'] += 35.0
                        base_features['r_shoulder_angle'] += 35.0
                        base_features['l_wrist_vel'] += (25 - t) * 0.22 + 1.7
                        base_features['r_wrist_vel'] += (25 - t) * 0.22 + 1.7
                
                # Append current frame's synthetic features
                frame_vector = [base_features[col] for col in FEATURE_COLUMNS]
                sequence_features.append(frame_vector)
                
                # Store frame level record
                frame_record = base_features.copy()
                frame_record['label'] = label
                frame_record['video_source'] = f"synthetic_{label}_{seq_idx}"
                frame_records.append(frame_record)
                
            # Store sequence level record
            sequence_records.append({
                'features': np.array(sequence_features),
                'label': label
            })
            
    # Save frame-level
    df = pd.DataFrame(frame_records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df)} synthetic frame records to {output_csv_path}")
    
    # Save sequence-level
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(sequence_records, f)
    print(f"Saved {len(sequence_records)} synthetic sequence records to {output_pkl_path}")
    
    return True

if __name__ == "__main__":
    # If run directly, generate synthetic data as default
    csv_path = "data/processed/pose_dataset.csv"
    pkl_path = "data/processed/sequence_dataset.pkl"
    generate_synthetic_dataset(csv_path, pkl_path)
