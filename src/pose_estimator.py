import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32), (27, 31), (28, 32),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10)
]

class CricketPoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # We assume the task model is in the root directory (where we downloaded it)
        model_path = os.path.abspath('pose_landmarker_lite.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please download it.")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False)
            
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
    def calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points.
        b is the vertex.
        a, b, c are [x, y] or [x, y, z] arrays.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def extract_landmarks(self, results):
        """
        Extract the raw 33 landmark points.
        Returns a list of landmarks for the first person.
        """
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks[0]

    def get_features(self, landmarks, prev_landmarks=None, dt=1.0/30.0):
        """
        Extract key angles and spatial features for classification.
        """
        if not landmarks:
            return None
            
        l_sh = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
        r_sh = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
        l_el = [landmarks[13].x, landmarks[13].y, landmarks[13].z]
        r_el = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
        l_wr = [landmarks[15].x, landmarks[15].y, landmarks[15].z]
        r_wr = [landmarks[16].x, landmarks[16].y, landmarks[16].z]
        
        l_hip = [landmarks[23].x, landmarks[23].y, landmarks[23].z]
        r_hip = [landmarks[24].x, landmarks[24].y, landmarks[24].z]
        l_kn = [landmarks[25].x, landmarks[25].y, landmarks[25].z]
        r_kn = [landmarks[26].x, landmarks[26].y, landmarks[26].z]
        l_an = [landmarks[27].x, landmarks[27].y, landmarks[27].z]
        r_an = [landmarks[28].x, landmarks[28].y, landmarks[28].z]
        
        l_elbow_angle = self.calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = self.calculate_angle(r_sh, r_el, r_wr)
        
        l_knee_angle = self.calculate_angle(l_hip, l_kn, l_an)
        r_knee_angle = self.calculate_angle(r_hip, r_kn, r_an)
        
        l_hip_angle = self.calculate_angle(l_sh, l_hip, l_kn)
        r_hip_angle = self.calculate_angle(r_sh, r_hip, r_kn)
        
        l_shoulder_angle = self.calculate_angle(l_hip, l_sh, l_el)
        r_shoulder_angle = self.calculate_angle(r_hip, r_sh, r_el)
        
        mid_shoulder = (np.array(l_sh) + np.array(r_sh)) / 2.0
        mid_hip = (np.array(l_hip) + np.array(r_hip)) / 2.0
        torso_vector = mid_shoulder - mid_hip
        vertical_vector = np.array([0, -1, 0])
        torso_angle = self.calculate_angle(mid_shoulder, mid_hip, mid_hip + vertical_vector)
        
        hip_width = np.linalg.norm(np.array(l_hip) - np.array(r_hip)) + 1e-6
        l_wr_to_hip = np.linalg.norm(np.array(l_wr) - np.array(l_hip)) / hip_width
        r_wr_to_hip = np.linalg.norm(np.array(r_wr) - np.array(r_hip)) / hip_width
        hand_dist = np.linalg.norm(np.array(l_wr) - np.array(r_wr)) / hip_width
        foot_stride = np.linalg.norm(np.array(l_an) - np.array(r_an)) / hip_width
        
        l_wrist_vel = 0.0
        r_wrist_vel = 0.0
        l_ankle_vel = 0.0
        r_ankle_vel = 0.0
        
        if prev_landmarks:
            pl_wr = [prev_landmarks[15].x, prev_landmarks[15].y, prev_landmarks[15].z]
            pr_wr = [prev_landmarks[16].x, prev_landmarks[16].y, prev_landmarks[16].z]
            pl_an = [prev_landmarks[27].x, prev_landmarks[27].y, prev_landmarks[27].z]
            pr_an = [prev_landmarks[28].x, prev_landmarks[28].y, prev_landmarks[28].z]
            
            l_wrist_vel = np.linalg.norm(np.array(l_wr) - np.array(pl_wr)) / dt
            r_wrist_vel = np.linalg.norm(np.array(r_wr) - np.array(pr_wr)) / dt
            l_ankle_vel = np.linalg.norm(np.array(l_an) - np.array(pl_an)) / dt
            r_ankle_vel = np.linalg.norm(np.array(r_an) - np.array(pr_an)) / dt
            
        features = {
            'l_elbow_angle': l_elbow_angle,
            'r_elbow_angle': r_elbow_angle,
            'l_knee_angle': l_knee_angle,
            'r_knee_angle': r_knee_angle,
            'l_hip_angle': l_hip_angle,
            'r_hip_angle': r_hip_angle,
            'l_shoulder_angle': l_shoulder_angle,
            'r_shoulder_angle': r_shoulder_angle,
            'torso_angle': torso_angle,
            'l_wr_to_hip': l_wr_to_hip,
            'r_wr_to_hip': r_wr_to_hip,
            'hand_dist': hand_dist,
            'foot_stride': foot_stride,
            'l_wrist_vel': l_wrist_vel,
            'r_wrist_vel': r_wrist_vel,
            'l_ankle_vel': l_ankle_vel,
            'r_ankle_vel': r_ankle_vel,
        }
        
        raw_coords = {
            'l_sh': l_sh, 'r_sh': r_sh,
            'l_el': l_el, 'r_el': r_el,
            'l_wr': l_wr, 'r_wr': r_wr,
            'l_hip': l_hip, 'r_hip': r_hip,
            'l_kn': l_kn, 'r_kn': r_kn,
            'l_an': l_an, 'r_an': r_an
        }
        
        return features, raw_coords

    def process_frame(self, frame, prev_landmarks=None):
        """
        Process an image frame, run MediaPipe, and return landmarks and features.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        
        landmarks = self.extract_landmarks(results)
        features = None
        raw_coords = None
        
        if landmarks:
            features, raw_coords = self.get_features(landmarks, prev_landmarks)
            
        return results, landmarks, features, raw_coords

    def draw_pose(self, frame, results):
        """
        Draw skeleton landmarks on frame using OpenCV manually.
        """
        if not results.pose_landmarks:
            return frame
            
        landmarks = results.pose_landmarks[0]
        h, w, _ = frame.shape
        
        # Convert to pixel coordinates
        pts = {}
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            pts[i] = (cx, cy)
            
        # Draw connections
        for connection in POSE_CONNECTIONS:
            p1 = connection[0]
            p2 = connection[1]
            if p1 in pts and p2 in pts:
                cv2.line(frame, pts[p1], pts[p2], (0, 255, 0), 2)
                
        # Draw keypoints
        for i, pt in pts.items():
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)
            
        return frame
