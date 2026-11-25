import cv2
import numpy as np

def estimate_shot_power(wrist_velocities, window_size=10):
    """
    Estimate shot power score (0 to 100) based on peak wrist velocities.
    """
    if not wrist_velocities:
        return 0.0
    
    # Take the maximum wrist velocity in the last window_size frames
    recent_vels = wrist_velocities[-window_size:]
    max_vel = max(recent_vels)
    
    # Scale max velocity (typically values range from 0.0 to 15.0 in coordinate space/sec)
    power_score = min(100.0, (max_vel / 12.0) * 100.0)
    return float(power_score)

def analyze_shot_timing(wrist_velocities, threshold=1.5):
    """
    Detect different phases of the cricket shot:
    - Stance/Address: speed near 0
    - Backswing/Trigger: speed starts rising
    - Strike/Impact: speed peaks
    - Follow-through: speed decreases
    """
    if len(wrist_velocities) < 15:
        return "Stance"
        
    recent_vels = list(wrist_velocities)[-15:]
    max_idx = np.argmax(recent_vels)
    max_vel = recent_vels[max_idx]
    
    # Simple threshold rules based on velocity profile
    if max_vel < threshold:
        return "Stance / Address"
    elif max_idx > len(recent_vels) - 5:
        return "Backswing / Trigger"
    elif len(recent_vels) - 10 <= max_idx < len(recent_vels) - 3:
        return "Strike / Contact"
    else:
        return "Follow-through"

def get_coaching_feedback(label, features):
    """
    Compare current user features with ideal ranges to provide real-time coaching suggestions.
    """
    feedback = []
    successes = []
    
    if not features:
        return ["No pose detected"], []
        
    l_elb = features.get('l_elbow_angle', 180)
    r_elb = features.get('r_elbow_angle', 180)
    l_knee = features.get('l_knee_angle', 180)
    r_knee = features.get('r_knee_angle', 180)
    stride = features.get('foot_stride', 0.5)
    torso = features.get('torso_angle', 0)
    
    # Feedback rules depending on selected shot
    if label == 'cover_drive':
        # Rule 1: front foot stride
        if stride < 1.0:
            feedback.append("Increase your front foot stride to reach the pitch of the ball.")
        else:
            successes.append("Excellent foot stride length!")
            
        # Rule 2: bend front knee
        if l_knee > 130:
            feedback.append("Bend your front knee more to transfer weight forward.")
        else:
            successes.append("Great forward knee bend.")
            
        # Rule 3: High lead elbow
        if r_elb > 130 and l_elb > 130:
            feedback.append("Keep your lead elbow high and dominant for control.")
        else:
            successes.append("Lead elbow position is high and controlled.")
            
    elif label == 'straight_drive':
        if stride < 0.9:
            feedback.append("Take a larger stride forward towards the line of the ball.")
        if torso > 30:
            feedback.append("Keep your head over the ball; avoid leaning too far back.")
        else:
            successes.append("Good head position over the ball.")
            
    elif label == 'pull_shot':
        # Pull is played on the back foot
        if r_knee < 120:
            successes.append("Good weight transfer onto the back foot.")
        else:
            feedback.append("Load your weight onto your back foot to execute the pull.")
            
        # Arms should extend horizontally
        if l_elb < 100 or r_elb < 100:
            feedback.append("Extend your arms fully to roll your wrists and keep the ball down.")
        else:
            successes.append("Excellent horizontal arm extension!")
            
    elif label == 'sweep_shot':
        # Drop down low
        if l_knee > 90 and r_knee > 90:
            feedback.append("Get lower to the ground by kneeling on your back leg.")
        else:
            successes.append("Great low stance for the sweep shot!")
            
    elif label == 'defensive_shot':
        # Stance close to body, soft hands
        if stride > 1.2:
            feedback.append("Keep your feet compact; do not over-stride on defensive shots.")
        if features.get('l_wr_to_hip', 0.8) > 0.6:
            feedback.append("Keep your hands close to your body and play with 'soft hands'.")
        else:
            successes.append("Perfect 'soft hands' defensive posture.")
            
    else:
        # Default encouragement
        successes.append("Form is solid. Maintain focus and follow through.")
        
    return feedback, successes

def draw_hud_overlay(frame, label, confidence, power, timing_phase, features=None):
    """
    Draw a professional analytics heads-up-display (HUD) overlay on the video frame.
    """
    h, w, _ = frame.shape
    
    # 1. Semi-transparent black background box for the telemetry HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (380, 220), (10, 10, 10), -1)
    # Apply transparency alpha blend
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Colors
    c_accent = (0, 255, 127)   # Spring Green
    c_text = (255, 255, 255)   # White
    c_gray = (180, 180, 180)   # Soft Gray
    
    # 2. Text layout
    cv2.putText(frame, "CRICKET ANALYTICS STUDIO", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_accent, 2)
    cv2.line(frame, (35, 55), (365, 55), c_accent, 1)
    
    cv2.putText(frame, "Shot:", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_gray, 1)
    cv2.putText(frame, f"{label.replace('_', ' ').title()}", (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_accent, 2)
    
    cv2.putText(frame, "Confidence:", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_gray, 1)
    cv2.putText(frame, f"{confidence * 100:.1f}%", (140, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_accent, 2)
    
    cv2.putText(frame, "Est. Power:", (35, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_gray, 1)
    cv2.putText(frame, f"{power:.1f}%", (140, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_accent, 2)
    
    cv2.putText(frame, "Timing Phase:", (35, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_gray, 1)
    cv2.putText(frame, f"{timing_phase}", (140, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_accent, 2)
    
    # 3. Draw mini confidence progress bar
    cv2.rectangle(frame, (35, 195), (365, 205), (50, 50, 50), -1)
    bar_width = int(330 * confidence)
    cv2.rectangle(frame, (35, 195), (35 + bar_width, 205), c_accent, -1)
    
    # 4. Optional: Draw specific joint angles in the top-right corner if feature dictionary is present
    if features:
        overlay_tr = frame.copy()
        cv2.rectangle(overlay_tr, (w - 240, 20), (w - 20, 150), (10, 10, 10), -1)
        cv2.addWeighted(overlay_tr, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, "JOINT TELEMETRY", (w - 225, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_accent, 1)
        cv2.line(frame, (w - 225, 52), (w - 35, 52), c_accent, 1)
        
        l_elb = features.get('l_elbow_angle', 0.0)
        r_elb = features.get('r_elbow_angle', 0.0)
        stride = features.get('foot_stride', 0.0)
        
        cv2.putText(frame, f"L-Elbow Angle: {l_elb:.1f} deg", (w - 225, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_text, 1)
        cv2.putText(frame, f"R-Elbow Angle: {r_elb:.1f} deg", (w - 225, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_text, 1)
        cv2.putText(frame, f"Foot Stride: {stride:.2f} xW", (w - 225, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c_text, 1)
        
    return frame
