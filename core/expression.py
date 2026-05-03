import numpy as np
from collections import deque

def get_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# ── Face helpers ────────────────────────────────────────────────────────────

def get_face_center_and_size(landmarks, w, h):
    """Return (cx, cy, face_width) using cheek landmarks."""
    l_face = (landmarks[234].x * w, landmarks[234].y * h)
    r_face = (landmarks[454].x * w, landmarks[454].y * h)
    cx = (l_face[0] + r_face[0]) / 2
    cy = (l_face[1] + r_face[1]) / 2
    face_width = get_dist(l_face, r_face)
    return cx, cy, face_width

def detect_frown(landmarks, w, h, threshold=0.4):
    """
    Detects a frown with adjustable threshold.
    Higher threshold = harder to trigger.
    """
    if not landmarks: return False
    l_corner_y = landmarks[61].y
    r_corner_y = landmarks[291].y
    center_y = landmarks[13].y
    lip_dist = abs(landmarks[13].y - landmarks[14].y)
    if lip_dist == 0: lip_dist = 0.01
    
    # Corners must be below center by (lip_dist * threshold)
    return (l_corner_y > center_y + lip_dist * threshold) and (r_corner_y > center_y + lip_dist * threshold)

# ── Hand helpers ─────────────────────────────────────────────────────────────

def detect_hand_near_face(hand_landmarks_list, face_cx, face_cy, face_width, w, h):
    if not hand_landmarks_list or face_width == 0:
        return False

    threshold = face_width * 1.0 
    for hand in hand_landmarks_list:
        lms = hand.landmark
        for idx in [0, 4, 8, 12, 16, 20]:
            px = lms[idx].x * w
            py = lms[idx].y * h
            dist = get_dist((px, py), (face_cx, face_cy))
            if dist < threshold:
                return True
    return False

# ── Motion Tracker ───────────────────────────────────────────────────────────

class ExpressionTracker:
    def __init__(self):
        self.monkey_match = False
        self.hamster_match = False
        self.sixseven_match = False
        self.pose_history = deque(maxlen=15) # Shorter history for faster response
        
    def update(self, face_lms, hand_lms, pose_lms, w, h, frown_sens=0.4):
        face_cx, face_cy, face_width = 0.0, 0.0, 0.0
        if face_lms:
            face_cx, face_cy, face_width = get_face_center_and_size(face_lms, w, h)
            
        # 1. Monkey: Hand near face
        self.monkey_match = detect_hand_near_face(hand_lms, face_cx, face_cy, face_width, w, h)
        
        # 2. Hamster: Frown
        self.hamster_match = detect_frown(face_lms, w, h, threshold=frown_sens)
        
        # 3. SixSeven: Arm motion
        self.sixseven_match = False
        
        if pose_lms:
            left_wrist = pose_lms.landmark[15]
            right_wrist = pose_lms.landmark[16]
            nose = pose_lms.landmark[0]
            
            self.pose_history.append((left_wrist.y, right_wrist.y))
            
            if len(self.pose_history) == self.pose_history.maxlen:
                l_y = [p[0] for p in self.pose_history]
                r_y = [p[1] for p in self.pose_history]
                
                l_range = max(l_y) - min(l_y)
                r_range = max(r_y) - min(r_y)
                
                # Check if hands are away from face
                l_dist_nose = get_dist((left_wrist.x, left_wrist.y), (nose.x, nose.y))
                r_dist_nose = get_dist((right_wrist.x, right_wrist.y), (nose.x, nose.y))
                
                # Arm swing check:
                # If hands are moving a lot AND aren't stuck to the face, it's a swing.
                # We allow monkey_match to be true briefly as arms pass by, 
                # but sixseven takes precedence if the movement range is very high.
                if (l_range > 0.2 or r_range > 0.2) and (l_dist_nose > 0.15 and r_dist_nose > 0.15):
                    self.sixseven_match = True
        else:
            self.pose_history.clear()

        return self.monkey_match or self.hamster_match or self.sixseven_match
