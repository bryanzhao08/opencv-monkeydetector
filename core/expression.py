import numpy as np

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


# ── Hand helpers ─────────────────────────────────────────────────────────────

def detect_hand_near_face(hand_landmarks_list, face_cx, face_cy, face_width, w, h):
    """
    Returns True if ANY hand's wrist or fingertip is within ~60% of face_width
    from the face centre — i.e. the hand is raised up to the face/chin area.
    The monkey in the photo has its hand touching its chin/cheek.
    """
    if not hand_landmarks_list or face_width == 0:
        return False

    threshold = face_width * 0.85   # generous — the monkey's hand is *on* its face

    for hand in hand_landmarks_list:
        lms = hand.landmark
        # Check wrist (0), index tip (8), middle tip (12), thumb tip (4)
        for idx in [0, 4, 8, 12]:
            px = lms[idx].x * w
            py = lms[idx].y * h
            dist = get_dist((px, py), (face_cx, face_cy))
            if dist < threshold:
                return True
    return False


# ── Tracker ──────────────────────────────────────────────────────────────────

class ExpressionTracker:
    def __init__(self):
        self.hand_near_face = False
        self.monkey_match = False
        self.match_duration = 0  # frames matched

    def update(self, face_lms, hand_lms, w, h):
        face_cx, face_cy, face_width = 0.0, 0.0, 0.0

        if face_lms:
            face_cx, face_cy, face_width = get_face_center_and_size(face_lms, w, h)

        self.hand_near_face = detect_hand_near_face(
            hand_lms, face_cx, face_cy, face_width, w, h
        )

        # Match condition: hand raised to face only
        if self.hand_near_face:
            self.monkey_match = True
            self.match_duration += 1
        else:
            self.monkey_match = False
            self.match_duration = 0

        return self.monkey_match
