import numpy as np
from collections import deque

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    # compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # compute the euclidean distance between the horizontal eye landmark
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    # compute the eye aspect ratio
    return (A + B) / (2.0 * C)

class BlinkTracker:
    def __init__(self):
        self.blink_count = 0
        self.was_closed = False
        self.ear_history = deque(maxlen=300)  # last 10s @ 30fps

    def update(self, ear):
        self.ear_history.append(ear)
        # Assuming an EAR < 0.25 indicates a closed eye
        if ear < 0.25 and not self.was_closed:
            self.blink_count += 1
            self.was_closed = True
        elif ear >= 0.25:
            self.was_closed = False
        return self.blink_count
