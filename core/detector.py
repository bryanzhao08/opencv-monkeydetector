import mediapipe as mp
import cv2

class FaceDetector:
    def __init__(self):
        # Use mp.solutions — works across mediapipe 0.10.x
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb)
        hand_results = self.hands.process(rgb)
        pose_results = self.pose.process(rgb)
        return face_results, hand_results, pose_results
