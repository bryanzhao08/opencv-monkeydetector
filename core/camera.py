import cv2
import sys

class CameraStream:
    def __init__(self, src=0):
        # On macOS, try AVFOUNDATION explicitly if default fails
        self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        
        if not self.cap.isOpened():
            # Fallback to default
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            print("ERROR: Could not open video device.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
