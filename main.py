from core.camera import CameraStream
from core.detector import FaceDetector
from core.blink import BlinkTracker, eye_aspect_ratio, LEFT_EYE, RIGHT_EYE
import cv2
import argparse
def draw_hud(frame, ear, blink_tracker):
    """Draw a basic HUD with focus metrics."""
    h, w = frame.shape[:2]
    
    # State logic based on EAR
    state = "DROWSY" if ear > 0 and ear < 0.20 else "AWAY" if ear == 0 else "FOCUSED"
    color = (0, 0, 255) if state == "DROWSY" else (100, 100, 100) if state == "AWAY" else (0, 255, 0)
    
    # Draw Background box for text
    cv2.rectangle(frame, (10, 10), (350, 140), (0, 0, 0), -1)
    
    cv2.putText(frame, f"State: {state}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Blinks: {blink_tracker.blink_count}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description="Attention Tracker")
    parser.add_argument('--camera', type=int, default=0, help="Camera index (0, 1, 2, etc.)")
    args = parser.parse_args()

    cam = CameraStream(src=args.camera)
    
    if not cam.cap.isOpened():
        print("\n[!] FATAL: Could not access camera.")
        print("Please check System Settings > Privacy & Security > Camera and ensure your Terminal is allowed.")
        return

    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"\n[!] FATAL: Failed to initialize MediaPipe: {e}")
        return

    blink_tracker = BlinkTracker()

    print("\n[✔] Attention tracker started! Press 'q' in the video window to quit.")

    while True:
        frame = cam.read()
        if frame is None:
            break

        h, w = frame.shape[:2]
        results = detector.process(frame)

        ear = 0
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            
            # Optional: draw some landmarks for debugging
            # for pt in LEFT_EYE + RIGHT_EYE:
            #     x = int(lms[pt].x * w)
            #     y = int(lms[pt].y * h)
            #     cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
            
            ear_l = eye_aspect_ratio(lms, LEFT_EYE, w, h)
            ear_r = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            ear = (ear_l + ear_r) / 2.0
            
            blink_tracker.update(ear)

        frame = draw_hud(frame, ear, blink_tracker)
        
        cv2.imshow("Attention Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
