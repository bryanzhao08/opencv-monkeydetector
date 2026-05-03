from core.camera import CameraStream
from core.detector import FaceDetector
from core.expression import ExpressionTracker
import cv2
import argparse
import numpy as np
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

def load_gif_as_bgr(path, max_size=200):
    """Load a GIF (or any image) using Pillow and return as BGR numpy array."""
    if _PIL_AVAILABLE:
        try:
            img = PILImage.open(path).convert('RGB')
            # Resize so it fits nicely as an overlay
            img.thumbnail((max_size, max_size), PILImage.LANCZOS)
            arr = np.array(img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[!] Pillow could not load {path}: {e}")
    # Fallback: try OpenCV directly (works for static frames)
    img = cv2.imread(path)
    if img is not None and max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    return img


def draw_hud(frame, tracker, monkey_img):
    """Display 5 monkey photos when pose is matched."""
    if tracker.monkey_match and monkey_img is not None:
        h, w = frame.shape[:2]
        mh, mw = monkey_img.shape[:2]
        
        # Calculate spacing for 5 monkeys
        # We'll make them a bit smaller to fit 5 across if needed, 
        # but for now let's just try to fit them with spacing.
        num_monkeys = 5
        padding = 10
        total_width = (mw * num_monkeys) + (padding * (num_monkeys + 1))
        
        # If they don't fit, scale them down
        display_mw = mw
        display_mh = mh
        if total_width > w:
            scale = (w - (padding * (num_monkeys + 1))) / (mw * num_monkeys)
            display_mw = int(mw * scale)
            display_mh = int(mh * scale)
            monkey_img_scaled = cv2.resize(monkey_img, (display_mw, display_mh))
        else:
            monkey_img_scaled = monkey_img
            
        start_x = (w - (display_mw * num_monkeys + padding * (num_monkeys - 1))) // 2
        y_offset = h - display_mh - 20
        
        if y_offset > 0:
            for i in range(num_monkeys):
                x_pos = start_x + i * (display_mw + padding)
                if x_pos + display_mw < w and x_pos >= 0:
                    frame[y_offset:y_offset+display_mh, x_pos:x_pos+display_mw] = monkey_img_scaled

    return frame

def main():
    parser = argparse.ArgumentParser(description="Monkey Face Detector")
    parser.add_argument('--camera', type=int, default=0, help="Camera index (usually 0 for built-in, try 1 if it shows your phone)")
    args = parser.parse_args()

    # Load monkey image (GIF requires Pillow; fallback to OpenCV)
    monkey_img = load_gif_as_bgr('monkey.gif')
    if monkey_img is None:
        print("[!] Warning: Could not load monkey.gif — install Pillow: pip install Pillow")
    else:
        print(f"[✔] Loaded monkey.gif as {monkey_img.shape[1]}x{monkey_img.shape[0]} overlay")

    print(f"\n[i] Opening camera {args.camera}...")
    cam = CameraStream(src=args.camera)
    
    if not cam.cap.isOpened():
        print(f"\n[!] FATAL: Could not access camera {args.camera}.")
        print("Tip: If you're on a Mac, try '--camera 1' if index 0 is your iPhone (Continuity Camera).")
        return

    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"\n[!] FATAL: Failed to initialize MediaPipe: {e}")
        print("\nPossible fix: Run './venv/bin/python -m pip install --upgrade mediapipe'")
        return

    tracker = ExpressionTracker()

    print("\n[✔] Monkey Matcher started! Match the monkey's pose to see the photo.")

    while True:
        frame = cam.read()
        if frame is None:
            break

        h, w = frame.shape[:2]
        face_results, hand_results = detector.process(frame)

        face_lms = None
        if face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0].landmark
            
        hands_lms = []
        if hand_results.multi_hand_landmarks:
            hands_lms = hand_results.multi_hand_landmarks

        tracker.update(face_lms, hands_lms, w, h)
        frame = draw_hud(frame, tracker, monkey_img)
        
        cv2.imshow("Monkey Matcher", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
