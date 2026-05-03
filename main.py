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

def on_trackbar(val):
    pass

def crop_black_bars(img, threshold=15):
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    return img

def load_and_prep_image(path, max_size=400, crop=False):
    img = None
    if _PIL_AVAILABLE and path.lower().endswith('.gif'):
        try:
            pil_img = PILImage.open(path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[!] Error loading GIF {path}: {e}")
    if img is None:
        img = cv2.imread(path)
    if img is not None:
        if crop: img = crop_black_bars(img)
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def draw_hud(frame, tracker, assets):
    """Display assets with priority: Monkey > SixSeven > Hamster."""
    h, w = frame.shape[:2]
    
    active_img = None
    # 1. Monkey (Hand to face) - Top Priority
    if tracker.monkey_match:
        active_img = assets.get('monkey')
    # 2. SixSeven (Arm motion)
    elif tracker.sixseven_match:
        active_img = assets.get('sixseven')
    # 3. Hamster (Frown)
    elif tracker.hamster_match:
        active_img = assets.get('hamster')
        
    if active_img is not None:
        ah, aw = active_img.shape[:2]
        x_offset = w - aw - 20
        y_offset = h - ah - 20
        if x_offset >= 0 and y_offset >= 0:
            frame[y_offset:y_offset+ah, x_offset:x_offset+aw] = active_img
            
    return frame

def main():
    parser = argparse.ArgumentParser(description="Expression Detector")
    parser.add_argument('--camera', type=int, default=0, help="Camera index")
    args = parser.parse_args()

    asset_size = 400
    assets = {
        'monkey': load_and_prep_image('monkey.gif', max_size=asset_size, crop=True),
        'hamster': load_and_prep_image('sadhamster.png', max_size=asset_size),
        'sixseven': load_and_prep_image('sixseven.jpeg', max_size=asset_size)
    }

    print(f"\n[i] Opening camera {args.camera}...")
    cam = CameraStream(src=args.camera)
    if not cam.cap.isOpened(): return

    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"FATAL: {e}")
        return

    tracker = ExpressionTracker()

    # Create window and trackbar for frown sensitivity
    cv2.namedWindow("Expression Matcher")
    # Range 0-100, mapped to 0.0-1.0 (default 0.4 -> 40)
    cv2.createTrackbar("Frown Sens", "Expression Matcher", 40, 100, on_trackbar)

    while True:
        frame = cam.read()
        if frame is None: break

        # Get sensitivity from slider
        sens_val = cv2.getTrackbarPos("Frown Sens", "Expression Matcher") / 100.0

        face_results, hand_results, pose_results = detector.process(frame)
        face_lms = face_results.multi_face_landmarks[0].landmark if face_results.multi_face_landmarks else None
        hands_lms = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []
        pose_lms = pose_results.pose_landmarks if pose_results.pose_landmarks else None

        h, w = frame.shape[:2]
        # Pass sensitivity to tracker
        tracker.update(face_lms, hands_lms, pose_lms, w, h, frown_sens=sens_val)
        frame = draw_hud(frame, tracker, assets)
        
        cv2.imshow("Expression Matcher", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
