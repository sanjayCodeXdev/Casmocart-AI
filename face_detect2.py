import cv2
import mediapipe as mp
import numpy as np
import json
import time
from config import CAMERA_SOURCE

# -----------------------------
# Image Enhancement (from face_detect.py)
# -----------------------------
def automatic_brightness_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0) / 2.0
    minimum_gray = 0
    while minimum_gray < hist_size and accumulator[minimum_gray] < clip_hist_percent: 
        minimum_gray += 1
        
    maximum_gray = hist_size - 1
    while maximum_gray >= 0 and accumulator[maximum_gray] >= (maximum - clip_hist_percent): 
        maximum_gray -= 1
    
    if maximum_gray <= minimum_gray or maximum_gray < 0 or minimum_gray >= hist_size:
        return image # Skip enhancement for flat/uniform frames
    
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

def gamma_correction(image, gamma=1.3):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_frame(frame):
    """Combines all enhancements for robust scanning."""
    enhanced = automatic_brightness_contrast(frame)
    enhanced = apply_clahe(enhanced)
    return gamma_correction(enhanced)


# -----------------------------
# Brightness Score
# -----------------------------
def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


# -----------------------------
# Shine Detection
# -----------------------------
def shine_detection(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    shine_pixels = np.sum(v > 220)
    return shine_pixels / v.size


# -----------------------------
# Texture Variance
# -----------------------------
def texture_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


# -----------------------------
# Mediapipe Setup
# -----------------------------
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def interpret_skin(regions):
    """Deep analysis based on multi-region scanning."""
    desc = []
    
    # Analyze Shine (Oiliness) across regions
    # T-Zone: Forehead + Nose
    t_zone_shine = (regions['forehead']['shine'] + regions['nose']['shine']) / 2
    # U-Zone: Cheeks
    u_zone_shine = (regions['left_cheek']['shine'] + regions['right_cheek']['shine']) / 2
    
    # Precise Skin Type Logic
    if t_zone_shine > 0.15 and u_zone_shine > 0.15:
        desc.append("OILY SKIN: Significant shine (sebum) detected globally across your T-Zone and cheeks.")
    elif t_zone_shine > 0.15 and u_zone_shine < 0.05:
        desc.append("COMBINATION SKIN: High shine in the T-Zone with drier Cheek areas detected.")
    elif t_zone_shine < 0.05 and u_zone_shine < 0.05:
        desc.append("DRY SKIN: Very low surface oil detected; skin appears matte or dehydrated.")
    else:
        desc.append("NORMAL/BALANCED SKIN: Healthy, moderate sebum levels detected across all zones.")

    # Analyze Texture (Roughness/Pores)
    avg_texture = sum(r['texture'] for r in regions.values()) / len(regions)
    if avg_texture > 250:
        desc.append("TEXTURED: High surface variance detected. This may indicate active breakouts or enlarged pores.")
    elif avg_texture > 120:
        desc.append("MODERATE TEXTURE: Some minor unevenness detected, possibly due to congestion.")
    else:
        desc.append("SMOOTH: Skin surface appears extremely even and clear.")

    return " | ".join(desc)

def get_region_features(frame, face_landmarks, indices):
    """Extracts features from a specific list of mediapipe landmark indices."""
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    points = []
    for idx in indices:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    roi = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Calculate features for this specific ROI
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_roi[mask > 0]) if np.any(mask > 0) else 0
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv_roi[:, :, 2]
    shine = np.sum((v > 220) & (mask > 0)) / np.sum(mask > 0) if np.any(mask > 0) else 0
    
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    texture = lap.var() if np.any(mask > 0) else 0
    
    return {"brightness": float(brightness), "shine": float(shine), "texture": float(texture)}

def extract_face_signature(face_landmarks):
    """Creates a unique list of ratios based on landmark distances."""
    def dist(p1_idx, p2_idx):
        p1 = face_landmarks.landmark[p1_idx]
        p2 = face_landmarks.landmark[p2_idx]
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

    # Key landmark indices (simplified)
    # 33 - Left eye outer, 133 - Left eye inner
    # 362 - Right eye inner, 263 - Right eye outer
    # 1 - Nose tip, 152 - Chin
    # 61 - mouth left, 291 - mouth right

    signatures = [
        dist(33, 133) / dist(33, 263),   # Eye width ratio
        dist(61, 291) / dist(33, 263),   # Mouth width / Face width
        dist(1, 152) / dist(10, 152),    # Nose-to-chin / Face height
        dist(33, 61) / dist(10, 152)     # Eye-to-mouth / Face height
    ]
    return [float(s) for s in signatures]

def run_face_scanner():
    face_mesh_module = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
    print("\n[CAMERA] Position your face and press 'S' to Scan or 'ESC' to Cancel.")
    
    captured_data = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ENHANCEMENT PIPELINE (Combining both detect files)
        frame = enhance_frame(frame)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_module.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Landmark Indices for key regions
                forehead_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127] # Outline for broad zones
                # Defining simpler boxes for speed
                regions = {
                    "forehead": get_region_features(frame, face_landmarks, [10, 67, 103, 109, 338, 297, 10]),
                    "left_cheek": get_region_features(frame, face_landmarks, [234, 93, 132, 58, 172, 136, 150, 234]),
                    "right_cheek": get_region_features(frame, face_landmarks, [454, 323, 361, 288, 397, 365, 379, 454]),
                    "nose": get_region_features(frame, face_landmarks, [168, 6, 197, 195, 5, 4, 168]),
                    "chin": get_region_features(frame, face_landmarks, [152, 148, 176, 149, 150, 136, 152])
                }

                interpretation = interpret_skin(regions)
                
                # Draw Face Mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Visual Feedback
                cv2.putText(frame, "AI ANALYZING ALL ZONES...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, interpretation.split(":")[0], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    signature = extract_face_signature(face_landmarks)
                    captured_data = {
                        "regions": regions,
                        "interpretation": interpretation,
                        "signature": signature
                    }
                    with open("skin_features.json", "w") as f:
                        json.dump(captured_data, f, indent=4)
                    print(f"✅ Full Face Scan Complete: {interpretation}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return captured_data

        cv2.imshow("AI For Her - Face Scanner", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return None

class VideoStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
        self.face_mesh_module = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.latest_data = None

    def get_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                # Keep trying instead of breaking the loop on Windows
                time.sleep(0.01)
                continue
            
            frame = enhance_frame(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh_module.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Logic for regions and drawing (same as run_face_scanner)
                    regions = {
                        "forehead": get_region_features(frame, face_landmarks, [10, 67, 103, 109, 338, 297, 10]),
                        "left_cheek": get_region_features(frame, face_landmarks, [234, 93, 132, 58, 172, 136, 150, 234]),
                        "right_cheek": get_region_features(frame, face_landmarks, [454, 323, 361, 288, 397, 365, 379, 454]),
                        "nose": get_region_features(frame, face_landmarks, [168, 6, 197, 195, 5, 4, 168]),
                        "chin": get_region_features(frame, face_landmarks, [152, 148, 176, 149, 150, 136, 152])
                    }
                    interpretation = interpret_skin(regions)
                    self.latest_data = {
                        "regions": regions, 
                        "interpretation": interpretation, 
                        "signature": extract_face_signature(face_landmarks)
                    }

                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_TESSELATION, None, mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    cv2.putText(frame, "AI ANALYSIS ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.latest_data = None
                cv2.putText(frame, "SEARCHING FOR FACE...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    run_face_scanner()