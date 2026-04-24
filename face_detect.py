import cv2
import mediapipe as mp
import numpy as np
from config import CAMERA_SOURCE

# -------------------------------
# Image Enhancement Functions
# -------------------------------

def automatic_brightness_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final


def gamma_correction(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# -------------------------------
# Mediapipe Setup
# -------------------------------

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------
# Camera Capture
# -------------------------------

cap = cv2.VideoCapture(CAMERA_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Auto brightness/contrast
    enhanced = automatic_brightness_contrast(frame)

    # Step 2: CLAHE enhancement
    enhanced = apply_clahe(enhanced)

    # Step 3: Gamma correction (low-light fix)
    enhanced = gamma_correction(enhanced, gamma=1.3)

    # Convert to RGB for Mediapipe
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                enhanced,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(255,0,0), thickness=1)
            )

    cv2.imshow("Low Light Robust Face Detection", enhanced)

    if cv2.waitKey(1) & 0xFF == 1:
        break

cap.release()
cv2.destroyAllWindows()