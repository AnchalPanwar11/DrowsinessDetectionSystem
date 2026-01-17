"""
head_to_wheel_monitor_alert.py
Head-to-steering-wheel proximity monitor (full-screen).
- Text alert only (small overlay for a few seconds when real lean-forward detected)
- Uses MediaPipe FaceMesh + OpenCV
- Cropped circle detection focuses on LOWER half of frame to ignore background arcs
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ----------------- User config (tune if needed) -----------------
WHEEL_DIAMETER_M = 0.38         # measured steering wheel / plate diameter (meters)
FACE_REAL_WIDTH_M = 0.16        # approximate face width (m) for optional focal usage
LEARN_FOCAL = False             # set True only if you want to calibrate focal
DISTANCE_ALERT_THRESHOLD_M = 0.32   # threshold (meters) for proximity alert
SUSTAIN_SECONDS = 1.5           # seconds the distance must remain below threshold
FRAME_WIDTH = 640               # keep small for i3 CPU
FRAME_HEIGHT = 360
# alert display params
ALERT_DISPLAY_SEC = 3.0         # how long on-screen alert remains
HEAD_DOWN_OFFSET_RATIO = 0.06   # nose must be this fraction lower than face center
ALERT_COOLDOWN = 4.0            # seconds between alerts
# ----------------------------------------------------------------

# MediaPipe FaceMesh init
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# Camera init
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(0.3)

# placeholders for wheel calibration
wheel_cx = wheel_cy = wheel_r = None
scale_m_per_px = None
focal_length_px = None

# buffers and timers
buffer_len = int(max(2.0, SUSTAIN_SECONDS) * 10)
dist_buffer = deque(maxlen=buffer_len)
last_alert_time = 0.0
alert_show_until = 0.0
alert_label = ""

# ----------------- Robust cropped circle detector -----------------
def detect_wheel_circle_cropped(gray, frame_bgr,
                               crop_fraction=0.6,
                               dp=1.2, minR=30, maxR=300,
                               hough_param2=28):
    """
    Detect circle only in lower `crop_fraction` of the frame (to avoid background).
    Returns (cx, cy, r) in full-frame coordinates or None.
    """
    h, w = gray.shape
    crop_y0 = int(h * (1.0 - crop_fraction))
    crop = gray[crop_y0:h, :]

    # preprocessing
    blur = cv2.medianBlur(crop, 5)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Hough on cropped area
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=dp, minDist=80,
                               param1=100, param2=hough_param2,
                               minRadius=minR, maxRadius=maxR)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        best = None
        best_score = -1
        for (cx_local, cy_local, r) in circles:
            angles = np.linspace(0, 2*np.pi, 180).astype(np.float32)
            xs = np.clip((cx_local + r * np.cos(angles)).astype(int), 0, edges.shape[1]-1)
            ys = np.clip((cy_local + r * np.sin(angles)).astype(int), 0, edges.shape[0]-1)
            coverage = np.mean(edges[ys, xs] > 0)
            if coverage > best_score:
                best_score = coverage
                best = (cx_local, cy_local, r, coverage)
        if best is not None and best[3] > 0.10:
            cx_full = int(best[0])
            cy_full = int(best[1] + crop_y0)
            return cx_full, cy_full, int(best[2])

    # fallback: contour-based
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    best_score = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-8))
        (cx_local, cy_local), r = cv2.minEnclosingCircle(cnt)
        if r < minR or r > maxR:
            continue
        score = circularity * np.log(area + 1)
        if score > best_score:
            best_score = score
            best_cnt = (int(cx_local), int(cy_local), int(r), score)
    if best_cnt is not None and best_cnt[3] > 0.6:
        cx_full = best_cnt[0]
        cy_full = best_cnt[1] + crop_y0
        return cx_full, cy_full, best_cnt[2]

    return None

# ----------------- Calibration routine -----------------
def calibrate_wheel_and_focal():
    global wheel_cx, wheel_cy, wheel_r, scale_m_per_px, focal_length_px
    print("Calibration: show a circular object (plate) in the LOWER HALF of the frame.")
    print("When the green circle aligns with your object, press 'c'. Press 'q' to skip.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = detect_wheel_circle_cropped(gray, frame, crop_fraction=0.6,
                                         minR=40, maxR=300, hough_param2=28)

        disp = frame.copy()
        if res:
            cx, cy, r = res
            cv2.circle(disp, (cx, cy), r, (0,255,0), 2)
            cv2.putText(disp, f"Wheel radius = {r}px", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(disp, "Press 'c' to accept or 'q' to skip",
                    (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # full-screen window
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Calibration", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and res:
            wheel_cx, wheel_cy, wheel_r = res
            scale_m_per_px = (WHEEL_DIAMETER_M / 2.0) / float(wheel_r)
            print("Wheel detected.")
            print(f"Scale = {scale_m_per_px:.6f} meters per pixel")
            break
        elif key == ord('q'):
            print("Calibration skipped by user.")
            break

    try:
        cv2.destroyWindow("Calibration")
    except:
        pass

    # optional focal calibration (skip by pressing Enter)
    if LEARN_FOCAL:
        try:
            Z_known = input("Enter known distance from face to camera (meters) or press Enter to skip: ").strip()
            if Z_known == "":
                print("Skipping focal calibration.")
                return
            Z_known = float(Z_known)
        except:
            print("Skipping focal calibration.")
            return

        print("Position face at that distance and press 'f' to capture.")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            cv2.namedWindow("FocalCal", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("FocalCal", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("FocalCal", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('f'):
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0]
                    xs = [int(p.x * FRAME_WIDTH) for p in lm.landmark]
                    face_width_px = max(xs) - min(xs)
                    if face_width_px > 0:
                        focal_length_px = (face_width_px * Z_known) / FACE_REAL_WIDTH_M
                        print(f"Focal length (px) ≈ {focal_length_px:.2f}")
                break
            elif key == ord('q'):
                break
        try:
            cv2.destroyWindow("FocalCal")
        except:
            pass

# ----------------- Run calibration -----------------
calibrate_wheel_and_focal()

print("System running. Position plate in lower half for best detection. Press 'q' or ESC to quit.")

# ----------------- Main loop -----------------
while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection
    results = face_mesh.process(rgb)
    face_detected = False
    nose_px = None
    fx = fy = None

    if results.multi_face_landmarks:
        face_detected = True
        lm = results.multi_face_landmarks[0]
        nose = lm.landmark[1]  # nose tip
        fx = int(nose.x * FRAME_WIDTH)
        fy = int(nose.y * FRAME_HEIGHT)
        nose_px = (fx, fy)
        cv2.circle(frame, nose_px, 4, (255, 0, 0), -1)

    # auto wheel detection if not calibrated
    if wheel_r is None:
        res = detect_wheel_circle_cropped(gray, frame, crop_fraction=0.6,
                                         minR=40, maxR=400, hough_param2=28)
        if res:
            wheel_cx, wheel_cy, wheel_r = res
            scale_m_per_px = (WHEEL_DIAMETER_M / 2.0) / float(wheel_r)
            print("Auto wheel detected during runtime.")
            print(f"Scale = {scale_m_per_px:.6f} m/px")

    # draw wheel if available
    if wheel_r is not None:
        cv2.circle(frame, (wheel_cx, wheel_cy), wheel_r, (0,255,0), 2)

    # compute planar distance nose->wheel (meters)
    fused_distance = None
    if face_detected and wheel_r is not None and scale_m_per_px is not None:
        d_px = np.hypot(fx - wheel_cx, fy - wheel_cy) - wheel_r
        fused_distance = max(0.0, d_px * scale_m_per_px)

    # append to buffer
    if fused_distance is not None:
        dist_buffer.append(fused_distance)

    # ---------- ALERT DECISION (sustained distance + head-down confirmation) ----------
    # compute head-down confirmation
    head_down_confirm = False
    if face_detected and results.multi_face_landmarks:
        ys = [p.y for p in results.multi_face_landmarks[0].landmark]
        face_center_y = np.mean(ys) * FRAME_HEIGHT
        face_height_px = (max(ys) - min(ys)) * FRAME_HEIGHT
        if fy is not None and face_height_px > 5:
            if (fy - face_center_y) > (HEAD_DOWN_OFFSET_RATIO * face_height_px):
                head_down_confirm = True

    # sustained-close check
    sustained_close = False
    SUSTAIN_FRAMES = max(3, int(SUSTAIN_SECONDS * 5))  # ~5 samples per second
    if len(dist_buffer) >= SUSTAIN_FRAMES:
        recent = np.array(dist_buffer)
        if np.all(recent[-SUSTAIN_FRAMES:] < DISTANCE_ALERT_THRESHOLD_M):
            sustained_close = True

    # final trigger: require both and cooldown
    if sustained_close and head_down_confirm and (time.time() - last_alert_time > ALERT_COOLDOWN):
        alert_show_until = time.time() + ALERT_DISPLAY_SEC
        alert_label = "Driver leaning forward!"
        last_alert_time = time.time()

    # overlay small on-screen alert if active
    if time.time() < alert_show_until:
        box_w, box_h = 420, 60
        alpha = 0.6
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, alert_label, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

    # show distance readout
    if fused_distance is not None:
        cv2.putText(frame, f"Dist: {fused_distance:.2f} m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, "Press 'q' or Esc to quit", (10, FRAME_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

    # full-screen display
    cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
