#!/usr/bin/env python3
"""
combined_single_file_monitor_fixed.py
Single-file driver monitor — tuned wheel detection + fullscreen display.
Update PREDICTOR_PATH to your dlib 68-landmarks file.
"""

import time, math
import cv2
import dlib
import imutils
import numpy as np
import mediapipe as mp
from collections import deque
from imutils import face_utils

# -------------------- CONFIG (tune these) --------------------
PREDICTOR_PATH = "./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat"
CAM_INDEX = 0

# Use a larger capture resolution for better wheel detection / fullscreen
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# EAR/MAR thresholds
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3

# Head-tilt
HEAD_TILT_THRESH = 20            # degrees
HEAD_TILT_CONSEC_FRAMES = 10

# Wheel/lean-forward params
WHEEL_DIAMETER_M = 0.38
FACE_REAL_WIDTH_M = 0.16
LEARN_FOCAL = False
DISTANCE_ALERT_THRESHOLD_M = 0.32
SUSTAIN_SECONDS = 1.5
HEAD_DOWN_OFFSET_RATIO = 0.06
ALERT_DISPLAY_SEC = 3.0
ALERT_COOLDOWN = 4.0

# buffers
BUFFER_LEN = int(max(2.0, SUSTAIN_SECONDS) * 10)

# -------------------- HELPERS: EAR / MAR / HeadPose --------------------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    try:
        A = np.linalg.norm(mouth[2] - mouth[10])
        B = np.linalg.norm(mouth[4] - mouth[8])
        C = np.linalg.norm(mouth[0] - mouth[6])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)
    except Exception:
        return 0.0

def getHeadTiltAndCoords(image_size, image_points, draw_length=FRAME_HEIGHT):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype="double")
    try:
        size_y, size_x = image_size
        focal_length = size_x
        center = (size_x/2.0, size_y/2.0)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1))
        if image_points.shape != (6,2):
            return None, (0,0), (0,0), (0,0)
        ok, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None, (0,0), (0,0), (0,0)
        R_mat, _ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(R_mat[0,0] * R_mat[0,0] + R_mat[1,0] * R_mat[1,0])
        singular = sy < 1e-6
        if not singular:
            x_rot = math.atan2(R_mat[2,1], R_mat[2,2])
            y_rot = math.atan2(-R_mat[2,0], sy)
            z_rot = math.atan2(R_mat[1,0], R_mat[0,0])
        else:
            x_rot = math.atan2(-R_mat[1,2], R_mat[1,1])
            y_rot = math.atan2(-R_mat[2,0], sy)
            z_rot = 0
        roll = math.degrees(z_rot)
        nose_end_3d = np.array([(0.0, 0.0, draw_length)], dtype="double").reshape((1,3))
        nose_point_2d, _ = cv2.projectPoints(nose_end_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        nose_pt = tuple(nose_point_2d.ravel().astype(int))
        start_pt = (int(image_points[0,0]), int(image_points[0,1]))
        side_3d = np.array([(draw_length, 0.0, 0.0)], dtype="double").reshape((1,3))
        side_pt_2d, _ = cv2.projectPoints(side_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        side_pt = tuple(side_pt_2d.ravel().astype(int))
        return roll, start_pt, nose_pt, side_pt
    except Exception:
        return None, (0,0), (0,0), (0,0)

# -------------------- Wheel detection (improved) --------------------
def detect_wheel_circle_cropped(gray, frame_bgr,
                               crop_fraction=0.75,
                               dp=1.2, minR=30, maxR=800,
                               hough_param2=22):
    """
    Detect wheel circle in lower portion of frame.
    Returns (cx_full, cy_full, r_full) in full-frame coords or None.
    """
    h, w = gray.shape
    crop_y0 = int(h * (1.0 - crop_fraction))
    crop = gray[crop_y0:h, :]

    # Preprocess
    blur = cv2.GaussianBlur(crop, (7,7), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # HoughCircles on cropped area
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=dp, minDist=80,
                               param1=100, param2=hough_param2,
                               minRadius=minR, maxRadius=maxR)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        best = None
        best_score = -1
        for (cx_local, cy_local, r) in circles:
            # coverage along circumference (robustness)
            angles = np.linspace(0, 2*np.pi, 180).astype(np.float32)
            xs = np.clip((cx_local + r * np.cos(angles)).astype(int), 0, edges.shape[1]-1)
            ys = np.clip((cy_local + r * np.sin(angles)).astype(int), 0, edges.shape[0]-1)
            coverage = np.mean(edges[ys, xs] > 0)
            score = coverage * r  # prefer larger circles with edge coverage
            if score > best_score:
                best_score = score
                best = (cx_local, cy_local, r, coverage)
        if best is not None and best[3] > 0.06:  # lowered threshold to catch faint wheels
            cx_full = int(best[0])
            cy_full = int(best[1] + crop_y0)
            r_full = int(best[2])
            # sanity clip to frame
            cx_full = np.clip(cx_full, 0, w-1)
            cy_full = np.clip(cy_full, 0, h-1)
            r_full = max(5, min(r_full, max(w,h)))
            return cx_full, cy_full, r_full

    # fallback: contour-based detection on edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    best_score = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-8))
        (cx_local, cy_local), r = cv2.minEnclosingCircle(cnt)
        if r < minR or r > maxR:
            continue
        score = circularity * math.log(area + 1) * r
        if score > best_score:
            best_score = score
            best_cnt = (int(cx_local), int(cy_local), int(r), score, circularity)
    if best_cnt is not None and best_cnt[4] > 0.45:
        cx_full = best_cnt[0]
        cy_full = best_cnt[1] + crop_y0
        r_full = best_cnt[2]
        cx_full = np.clip(cx_full, 0, w-1)
        cy_full = np.clip(cy_full, 0, h-1)
        r_full = max(5, min(r_full, max(w,h)))
        return cx_full, cy_full, r_full

    return None

# -------------------- INIT --------------------
print("[INFO] loading models...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

print("[INFO] initializing MediaPipe FaceMesh...")
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

print("[INFO] opening camera...")
cap = cv2.VideoCapture(CAM_INDEX)
# set capture resolution (request to camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(0.5)

# state variables
wheel_cx = wheel_cy = wheel_r = None
scale_m_per_px = None
focal_length_px = None

dist_buffer = deque(maxlen=BUFFER_LEN)
last_alert_time = 0.0
alert_show_until = 0.0
alert_label = ""

eye_counter = 0
head_counter = 0

image_points = np.zeros((6,2), dtype="double")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# -------------------- Wheel calibration (optional) --------------------
def calibrate_wheel_and_focal():
    global wheel_cx, wheel_cy, wheel_r, scale_m_per_px, focal_length_px
    print("[CAL] Wheel calibration: put wheel/plate in lower half. Press 'c' to accept, 'q' to skip.")
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = imutils.resize(frame, width=FRAME_WIDTH)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = detect_wheel_circle_cropped(gray, frame, crop_fraction=0.75, minR=30, maxR=600, hough_param2=22)
        disp = frame.copy()
        if res:
            cx, cy, r = res
            cv2.circle(disp, (cx, cy), r, (0,255,0), 4)
            cv2.putText(disp, f"Wheel radius = {r}px", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        cv2.putText(disp, "Press 'c' to accept or 'q' to skip", (20, FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow("Calibration", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and res:
            wheel_cx, wheel_cy, wheel_r = res
            scale_m_per_px = (WHEEL_DIAMETER_M / 2.0) / float(wheel_r)
            print("[CAL] Wheel detected. Scale (m/px) = {:.6f}".format(scale_m_per_px))
            break
        elif key == ord('q'):
            print("[CAL] Calibration skipped.")
            break
    try:
        cv2.destroyWindow("Calibration")
    except:
        pass

# Run calibration once (optional)
calibrate_wheel_and_focal()
print("[INFO] System running. Press q or ESC to quit.")

# make display window fullscreen
win_name = "Driver Monitor - single file (fixed)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# -------------------- MAIN LOOP --------------------
while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        continue

    # no forced resize to keep original capture proportions. If needed, use imutils.resize
    frame = imutils.resize(frame, width=FRAME_WIDTH)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    size = gray.shape  # (rows, cols)

    # dlib faces
    rects = detector(gray, 0)
    if len(rects) > 0:
        cv2.putText(frame, f"{len(rects)} face(s) found", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # mediapipe for nose & head-down confirmation
    mp_results = face_mesh.process(rgb)
    face_detected_mp = False
    fx = fy = None
    if mp_results.multi_face_landmarks:
        face_detected_mp = True
        lm = mp_results.multi_face_landmarks[0]
        nose = lm.landmark[1]
        fx = int(nose.x * w)
        fy = int(nose.y * h)
        cv2.circle(frame, (fx, fy), 4, (255,0,0), -1)

    # auto wheel detection (if not calibrated)
    if wheel_r is None:
        res = detect_wheel_circle_cropped(gray, frame, crop_fraction=0.75, minR=30, maxR=int(min(w,h)*0.8), hough_param2=22)
        if res:
            wheel_cx, wheel_cy, wheel_r = res
            scale_m_per_px = (WHEEL_DIAMETER_M / 2.0) / float(wheel_r)
            print("[INFO] Auto wheel detected. Scale (m/px) = {:.6f}".format(scale_m_per_px))

    # draw wheel if valid and within frame
    if wheel_r is not None:
        cx = int(np.clip(wheel_cx, 0, w-1))
        cy = int(np.clip(wheel_cy, 0, h-1))
        rr = int(np.clip(wheel_r, 3, max(w,h)))
        # only draw if center is reasonably inside frame
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(frame, (cx, cy), rr, (0,255,0), 4)
            # small filled dot at center for clarity
            cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)

    # process dlib-detected faces for EAR/MAR/head-pose
    for rect in rects:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        face_box_color = (0,255,0)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # EAR
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)

        if ear < EYE_AR_THRESH:
            eye_counter += 1
            if eye_counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Closed!", (int(w*0.45), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else:
            eye_counter = 0

        # MAR
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0,255,0), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (int(w*0.64), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (int(w*0.78), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # head-pose points
        key_indices = [33, 8, 36, 45, 48, 54]
        for idx_j, lm_idx in enumerate(key_indices):
            x,y = shape[lm_idx]
            image_points[idx_j] = np.array([x,y], dtype='double')

        # minimal landmark draw
        for i,(x,y) in enumerate(shape):
            color = (0,255,0) if i in set(key_indices) else (0,0,255)
            cv2.circle(frame, (x,y), 1, color, -1)
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        head_roll, start_point, end_point, end_point_alt = getHeadTiltAndCoords((h,w), image_points, FRAME_HEIGHT)
        line_color = (255,0,0); alt_color = (0,0,255)
        if head_roll is not None:
            angle = float(head_roll)
            cv2.putText(frame, f'Head Tilt: {angle:.2f}', (int(w*0.13), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if abs(angle) > HEAD_TILT_THRESH:
                head_counter += 1
                line_color = (0,0,255); alt_color = (0,0,255)
            else:
                head_counter = 0
            if head_counter >= HEAD_TILT_CONSEC_FRAMES:
                cv2.putText(frame, "Head Tilted!", (int(w*0.45), 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                face_box_color = (0,0,255)
        else:
            head_counter = 0

        try:
            cv2.line(frame, start_point, end_point, line_color, 3)
            cv2.line(frame, start_point, end_point_alt, alt_color, 2)
            cv2.rectangle(frame, (bX,bY), (bX+bW, bY+bH), face_box_color, 2)
        except Exception:
            pass

    # ----------------- head-to-wheel proximity computations -----------------
    fused_distance = None
    if 'fx' in locals() and fx is not None and wheel_r is not None and scale_m_per_px is not None:
        d_px = np.hypot(fx - wheel_cx, fy - wheel_cy) - wheel_r
        fused_distance = max(0.0, d_px * scale_m_per_px)
    if fused_distance is not None:
        dist_buffer.append(fused_distance)

    # head-down confirmation via MediaPipe
    head_down_confirm = False
    if mp_results.multi_face_landmarks:
        ys = [p.y for p in mp_results.multi_face_landmarks[0].landmark]
        face_center_y = np.mean(ys) * h
        face_height_px = (max(ys) - min(ys)) * h
        if fy is not None and face_height_px > 5:
            if (fy - face_center_y) > (HEAD_DOWN_OFFSET_RATIO * face_height_px):
                head_down_confirm = True

    # sustained-close
    sustained_close = False
    SUSTAIN_FRAMES = max(3, int(SUSTAIN_SECONDS * 5))
    if len(dist_buffer) >= SUSTAIN_FRAMES:
        recent = np.array(dist_buffer)
        if np.all(recent[-SUSTAIN_FRAMES:] < DISTANCE_ALERT_THRESHOLD_M):
            sustained_close = True

    # final trigger + cooldown
    lean_alert_active = False
    if sustained_close and head_down_confirm and (time.time() - last_alert_time > ALERT_COOLDOWN):
        alert_show_until = time.time() + ALERT_DISPLAY_SEC
        alert_label = "Driver leaning forward!"
        last_alert_time = time.time()

    if time.time() < alert_show_until:
        lean_alert_active = True

    # overlay lean alert (top-left)
    if lean_alert_active:
        box_w, box_h = int(w*0.4), 80
        alpha = 0.75
        overlay = frame.copy()
        cv2.rectangle(overlay, (10,10), (10+box_w, 10+box_h), (0,0,0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        cv2.putText(frame, alert_label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,200,255), 3, cv2.LINE_AA)

    # distance readout
    if fused_distance is not None:
        cv2.putText(frame, f"WheelDist: {fused_distance:.2f} m", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    # footer and show (fullscreen window)
    cv2.putText(frame, "Press 'q' or ESC to quit", (w-350, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)
    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
