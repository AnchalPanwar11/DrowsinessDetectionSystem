#!/usr/bin/env python
import time
import math
import cv2
import dlib
import imutils
import numpy as np

from imutils.video import VideoStream
from imutils import face_utils

from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

# -------------------- Config --------------------
PREDICTOR_PATH = './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat'
CAM_INDEX = 0  # change to 0 if your primary webcam is index 0

FRAME_WIDTH = 1024
FRAME_HEIGHT = 576   # used by head-pose helper only (drawing length)

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3

# -------------------- Init ----------------------
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

print("[INFO] initializing camera...")
vs = VideoStream(src=CAM_INDEX).start()
# vs = VideoStream(usePiCamera=True).start()  # Raspberry Pi
time.sleep(2.0)

COUNTER = 0

# Pre-allocate 2D image points (updated per frame from landmarks)
# Order: [nose tip(33), chin(8), left eye corner(36), right eye corner(45),
#         left mouth corner(48), right mouth corner(54)]
image_points = np.zeros((6, 2), dtype="double")

# landmark index ranges
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]  # (48, 68)

# -------------------- Loop ----------------------
while True:
    frame = vs.read()
    if frame is None:
        # camera not ready or failed
        print("[WARN] No frame received from camera. Check CAM_INDEX.")
        break

    # keep aspect ratio; height computed automatically
    frame = imutils.resize(frame, width=FRAME_WIDTH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    rects = detector(gray, 0)

    if len(rects) > 0:
        text = f"{len(rects)} face(s) found"
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for rect in rects:
        # Face bbox
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

        # 68 landmarks -> numpy
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # --- Eyes / EAR ---
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Closed!", (int(FRAME_WIDTH*0.45), 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

        # --- Mouth / MAR ---
        mouth = shape[mStart:mEnd]  # 48..67
        mar = mouth_aspect_ratio(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        cv2.putText(frame, f"MAR: {mar:.2f}", (int(FRAME_WIDTH*0.64), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (int(FRAME_WIDTH*0.78), 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Select 6 key 2D points for head pose ---
        # indices we want: 33, 8, 36, 45, 48, 54
        key_indices = [33, 8, 36, 45, 48, 54]
        for idx_j, lm_idx in enumerate(key_indices):
            x, y = shape[lm_idx]
            image_points[idx_j] = np.array([x, y], dtype='double')

        # Visualize all landmarks (green for the 6 we use)
        key_set = set(key_indices)
        for i, (x, y) in enumerate(shape):
            color = (0, 255, 0) if i in key_set else (0, 0, 255)
            cv2.circle(frame, (x, y), 1, color, -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Draw the determinant image points (slightly larger)
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        # --- Head Pose / Tilt ---
        head_tilt_degree, start_point, end_point, end_point_alt = \
            getHeadTiltAndCoords(size, image_points, FRAME_HEIGHT)

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

        if head_tilt_degree is not None:
            cv2.putText(frame, f'Head Tilt Degree: {head_tilt_degree[0]}',
                        (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

    # --- Show frame ---
    cv2.imshow("Driver Drowsiness Monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# -------------------- Cleanup -------------------
cv2.destroyAllWindows()
vs.stop()
