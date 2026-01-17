#!/usr/bin/env python3
"""
Driver Drowsiness Monitor - complete script

Requirements:
  - dlib (with the 68-landmarks predictor file)
  - imutils
  - opencv-python
  - numpy

Place dlib's shape_predictor_68_face_landmarks.dat and update PREDICTOR_PATH below.
"""

import time
import math
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream

# -------------------- Config --------------------
PREDICTOR_PATH = "./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat"
CAM_INDEX = 0  # 0 for primary webcam

FRAME_WIDTH = 1024
FRAME_HEIGHT = 576  # used for drawing/projecting length

# EAR / MAR thresholds & counters
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3

# Head tilt (roll) threshold and consecutive frames before warning
HEAD_TILT_THRESH = 20           # degrees; tune 15-30
HEAD_TILT_CONSEC_FRAMES = 10

# -------------------- Helper functions --------------------
def eye_aspect_ratio(eye):
    """
    eye: array of 6 (x, y) points
    returns EAR value (float)
    """
    # vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])
    # avoid division by zero
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """
    mouth: array of 20 (x, y) points (48..67)
    returns MAR value (float)
    Typical simple MAR: (vertical_avg) / horizontal
    """
    # Using open-mouth measure: use points:
    # top inner lip (51), bottom inner lip (57) and horizontal (48, 54)
    # indices relative to mouth array (0..19): 3 -> point 51, 9 -> 57, 0 -> 48, 6 -> 54
    A = np.linalg.norm(mouth[2] - mouth[10])  # 50 - 58
    B = np.linalg.norm(mouth[4] - mouth[8])   # 52 - 56
    C = np.linalg.norm(mouth[0] - mouth[6])   # 48 - 54 (horizontal)
    if C == 0:
        return 0.0
    mar = (A + B) / (2.0 * C)
    return mar

def getHeadTiltAndCoords(image_size, image_points, draw_length=FRAME_HEIGHT):
    """
    image_points: 6x2 numpy array containing:
        [nose tip(33), chin(8), left eye corner(36), right eye corner(45),
         left mouth corner(48), right mouth corner(54)]
    Returns:
        roll_deg (float scalar) OR None on failure,
        start_point (tuple) - usually nose tip,
        end_point (tuple) - projected forward vector
        end_point_alt (tuple) - projected sideways vector (for visualization)
    """
    # 3D model points of a generic head in mm
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -330.0, -65.0),   # Chin
        (-225.0, 170.0, -135.0),# Left eye left corner
        (225.0, 170.0, -135.0), # Right eye right corner
        (-150.0, -150.0, -125.0),# Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype="double")

    try:
        # camera matrix: fx = fy = focal length approx = width (or height)
        size_y, size_x = image_size
        focal_length = size_x
        center = (size_x / 2.0, size_y / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        # ensure proper shape (6,2)
        if image_points.shape != (6, 2):
            return None, (0,0), (0,0), (0,0)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, (0,0), (0,0), (0,0)

        # Rodrigues to rotation matrix
        R_mat, _ = cv2.Rodrigues(rotation_vector)

        # Compute Euler angles from rotation matrix
        # Using the convention: roll (x-axis), pitch (y-axis), yaw (z-axis)
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

        # Convert radians to degrees
        pitch = math.degrees(x_rot)
        yaw = math.degrees(y_rot)
        roll = math.degrees(z_rot)

        # For visualization: project a 3D point in front of the nose
        nose_end_3d = np.array([(0.0, 0.0, draw_length)], dtype="double").reshape((1, 3))
        nose_point_2d, _ = cv2.projectPoints(nose_end_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        nose_point_2d = tuple(nose_point_2d.ravel().astype(int))

        # start point is the nose tip in image_points[0]
        start_point = (int(image_points[0, 0]), int(image_points[0, 1]))

        # also project a sideways point for alt line (for better visualization)
        side_3d = np.array([(draw_length, 0.0, 0.0)], dtype="double").reshape((1, 3))
        side_point_2d, _ = cv2.projectPoints(side_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        side_point_2d = tuple(side_point_2d.ravel().astype(int))

        # return roll angle (tilt), and coordinates
        # roll is the rotation around z-axis; sign indicates left/right tilt
        return roll, start_point, nose_point_2d, side_point_2d

    except Exception as e:
        # on any failure return None so caller can handle
        # print("Head pose exception:", e)
        return None, (0,0), (0,0), (0,0)


# -------------------- Init ----------------------
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

print("[INFO] initializing camera...")
vs = VideoStream(src=CAM_INDEX).start()
time.sleep(2.0)

# counters
eye_counter = 0
head_counter = 0

# Pre-allocate image points for head pose (6 points)
image_points = np.zeros((6, 2), dtype="double")

# landmark index ranges
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]  # (48, 68)

print("[INFO] starting main loop. Press 'q' to quit.")
while True:
    frame = vs.read()
    if frame is None:
        print("[WARN] No frame received from camera. Exiting.")
        break

    # resize (keep aspect)
    frame = imutils.resize(frame, width=FRAME_WIDTH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape  # (rows, cols)

    rects = detector(gray, 0)

    if len(rects) > 0:
        text = f"{len(rects)} face(s) found"
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for rect in rects:
        # Face bounding box
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        face_box_color = (0, 255, 0)  # may change to red when alert
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), face_box_color, 1)

        # facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # --- Eyes / EAR ---
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # draw eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            eye_counter += 1
            if eye_counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Closed!", (int(FRAME_WIDTH*0.45), 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            eye_counter = 0

        # --- Mouth / MAR ---
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        cv2.putText(frame, f"MAR: {mar:.2f}", (int(FRAME_WIDTH*0.64), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (int(FRAME_WIDTH*0.78), 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Head pose: fill 6 image_points in order ---
        # indices we want: 33, 8, 36, 45, 48, 54  (0-based indexes accepted)
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

        # draw the selected image points slightly larger
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        # --- Head Pose / Tilt ---
        head_roll, start_point, end_point, end_point_alt = getHeadTiltAndCoords(size, image_points, FRAME_HEIGHT)

        # default line colors
        line_color = (255, 0, 0)
        alt_color = (0, 0, 255)

        if head_roll is not None:
            # Use head_roll (degrees). Convert to absolute tilt magnitude for thresholding
            angle = float(head_roll)
            cv2.putText(frame, f'Head Roll: {angle:.2f}', (170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if abs(angle) > HEAD_TILT_THRESH:
                head_counter += 1
                line_color = (0, 0, 255)  # red when tilted
                alt_color = (0, 0, 255)
            else:
                head_counter = 0

            if head_counter >= HEAD_TILT_CONSEC_FRAMES:
                cv2.putText(frame, "Head Tilted!", (int(FRAME_WIDTH*0.45), 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # make bounding box red when alert
                face_box_color = (0, 0, 255)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), face_box_color, 2)
        else:
            head_counter = 0

        # draw head orientation lines (colored per tilt)
        try:
            cv2.line(frame, start_point, end_point, line_color, 2)
            cv2.line(frame, start_point, end_point_alt, alt_color, 1)
        except Exception:
            pass

    # --- Show frame ---
    cv2.imshow("Driver Drowsiness Monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# -------------------- Cleanup -------------------
cv2.destroyAllWindows()
vs.stop()
