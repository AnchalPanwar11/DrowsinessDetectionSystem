import numpy as np
import math
import cv2

# 3D model points (mm). Order MUST match your 2D image_points:
# [nose tip(33), chin(8), left eye corner(36), right eye corner(45),
#  left mouth corner(48), right mouth corner(54)]
model_points = np.array([
    (0.0,    0.0,    0.0),    # Nose tip
    (0.0,  -330.0,  -65.0),   # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0,  170.0, -135.0),  # Right eye right corner
    (-150.0,-150.0, -125.0),  # Left mouth corner
    (150.0,-150.0, -125.0)    # Right mouth corner
], dtype=np.float64)

def isRotationMatrix(R: np.ndarray) -> bool:
    Rt = R.T
    shouldBeIdentity = Rt @ R
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R: np.ndarray) -> np.ndarray:
    """
    Returns angles (x, y, z) in radians using the same convention as your code:
    x ~ pitch (rotation around X), y ~ yaw (around Y), z ~ roll (around Z).
    """
    assert isRotationMatrix(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)

def getHeadTiltAndCoords(size, image_points, frame_height):
    """
    Args:
        size: gray.shape (h, w) from the current frame.
        image_points: np.ndarray shape (6, 2) in the exact order of model_points.
        frame_height: only used for drawing lengths in your app.
    Returns:
        head_tilt_degree: np.array([abs_pitch_deg])  # to match your existing usage head_tilt_degree[0]
        start_point: (x,y) at the nose tip in the image
        end_point: (x,y) for the projected "nose direction" (long Z+ axis)
        end_point_alt: (x,y) for an alternate axis projection (short X axis)
    """
    image_points = np.asarray(image_points, dtype=np.float64)
    h, w = size[:2]

    # Simple intrinsic guess: focal length ~ width in pixels, principal point at image center
    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0,         center[0]],
        [0.0,          focal_length, center[1]],
        [0.0,          0.0,          1.0      ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assuming no lens distortion

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, (0, 0), (0, 0), (0, 0)

    # Project a forward point from the nose to visualize orientation
    axis_3d = np.array([
        (0.0, 0.0, 1000.0),  # forward (Z+)
        (100.0, 0.0, 0.0),   # X axis to the right
        (0.0, 100.0, 0.0),   # Y axis up
    ], dtype=np.float64)

    axis_2d, _ = cv2.projectPoints(axis_3d, rotation_vector, translation_vector,
                                   camera_matrix, dist_coeffs)

    # Convert rvec -> R and compute Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    euler_rad = rotationMatrixToEulerAngles(rotation_matrix)
    euler_deg = np.degrees(euler_rad)  # [pitch, yaw, roll] in degrees (x, y, z)

    # "Head tilt degree" for your overlay: use absolute pitch in degrees
    abs_pitch_deg = float(abs(euler_deg[0]))
    head_tilt_degree = np.array([abs_pitch_deg], dtype=np.float64)

    # Start point is nose tip (first of the 6 image points)
    start_point = (int(image_points[0][0]), int(image_points[0][1]))

    nose_dir = axis_2d[0][0]
    x_axis = axis_2d[1][0]
    # (We keep only one alternate as your original API expects four returns.)
    end_point = (int(nose_dir[0]), int(nose_dir[1]))
    end_point_alternate = (int(x_axis[0]), int(x_axis[1]))

    return head_tilt_degree, start_point, end_point, end_point_alternate

# ---------------------------
# Optional: if you ever want full angles without changing your existing call-site,
# you could add a helper like:
#
# def getEulerAnglesDeg(size, image_points):
#     ... # same solvePnP
#     return float(pitch_deg), float(yaw_deg), float(roll_deg)
#
# and use it just for logging/tuning thresholds.
