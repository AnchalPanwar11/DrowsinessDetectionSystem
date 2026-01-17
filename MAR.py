from scipy.spatial import distance as dist
import numpy as np

def mouth_aspect_ratio(mouth):
    """
    Compute the Mouth Aspect Ratio (MAR) using 3 vertical distances and one horizontal.
    Expected input: 20 mouth landmark points (indices 48–67 from dlib 68-point model).
    """
    # Vertical distances
    A = dist.euclidean(mouth[2], mouth[10])  # 50–58
    B = dist.euclidean(mouth[4], mouth[8])   # 52–56
    C = dist.euclidean(mouth[0], mouth[6])   # 48–54
    D = dist.euclidean(mouth[3], mouth[9])   # 51–57 (optional middle pair)

    mar = (A + B + D) / (3.0 * (C + 1e-6))  # prevent division by zero

    return mar
