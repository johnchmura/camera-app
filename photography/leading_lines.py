import numpy as np
from cv2 import (
    cvtColor, equalizeHist, GaussianBlur, rectangle, bitwise_and,
    Canny, HoughLinesP, line, circle, COLOR_BGR2GRAY
)

def get_person_bounding_box(landmarks, image_shape):
    """Computes the bounding box of a person using pose landmarks."""
    if not landmarks:
        return None  # No person detected

    h, w, _ = image_shape
    x_coords = np.array(landmarks[0::3]) * w  # Extract all x coordinates
    y_coords = np.array(landmarks[1::3]) * h  # Extract all y coordinates

    # Compute bounding box with margin
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    margin_x = int((x_max - x_min) * 0.2)  # 20% margin
    margin_y = int((y_max - y_min) * 0.1)  # 10% margin

    x_min = max(0, int(x_min - margin_x))
    x_max = min(w, int(x_max + margin_x))
    y_min = max(0, int(y_min - margin_y))
    y_max = min(h, int(y_max + margin_y))

    return (x_min, y_min, x_max - x_min, y_max - y_min)  # (x, y, w, h)

def detect_leading_lines(image, pose_landmarks=None):
    """
    Detects leading lines in an image and calculates the convergence point.

    Args:
        image (numpy array): The input image.
        pose_landmarks (list): List of pose landmarks for bounding box calculation.

    Returns:
        tuple: (lines, circle_center, bounding_box)
    """
    hitbox = get_person_bounding_box(pose_landmarks, image.shape) if pose_landmarks else None

    # Convert to grayscale and apply contrast enhancement
    gray = cvtColor(image, COLOR_BGR2GRAY)
    gray = equalizeHist(gray)
    blurred = GaussianBlur(gray, (3, 3), 0)

    # Mask out the bounding box area if detected
    if hitbox:
        x, y, w, h = hitbox
        blurred[y:y+h, x:x+w] = 0  # Directly zero out pixels instead of bitwise_and

    # Canny edge detection
    edges = Canny(blurred, 50, 150, apertureSize=3)

    # Detect lines using Probabilistic Hough Transform
    lines = HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # Compute the convergence point (center of detected lines)
    if lines is not None and len(lines) > 0:
        midpoints = np.mean(lines[:, 0, :2] + lines[:, 0, 2:], axis=0).astype(int)
        circle_center = tuple(midpoints)
    else:
        circle_center = None

    return lines, circle_center, hitbox

def draw_detected_lines(image, lines, bounding_box=None, circle_center=None):
    """
    Draws detected leading lines, bounding box, and convergence point on an image.

    Args:
        image (numpy array): The image to draw on.
        lines (list): List of detected lines.
        bounding_box (tuple): Bounding box of the person (x, y, w, h).
        circle_center (tuple): Convergence point of leading lines.

    Returns:
        numpy array: Image with drawn lines.
    """
    output_image = image.copy()

    # Draw detected lines
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:  # Efficiently unpack lines
            line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw bounding box if available
    if bounding_box:
        x, y, w, h = bounding_box
        rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box

    # Draw the convergence circle
    if circle_center:
        circle(output_image, circle_center, 10, (0, 0, 255), -1)  # Red dot

    return output_image
