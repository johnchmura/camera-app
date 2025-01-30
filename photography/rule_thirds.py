import cv2
import numpy as np

def get_rule_thirds(img, grid_size=(3, 3), color=(0, 255, 0), thickness=1, circle_radius=5, highlight_point=None, posture="forward"):
    """
    Draws a grid overlay on an image with circles at the four main rule-of-thirds intersections.
    If `highlight_point` is provided, the nearest intersection is drawn larger, unless a specific posture is given.

    Args:
        img: The input image.
        grid_size: A tuple specifying the number of rows and columns of the grid.
        color: The color of the grid lines and circles in BGR format.
        thickness: The thickness of the grid lines.
        circle_radius: The radius of the grid intersection circles.
        highlight_point: A tuple (x, y) representing a point to highlight the nearest intersection.
        posture: A string indicating the posture classification ("forward", "backward", "skewed_left", "skewed_right", "over_shoulder_left", "over_shoulder_right").

    Returns:
        Tuple: (Modified image, list of the four rule-of-thirds points)
    """
    height, width, _ = img.shape

    # Calculate third divisions (not full grid, only 3x3)
    dx = width // 3
    dy = height // 3

    # Draw vertical lines at 1/3 and 2/3 of the width
    for x in [dx, 2 * dx]:
        cv2.line(img, (x, 0), (x, height), color, thickness)

    # Draw horizontal lines at 1/3 and 2/3 of the height
    for y in [dy, 2 * dy]:
        cv2.line(img, (0, y), (width, y), color, thickness)

    # Rule of thirds intersection points (inner 4 only)
    points = [
        (dx, dy), (2 * dx, dy),   # Top-left, Top-right
        (dx, 2 * dy), (2 * dx, 2 * dy)  # Bottom-left, Bottom-right
    ]

    # Determine which points to highlight based on posture
    if posture in ["forward", "backwards"]:
        # Default behavior: highlight the closest point if highlight_point is provided
        closest_point = None
        if highlight_point:
            distances = [np.linalg.norm(np.array(pt) - np.array(highlight_point)) for pt in points]
            closest_idx = np.argmin(distances)
            closest_point = points[closest_idx]
    elif posture in ["skewed left", "over shoulder left"]:
        # Highlight both right-side points
        closest_point = None
        points_to_highlight = [points[1], points[3]]  # Top-right, Bottom-right
    elif posture in ["skewed right", "over shoulder right"]:
        # Highlight both left-side points
        closest_point = None
        points_to_highlight = [points[0], points[2]]  # Top-left, Bottom-left
    else:
        raise ValueError(f"Unknown posture: {posture}")
    points_to_highlight = []
    # Draw circles at intersections
    if posture in ["forward", "backward"]:
        for point in points:
            radius = circle_radius * 2 if point == closest_point else circle_radius  # Make closest point larger
            cv2.circle(img, point, radius, color, -1)
    else:
        for point in points_to_highlight:
            cv2.circle(img, point, circle_radius * 2, color, -1)  # Highlight selected points

    return img, points