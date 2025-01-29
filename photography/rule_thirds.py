import cv2
import numpy as np

def get_rule_thirds(img, grid_size=(3, 3), color=(0, 255, 0), thickness=1, circle_radius=5, highlight_point=None):
    """
    Draws a grid overlay on an image with circles at the four main rule-of-thirds intersections.
    If `highlight_point` is provided, the nearest intersection is drawn larger.

    Args:
        img: The input image.
        grid_size: A tuple specifying the number of rows and columns of the grid.
        color: The color of the grid lines and circles in BGR format.
        thickness: The thickness of the grid lines.
        circle_radius: The radius of the grid intersection circles.
        highlight_point: A tuple (x, y) representing a point to highlight the nearest intersection.

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

    # Find closest intersection if highlight_point is provided
    closest_point = None
    if highlight_point:
        distances = [np.linalg.norm(np.array(pt) - np.array(highlight_point)) for pt in points]
        closest_idx = np.argmin(distances)
        closest_point = points[closest_idx]

    # Draw circles at intersections
    for point in points:
        radius = circle_radius * 2 if point == closest_point else circle_radius  # Make closest point larger
        cv2.circle(img, point, radius, color, -1)

    return img, points
