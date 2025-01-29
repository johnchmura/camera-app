import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_person_bounding_box(landmarks, image_shape):
    """
    Computes the bounding box of a person using pose landmarks.

    Args:
        landmarks (list): List of MediaPipe pose landmarks as floats.
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        tuple: Bounding box (x, y, w, h) or None if no valid landmarks.
    """
    if not landmarks:
        return None  # No person detected

    h, w, _ = image_shape

    x_coords = landmarks[0::3]  # Extract all x coordinates (every 3rd value starting from 0)
    y_coords = landmarks[1::3]  # Extract all y coordinates (every 3rd value starting from 1)

    # Get bounding box limits
    x_min, x_max = min(x_coords)*w, max(x_coords)*w
    y_min, y_max = min(y_coords)*h, max(y_coords)*h

    # Expand the bounding box slightly for better coverage
    margin_x = int((x_max - x_min) * 0.2)  # 15% margin
    margin_y = int((y_max - y_min) * 0.1)   # 20% margin

    x_min = max(0, int(x_min) - margin_x)
    x_max = min(w, int(x_max) + margin_x)
    y_min = max(0, int(y_min) - margin_y)
    y_max = min(h, int(y_max) + margin_y)

    return (x_min, y_min, x_max - x_min, y_max - y_min)  # (x, y, w, h)



def detect_leading_lines(image_path: str, pose_landmarks=None, visualize=False):
    """
    Detects leading lines in an image, calculates the convergence circle's center,
    and avoids detecting lines within a person's bounding box.

    Args:
        image_path (str): Path to the input image.
        pose_landmarks (list): List of pose landmarks to determine the bounding box.
        visualize (bool): If True, displays the processed image with detected lines and circle.

    Returns:
        circle_center (tuple): Coordinates of the circle center (or None if no lines are detected).
    """
    # Step 1: Load the image
    image = cv2.imread(image_path)

    # Step 2: Compute the bounding box if pose landmarks are provided
    hitbox = None
    if pose_landmarks:
        hitbox = get_person_bounding_box(pose_landmarks, image.shape)

    # Step 3: Convert to grayscale and apply contrast enhancement
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 4: Create a mask to exclude the person’s bounding box
    if hitbox:
        mask = np.ones_like(blurred, dtype=np.uint8) * 255  # White mask
        x, y, w, h = hitbox
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)  # Black rectangle
        blurred = cv2.bitwise_and(blurred, blurred, mask=mask)

    # Step 5: Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Step 6: Detect lines using Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # Step 7: Calculate midpoints of lines
    midpoints = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate and store the midpoint
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            midpoints.append((mid_x, mid_y))

    # Step 8: Find the center of leading lines
    circle_center = None
    if midpoints:
        midpoints = np.array(midpoints)
        circle_center = tuple(np.mean(midpoints, axis=0).astype(int))

    # Step 9: Visualize results if requested
    if visualize:
        # Create a copy of the image to draw the detected elements
        output_image = image.copy()

        # Draw detected lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the bounding box (person's hitbox)
        if hitbox:
            x, y, w, h = hitbox
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle

        # Draw the circle at the convergence point
        if circle_center:
            cv2.circle(output_image, circle_center, 10, (0, 0, 255), -1)  # Red circle

        # Display results
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title("Leading Lines & Bounding Box")
        plt.axis("off")

        plt.show()

    return circle_center

