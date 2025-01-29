import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from cv2 import imread,cvtColor,COLOR_BGR2RGB
from utilities.images import read_image_from_memory

# Singleton pattern for efficient PoseLandmarker initialization
_landmarker = None

def get_pose_landmarker():
    """Initializes the PoseLandmarker model only once (singleton)."""
    global _landmarker
    if _landmarker is None:
        base_options = python.BaseOptions(model_asset_path='models/mediapipe/pose_landmarker_lite.task')
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
        _landmarker = vision.PoseLandmarker.create_from_options(options)
    return _landmarker

def extract_landmarks(rgb_image):
    """
    Extracts pose landmarks from an RGB image using MediaPipe PoseLandmarker.

    :param rgb_image: RGB image as a NumPy array.
    :return: Flattened list of (x, y, z) landmarks or None if no landmarks detected.
    """
    detector = get_pose_landmarker()
    
    # Convert OpenCV image (RGB) to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Detect pose landmarks
    detection_result = detector.detect(mp_image)

    # Return landmarks as a NumPy array (flattened)
    if detection_result.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in detection_result.pose_landmarks[0]]).flatten().tolist()
    
    print("No landmarks detected in the image.")
    return None

def extract_pose_data(image):
    """
    Extracts pose landmarks from an image (file path, NumPy array, or raw bytes).

    :param image: Image input (str: file path, np.ndarray: OpenCV frame, bytes: raw image).
    :return: Flattened list of (x, y, z) landmarks or None.
    """
    input_handlers = {
        str: lambda img: imread(img),  # File path
        np.ndarray: lambda img: img,       # Already a NumPy array (frame from OpenCV)
        bytes: lambda img: read_image_from_memory(img)  # Raw image data
    }

    # Select appropriate handler based on input type
    bgr_image = input_handlers.get(type(image), lambda _: None)(image)

    if bgr_image is None:
        print("Failed to load image.")
        return None

    # Convert BGR to RGB for MediaPipe processing
    rgb_image = cvtColor(bgr_image, COLOR_BGR2RGB)

    # Extract and return pose landmarks
    return extract_landmarks(rgb_image)
