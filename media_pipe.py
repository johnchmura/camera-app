import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np


def extract_landmarks(rgb_image, detector):
        # Convert the OpenCV image to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect pose landmarks
        detection_result = detector.detect(mp_image)

        # If no landmarks are detected, return None
        if not detection_result.pose_landmarks:
            return None

        # Extract landmarks (x, y, z)
        landmarks = []
        for landmark in detection_result.pose_landmarks[0]:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        return landmarks
    
# Function to extract pose landmarks from a single image
def extract_pose_data_from_image(image_path):

    # Initialize PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='models/mediapipe/pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Read the image
    bgr_image = cv2.imread(image_path)

    if bgr_image is None:
        print(f"Failed to read the image from {image_path}")
        return None

    # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Extract landmarks
    landmarks = extract_landmarks(rgb_image, detector)

    if landmarks:
        return landmarks
    else:
        print(f"No landmarks detected in the image: {image_path}")
        return None

def extract_pose_data_from_image_data(image_data):
    """
    Extracts pose landmarks from raw image data (JPEG/PNG bytes) in memory.
    
    :param image_data: Raw image data (e.g., JPEG or PNG bytes)
    :return: List of landmarks [x, y, z] or None if no landmarks are detected
    """
    # Initialize PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='models/mediapipe/pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Read the image from memory
    bgr_image = read_image_from_memory(image_data)

    if bgr_image is None:
        print("Failed to decode the image from memory.")
        return None

    # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Extract landmarks
    landmarks = extract_landmarks(rgb_image, detector)

    if landmarks:
        return landmarks
    else:
        print("No landmarks detected in the image.")
        return None
    
def read_image_from_memory(image_data):
    """
    Reads an image from raw bytes in memory using OpenCV.
    
    :param image_data: Raw image data (e.g., bytes)
    :return: Decoded BGR image or None if decoding fails
    """
    # Convert raw bytes into a NumPy array
    image_array = np.frombuffer(image_data, np.uint8)
    
    # Decode the image
    bgr_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    return bgr_image
