import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from utilities.images import read_image_from_memory

def extract_landmarks(rgb_image, detector):
    """
    Extracts pose landmarks (x, y, z coordinates) from a given RGB image using a MediaPipe PoseLandmarker.
    
    :param rgb_image: The image in RGB format (numpy array).
    :param detector: A MediaPipe PoseLandmarker instance to detect landmarks.
    :return: A list of landmarks in the form [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn] 
             or None if no landmarks are detected.
    """
    # Convert the OpenCV image (RGB) to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Detect pose landmarks in the image
    detection_result = detector.detect(mp_image)

    # If no landmarks are detected, return None
    if not detection_result.pose_landmarks:
        return None

    # Extract and return the landmarks as a flattened list of x, y, z coordinates
    landmarks = []
    for landmark in detection_result.pose_landmarks[0]:
        landmarks.extend([landmark.x, landmark.y, landmark.z])

    return landmarks
    
def extract_pose_data_from_image(image_path):
    """
    Extracts pose landmarks from a single image file located at the specified path.
    
    :param image_path: Path to the image file (string).
    :return: A list of landmarks [x, y, z] or None if no landmarks are detected.
    """
    # Initialize the PoseLandmarker with the model path for lightweight pose detection
    base_options = python.BaseOptions(model_asset_path='models/mediapipe/pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)  # We do not need segmentation masks
    detector = vision.PoseLandmarker.create_from_options(options)

    # Read the image from the file path
    bgr_image = imread(image_path)

    if bgr_image is None:
        print(f"Failed to read the image from {image_path}")
        return None

    # Convert the image from BGR to RGB format (as required by MediaPipe)
    rgb_image = cvtColor(bgr_image, COLOR_BGR2RGB)

    # Extract and return pose landmarks from the image
    landmarks = extract_landmarks(rgb_image, detector)

    if landmarks:
        return landmarks
    else:
        print(f"No landmarks detected in the image: {image_path}")
        return None

def extract_pose_data_from_image_data(image_data):
    """
    Extracts pose landmarks from raw image data (such as JPEG or PNG bytes) in memory.
    
    :param image_data: Raw image data in bytes (e.g., JPEG/PNG).
    :return: A list of landmarks [x, y, z] or None if no landmarks are detected.
    """
    # Initialize the PoseLandmarker with the model path for lightweight pose detection
    base_options = python.BaseOptions(model_asset_path='models/mediapipe/pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)  # We do not need segmentation masks
    detector = vision.PoseLandmarker.create_from_options(options)

    # Read the image from memory (e.g., JPEG/PNG bytes)
    bgr_image = read_image_from_memory(image_data)

    if bgr_image is None:
        print("Failed to decode the image from memory.")
        return None

    # Convert the image from BGR to RGB format (as required by MediaPipe)
    rgb_image = cvtColor(bgr_image, COLOR_BGR2RGB)

    # Extract and return pose landmarks from the image
    landmarks = extract_landmarks(rgb_image, detector)

    if landmarks:
        return landmarks
    else:
        print("No landmarks detected in the image.")
        return None
