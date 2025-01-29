import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from cv2 import imread,cvtColor,COLOR_BGR2RGB
from utilities.images import read_image_from_memory

# Singleton pattern for efficient PoseLandmarker initialization
_landmarker = None

def get_pose_landmarker(running_mode='IMAGE'):
    """Initializes the PoseLandmarker model only once (singleton)."""
    global _landmarker
    if _landmarker is None:
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Determine the running mode
        if running_mode.upper() == 'IMAGE':
            mode = VisionRunningMode.IMAGE
        elif running_mode.upper() == 'VIDEO':
            mode = VisionRunningMode.VIDEO
        else:
            raise ValueError("Invalid running mode. Use 'IMAGE' or 'VIDEO'.")

        # Create a pose landmarker instance with the specified mode:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='models/mediapipe/pose_landmarker_lite.task'),
            running_mode=mode
        )
        _landmarker = vision.PoseLandmarker.create_from_options(options)
    return _landmarker

def extract_landmarks(rgb_image, running_mode='IMAGE', frame_count=None):
    """
    Extracts pose landmarks from an RGB image using MediaPipe PoseLandmarker.

    :param rgb_image: RGB image as a NumPy array.
    :param running_mode: 'IMAGE' or 'VIDEO' depending on the context.
    :param frame_count: Frame count for VIDEO mode (required if running_mode is 'VIDEO').
    :return: Flattened list of (x, y, z) landmarks or None if no landmarks detected.
    """
    detector = get_pose_landmarker(running_mode)
    
    # Convert OpenCV image (RGB) to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Detect pose landmarks based on the running mode
    if running_mode.upper() == 'IMAGE':
        detection_result = detector.detect(mp_image)
    elif running_mode.upper() == 'VIDEO':
        if frame_count is None:
            raise ValueError("frame_count must be provided for VIDEO mode.")
        detection_result = detector.detect_for_video(mp_image, frame_count)
    else:
        raise ValueError("Invalid running mode. Use 'IMAGE' or 'VIDEO'.")

    # Return landmarks as a NumPy array (flattened)
    if detection_result.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in detection_result.pose_landmarks[0]]).flatten().tolist()
    
    print("No landmarks detected in the image.")
    return None

def extract_pose_data(image, running_mode='IMAGE', frame_count=None):
    """
    Extracts pose landmarks from an image (file path, NumPy array, or raw bytes).

    :param image: Image input (str: file path, np.ndarray: OpenCV frame, bytes: raw image).
    :param running_mode: 'IMAGE' or 'VIDEO' depending on the context.
    :param frame_count: Frame count for VIDEO mode (required if running_mode is 'VIDEO').
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
    return extract_landmarks(rgb_image, running_mode, frame_count)


def find_roll(landmarks):
    """
    Calculate the roll of the face (rotation around the Z-axis).

    Parameters:
    ----------
    landmarks : list of float
        Flattened list of (x, y, z) coordinates for each landmark.

    Returns:
    -------
    float
        Roll angle (rotation around the Z-axis).
    """
    # Indices for left and right eye landmarks
    left_eye = landmarks[3 * 3:3 * 3 + 3]  # Left eye outer corner (landmark 3)
    right_eye = landmarks[6 * 3:6 * 3 + 3]  # Right eye outer corner (landmark 6)

    # Difference between y-coordinates of the eyes
    return left_eye[1] - right_eye[1]

def find_yaw(landmarks):
    """
    Calculate the yaw of the face (rotation around the Y-axis).

    Parameters:
    ----------
    landmarks : list of float
        Flattened list of (x, y, z) coordinates for each landmark.

    Returns:
    -------
    float
        Yaw angle (rotation around the Y-axis).
    """
    # Indices for left eye, right eye, and nose tip landmarks
    left_eye = landmarks[3 * 3:3 * 3 + 3]  # Left eye outer corner (landmark 3)
    right_eye = landmarks[6 * 3:6 * 3 + 3]  # Right eye outer corner (landmark 6)
    nose_tip = landmarks[0 * 3:0 * 3 + 3]  # Nose tip (landmark 0)

    # Distance from left eye to nose and right eye to nose
    le2n = nose_tip[0] - left_eye[0]
    re2n = right_eye[0] - nose_tip[0]

    # Difference between left and right distances
    return le2n - re2n

def find_pitch(landmarks):
    """
    Calculate the pitch of the face (rotation around the X-axis).

    Parameters:
    ----------
    landmarks : list of float
        Flattened list of (x, y, z) coordinates for each landmark.

    Returns:
    -------
    float
        Pitch angle (rotation around the X-axis).
    """
    # Indices for eye and mouth landmarks
    left_eye = landmarks[3 * 3:3 * 3 + 3]  # Left eye outer corner (landmark 3)
    right_eye = landmarks[6 * 3:6 * 3 + 3]  # Right eye outer corner (landmark 6)
    nose_tip = landmarks[0 * 3:0 * 3 + 3]  # Nose tip (landmark 0)
    mouth_left = landmarks[9 * 3:9 * 3 + 3]  # Mouth left corner (landmark 9)
    mouth_right = landmarks[10 * 3:10 * 3 + 3]  # Mouth right corner (landmark 10)

    # Average y-coordinate of the eyes and mouth
    eye_y = (left_eye[1] + right_eye[1]) / 2
    mou_y = (mouth_left[1] + mouth_right[1]) / 2

    # Distance from eyes to nose and nose to mouth
    e2n = eye_y - nose_tip[1]
    n2m = nose_tip[1] - mou_y

    # Ratio of distances
    return e2n / n2m if n2m != 0 else 0

import numpy as np

def find_pose(landmarks):
    """
    Calculate the roll, yaw, and pitch of the face using a geometric approach.
    Uses the first 10 landmarks for the face, and the rest for the body.

    Parameters:
    ----------
    landmarks : list of float
        Flattened list of (x, y, z) coordinates for each landmark.

    Returns:
    -------
    tuple
        A tuple containing:
        - roll: Rotation around the Z-axis (in degrees).
        - yaw: Rotation around the Y-axis (in degrees).
        - pitch: Rotation around the X-axis (in degrees).
    """
    # Extract first 10 landmarks for the face (x, y, z)
    face_landmarks = landmarks[:30]  # First 10 landmarks (3 values each)
    LMx = np.array([face_landmarks[i * 3] for i in range(len(face_landmarks) // 3)])  # x-coordinates
    LMy = np.array([face_landmarks[i * 3 + 1] for i in range(len(face_landmarks) // 3)])  # y-coordinates

    # Calculate roll (rotation around Z-axis)
    dPx_eyes = max((LMx[3] - LMx[1]), 1)  # Horizontal distance between eyes (landmarks 1 and 3)
    dPy_eyes = (LMy[3] - LMy[1])  # Vertical distance between eyes
    angle = np.arctan(dPy_eyes / dPx_eyes)  # Angle for rotation based on slope of eyes
    roll = angle * 180 / np.pi  # Convert radians to degrees

    # Calculate yaw and pitch using geometric transformations
    alpha = np.cos(angle)
    beta = np.sin(angle)
    LMxr = (alpha * LMx + beta * LMy)  # Rotated x-coordinates
    LMyr = (-beta * LMx + alpha * LMy)  # Rotated y-coordinates

    # Average distance between eyes and mouth (landmarks 2 and 4 for mouth)
    dXtot = (LMxr[3] - LMxr[1] + LMxr[4] - LMxr[2]) / 2
    dYtot = (LMyr[2] - LMyr[1] + LMyr[4] - LMyr[3]) / 2

    # Average distance between nose and eyes (landmark 0 for nose)
    dXnose = (LMxr[3] - LMxr[0] + LMxr[4] - LMxr[0]) / 2
    dYnose = (LMyr[2] - LMyr[0] + LMyr[4] - LMyr[0]) / 2

    # Calculate yaw and pitch
    yaw = (-90 + 90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0  # Yaw angle
    pitch = (-90 + 90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0  # Pitch angle

    return roll, yaw, pitch
