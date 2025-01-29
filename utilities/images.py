from cv2 import imdecode, IMREAD_COLOR
import numpy as np

def read_image_from_memory(image_data):
    """
    Reads an image from raw bytes in memory using OpenCV.
    
    :param image_data: Raw image data (e.g., bytes)
    :return: Decoded BGR image or None if decoding fails
    """
    # Convert raw bytes into a NumPy array
    image_array = np.frombuffer(image_data, np.uint8)
    
    # Decode the image
    bgr_image = imdecode(image_array, IMREAD_COLOR)
    
    return bgr_image