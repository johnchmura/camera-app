import joblib
from typing import List
from utilities.media_pipe import extract_pose_data

def make_prediction_image(image_path: str):

    pose_data = extract_pose_data(image_path)
    
    if pose_data:
        print("Pose data extracted successfully:")
        print(pose_data)
        
        # Load the model from the .pkl file
        model = joblib.load('./models/pose-classifier/pose_classifier_test.pkl')
        label_encoder = joblib.load('./models/pose-classifier/label_encoder_test.pkl')
        scaler = joblib.load("./models/pose-classifier/scaler_test.pkl")
        # Reshape pose_data to match the model input
        pose_data_reshaped = scaler.transform([pose_data])  # Reshape to (1, number_of_features)

        # Use the model for predictions
        predictions = model.predict(pose_data_reshaped)  # Model expects 2D array
        predicted_label = label_encoder.inverse_transform(predictions)[0]
        print("Predicted pose class:", predicted_label)
        return predicted_label
    else:
        print("No pose data extracted.")
        return None

def make_prediction_data(pose_data: List[float]):
    if pose_data:
        
        # Load the model from the .pkl file
        model = joblib.load('./models/pose-classifier/pose_classifier_test.pkl')
        label_encoder = joblib.load('./models/pose-classifier/label_encoder_test.pkl')
        scaler = joblib.load("./models/pose-classifier/scaler.pkl")
        # Reshape pose_data to match the model input
        pose_data_reshaped = scaler.transform([pose_data])  # Reshape to (1, number_of_features)

        # Use the model for predictions
        predictions = model.predict(pose_data_reshaped)  # Model expects 2D array
        predicted_label = label_encoder.inverse_transform(predictions)[0]
        return predicted_label
    else:
        return None

   
def save_pose_data_to_file(image_path: str, output_file: str):
    # Extract pose data from the image
    pose_data = extract_pose_data(image_path)
    
    if pose_data:
        print("Pose data extracted successfully:")
        print(pose_data)
        
        # Write the pose data to a text file
        with open(output_file, 'w') as f:
            f.write(', '.join(map(str, pose_data)))  # Convert list of floats to a comma-separated string
            
        print(f"Pose data saved to {output_file}")
    else:
        print("No pose data extracted.")


def get_gaze_direction(yaw):
    """
    Classifies whether a person is looking left, right, or straight based on the yaw angle.

    Parameters:
    ----------
    yaw

    Returns:
    -------
    str
        A string indicating the gaze direction: "left", "right", or "straight".
    """

    # Define a threshold for yaw to determine left/right gaze
    yaw_threshold = 35

    if yaw < -yaw_threshold:
        return "left"
    elif yaw > yaw_threshold:
        return "right"
    else:
        return "straight"