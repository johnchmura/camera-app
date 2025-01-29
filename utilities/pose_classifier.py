import joblib
from typing import List
from utilities.media_pipe import extract_pose_data_from_image

def make_prediction_image(image_path: str):

    pose_data = extract_pose_data_from_image(image_path)
    
    if pose_data:
        print("Pose data extracted successfully:")
        print(pose_data)
        
        # Load the model from the .pkl file
        model = joblib.load('./models/pose-classifier/pose_classifier.pkl')
        label_encoder = joblib.load('./models/pose-classifier/label_encoder.pkl')
        scaler = joblib.load("./models/pose-classifier/scaler.pkl")
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
        print("Pose data extracted successfully:")
        print(pose_data)
        
        # Load the model from the .pkl file
        model = joblib.load('./models/pose-classifier/pose_classifier.pkl')
        label_encoder = joblib.load('./models/pose-classifier/label_encoder.pkl')
        scaler = joblib.load("./models/pose-classifier/scaler.pkl")
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

   
def save_pose_data_to_file(image_path: str, output_file: str):
    # Extract pose data from the image
    pose_data = extract_pose_data_from_image(image_path)
    
    if pose_data:
        print("Pose data extracted successfully:")
        print(pose_data)
        
        # Write the pose data to a text file
        with open(output_file, 'w') as f:
            f.write(', '.join(map(str, pose_data)))  # Convert list of floats to a comma-separated string
            
        print(f"Pose data saved to {output_file}")
    else:
        print("No pose data extracted.")

