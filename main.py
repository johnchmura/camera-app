from fastapi import FastAPI, HTTPException
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel
from typing import List
from pose_classifier import make_prediction_data  # Ensure this is implemented correctly
from media_pipe import extract_pose_data_from_image
# Creating FastAPI instance
app = FastAPI()

# Defining the Pydantic model for request body
class RequestBody(BaseModel):
    numbers: List[float]  # Expecting an array of floats

class ImageRequest(BaseModel):
    image_name: str


@app.post('/predict')
def predict(data: RequestBody):
    # Validate the input (e.g., ensure the array is not empty)
    if not data.numbers:
        raise HTTPException(status_code=400, detail="The numbers array cannot be empty.")

    try:
        # Pass the data to the prediction function
        prediction = make_prediction_data(data.numbers)
    except Exception as e:
        # Handle any exceptions from the prediction function
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Return the result
    return {'class': prediction}

@app.post('/pose-data')
def extract_pose_data(request: ImageRequest):
    """
    Extract pose data from the given image name.
    """
    try:
        # Process the image using Mediapipe
        pose_data = extract_pose_data_from_image(request.image_name)

        if not pose_data:
            raise HTTPException(status_code=404, detail="No pose data could be extracted from the image.")

        # Return the extracted pose data
        return {"image_name": request.image_name, "pose_data": pose_data}

    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")