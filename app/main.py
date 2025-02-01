from fastapi import FastAPI, HTTPException
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel
from typing import List
from utilities.pose_classifier import make_prediction_data, get_gaze_direction
from utilities.media_pipe import extract_pose_data, find_pose
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict

# Creating FastAPI instance
app = FastAPI()

# Defining the Pydantic model for request body
class RequestBody(BaseModel):
    numbers: List[float]  # Expecting an array of floats

class ImageRequest(BaseModel):
    image_name: str

@app.post('/pose-data')
def get_pose_data(request: ImageRequest):
    """
    Extract pose data from the given image name.
    """
    try:
        # Process the image using Mediapipe
        pose_data = extract_pose_data(request.image_name)

        if not pose_data:
            raise HTTPException(status_code=404, detail="No pose data could be extracted from the image.")

        # Return the extracted pose data
        return {"image_name": request.image_name, "pose_data": pose_data}

    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")



@app.post('/prediction')
async def get_prediction(image: UploadFile = File(...)) -> Dict:
    """
    Extract pose data from the given image.
    The image is expected to be uploaded as a file.
    """
    try:
        # Read the image data from the uploaded file
        image_data = await image.read()

        # Process the image using Mediapipe
        pose_data = extract_pose_data(image_data)
        prediction = make_prediction_data(pose_data)
        roll, yaw, pitch = find_pose(pose_data)
        look = get_gaze_direction(yaw)
        prediction = prediction +"direction: "+ look
        if not pose_data:
            raise HTTPException(status_code=404, detail="No pose data could be extracted from the image.")

        # Return the extracted pose data
        return {"prediction": prediction}

    except Exception as e:
        # Handle any unexpected errors
        return {"prediction": "No one found."}
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

