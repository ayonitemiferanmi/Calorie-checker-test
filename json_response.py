import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLOv10
import io


# make sure you install fastapi, pip install fastapi uvicorn
# to run and test the code   ====>  uvicorn main:predict and go to http://127.0.0.1:8000/docs to test the model with the endpoint with an image. 

app = FastAPI()

# I am assuming that this is the model that you trained
model_path = "best.pt"

# Instantiating the model
model = YOLOv10(model=model_path, task="detect")

# Welcome page of the Fast API
@app.get("/")
def hello():
    return "Welcome to this fastapi"

# @app.post("/detect")
# async def predict(file: UploadFile = File(...), confidence: float = 0.05)
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), confidence: float = 0.05):
    # Read the uploaded image file
    image_bytes = await file.read()
    
    # Open the image using PIL
    input_image = Image.open(io.BytesIO(image_bytes))
    
    # Convert PIL image to numpy array for model processing
    image_np = np.array(input_image)
    
    # Perform object detection using YOLOv10 model
    results = model(source=image_np, conf=confidence, save=False)
    
    # Extract detection data
    detections = []
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            detections.append({
                "box": box.tolist(),  # [x_min, y_min, x_max, y_max]
                "class": int(cls),
                "confidence": float(conf)
            })
    
    # Return the detections as JSON
    return {"detections": detections}
