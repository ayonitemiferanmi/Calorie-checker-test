# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:34:38 2024

@author: Rise Networks
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
from ultralytics import YOLOv10
import io
import requests

app = FastAPI()

# temp_file = requests.get("https://huggingface.co/Ayonitemi-Feranmi/calorie_tester/resolve/main/best.pt")
# model_path_1 = ""
# with open(model_path_1, "wb") as f:
#     f.write(temp_file.content)

# Load the YOLO model (you can also include logic to download it from Hugging Face if not available)
model_path = "best.pt"
model = YOLOv10(model_path_1, task='detect')

@app.get("/")
def hello():
    return "Welcome to this fastapi"

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
    
    # Plot the detected objects on the image
    result_image_np = results[0].plot()  # This is a numpy array with detections

    # Convert the result back to a PIL image
    result_image_pil = Image.fromarray(result_image_np)
    
    # Prepare the image for response by saving it to a byte stream
    img_byte_arr = io.BytesIO()
    result_image_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Return the image with detections as a StreamingResponse
    return StreamingResponse(img_byte_arr, media_type="image/png")

# To run the app:
# uvicorn fastapi_app:app --reload
