# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:32:16 2024

@author: Feranmi Ayonitemi
"""

import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def app():
    st.header('Object Detection Web App')
    st.subheader('Powered by YOLOv10')
    st.write('Welcome!')
    model_path = 'best.pt'
    model = YOLO(model_path, task='detect', )
    #object_names = list(model.names.values())

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        #selected_objects = st.multiselect('Choose objects to detect', object_names, default=['person']) 
        min_confidence = st.slider('Confidence score', 0.0, 1.0, value = 0.05)
        submit_button = st.form_submit_button(label='Submit')
            
    if uploaded_file is not None and submit_button:
        
        # Load the Image
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
        
        # Convert PIL Image to OpenCV format
        image_np = np.array(input_image)
        
        # Perform Object Detection using Yolov10
        with st.spinner('Processing image...'):
            results = model(source=image_np, conf=min_confidence, save=True)
        
        
        # Convert back to PIL image to display with Streamlit
        result_image_pil = Image.fromarray(results[0].orig_img)
        st.write(result_image_pil)
        
        # Display the output
        results[0].show()#save=True, filename='eba_res.png', conf=True)
        #st.image(results[0].show(), caption="Detected Objects", use_column_width=True)
        #st.image(read_img, use_column_width=True)
if __name__ == "__main__":
    app()