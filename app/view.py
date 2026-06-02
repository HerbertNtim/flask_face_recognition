import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request

UPLOAD_FOLDER = os.path.join('static', 'upload')
PREDICT_FOLDER = os.path.join('static', 'predict')

def index():
  return render_template('index.html')

def app():
  return render_template('app.html')

def gender():
  if request.method == 'POST':
    f = request.files['image_name']
    filename = f.filename
    
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PREDICT_FOLDER, exist_ok=True)
    
    # Save image in upload folder
    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)   
    # Get predictions
    pred_image, predictions = faceRecognitionPipeline(path)
    pred_filename = 'prediction_image.jpg'
    pred_path = os.path.join(PREDICT_FOLDER, pred_filename)
    
    cv2.imwrite(pred_path, pred_image)
    
    print('ML model predicted successfully')
    
    
  return render_template('gender.html')
