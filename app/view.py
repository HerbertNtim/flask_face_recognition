import os
import cv2
from app.face_recognition import faceRecognitionPipeline
import matplotlib.image as matimg
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
    
    # Generate report
    report = []
    for i, obj in enumerate(predictions):
      gray_image = obj['roi']
      eigen_image = obj['eig_img'].reshape(100, 100)
      gender_name = obj['prediction_name']
      score = round(obj['score']*100, 2)
      
      # Save grayscale and eigen in predict folder
      gray_image_name = f'roi_{i}.jpg'
      eigen_image_name = f'eigen_{i}.jpg'
      gray_path = os.path.join(PREDICT_FOLDER, gray_image_name)
      eigen_path = os.path.join(PREDICT_FOLDER, eigen_image_name)
      matimg.imsave(gray_path, gray_image, cmap='gray')
      matimg.imsave(eigen_path, eigen_image cmap='gray')
            
      # Save report
      report.append([gray_image_name, eigen_image_name, gender_name, score])
    
    
  return render_template('gender.html')





