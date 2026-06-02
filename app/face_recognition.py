import numpy as np 
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import cv2

# Load the models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml') # Cascade Classifier
model_svm = pickle.load(open('./model/model_svm.pickle', mode='rb')) # ml model (SVM)
pca_models = pickle.load(open('./model/pca_dict.pickle', mode='rb')) # pca dictionary
model_pca = pca_models['pca']
mean_face_array = pca_models['mean_face']

def faceRecognitionPipeline(filepath, path = True):
    # Step 01 - Read Image
    if path:
        img = cv2.imread(filepath)
    else:
        img = filepath # Array

    # Step 02 - Convert Image into gray Scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 03 - Crop the face (using haar cascase classifier )
    faces = haar.detectMultiScale(gray,1.5,3)
    predictions = []
    for x,y,w,h in faces:
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        roi = gray[y:y+h, x:x+w]
        
        # Step 04: Normalization (0, 1) Changing the value of the image data to be between 0 and 1
        roi = roi / 255.0
        
        # Step 05: Resize the Images (100, 100)
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)
    
        # Step 06: Flattening Images (1x10000)
        roi_reshape = roi_resize.reshape(1, 10000)
    
        # Step 07: Subtract with the mean
        roi_mean = roi_reshape - mean_face_array
    
        # Step 08: Get Eigen Image (apply roi_mean to pca)
        eigen_image = model_pca.transform(roi_mean)
    
        # Step 09: Eigen Image Visualization
        eigen_img = model_pca.inverse_transform(eigen_image)
    
        # Step 10: Pass the ML model (SVM) and get Predictions
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
    
        # Step 11: Generate the Report
        text = "%s : %d"%(results[0], prob_score_max*100)
    
        # Defining color based on results
        if results[0] == 'male':
            color = (255, 255, 0)
        else:
            color = (0, 255, 255)
    
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.rectangle(img, (x,y - 40), (x+w, y), color, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)
    
        output = {
            'roi': roi,
            'eig_img': eigen_img,
            'prediction_name': results[0],
            'score': prob_score_max
        }
    
        predictions.append(output)

    return img, predictions
