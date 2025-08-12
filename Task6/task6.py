import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import time

custom_objects = {
    'mse': MeanSquaredError(),
    'mae': MeanAbsoluteError()
}

face_model = load_model(r'C:\Users\ROY\Documents\python_proj\Task6\facedetectortask6.h5', custom_objects=custom_objects)
age_model = load_model(r'C:\Users\ROY\Documents\python_proj\Task6\agetask6.h5', custom_objects=custom_objects)
gender_model = load_model(r'C:\Users\ROY\Documents\python_proj\Task6\gendertask6.h5', custom_objects=custom_objects)

IMG_SIZE = (128, 128)
MAX_AGE = 116.0
CSV_FILE = r'C:\Users\ROY\Documents\python_proj\Task6\entry_log.csv'
FRAME_INTERVAL = 15.0  #Logging frame every 15 seconds
BOX_EXPANSION = 0.35  #Expanding bounding box by 35% to account for MAE
EXIT_KEY = 'q'  #Key to exit the loop

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=['Entry_Time', 'Age', 'Gender', 'Label']).to_csv(CSV_FILE, index=False)

#Initializing webcam for real-time footage
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam at index 0")
    exit()

#Preprocessing image for face detection model
def preprocess_face_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img).resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img_pil) / 255.0
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

#Preprocessing image for age and gender models
def preprocess_age_gender(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pil = Image.fromarray(img).resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img_pil) / 255.0
    return np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 128, 128, 1)

#Initializing state for persistent red rectangles
last_box = None
last_age = None
last_gender = None
last_label = None
last_processed_time = 0.0

#Loop for real-time face detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        continue

    #Current time
    current_time = time.time()

    display_frame = frame.copy()
    height, width = frame.shape[:2]
    frame_array = preprocess_face_detection(frame)

    #Detecting faces
    try:
        bboxes = face_model.predict(frame_array, verbose=0)[0]
        print(f"Raw bounding box: {bboxes}, shape: {bboxes.shape}")
    except Exception as e:
        print(f"Error during face detection: {e}")
        bboxes = None

    #Processing bounding box
    if bboxes is not None and len(bboxes) == 4:
        x1 = int(bboxes[0] * width)
        y1 = int(bboxes[1] * height)
        x2 = int(bboxes[2] * width)
        y2 = int(bboxes[3] * height)
        scaling_type = "normalized [0, 1]"

        #Validating coordinates else fallback to pixel scaling ([0, 128])
        if max(x1, y1, x2, y2) > max(width, height) or min(x1, y1, x2, y2) < 0:
            x1 = int(bboxes[0] * width / IMG_SIZE[0])
            y1 = int(bboxes[1] * height / IMG_SIZE[1])
            x2 = int(bboxes[2] * width / IMG_SIZE[0])
            y2 = int(bboxes[3] * height / IMG_SIZE[1])
            scaling_type = "pixel [0, 128]"
            print("Warning: Normalized scaling produced invalid coordinates, using pixel scaling")

        #Expanding bounding box
        box_width = x2 - x1
        box_height = y2 - y1
        x1 = max(0, x1 - int(BOX_EXPANSION * box_width))
        y1 = max(0, y1 - int(BOX_EXPANSION * box_height))
        x2 = min(width, x2 + int(BOX_EXPANSION * box_width))
        y2 = min(height, y2 + int(BOX_EXPANSION * box_height))
        print(f"Scaled and expanded bounding box ({scaling_type}): x1={x1}, y1={y1}, x2={x2}, y2={y2}, frame_size={width}x{height}")

        if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
            last_box = [x1, y1, x2, y2]
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                print("Empty face region, skipping...")
                last_box = None
            else:
                #Predicting age and gender
                face_array = preprocess_age_gender(face)
                try:
                    age_pred = age_model.predict(face_array, verbose=0)[0][0] * MAX_AGE
                    last_age = int(round(age_pred))
                    print(f"Predicted age: {last_age}")
                except Exception as e:
                    print(f"Error during age prediction: {e}")
                    last_age = None

                try:
                    gender_pred = gender_model.predict(face_array, verbose=0)[0][0]
                    last_gender = 'Female' if gender_pred >= 0.5 else 'Male'
                except Exception as e:
                    print(f"Error during gender prediction: {e}")
                    last_gender = None

                last_label = 'Not allowed' if last_age is not None and (last_age < 13 or last_age > 60) else 'Allowed'
        else:
            print("Invalid bounding box, skipping...")
            last_box = None
    else:
        print("No valid bounding box detected, skipping...")
        last_box = None

    #Logging to CSV every 15 seconds
    if current_time - last_processed_time >= FRAME_INTERVAL and last_box and last_age is not None and last_gender is not None:
        entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_data = [{
            'Entry_Time': entry_time,
            'Age': last_age,
            'Gender': last_gender,
            'Label': last_label
        }]
        try:
            pd.DataFrame(log_data).to_csv(CSV_FILE, mode='a', header=False, index=False)
            print(f"Logged to CSV: {log_data}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
        last_processed_time = current_time

    #Drawing persistent red rectangle
    if last_box and last_label == 'Not allowed':
        x1, y1, x2, y2 = [int(coord) for coord in last_box]
        color = (0, 0, 255)  # Red
        label_text = f"Age: {last_age}, {last_gender} (Not allowed)" if last_age and last_gender else "Not allowed"
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Face Detection', display_frame)
    #Breaking loop on exit key press
    if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
        break

cap.release()
cv2.destroyAllWindows()