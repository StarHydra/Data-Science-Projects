import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from datetime import datetime
import pytz
from itertools import permutations
import time

st.set_page_config(page_title="Sign Language Detector", page_icon="Task7/hello.png", layout="centered")

st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] {
    background: radial-gradient(#EDEFA9,#10911A);}
    .main-header { font-size: 4.5rem; font-weight: 800; color: #540f60; text-align: center; margin-bottom: 5rem; }
    .sub-header {font-size: 1.89rem; font-weight: 500; color: #540f60; margin: 2.5rem 0 2.5rem; text-align:center; }
    .result-text {font-size: 2rem; font-weight: bolder; border-radius: 1rem; margin-bottom: 1rem; text-align:center; }
    .image-container { margin-bottom: 1rem; border-radius: 1rem; text-align: center; background-color: #44b4c5; }
    .stButton>button { background-color: #267384; color: #D890f4; font-weight: 700; border-radius: 0.5rem; padding: 0.5rem 1rem; }
    .stButton>button:hover { background-color: #1a5460; }
    .error-text { font-size: 1.3rem; font-weight: 200; color: #540f60; border-radius: 1rem; margin-bottom: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_models():
    try:
        sign_language_model = load_model(r"Task7\signlan_task7.h5")
        return sign_language_model
    except Exception as e:
        st.error(f"Error loading sign language model: {e}")
        return None

def preprocess_sign_image(image, target_size=(256, 256)):
    try:
        img_array = np.array(image.convert("RGB"))[..., ::-1]  
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        if img_array.shape != target_size:
            img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_NEAREST)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1) 
        return img_array
    except Exception as e:
        st.error(f"Error in sign language image preprocessing: {e}")
        return None

def predict_sign_language(model, image_features):
    try:
        sign_classes = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
                        9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G', 16: 'H',
                        17: 'I', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P',
                        25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X',
                        33: 'Y', 34: 'Z'}
        prediction = model.predict(image_features[np.newaxis, ...], verbose=0)
        sign_id = np.argmax(prediction, axis=1)[0]
        return sign_classes.get(sign_id, "Unknown")
    except Exception as e:
        st.error(f"Error predicting sign language: {e}")
        return None

def check_word_match(predictions):
    target_words = ['SANG', 'RING', 'KING', 'BOOK', 'FIVE']
    for perm in permutations(predictions):
        word = ''.join(perm)
        if word in target_words:
            return word
    return None

def process_webcam(model, duration=60):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Ensure a camera is connected and permissions are granted.")
            return None

        predictions = []
        start_time = time.time()
        frame_count = 0
        valid_frame_count = 0

        st.write("Capturing webcam footage for 60 seconds. Show sign language gestures clearly.")
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture a frame. Continuing to next attempt...")
                time.sleep(0.1)
                continue

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sign_image = preprocess_sign_image(image)
            if sign_image is not None:
                prediction = predict_sign_language(model, sign_image)
                frame_count += 1
                if prediction != "Unknown":
                    if prediction not in predictions:  #Avoiding duplicate predictions
                        predictions.append(prediction)
                        valid_frame_count += 1
                        st.write(f"Detected: {prediction}, Valid frames: {valid_frame_count}")

            #Safety limit to avoid infinite loop
            if frame_count >= 600:
                break

            time.sleep(0.1) #Controlling the frame rate

        cap.release()

        if valid_frame_count < 4:
            st.error(f"Only captured {valid_frame_count} valid frames. Need at least 4. Please try again with clear gestures.")
            return None
        return predictions

    except Exception as e:
        st.error(f"Error processing webcam: {e}")
        return None

def main():
    st.markdown('<div class="main-header">Sign Language Detector</div>', unsafe_allow_html=True)
    with st.spinner("Loading sign language model..."):
        sign_language_model = load_models()
    if sign_language_model is None:
        st.warning("Ensure the model file 'signlan_task7.h5' exists at the specified path.")
        return
    st.markdown('<div class="sub-header">Upload Exactly 4 Image Files or Use Webcam</div>', unsafe_allow_html=True)

    #Getting the current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    current_hour = current_time.hour
    current_minute = current_time.minute

    #To check if the current time is between 9:00 AM and 4:00 PM IST
    if not (9 <= current_hour < 16) or (current_hour == 16 and current_minute > 0):
        st.error("This application is available only between 9:00 AM and 4:00 PM IST.")
        return

    #For selecting either image upload or webcam capture
    option = st.radio("Select input method:", ("Upload Images", "Use Webcam"))

    if option == "Upload Images":
        uploaded_files = st.file_uploader("Choose image files (.jpg, .png)", type=["jpg", "png"], accept_multiple_files=True)
        if st.button("Detect Sign Language"):
            if uploaded_files is None or len(uploaded_files) != 4:
                st.error("Please upload exactly 4 image files.")
            else:
                with st.spinner("Analyzing images"):
                    predictions = []
                    for uploaded_file in uploaded_files:
                        image = Image.open(uploaded_file)
                        sign_image = preprocess_sign_image(image)
                        if sign_image is None:
                            st.error("Failed to preprocess sign language image.")
                            return
                        sign_prediction = predict_sign_language(sign_language_model, sign_image)
                        if sign_prediction is None:
                            st.error("Failed to predict sign language.")
                            return
                        predictions.append(sign_prediction)

                    matched_word = check_word_match(predictions)
                    if matched_word:
                        st.markdown(f'<div class="result-text"><strong>{matched_word}</strong></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-text"><strong>No word detected</strong></div>', unsafe_allow_html=True)

    elif option == "Use Webcam":
        if st.button("Start Webcam Capture (60 seconds)"):
            with st.spinner("Capturing and analyzing webcam footage"):
                predictions = process_webcam(sign_language_model)
                if predictions is None or len(predictions) < 4:
                    st.error("Failed to capture enough valid frames. Please try again with clear gestures.")
                else:
                    matched_word = check_word_match(predictions)
                    if matched_word:
                        st.markdown(f'<div class="result-text"><strong>{matched_word}</strong></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-text"><strong>No word detected</strong></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()