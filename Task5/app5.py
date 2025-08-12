import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from PIL import Image
import cv2
from matplotlib import colors as mcolors

st.set_page_config(page_title="Person Attribute and Clothing Detector", page_icon="Task5/face-scan.png", layout="centered")

st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] {
    background: radial-gradient(#EDEFA9,#10911A);}
    .main-header { font-size: 4.5rem; font-weight: 800; color: #540f60; text-align: center; margin-bottom: 5rem; }
    .sub-header {font-size: 1.89rem; font-weight: 500; color: #540f60; margin: 2.5rem 0 2.5rem; text-align:center; }
    .result-text {font-size: 1.6rem; font-weight: bolder; border-radius: 1rem; margin-bottom: 1rem; text-align:center; }
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
    custom_objects = {
        'mse': MeanSquaredError(),
        'mae': MeanAbsoluteError(),
    }
    try:
        face_model = load_model(r"Task5/facedetectortask5.h5", custom_objects=custom_objects)
        nationality_model = load_model(r"Task5/nationalitytask5.h5")
        emotion_model = load_model(r"Task5/emotiontask5.h5")
        clothing_model = load_model(r"Task5/clothingtask5.h5", custom_objects=custom_objects)  
        age_model = load_model(r"Task5/agetask5.h5", custom_objects=custom_objects)
        return face_model, nationality_model, emotion_model, clothing_model, age_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

def detect_and_crop_face(image, face_detector):
    try:
        img_array = np.array(image.convert("RGB"))
        orig_height, orig_width = img_array.shape[:2]
        img_resized = cv2.resize(img_array, (128, 128))
        img_normalized = img_resized.astype(np.float32) / 255.0
        bbox = face_detector.predict(img_normalized[np.newaxis, ...], verbose=0)[0]
        x1, y1, x2, y2 = bbox
        #Validating bounding box
        if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
            st.error("Invalid bounding box coordinates (outside [0, 1]).")
            return None, None, None
        if x2 <= x1 or y2 <= y1:
            st.error("Invalid bounding box (x2 <= x1 or y2 <= y1).")
            return None, None, None
        #Buffer to ensure face is not cropped too tightly 
        buffer = 10
        x1 = int(max(0, x1 * orig_width - buffer))
        y1 = int(max(0, y1 * orig_height - buffer))
        x2 = int(min(orig_width, x2 * orig_width + buffer))
        y2 = int(min(orig_height, y2 * orig_height + buffer))
        #Ensuring coordinates are valid after buffer
        if x2 <= x1 or y2 <= y1:
            st.error("Invalid bounding box after applying buffer.")
            return None, None, None
        #Cropping face
        face_crop = img_array[y1:y2, x1:x2]
        if face_crop.size == 0:
            st.error("Empty face crop detected.")
            return None, None, None
        #Resizing image for different models
        face_128_rgb = cv2.resize(face_crop, (128, 128))
        face_224_rgb = cv2.resize(face_crop, (224, 224))
        face_128_gray = cv2.cvtColor(face_128_rgb, cv2.COLOR_RGB2GRAY)
        face_128_gray = face_128_gray[..., np.newaxis]
        return face_128_rgb, face_224_rgb, face_128_gray
    except Exception as e:
        st.error(f"Error in face detection: {e}")
        return None, None, None

def detect_and_crop_clothing(image, clothing_model):
    try:
        #Resize original image to 224x224
        img_array = np.array(image.convert("RGB"))
        img_resized = tf.image.resize(img_array, [224, 224]).numpy()
        img_normalized = img_resized.astype(np.float32) / 255.0 
        #Predicting clothing bounding box
        clothing_bbox = clothing_model.predict(img_normalized[np.newaxis, ...], verbose=0)[0]
        cx1, cy1, cx2, cy2 = clothing_bbox
        #Validating clothing bounding box
        if not (0 <= cx1 <= 1 and 0 <= cy1 <= 1 and 0 <= cx2 <= 1 and 0 <= cy2 <= 1):
            st.warning("Invalid clothing bounding box coordinates; using full image region instead.")
            return img_resized
        if cx2 <= cx1 or cy2 <= cy1:
            st.warning("Invalid clothing bounding box; using full image region instead.")
            return img_resized
        orig_height, orig_width = img_array.shape[:2]
        buffer = 20  #Adjust based on MAE
        cx1 = int(max(0, cx1 * 224 - buffer))  
        cy1 = int(max(0, cy1 * 224 - buffer))
        cx2 = int(min(224, cx2 * 224 + buffer))
        cy2 = int(min(224, cy2 * 224 + buffer))
        #Ensuring valid coordinates after buffer within 224x224
        if cx2 <= cx1 or cy2 <= cy1:
            st.warning("Invalid clothing bounding box after applying buffer; using full image region instead.")
            return img_resized
        #Scaling bounding box to original image
        scale_x = orig_width / 224
        scale_y = orig_height / 224
        clothing_x1 = int(cx1 * scale_x)
        clothing_y1 = int(cy1 * scale_y)
        clothing_x2 = int(cx2 * scale_x)
        clothing_y2 = int(cy2 * scale_y)
        #Cropping clothing region from original image
        clothing_crop = img_array[clothing_y1:clothing_y2, clothing_x1:clothing_x2]
        if clothing_crop.size == 0:
            st.warning("Empty clothing crop detected; using full image region instead.")
            return img_resized
        return clothing_crop
    except Exception as e:
        st.error(f"Error in clothing detection: {e}")
        return img_resized

def get_face_bbox(image, face_detector):
    img_array = np.array(image.convert("RGB"))
    orig_height, orig_width = img_array.shape[:2]
    img_resized = cv2.resize(img_array, (128, 128))
    img_normalized = img_resized.astype(np.float32) / 255.0
    bbox = face_detector.predict(img_normalized[np.newaxis, ...], verbose=0)[0]
    x1, y1, x2, y2 = bbox
    buffer = 10
    x1 = int(max(0, x1 * orig_width - buffer))
    y1 = int(max(0, y1 * orig_height - buffer))
    x2 = int(min(orig_width, x2 * orig_width + buffer))
    y2 = int(min(orig_height, y2 * orig_height + buffer))
    return x1, y1, x2, y2

def preprocess_nationality_emotion_image(image):
    try:
        return image.astype(np.float32) / 255.0
    except Exception as e:
        st.error(f"Error in nationality/emotion preprocessing: {e}")
        return None

def preprocess_clothing_image(image):
    try:
        return image.astype(np.float32) / 255.0
    except Exception as e:
        st.error(f"Error in clothing preprocessing: {e}")
        return None

def preprocess_age_image(image):
    try:
        return image.astype(np.float32) / 255.0
    except Exception as e:
        st.error(f"Error in age preprocessing: {e}")
        return None

def predict_nationality(model, image_features):
    try:
        nationality_classes = {0: "USA", 1: "Indian", 2: "African", 3: "Other"}
        prediction = model.predict(image_features[np.newaxis, ...], verbose=0)
        nationality_id = np.argmax(prediction, axis=1)[0]
        return nationality_classes.get(nationality_id, "Unknown")
    except Exception as e:
        st.error(f"Error predicting nationality: {e}")
        return None

def predict_emotion(model, image_features):
    try:
        emotion_classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        prediction = model.predict(image_features[np.newaxis, ...], verbose=0)
        if prediction.shape[1] != len(emotion_classes):
            st.error(f"Unexpected number of emotion classes: expected {len(emotion_classes)}, got {prediction.shape[1]}")
            return None
        emotion_id = np.argmax(prediction, axis=1)[0]
        if emotion_id >= len(emotion_classes):
            st.error(f"Invalid emotion index: {emotion_id}")
            return "Unknown"
        return emotion_classes[emotion_id]
    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return None

def closest_color_name(rgb):
    input_rgb = np.array(rgb)
    min_dist = float('inf')
    closest_name = None
    for name, hex_code in mcolors.XKCD_COLORS.items():
        rgb_array = np.array(mcolors.hex2color(hex_code)) * 255
        dist = np.sum((input_rgb - rgb_array) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_name = name.replace("xkcd:", "")

    return closest_name

def predict_clothing_color(image_features):
    try:
        if image_features.size == 0:
            return "Unknown (no clothing region detected)"
        clothing_pixels_int = (image_features * 255).round().astype(int)
        pixels, counts = np.unique(clothing_pixels_int, axis=0, return_counts=True)
        most_frequent_rgb = pixels[np.argmax(counts)].tolist()
        color_name = closest_color_name(most_frequent_rgb)
        return color_name.title()
    except Exception as e:
        st.error(f"Error predicting clothing color: {e}")
        return None

def predict_age(model, image_features, age_max=116):
    try:
        prediction = model.predict(image_features[np.newaxis, ...], verbose=0)[0][0]
        return round(prediction * age_max)
    except Exception as e:
        st.error(f"Error predicting age: {e}")
        return None

def main():
    st.markdown('<div class="main-header">Person Attribute Detector</div>', unsafe_allow_html=True)
    with st.spinner("Loading models..."):
        face_detector, nationality_model, emotion_model, clothing_model, age_model = load_models()
    if None in (face_detector, nationality_model, emotion_model, clothing_model, age_model):
        st.warning("Ensure model files exist at specified paths (Task5/).")
        return
    st.markdown('<div class="sub-header">Upload Image Files</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose image files (.jpg, .png)...", type=["jpg", "png"], accept_multiple_files=True)
    if st.button("Detect Attributes"):
        if uploaded_files:
            with st.spinner("Analyzing images..."):
                for i, uploaded_file in enumerate(uploaded_files):
                    st.markdown(f'<div class="image-container"><h3>Image {i+1}</h3>', unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 1])
                    image = Image.open(uploaded_file)
                    col1.image(image, caption=f"Image {i+1}: {uploaded_file.name}", use_container_width=True)

                    #Detecting and cropping face
                    face_128_rgb, face_224_rgb, face_128_gray = detect_and_crop_face(image, face_detector)
                    if any(f is None for f in [face_128_rgb, face_224_rgb, face_128_gray]):
                        col2.error("Failed to detect or crop face.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue

                    #Detecting and cropping clothing
                    clothing_crop = detect_and_crop_clothing(image, clothing_model)
                    if clothing_crop.size == 0:
                        st.warning("Failed to detect clothing region; using face region instead.")
                        clothing_crop = face_224_rgb

                    #Preprocess for different models
                    nationality_emotion_features = preprocess_nationality_emotion_image(face_128_rgb)
                    clothing_features = preprocess_clothing_image(clothing_crop)
                    age_features = preprocess_age_image(face_128_gray)
                    
                    if any(f is None for f in [nationality_emotion_features, clothing_features, age_features]):
                        col2.error("Failed to preprocess features.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue

                    #Predicting nationality
                    nationality = predict_nationality(nationality_model, nationality_emotion_features)
                    if nationality is None:
                        col2.error("Failed to predict nationality.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue

                    #Predicting emotion
                    emotion = predict_emotion(emotion_model, nationality_emotion_features)
                    if emotion is None:
                        col2.error("Failed to predict emotion.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue

                    col2.markdown('<div class="sub-header">Results:</div>', unsafe_allow_html=True)
                    col2.markdown(f'<div class="result-text">Nationality: {nationality}</div>', unsafe_allow_html=True)
                    col2.markdown(f'<div class="result-text">Emotion: {emotion}</div>', unsafe_allow_html=True)

                    #Conditional predictions based on nationalities
                    if nationality == "African":
                        clothing_color = predict_clothing_color(clothing_features)
                        if clothing_color is None:
                            col2.error("Failed to predict clothing color.")
                        else:
                            col2.markdown(f'<div class="result-text">Dress Color: {clothing_color}</div>', unsafe_allow_html=True)
                    elif nationality == "Indian":
                        age = predict_age(age_model, age_features)
                        clothing_color = predict_clothing_color(clothing_features)
                        if age is None or clothing_color is None:
                            col2.error("Failed to predict age or clothing color.")
                        else:
                            col2.markdown(f'<div class="result-text">Age: {age} years</div>', unsafe_allow_html=True)
                            col2.markdown(f'<div class="result-text">Dress Color: {clothing_color}</div>', unsafe_allow_html=True)
                    elif nationality == "USA":
                        age = predict_age(age_model, age_features)
                        if age is None:
                            col2.error("Failed to predict age.")
                        else:
                            col2.markdown(f'<div class="result-text">Age: {age} years</div>', unsafe_allow_html=True)
                    else:
                        pass

                    st.markdown("</div>", unsafe_allow_html=True)
                    if i < len(uploaded_files) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.info("Upload image files first.")

if __name__ == "__main__":
    main()