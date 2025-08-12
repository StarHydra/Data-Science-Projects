import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Gender of Person in Image Detector", page_icon="Task4\hair.png", layout="centered")

st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] {
    background: radial-gradient(#EDEFA9,#10911A);}
    .main-header { font-size: 4.5rem; font-weight: 800; color: #540f60; text-align: center; margin-bottom: 5rem; }
    .sub-header {font-size: 1.89rem; font-weight: 500; color: #540f60; margin: 2.5rem 0 2.5rem;text-align:center; }
    .result-text {font-size: 1.6rem; font-weight: bolder; border-radius: 1rem; margin-bottom: 1rem;text-align:center; }
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
        gender_model = load_model("Task4/gendertask4.h5")
        age_model = load_model("Task4/agetask4.h5")
        hairlength_model = load_model("Task4/hairlengthtask4.h5")
        return gender_model, age_model, hairlength_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def preprocess_gender_age_image(image):
    try:
        image = image.resize((128, 128)).convert("L") 
        image_array = np.array(image) / 255.0  
        image_array = image_array[..., np.newaxis]  
        return image_array.astype(np.float32)
    except Exception as e:
        st.error(f"Error in gender/age preprocessing: {e}")
        return None

def preprocess_hairlength_image(image):
    try:
        image = image.resize((224, 224))
        image = image.convert("RGB")  
        image_array = np.array(image) / 255.0  
        image_array = image_array.astype(np.float32) 
        return image_array
    except Exception as e:
        st.error(f"Error in hair length preprocessing: {e}")
        return None

def predict_gender(model, image_features):
    try:
        prediction = model.predict(image_features[np.newaxis, ...], verbose=0)[0][0]
        gender = "Female" if prediction > 0.5 else "Male"
        return gender
    except Exception as e:
        st.error(f"Error predicting gender: {e}")
        return None

def predict_age(model, image_features, age_max=116):
    try:
        prediction = model.predict(image_features[np.newaxis, ...], verbose=0)[0][0]
        return round(prediction * age_max)  
    except Exception as e:
        st.error(f"Error predicting age: {e}")
        return None

def predict_hairlength(model, image_features):
    try:
        prediction = model.predict(image_features[np.newaxis, ...], verbose=0)[0][0]
        hairlength = "Long" if prediction > 0.5 else "Short"
        return hairlength
    except Exception as e:
        st.error(f"Error predicting hair length: {e}")
        return None

def determine_gender(hairlength=None, gender_model = None, gender_features=None):
    if hairlength is not None:
        return "Female" if hairlength == "Long" else "Male"
    return predict_gender(gender_model, gender_features)

def main():
    st.markdown('<div class="main-header">Image Gender Detector</div>', unsafe_allow_html=True)
    with st.spinner("Loading models..."):
        gender_model, age_model, hairlength_model = load_models()
    if None in (gender_model, age_model, hairlength_model):
        st.warning("Ensure model files exist at specified paths.")
        return
    st.markdown('<div class="sub-header">Upload Image Files</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose passport image files (.jpg, .png)...", type=["jpg", "png"], accept_multiple_files=True)
    if st.button("Detect Gender"):
        if uploaded_files:
            with st.spinner("Analyzing images..."):
                for i, uploaded_file in enumerate(uploaded_files):
                    st.markdown(f'<div class="image-container"><h3>Image {i+1}</h3>', unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 1])
                    image = Image.open(uploaded_file)
                    col1.image(image, caption=f"Image {i+1}: {uploaded_file.name}", use_container_width=True)
                    features = preprocess_gender_age_image(image)
                    if features is None:
                        col2.error("Failed to preprocess features.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue
                    age = predict_age(age_model, features)
                    if age is None:
                        col2.error("Failed to predict age.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue

                    col2.markdown('<div class="sub-header">Results:</div>', unsafe_allow_html=True)

                    if 20 <= age <= 30:
                        hairlength_features = preprocess_hairlength_image(image)
                        if hairlength_features is None:
                            col2.error("Failed to preprocess hair length features.")
                            st.markdown("</div>", unsafe_allow_html=True)
                            if i < len(uploaded_files) - 1:
                                st.markdown("<hr>", unsafe_allow_html=True)
                            continue
                        hairlength = predict_hairlength(hairlength_model, hairlength_features)
                        if hairlength is None:
                            col2.error("Failed to predict hair length.")
                            st.markdown("</div>", unsafe_allow_html=True)
                            if i < len(uploaded_files) - 1:
                                st.markdown("<hr>", unsafe_allow_html=True)
                            continue
                        gender = determine_gender(hairlength=hairlength)
                    else:
                        gender = determine_gender(gender_model=gender_model, gender_features=features)
                        if gender is None:
                            col2.error("Failed to predict gender.")
                            st.markdown("</div>", unsafe_allow_html=True)
                            if i < len(uploaded_files) - 1:
                                st.markdown("<hr>", unsafe_allow_html=True)
                            continue

                    col2.markdown(f'<div class="result-text">Gender: {gender}</div>', unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
                    if i < len(uploaded_files) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.info("Upload image files first.")

if __name__ == "__main__":
    main()