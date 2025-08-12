import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, Layer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
import librosa
import os
import pickle

class MultiHeadAttentionWrapper(Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(MultiHeadAttentionWrapper, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        return self.attention(inputs, inputs) 

    def compute_output_shape(self, input_shape):
        return input_shape

st.set_page_config(page_title="Gender, Emotion & Age Audio Detector", page_icon="Task3\microphone.png", layout="centered")

st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] {
    background: radial-gradient(#EDEFA9,#10911A);}
    .main-header { font-size: 4.5rem; font-weight: 800; color: #540f60; text-align: center; margin-bottom: 5rem; }
    .sub-header {font-size: 1.89rem; font-weight: 500; color: #540f60; margin: 2.5rem 0 2.5rem;text-align:center; }
    .result-text {font-size: 1.6rem; font-weight: bolder; border-radius: 1rem; margin-bottom: 1rem;text-align:center; }
    .audio-container { margin-bottom: 1rem; border-radius: 1rem; text-align: center; background-color: #44b4c5  ; }
    .stButton>button { background-color: #267384; font-color:#D890f4 ; font-weight: 700; border-radius: 0.5rem; padding: 0.5rem 1rem; }
    .stButton>button:hover { background-color: #;}
    .error-text { font-size: 1.3rem; font-weight: 200; color: #540f60; border-radius: 1rem; margin-bottom: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_models_and_scaler():
    try:
        custom_objects = {
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'binary_crossentropy': BinaryCrossentropy(),
            'categorical_crossentropy': CategoricalCrossentropy(),
            'MultiHeadAttentionWrapper': MultiHeadAttentionWrapper
        }
        gender_model = load_model(r"C:\Users\ROY\Documents\python_proj\Task3\Task3\gendertask3.h5", custom_objects=custom_objects)
        emotion_model = load_model(r"C:\Users\ROY\Documents\python_proj\Task3\Task3\emotiontask3.h5", custom_objects=custom_objects)
        age_model = load_model(r"C:\Users\ROY\Documents\python_proj\Task3\Task3\agetask3.h5", custom_objects=custom_objects)
        with open(r"C:\Users\ROY\Documents\python_proj\Task3\Task3\emotion_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return gender_model, emotion_model, age_model, scaler
    except Exception as e:
        st.error(f"Error loading models or scaler: {e}")
        return None, None, None, None

def augment_audio(audio, sr=16000, is_young=False):
    try:
        noise = np.random.randn(len(audio))
        audio = audio + 0.005 * noise
        pitch_shift = np.random.uniform(-4, 4) if is_young else np.random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
        audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
        gain_factor = np.random.uniform(0.7, 1.3)
        audio = audio * gain_factor
        return audio
    except Exception as e:
        st.error(f"Error in audio augmentation: {e}")
        return None

def preprocess_gender_audio(file_path, max_length=4.0, sr=16000):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        target_samples = int(max_length * sr)
        y = np.pad(y, (0, target_samples - len(y)), mode='constant') if len(y) < target_samples else y[:target_samples]
    
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, hop_length=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 501 - mel_spec_db.shape[1])), mode='constant') if mel_spec_db.shape[1] < 501 else mel_spec_db[:, :501]
        mel_spec_db = mel_spec_db[..., np.newaxis]  

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = pitches[magnitudes > 0].mean() if magnitudes.sum() > 0 else 0
        pitch_std = pitches[magnitudes > 0].std() if magnitudes.sum() > 0 else 0
        pitch = (pitch - 50) / (500 - 50)
        pitch_std = (pitch_std - 0) / (500 - 0)
        
        rmse = librosa.feature.rms(y=y)[0]
        rmse_mean = np.mean(rmse)
        rmse_std = np.std(rmse)
        rmse_mean = (rmse_mean - 0) / (1 - 0)
        rmse_std = (rmse_std - 0) / (1 - 0)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs))
        mfcc_means = np.mean(mfccs, axis=1)
        
        y_preemph = librosa.effects.preemphasis(y)
        lpc_order = 4 + int(sr / 1000)
        lpc_coeffs = librosa.lpc(y_preemph, order=lpc_order)
        roots = np.roots(lpc_coeffs)
        roots = roots[np.imag(roots) >= 0]
        formant_freqs = np.angle(roots) * (sr / (2 * np.pi))
        formant_freqs = formant_freqs[formant_freqs > 0][:2] if len(formant_freqs) >= 2 else np.array([0, 0])
        formant_freqs = (formant_freqs - 200) / (2000 - 200)
        
        extra_features = np.array([pitch, pitch_std, rmse_mean, rmse_std] + formant_freqs.tolist() + mfcc_means.tolist())
        extra_features = extra_features[:, np.newaxis, np.newaxis] 
        extra_features = np.repeat(extra_features, 501, axis=1)  
        
        combined_features = np.concatenate([mel_spec_db, extra_features], axis=0)  # Shape: (275, 501, 1)
        return combined_features.astype(np.float32)
    except Exception as e:
        st.error(f"Error in gender preprocessing: {e}")
        return None

def preprocess_emotion_audio(file_path, scaler, n_mfcc=13, max_length=100, sr=22050):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.full((1, mfcc.shape[1]), pitches[magnitudes > 0].mean() if magnitudes.sum() > 0 else 0)
        energy = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4)
        
        features = [mfcc, delta_mfcc, energy, zcr, pitch, spectral_contrast]
        for i, feature in enumerate(features):
            if feature.shape[1] < max_length:
                features[i] = np.pad(feature, ((0, 0), (0, max_length - feature.shape[1])), mode='constant')
            else:
                features[i] = feature[:, :max_length]
        
        stacked_features = np.vstack(features) 
        
        stacked_features_reshaped = stacked_features.reshape(-1, stacked_features.shape[0] * stacked_features.shape[1])
        stacked_features_normalized = scaler.transform(stacked_features_reshaped)
        stacked_features = stacked_features_normalized.reshape(stacked_features.shape)
        
        return stacked_features[..., np.newaxis].astype(np.float32) 
    except Exception as e:
        st.error(f"Error in emotion preprocessing: {e}")
        return None

def preprocess_age_audio(file_path, target_length=4.0, sr=16000, n_mfcc=40, hop_length=128):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        # Apply augmentation (assume not young for inference, adjust if age known)
        audio = augment_audio(audio, sr=sr, is_young=False)
        if audio is None:
            return None
        
        target_samples = int(target_length * sr)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        else:
            audio = audio[:target_samples]
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        pitch, _ = librosa.piptrack(y=audio, sr=sr, hop_length=hop_length)
        pitch = pitch[0, :].reshape(-1, 1)
        
        features = np.concatenate([mfcc.T, delta.T, delta2.T, pitch], axis=1)
        
        if np.random.random() < 0.5:
            time_mask = np.random.randint(0, 20)
            freq_mask = np.random.randint(0, 5)
            features = np.copy(features)
            for _ in range(2):
                t_start = np.random.randint(0, features.shape[0] - time_mask)
                f_start = np.random.randint(0, features.shape[1] - freq_mask)
                features[t_start:t_start+time_mask, :] = 0
                features[:, f_start:f_start+freq_mask] = 0
        
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        return features[..., np.newaxis].astype(np.float32)
    except Exception as e:
        st.error(f"Error in age preprocessing: {e}")
        return None

def predict_gender(model, audio_features):
    try:
        prediction = model.predict(audio_features[np.newaxis, ...], verbose=0)[0][0]
        gender = "Female" if prediction > 0.5 else "Male"
        return gender
    except:
        return None, None

def predict_emotion(model, audio_features):
    try:
        emotion_dict = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sadness', 5: 'Neutral'}
        prediction = model.predict(audio_features[np.newaxis, ...], verbose=0)[0]
        emotion_idx = np.argmax(prediction)
        return emotion_dict[emotion_idx]
    except:
        return None, None

def predict_age(model, audio_features, age_min=18, age_max=80):
    try:
        prediction = model.predict(audio_features[np.newaxis, ...], verbose=0)[0][0]
        return round(prediction * (age_max - age_min) + age_min)
    except:
        return None

def main():
    st.markdown('<div class="main-header">Emotion & Age Audio Detector for Male</div>', unsafe_allow_html=True)
    with st.spinner("Loading models and scaler..."):
        gender_model, emotion_model, age_model, scaler = load_models_and_scaler()
    if None in (gender_model, emotion_model, age_model, scaler):
        st.warning("Ensure model and scaler files exist at specified paths.")
        return
    st.markdown('<div class="sub-header">Upload Audio Files</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose .wav files...", type=["wav"], accept_multiple_files=True)
    if st.button("Detect Age & Emotion"):
        if uploaded_files:
            with st.spinner("Analyzing audio..."):
                for i, uploaded_file in enumerate(uploaded_files):
                    st.markdown(f'<div class="audio-container"><h3>Audio {i+1}</h3>', unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 1])
                    col1.audio(uploaded_file, format="audio/wav")
                    col1.caption(f"Audio {i+1}: {uploaded_file.name}")
                    temp_file_path = f"temp_audio_{i}.wav"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    gender_features = preprocess_gender_audio(temp_file_path)
                    if gender_features is None:
                        col2.error("Failed to preprocess gender features.")
                        os.remove(temp_file_path)
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue
                    gender = predict_gender(gender_model, gender_features)
                    if gender is None:
                        col2.error("Failed to predict gender.")
                        os.remove(temp_file_path)
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue
                    
                    col2.markdown('<div class="sub-header">Results:</div>', unsafe_allow_html=True)
                    if gender == "Female":
                        col2.markdown(
                            '<div class="error-text">The audio is predicted as female. Please upload a male audio .wav file.</div>',
                            unsafe_allow_html=True,
                        )
                        os.remove(temp_file_path)
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue
                    
                    age_features = preprocess_age_audio(temp_file_path)
                    if age_features is None:
                        col2.error("Failed to preprocess age features.")
                        os.remove(temp_file_path)
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue
                    age = predict_age(age_model, age_features)
                    if age is None:
                        col2.error("Failed to predict age.")
                        os.remove(temp_file_path)
                        st.markdown("</div>", unsafe_allow_html=True)
                        if i < len(uploaded_files) - 1:
                            st.markdown("<hr>", unsafe_allow_html=True)
                        continue
                    
                    col2.markdown(
                        f'<div class="result-text">Age: {age}</div>',
                        unsafe_allow_html=True,
                    )
                    
                    if age > 60:
                        emotion_features = preprocess_emotion_audio(temp_file_path, scaler)
                        if emotion_features is None:
                            col2.error("Failed to preprocess emotion features.")
                            os.remove(temp_file_path)
                            st.markdown("</div>", unsafe_allow_html=True)
                            if i < len(uploaded_files) - 1:
                                st.markdown("<hr>", unsafe_allow_html=True)
                            continue
                        emotion = predict_emotion(emotion_model, emotion_features)
                        if emotion is None:
                            col2.error("Failed to predict emotion.")
                            os.remove(temp_file_path)
                            st.markdown("</div>", unsafe_allow_html=True)
                            if i < len(uploaded_files) - 1:
                                st.markdown("<hr>", unsafe_allow_html=True)
                            continue
                        col2.markdown(
                            f'<div class="result-text">Emotion: {emotion}</div>',
                            unsafe_allow_html=True,
                        )
                    
                    os.remove(temp_file_path)
                    st.markdown("</div>", unsafe_allow_html=True)
                    if i < len(uploaded_files) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.info("Upload audio files first.")

if __name__ == "__main__":
    main()