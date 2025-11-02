import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
import io

st.title("🎙️ Identifikasi Suara (Buka / Tutup) Menggunakan Fitur Statistik Time Series")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

def extract_time_series_features_from_array(y, sr):
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = (y - np.mean(y)) / np.std(y)

    mean = np.mean(y)
    std = np.std(y)
    max_val = np.max(y)
    min_val = np.min(y)
    median = np.median(y)
    skewness = np.mean((y - mean)**3) / (std**3)
    kurtosis = np.mean((y - mean)**4) / (std**4)
    rms = np.sqrt(np.mean(y**2))
    zero_crossing_rate = np.mean(librosa.zero_crossings(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    features = np.array([
        mean, std, max_val, min_val, median, skewness, kurtosis,
        rms, zero_crossing_rate,
        spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness
    ])
    return features.reshape(1, -1)

uploaded_file = st.file_uploader("Upload file suara (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.info("⏳ Sedang memproses fitur statistik time series...")
    try:
        y, sr = sf.read(io.BytesIO(uploaded_file.read()))
        features = extract_time_series_features_from_array(y, sr)
        prediction = model.predict(features)[0]
        st.success(f"🎧 Hasil Prediksi: **{prediction.upper()}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
