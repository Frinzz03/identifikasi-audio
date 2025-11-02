import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import joblib
import io
import tempfile

st.title("🎙️ Identifikasi Suara (Buka / Tutup) Menggunakan Fitur Statistik Time Series")
st.write("Kamu bisa **merekam suara langsung** atau **upload file .wav** untuk dikenali oleh model.")

# ====== 1. Load Model ======
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ====== 2. Ekstraksi Fitur Statistik Time Series ======
def extract_time_series_features_from_array(y, sr):
    # pastikan mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Normalisasi sinyal
    y = (y - np.mean(y)) / np.std(y)

    # ==== Fitur Time Series ====
    mean = np.mean(y)
    std = np.std(y)
    max_val = np.max(y)
    min_val = np.min(y)
    median = np.median(y)
    skewness = np.mean((y - mean)**3) / (std**3)
    kurtosis = np.mean((y - mean)**4) / (std**4)
    rms = np.sqrt(np.mean(y**2))
    zero_crossing_rate = np.mean(librosa.zero_crossings(y))

    # ==== Fitur Spektral ====
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

# ====== 3. Pilihan Input ======
option = st.radio("Pilih sumber suara:", ["🎤 Rekam Langsung", "📂 Upload File .wav"])

# --- Jika pilih rekam ---
if option == "🎤 Rekam Langsung":
    duration = st.slider("Durasi rekaman (detik)", 1, 5, 2)
    sr = 22050

    if st.button("Mulai Rekam"):
        st.info("🎙️ Rekaman dimulai... silakan ucapkan 'buka' atau 'tutup'.")
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        st.success("✅ Rekaman selesai!")

        # Simpan sementara file audio
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmpfile.name, recording, sr)
        st.audio(tmpfile.name, format="audio/wav")

        # Ekstraksi fitur dan prediksi
        try:
            features = extract_time_series_features_from_array(recording.flatten(), sr)
            prediction = model.predict(features)[0]
            st.success(f"🎧 Hasil Prediksi: **{prediction.upper()}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses rekaman: {e}")

# --- Jika pilih upload file ---
elif option == "📂 Upload File .wav":
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
