# train_model.py
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

# Contoh data fitur (13 fitur time series, seperti yang digunakan di app.py)
# misal 10 contoh suara 'buka' dan 10 contoh suara 'tutup'
X = np.random.rand(20, 13)
y = ['buka'] * 10 + ['tutup'] * 10

# Latih model sederhana
model = RandomForestClassifier()
model.fit(X, y)

# Simpan ke file
joblib.dump(model, "model.pkl")
print("✅ model.pkl berhasil dibuat!")
