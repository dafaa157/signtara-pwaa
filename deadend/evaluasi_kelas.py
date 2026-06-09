import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load Data Test dan Model
print("Memuat data dan model...")
output_dir = "output"
X_test = np.load(os.path.join(output_dir, "X_test.npy"))
y_test = np.load(os.path.join(output_dir, "y_test.npy"))

model = tf.keras.models.load_model('model_signtara_bilstm.keras')

with open(os.path.join(output_dir, "label_map.json"), "r") as f:
    label_map = json.load(f)

# Ubah format label_map dari dictionary menjadi list nama kelas secara berurutan
# Misal: ["Belajar", "Maaf", "Makan", "Motor", "Saya"]
class_names = [label_map[str(i)] for i in range(len(label_map))]

# 2. Lakukan Prediksi
print("Melakukan prediksi pada data pengujian...")
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1) # Ambil indeks dengan probabilitas tertinggi

# 3. Print Classification Report di Terminal
print("\n=== LAPORAN DETAIL PER KELAS ===")
print(classification_report(y_test, y_pred, target_names=class_names))

# 4. Gambar dan Simpan Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - Deteksi Kelas")
plt.tight_layout()

# Simpan gambar
grafik_path = "confusion_matrix.png"
plt.savefig(grafik_path)
print(f"\n[INFO] Gambar Confusion Matrix berhasil disimpan di: {grafik_path}")