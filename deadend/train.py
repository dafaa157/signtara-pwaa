import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import json
import os

# 1. Load Data
# Pastikan path ini sesuai dengan output dari preprocess.py
output_dir = "output"
X_train = np.load(os.path.join(output_dir, "X_train.npy"))
y_train = np.load(os.path.join(output_dir, "y_train.npy"))
X_val = np.load(os.path.join(output_dir, "X_val.npy"))
y_val = np.load(os.path.join(output_dir, "y_val.npy"))

with open(os.path.join(output_dir, "dataset_info.json"), "r") as f:
    dataset_info = json.load(f)

max_frames = dataset_info["max_frames"]
feature_dim = dataset_info["feature_dim"]
n_classes = dataset_info["n_classes"]

print(f"Bentuk X_train: {X_train.shape}")
print(f"Bentuk y_train: {y_train.shape}")

# 2. Arsitektur Model BiLSTM Ringan
model = Sequential([
    # Layer BiLSTM Pertama: Membaca gerakan maju & mundur
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(max_frames, feature_dim)),
    Dropout(0.5),
    
    # Layer BiLSTM Kedua: Mengekstrak pola yang lebih dalam
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.5),
    
    # Layer Klasifikasi
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# prosess
# 3. Setup Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=15,          # Berhenti jika val_loss tidak turun selama 15 epoch
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'model_signtara_bilstm.keras', # Simpan model terbaik ke file ini
        monitor='val_loss', 
        save_best_only=True,
        verbose=1
    )
]

# 4. Mulai Training
print("\nMemulai proses training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,           # Set tinggi saja, akan distop otomatis oleh EarlyStopping
    batch_size=16,        # Batch size kecil lebih baik untuk data kecil
    callbacks=callbacks
)

# eval

# 5. Visualisasi Hasil Training
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot Accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')

# Plot Loss
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')

# PERUBAHAN DI SINI: Simpan sebagai file PNG, bukan plt.show()
grafik_path = os.path.join(output_dir, "grafik_training.png")
plt.savefig(grafik_path)
print(f"\n[INFO] Grafik training berhasil disimpan di: {grafik_path}")

# Bersihkan memori figure setelah disimpan
plt.close()

# 6. Test di Data Testing yang Belum Pernah Dilihat Model
X_test = np.load(os.path.join(output_dir, "X_test.npy"))
y_test = np.load(os.path.join(output_dir, "y_test.npy"))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAkurasi pada Data Test: {test_acc * 100:.2f}%")