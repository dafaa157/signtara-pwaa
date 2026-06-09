import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# 1. Load Data
print("[INFO] Memuat dataset...")
output_dir = "output"
X_train = np.load(os.path.join(output_dir, "X_train.npy"))
y_train = np.load(os.path.join(output_dir, "y_train.npy"))
X_val = np.load(os.path.join(output_dir, "X_val.npy"))
y_val = np.load(os.path.join(output_dir, "y_val.npy"))
X_test = np.load(os.path.join(output_dir, "X_test.npy"))
y_test = np.load(os.path.join(output_dir, "y_test.npy"))

with open(os.path.join(output_dir, "dataset_info.json"), "r") as f:
    dataset_info = json.load(f)

max_frames = dataset_info["max_frames"]
feature_dim = dataset_info["feature_dim"]
n_classes = dataset_info["n_classes"]

print(f"Data Train siap! Shape: {X_train.shape}")

# 2. Arsitektur Model (PERBAIKAN: Dropout lebih kecil agar model lebih teliti)
model = Sequential([
    # Layer 1
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(max_frames, feature_dim)),
    Dropout(0.2), # Diturunkan agar memori tentang detail jari tidak banyak terbuang
    
    # Layer 2
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.2), # Diturunkan
    
    # Dense Layer
    Dense(32, activation='relu'),
    Dropout(0.2), # Diturunkan
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# 3. Setup Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'model_signtara_bilstm.keras', # Ini akan otomatis menimpa model yang lama
        monitor='val_loss', 
        save_best_only=True,
        verbose=1
    )
]

# 4. Mulai Training
print("\n[INFO] Memulai proses training ulang...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=16,
    callbacks=callbacks
)

# 5. Evaluasi pada Data Test
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n=========================================")
print(f" HASIL AKURASI TEST FINAL: {test_acc * 100:.2f}%")
print(f"=========================================\n")

# 6. Simpan Grafik Visualisasi ke PNG
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Akurasi Model')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Akurasi')
ax1.legend(loc='lower right')

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Loss Model')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')

grafik_path = os.path.join(output_dir, "grafik_training_revisi.png")
plt.savefig(grafik_path)
plt.close()
print(f"[INFO] Grafik training berhasil disimpan di: {grafik_path}")