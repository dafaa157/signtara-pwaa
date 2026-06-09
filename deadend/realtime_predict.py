import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
from preprocess2 import extract_frame_features 

# --- 1. Fungsi Bantuan untuk Menyamakan Frame ---
def process_recorded_sequence(sequence, max_frames=20):
    """
    Fungsi ini melakukan downsampling atau padding agar 
    rekaman gerakan yang panjang/pendeknya bervariasi
    menjadi pas 20 frame untuk masuk ke model BiLSTM.
    """
    n = len(sequence)
    if n >= max_frames:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        final_seq = [sequence[i] for i in indices]
    else:
        final_seq = list(sequence)
        while len(final_seq) < max_frames:
            final_seq.append(sequence[-1]) # Padding duplikasi frame terakhir
    return np.array(final_seq)


# --- 2. Load Model & Konfigurasi ---
model = tf.keras.models.load_model('model_signtara_bilstm.keras')
with open('output/label_map.json', 'r') as f:
    label_map = json.load(f)

MAX_FRAMES = 20
THRESHOLD = 0.98
MOTION_THRESHOLD = 0.015  # Sesuaikan jika kurang/terlalu sensitif

# Variabel State Machine
recent_features = []       # Buffer kecil untuk menghitung kecepatan saat ini
recorded_sequence = []     # Tempat menyimpan frame selama tangan bergerak
is_signing = False         # Status apakah user sedang bergerak
last_prediction = "-"      # Hasil terjemahan terakhir

mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

print("[INFO] Kamera menyala. Silakan mulai berisyarat.")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ekstraksi 156 fitur untuk frame saat ini
        feat = extract_frame_features(results)
        recent_features.append(feat)

        # Pertahankan buffer recent_features hanya sepanjang 5 frame 
        # untuk menghitung kecepatan instan secara responsif
        if len(recent_features) > 5:
            recent_features.pop(0)

        mean_velocity = 0.0
        status_text = "Diam"
        box_color = (150, 150, 150) # Abu-abu untuk status Diam

        if len(recent_features) == 5:
            # Hitung kecepatan tangan dari 5 frame terakhir
            hand_features = np.array(recent_features)[:, 30:156]
            velocities = np.linalg.norm(np.diff(hand_features, axis=0), axis=1)
            mean_velocity = np.mean(velocities)

            # Logika State Machine
            if mean_velocity > MOTION_THRESHOLD:
                # [STATE 1] TANGAN SEDANG BERGERAK
                status_text = "Merekam..."
                box_color = (0, 165, 255) # Oranye untuk status Merekam

                if not is_signing:
                    # Tangan baru saja mulai bergerak dari posisi diam
                    is_signing = True
                    # Masukkan buffer awal agar awal gerakan tidak terpotong
                    recorded_sequence = list(recent_features) 
                else:
                    # Tangan masih terus bergerak
                    recorded_sequence.append(feat)
                    
                    # Batasan pengaman agar memori tidak bocor jika ada noise terus-menerus
                    if len(recorded_sequence) > 150: 
                        recorded_sequence.pop(0)

            else:
                # [STATE 2] TANGAN DIAM
                if is_signing:
                    # Tangan baru saja berhenti. Artinya 1 isyarat telah selesai!
                    is_signing = False
                    
                    # Pastikan gerakannya bukan sekadar getaran singkat (minimal 10 frame / ~0.3 detik)
                    if len(recorded_sequence) >= 10:
                        # 1. Sesuaikan menjadi 20 frame
                        processed_seq = process_recorded_sequence(recorded_sequence, MAX_FRAMES)
                        
                        # 2. Prediksi
                        input_data = np.expand_dims(processed_seq, axis=0)
                        res = model.predict(input_data, verbose=0)[0]
                        
                        predicted_idx = np.argmax(res)
                        confidence = res[predicted_idx]

                        if confidence > THRESHOLD:
                            last_prediction = f"{label_map[str(predicted_idx)]} ({confidence*100:.0f}%)"
                        else:
                            # Jika akurasi rendah, jangan paksa tampilkan kata yang salah
                            last_prediction = "(Gerakan tidak dikenali)"
                    
                    # Kosongkan wadah rekaman untuk kata selanjutnya
                    recorded_sequence = []

        # --- Tampilan Visual ---
        # 1. Kotak Status Gerakan (Pojok Kiri Atas)
        cv2.rectangle(image, (0, 0), (200, 40), box_color, -1)
        cv2.putText(image, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 2. Kotak Hasil Terjemahan (Bagian Bawah)
        h, w, _ = image.shape
        cv2.rectangle(image, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.putText(image, f"Terjemahan: {last_prediction}", (20, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 3. Indikator Kecepatan (Kecil di bawah kotak status)
        cv2.putText(image, f"V: {mean_velocity:.4f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('Signtara AI - Action Spotting', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()