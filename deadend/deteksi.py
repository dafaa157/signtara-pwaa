"""
landmark_detector.py
--------------------
Detektor landmark MediaPipe Holistic dengan seleksi fitur spesifik untuk BISINDO.

Fitur yang diekstrak (per frame):
  - Pose  : hidung (0), telinga kiri (7), telinga kanan (8),
            bahu kiri (11), bahu kanan (12)            → 5 titik × 3 = 15 nilai
  - Face  : dahi (10), mulut 4 titik (0,17,61,291)    → 5 titik × 3 = 15 nilai
  - Tangan kiri  : 21 titik                           → 21 titik × 3 = 63 nilai
  - Tangan kanan : 21 titik                           → 21 titik × 3 = 63 nilai

Total per frame: 15 + 15 + 63 + 63 = 156 nilai (float32)

Penggunaan:
  python landmark_detector.py                    → pakai kamera (default id=0)
  python landmark_detector.py --source video.mp4  → pakai file video
  python landmark_detector.py --source 1          → kamera id 1
  python landmark_detector.py --save              → simpan hasil ke file .npy
"""

import argparse
import sys
import cv2
import mediapipe as mp
import numpy as np

# ─── Konfigurasi Indeks Landmark ─────────────────────────────────────────────

POSE_INDICES  = [0, 7, 8, 11, 12]          # hidung, telinga, bahu
FACE_INDICES  = [10, 0, 17, 61, 291]       # dahi, mulut (atas,bawah,kiri,kanan)
HAND_COUNT    = 21                         # seluruh titik tangan

# Warna overlay (BGR)
COLOR_POSE   = (0, 255, 128)
COLOR_FACE   = (255, 200, 0)
COLOR_LHAND  = (255, 80, 80)
COLOR_RHAND  = (80, 80, 255)
COLOR_TEXT   = (255, 255, 255)
COLOR_BG     = (30, 30, 30)

FEATURE_DIM = len(POSE_INDICES)*3 + len(FACE_INDICES)*3 + HAND_COUNT*3*2
# 5*3 + 5*3 + 21*3 + 21*3 = 15+15+63+63 = 156


# ─── Fungsi Ekstraksi Fitur ───────────────────────────────────────────────────

def extract_features(results) -> np.ndarray:
    """
    Ekstrak dan flatten landmark terpilih dari hasil MediaPipe Holistic.
    Mengembalikan array float32 sepanjang FEATURE_DIM (156).
    Jika landmark tidak terdeteksi, titik tersebut diisi nol.
    """
    feat = []

    # --- Pose ---
    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark
        for idx in POSE_INDICES:
            lm = lms[idx]
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * (len(POSE_INDICES) * 3))

    # --- Face Mesh ---
    if results.face_landmarks:
        lms = results.face_landmarks.landmark
        for idx in FACE_INDICES:
            lm = lms[idx]
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * (len(FACE_INDICES) * 3))

    # --- Tangan Kiri ---
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * (HAND_COUNT * 3))

    # --- Tangan Kanan ---
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * (HAND_COUNT * 3))

    return np.array(feat, dtype=np.float32)


def hands_detected(results) -> bool:
    return (results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None)


# ─── Fungsi Gambar Overlay ────────────────────────────────────────────────────

def draw_selected_landmarks(frame, results):
    h, w = frame.shape[:2]

    def px(lm):
        return int(lm.x * w), int(lm.y * h)

    # Pose
    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark
        for idx in POSE_INDICES:
            cv2.circle(frame, px(lms[idx]), 6, COLOR_POSE, -1)
            cv2.circle(frame, px(lms[idx]), 7, (0, 0, 0), 1)

    # Face
    if results.face_landmarks:
        lms = results.face_landmarks.landmark
        for idx in FACE_INDICES:
            cv2.circle(frame, px(lms[idx]), 5, COLOR_FACE, -1)
            cv2.circle(frame, px(lms[idx]), 6, (0, 0, 0), 1)

    # Tangan Kiri
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            cv2.circle(frame, px(lm), 4, COLOR_LHAND, -1)

    # Tangan Kanan
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            cv2.circle(frame, px(lm), 4, COLOR_RHAND, -1)


def draw_info_panel(frame, frame_idx, feat, recording, saved_count):
    h, w = frame.shape[:2]

    # Panel latar
    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 90), (50, 50, 50), 1)

    # Baris 1: info umum
    nonzero = int(np.count_nonzero(feat) / 3)
    cv2.putText(frame,
        f"Frame: {frame_idx:04d}   Aktif: {nonzero}/{FEATURE_DIM//3} titik   Dim: {FEATURE_DIM}",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    # Baris 2: legenda warna
    cv2.circle(frame, (14, 45), 5, COLOR_POSE, -1)
    cv2.putText(frame, "Pose", (22, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_POSE, 1, cv2.LINE_AA)
    cv2.circle(frame, (74, 45), 5, COLOR_FACE, -1)
    cv2.putText(frame, "Face", (82, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_FACE, 1, cv2.LINE_AA)
    cv2.circle(frame, (134, 45), 5, COLOR_LHAND, -1)
    cv2.putText(frame, "L.Hand", (142, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_LHAND, 1, cv2.LINE_AA)
    cv2.circle(frame, (214, 45), 5, COLOR_RHAND, -1)
    cv2.putText(frame, "R.Hand", (222, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_RHAND, 1, cv2.LINE_AA)

    # Baris 3: kontrol
    rec_color = (0, 60, 200) if recording else (80, 80, 80)
    rec_label = f"[R] REC ({saved_count} frame tersimpan)" if recording else "[R] Mulai Rekam"
    cv2.putText(frame, rec_label, (10, 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, rec_color if not recording else (60, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "  [S] Simpan .npy   [Q] Keluar",
        (250, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run(source, auto_save: bool, output_path: str):
    # Buka sumber video
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        window_title = f"Landmark Detector — Kamera {source}"
    else:
        cap = cv2.VideoCapture(source)
        window_title = f"Landmark Detector — {source}"

    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka sumber: {source}")
        sys.exit(1)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_features = []      # buffer rekaman
    recording   = False
    frame_idx   = 0

    print(f"\n{'─'*52}")
    print(f"  SignTara — Landmark Detector")
    print(f"  Fitur per frame : {FEATURE_DIM} nilai (float32)")
    print(f"  Sumber          : {source}")
    print(f"{'─'*52}")
    print("  Kontrol:")
    print("    [R]   — Mulai/Stop rekam")
    print("    [S]   — Simpan buffer ke .npy")
    print("    [Q]   — Keluar")
    print(f"{'─'*52}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream selesai.")
            break

        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        feat = extract_features(results)
        draw_selected_landmarks(frame, results)
        draw_info_panel(frame, frame_idx, feat, recording, len(all_features))

        if recording:
            all_features.append(feat)
            # Indikator merah rekaman
            cv2.circle(frame, (frame.shape[1] - 20, 20), 7, (0, 0, 255), -1)

        cv2.imshow(window_title, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord('r'):
            recording = not recording
            state = "MULAI" if recording else "STOP"
            print(f"[REC] {state} — {len(all_features)} frame tersimpan di buffer")

        elif key == ord('s'):
            if all_features:
                save_features(all_features, output_path)
                all_features.clear()
            else:
                print("[WARN] Buffer kosong, tidak ada yang disimpan.")

    # Auto-save saat keluar jika flag aktif
    if auto_save and all_features:
        save_features(all_features, output_path)

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Selesai. Total frame diproses: {frame_idx}")


def save_features(buffer: list, path: str):
    arr = np.array(buffer, dtype=np.float32)  # shape: (N, 156)
    np.save(path, arr)
    print(f"[SAVE] Tersimpan → {path}  shape={arr.shape}  dtype={arr.dtype}")
    return arr


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detektor landmark BISINDO — MediaPipe Holistic (fitur terpilih)")
    p.add_argument("--source", default="0",
        help="Sumber video: '0' kamera default, '1' kamera lain, atau path ke file video")
    p.add_argument("--save", action="store_true",
        help="Auto-simpan semua frame saat keluar")
    p.add_argument("--output", default="features.npy",
        help="Nama file output .npy (default: features.npy)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(source=args.source, auto_save=args.save, output_path=args.output)