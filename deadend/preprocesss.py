"""
preprocess.py
-------------
Ekstraksi fitur landmark MediaPipe Holistic dari dataset video BISINDO.

Format nama file : [signerID]_[labelID]_[sampleID].mp4
Kelas yang diproses: Belajar(1), Maaf(6), Makan(7), Motor(8), Saya(9)

Output:
  output/X_train.npy  shape: (N_train, MAX_FRAMES, 156)
  output/X_val.npy    shape: (N_val,   MAX_FRAMES, 156)
  output/X_test.npy   shape: (N_test,  MAX_FRAMES, 156)
  output/y_train.npy  shape: (N_train,)
  output/y_val.npy    shape: (N_val,)
  output/y_test.npy   shape: (N_test,)
  output/label_map.json   { "0":"Belajar", "1":"Maaf", ... }
  output/dataset_info.json  ringkasan dataset

Penggunaan:
  python preprocess.py --dataset_dir /path/ke/folder/video
  python preprocess.py --dataset_dir ./data --max_frames 30
  python preprocess.py --dataset_dir ./data --val_size 0.15 --test_size 0.15
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split

# ─── Konfigurasi ─────────────────────────────────────────────────────────────

TARGET_LABELS = {1: "Belajar", 6: "Maaf", 7: "Makan", 8: "Motor", 9: "Saya"}

# Indeks landmark yang diekstrak
POSE_IDX = [0, 7, 8, 11, 12]       # hidung, telinga, bahu
FACE_IDX = [10, 0, 17, 61, 291]    # dahi, mulut
HAND_N   = 21

FEATURE_DIM = len(POSE_IDX)*3 + len(FACE_IDX)*3 + HAND_N*3*2  # 156


# ─── Ekstraksi Fitur ──────────────────────────────────────────────────────────

def extract_frame_features(results) -> np.ndarray:
    feat = []

    if results.pose_landmarks:
        for i in POSE_IDX:
            lm = results.pose_landmarks.landmark[i]
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * (len(POSE_IDX) * 3))

    if results.face_landmarks:
        for i in FACE_IDX:
            lm = results.face_landmarks.landmark[i]
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * (len(FACE_IDX) * 3))

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * (HAND_N * 3))

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * (HAND_N * 3))

    return np.array(feat, dtype=np.float32)


def process_video(video_path: str, holistic, max_frames: int) -> np.ndarray | None:
    """
    Proses satu file video → array shape (max_frames, FEATURE_DIM).
    - Jika frame < max_frames: padding nol di akhir.
    - Jika frame > max_frames: ambil frame secara merata (uniform sampling).
    Mengembalikan None jika video tidak bisa dibaca.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Baca semua frame dulu
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) == 0:
        return None

    # Uniform sampling jika terlalu panjang
    n = len(raw_frames)
    if n >= max_frames:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        selected = [raw_frames[i] for i in indices]
    else:
        selected = raw_frames

    # Ekstraksi fitur per frame
    sequence = []
    for frame in selected:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        feat = extract_frame_features(results)
        sequence.append(feat)

    # Padding jika kurang dari max_frames
    while len(sequence) < max_frames:
        sequence.append(np.zeros(FEATURE_DIM, dtype=np.float32))

    return np.array(sequence, dtype=np.float32)  # (max_frames, 156)


# ─── Scan Dataset ─────────────────────────────────────────────────────────────

def parse_filename(stem: str):
    """
    Support dua format nama file:
      - signer0_label1_sample2   → signer_id=0, label_id=1, sample_id=2
      - 0_1_2                    → signer_id=0, label_id=1, sample_id=2
    Mengembalikan (signer_id, label_id, sample_id) atau None jika tidak cocok.
    """
    import re
    # Format: signer0_label1_sample2
    m = re.fullmatch(r'signer(\d+)_label(\d+)_sample(\d+)', stem, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Format: 0_1_2
    parts = stem.split('_')
    if len(parts) == 3:
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            pass
    return None


def scan_dataset(dataset_dir: str) -> list[dict]:
    """
    Scan folder dataset dan kembalikan list entry video yang sesuai target label.
    Format nama yang didukung:
      - signer0_label1_sample2.mp4
      - 0_1_2.mp4
    """
    entries = []
    skipped = []

    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.lower().endswith('.mp4'):
            continue

        stem = fname[:-4]
        parsed = parse_filename(stem)
        if parsed is None:
            skipped.append(fname)
            continue

        signer_id, label_id, sample_id = parsed

        if label_id not in TARGET_LABELS:
            continue

        entries.append({
            'path'     : os.path.join(dataset_dir, fname),
            'fname'    : fname,
            'signer_id': signer_id,
            'label_id' : label_id,
            'sample_id': sample_id,
            'gloss'    : TARGET_LABELS[label_id],
        })

    if skipped:
        print(f"  [SKIP] {len(skipped)} file diabaikan (format tidak sesuai)")

    return entries


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'─'*55}")
    print(f"  SignTara — Preprocessing & Feature Extraction")
    print(f"{'─'*55}")
    print(f"  Dataset dir : {args.dataset_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Max frames  : {args.max_frames}")
    print(f"  Val size    : {args.val_size}")
    print(f"  Test size   : {args.test_size}")
    print(f"  Kelas       : {list(TARGET_LABELS.values())}")
    print(f"  Fitur/frame : {FEATURE_DIM}")
    print(f"{'─'*55}\n")

    # 1. Scan dataset
    print("[1/4] Scanning dataset...")
    entries = scan_dataset(args.dataset_dir)
    if not entries:
        print("[ERROR] Tidak ada video yang ditemukan. Cek path dan format nama file.")
        sys.exit(1)

    # Ringkasan per kelas
    per_label = defaultdict(list)
    for e in entries:
        per_label[e['label_id']].append(e)

    print(f"       Total video ditemukan: {len(entries)}")
    for lid, items in sorted(per_label.items()):
        print(f"       Label {lid:2d} ({TARGET_LABELS[lid]:12s}): {len(items)} video")

    # 2. Inisialisasi MediaPipe
    print("\n[2/4] Inisialisasi MediaPipe Holistic...")
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

    # 3. Ekstraksi fitur
    print(f"\n[3/4] Ekstraksi fitur dari {len(entries)} video...")
    X_all, y_all = [], []
    failed = []
    t0 = time.time()

    # Buat label map kontinu: label_id → index 0..N-1
    label_ids = sorted(TARGET_LABELS.keys())
    label_to_idx = {lid: idx for idx, lid in enumerate(label_ids)}
    idx_to_gloss = {idx: TARGET_LABELS[lid] for lid, idx in label_to_idx.items()}

    for i, entry in enumerate(entries):
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (len(entries) - i - 1) if i > 0 else 0
        print(f"  [{i+1:3d}/{len(entries)}] {entry['fname']:<30s} "
              f"ETA: {eta:.0f}s", end='\r')

        seq = process_video(entry['path'], holistic, args.max_frames)

        if seq is None:
            failed.append(entry['fname'])
            print(f"\n  [WARN] Gagal proses: {entry['fname']}")
            continue

        X_all.append(seq)
        y_all.append(label_to_idx[entry['label_id']])

    holistic.close()
    print(f"\n  Selesai dalam {time.time()-t0:.1f}s. "
          f"Berhasil: {len(X_all)}, Gagal: {len(failed)}")

    if not X_all:
        print("[ERROR] Tidak ada data yang berhasil diekstrak.")
        sys.exit(1)

    X = np.array(X_all, dtype=np.float32)  # (N, max_frames, 156)
    y = np.array(y_all, dtype=np.int32)    # (N,)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    # 4. Split & Simpan
    print(f"\n[4/4] Split train/val/test dan simpan...")

    # Split: train | (val+test)
    test_ratio_adjusted = args.test_size / (1.0 - args.val_size)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=args.val_size + args.test_size,
        random_state=42,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=test_ratio_adjusted,
        random_state=42,
        stratify=y_tmp
    )

    splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_val'  : X_val,   'y_val'  : y_val,
        'X_test' : X_test,  'y_test' : y_test,
    }

    for name, arr in splits.items():
        path = os.path.join(args.output_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"  Saved {name}.npy  shape={arr.shape}")

    # Label map
    label_map = {str(idx): gloss for idx, gloss in idx_to_gloss.items()}
    label_map_path = os.path.join(args.output_dir, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"  Saved label_map.json: {label_map}")

    # Dataset info
    info = {
        'total_samples'  : int(len(X_all)),
        'feature_dim'    : FEATURE_DIM,
        'max_frames'     : args.max_frames,
        'n_classes'      : len(TARGET_LABELS),
        'label_map'      : label_map,
        'split'          : {
            'train': int(len(X_train)),
            'val'  : int(len(X_val)),
            'test' : int(len(X_test)),
        },
        'per_class_total': {
            TARGET_LABELS[lid]: int(np.sum(y == label_to_idx[lid]))
            for lid in label_ids
        },
        'failed_videos'  : failed,
    }
    info_path = os.path.join(args.output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n{'─'*55}")
    print(f"  SELESAI")
    print(f"  Train : {len(X_train)} sampel")
    print(f"  Val   : {len(X_val)} sampel")
    print(f"  Test  : {len(X_test)} sampel")
    print(f"  Output: {args.output_dir}/")
    print(f"{'─'*55}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BISINDO dataset preprocessor")
    p.add_argument("--dataset_dir", required=True,
                   help="Folder berisi file .mp4 dataset")
    p.add_argument("--output_dir", default="output",
                   help="Folder output .npy (default: output/)")
    p.add_argument("--max_frames", type=int, default=20,
                   help="Jumlah frame per sampel (default: 20)")
    p.add_argument("--val_size", type=float, default=0.15,
                   help="Proporsi validasi (default: 0.15)")
    p.add_argument("--test_size", type=float, default=0.15,
                   help="Proporsi test (default: 0.15)")
    args = p.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"[ERROR] Dataset dir tidak ditemukan: {args.dataset_dir}")
        sys.exit(1)

    main(args)