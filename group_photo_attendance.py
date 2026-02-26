"""
group_photo_attendance.py
--------------------------
Detects faces in a group photo (left to right), assigns student names,
saves cropped faces to student_faces/, trains the LBPH face recognition model,
and produces a printed attendance report.

Usage:
    python group_photo_attendance.py --photo group_photo.jpg
"""

import cv2
import numpy as np
import os
import argparse
import sys

# ── Student names in the order they appear LEFT to RIGHT in the photo ──
STUDENT_NAMES = ["Abhijeeth", "Kavishik", "Aryan", "Anand", "Shashank"]

FACE_DIR = "student_faces"
MODEL_PATH = "face_model.yml"


def detect_faces(img_bgr):
    """Return face bounding boxes in (x, y, w, h) sorted left-to-right."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Try DNN-based detector first for better accuracy
    dnn_proto = "deploy.prototxt"
    dnn_model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    
    if os.path.exists(dnn_proto) and os.path.exists(dnn_model):
        net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                fw, fh = x2 - x1, y2 - y1
                if fw > 20 and fh > 20:
                    faces.append((x1, y1, fw, fh))
        if faces:
            faces.sort(key=lambda b: b[0])  # left-to-right
            return faces, gray

    # Fallback: Haar cascade
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        # Try with relaxed params
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    faces = sorted(faces, key=lambda b: b[0])
    return list(faces), gray


def save_faces(img_bgr, faces, gray, names):
    """Crop & save each face to student_faces/<name>/1.jpg"""
    os.makedirs(FACE_DIR, exist_ok=True)
    saved = []
    for idx, (x, y, w, h) in enumerate(faces):
        if idx >= len(names):
            break
        name = names[idx]
        person_dir = os.path.join(FACE_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        # Expand the crop slightly for better recognition
        pad = int(0.15 * min(w, h))
        ih, iw = img_bgr.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad)
        y2 = min(ih, y + h + pad)

        face_crop = gray[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (200, 200))
        
        save_path = os.path.join(person_dir, "1.jpg")
        cv2.imwrite(save_path, face_crop)
        saved.append((name, save_path))
        print(f"  ✅ Saved face: {name} → {save_path}")
    return saved


def train_model(names):
    """Train LBPH recognizer on saved face images."""
    faces_data, labels = [], []
    for label, name in enumerate(names):
        person_dir = os.path.join(FACE_DIR, name)
        if not os.path.exists(person_dir):
            print(f"  ⚠️  No folder for {name}, skipping.")
            continue
        for fname in os.listdir(person_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(person_dir, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (200, 200))
                    faces_data.append(img)
                    labels.append(label)

    if not faces_data:
        print("  ❌ No face images found. Training aborted.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces_data, np.array(labels))
    recognizer.save(MODEL_PATH)
    print(f"  ✅ Model trained with {len(faces_data)} face(s) → saved to {MODEL_PATH}")
    return True


def annotate_photo(img_bgr, faces, names):
    """Draw bounding boxes + names on the image."""
    annotated = img_bgr.copy()
    colors = [
        (0, 200, 80),    # green
        (255, 140, 0),   # orange
        (0, 180, 255),   # cyan-blue
        (200, 0, 200),   # purple
        (0, 80, 255),    # deep blue
    ]
    for idx, (x, y, w, h) in enumerate(faces):
        if idx >= len(names):
            break
        color = colors[idx % len(colors)]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        label = names[idx]
        font_scale = max(0.6, w / 200)
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        # Background pill for text
        cv2.rectangle(annotated, (x, y - th - 14), (x + tw + 10, y), color, -1)
        cv2.putText(annotated, label, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness)
    return annotated


def print_attendance(names, status="Present"):
    print("\n" + "=" * 45)
    print("       📋  ATTENDANCE REPORT")
    print("=" * 45)
    print(f"  {'#':<4} {'Student Name':<20} {'Status'}")
    print("  " + "-" * 38)
    for i, name in enumerate(names, 1):
        badge = "✅ Present" if status == "Present" else "❌ Absent"
        print(f"  {i:<4} {name:<20} {badge}")
    print("=" * 45)
    print(f"  Total: {len(names)}  |  Present: {len(names)}  |  Absent: 0")
    print("=" * 45 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Group Photo Attendance")
    parser.add_argument("--photo", default="group_photo.jpg", help="Path to group photo")
    parser.add_argument("--no-train", action="store_true", help="Skip model training")
    args = parser.parse_args()

    photo_path = args.photo
    if not os.path.exists(photo_path):
        print(f"❌ Photo not found: {photo_path}")
        sys.exit(1)

    print(f"\n🖼  Loading photo: {photo_path}")
    img = cv2.imread(photo_path)
    if img is None:
        print("❌ Failed to read image.")
        sys.exit(1)

    # Resize if too large
    h, w = img.shape[:2]
    if max(h, w) > 1920:
        scale = 1920 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        print(f"  ℹ️  Resized to {img.shape[1]}x{img.shape[0]}")

    print("🔍 Detecting faces...")
    faces, gray = detect_faces(img)
    print(f"  Found {len(faces)} face(s)")

    if len(faces) == 0:
        print("❌ No faces detected. Please check the photo.")
        sys.exit(1)

    # If more faces than names, keep only the first N
    if len(faces) > len(STUDENT_NAMES):
        faces = faces[:len(STUDENT_NAMES)]
        print(f"  ℹ️  Using only first {len(STUDENT_NAMES)} faces (left-to-right)")

    names_used = STUDENT_NAMES[:len(faces)]
    print(f"\n👥 Assigning names (left → right):")
    for i, name in enumerate(names_used):
        x, y, w_f, h_f = faces[i]
        print(f"  Face {i+1}: {name}  (x={x})")

    print("\n💾 Saving cropped faces...")
    save_faces(img, faces, gray, names_used)

    if not args.no_train:
        print("\n🧠 Training face recognition model...")
        train_model(STUDENT_NAMES)

    # Save annotated photo
    annotated = annotate_photo(img, faces, names_used)
    out_path = "attendance_result.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\n📸 Annotated photo saved → {out_path}")

    # Print attendance report
    print_attendance(names_used, status="Present")


if __name__ == "__main__":
    main()
