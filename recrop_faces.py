"""
recrop_faces.py — DNN auto-detect + manual fallback
Looks for: group_photo.jpg first, then new_group.png
Run: python3 recrop_faces.py
"""

import cv2
import os
import numpy as np

NAMES      = ["Abhijeeth", "Kavishik", "Aryan", "Anand", "Shashank"]
OUTPUT_DIR = "student_faces"
MODEL_PATH = "face_model.yml"
DEBUG_PATH = "debug_faces.jpg"
DNN_PROTO  = "deploy.prototxt"
DNN_MODEL  = "res10_300x300_ssd_iter_140000_fp16.caffemodel"


def detect_faces_dnn(img, conf_threshold=0.35):
    net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                  (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()

    boxes = []
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf < conf_threshold:
            continue
        b = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = b.astype(int)
        pad = int((y2 - y1) * 0.30)
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: b[0])   # sort left → right
    return boxes


def manual_crop_boxes(h, w):
    """
    Fallback manual crop if DNN misses faces.
    Calibrated for an ~840x1120 portrait phone photo
    (5 people standing left to right under a tent).
    Coordinates scale proportionally.
    """
    sx, sy = w / 840.0, h / 1120.0

    raw = [
        (55,  170, 225, 380),   # Abhijeeth  – far left, dark shirt under white
        (200, 120, 400, 380),   # Kavishik   – 2nd from left, taller
        (370, 190, 560, 420),   # Aryan      – middle, patterned shirt
        (530, 165, 720, 390),   # Anand      – 2nd from right, watch
        (690, 140, 855, 375),   # Shashank   – far right, dark check shirt
    ]
    return [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
            for (x1, y1, x2, y2) in raw]


def crop_and_save(img):
    h, w = img.shape[:2]
    print(f"📷  Image: {w}×{h}")

    boxes = detect_faces_dnn(img)
    print(f"🔍  DNN → {len(boxes)} face(s)")

    if len(boxes) < 5:
        print(f"⚠️   Switching to manual calibrated boxes …")
        boxes = manual_crop_boxes(h, w)

    boxes = boxes[:5]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    debug = img.copy()
    COLS = [(0,255,0),(255,165,0),(0,165,255),(255,0,255),(0,255,255)]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        name   = NAMES[i]
        crop   = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"⚠️  Empty crop for {name}")
            continue
        out = os.path.join(OUTPUT_DIR, f"{name}.jpg")
        cv2.imwrite(out, cv2.resize(crop, (200, 200)))
        print(f"✅  {name:12s}  [{x1},{y1}→{x2},{y2}]")
        cv2.rectangle(debug, (x1,y1),(x2,y2), COLS[i], 3)
        cv2.putText(debug, name, (x1, max(y1-10,20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLS[i], 2)

    cv2.imwrite(DEBUG_PATH, debug)
    print(f"🖼   Debug → {DEBUG_PATH}")


def train_model():
    recognizer   = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces, labels = [], []
    print("\n🔄  Training …")

    for label, name in enumerate(NAMES):
        p = os.path.join(OUTPUT_DIR, f"{name}.jpg")
        if not os.path.exists(p):
            print(f"⚠️  Missing {p}"); continue

        gray = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        det  = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20,20))
        if len(det) > 0:
            x,y,fw,fh = max(det, key=lambda r: r[2]*r[3])
            gray = gray[y:y+fh, x:x+fw]
        gray = cv2.resize(gray, (100, 100))
        faces.append(gray); labels.append(label)

    if not faces:
        print("❌  No faces found"); return

    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_PATH)
    print(f"✅  Model saved → {MODEL_PATH}")
    for i,n in enumerate(NAMES): print(f"   {i} → {n}")


if __name__ == "__main__":
    # Try group_photo.jpg first, then new_group.png
    for candidate in ["group_photo.jpg", "group_photo.png",
                      "new_group.jpg", "new_group.png"]:
        if os.path.exists(candidate):
            img = cv2.imread(candidate)
            if img is not None:
                print(f"📂  Using: {candidate}")
                break
    else:
        print("❌  No photo found!")
        print("👉  Save the group photo as:  group_photo.jpg  in this folder")
        exit(1)

    crop_and_save(img)
    train_model()
    print("\n🎉  Done! Restart app.py")
