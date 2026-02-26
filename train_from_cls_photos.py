"""
train_from_cls_photos.py  ─ MAXIMUM ACCURACY (Eye-Aligned + Ensemble)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline:
  1. DNN SSD face detection (+ profile + frontal Haar fallback)
  2. Eye-based face ALIGNMENT → canonical 100×100 face patch
  3. Rich augmentation per aligned face (~44 variants)
  4. Train TWO models:
       • face_model.yml   – LBPH   (texture, illumination robust)
       • eigen_model.yml  – EigenFace (global appearance)
  Prediction uses ensemble vote of both models.
"""

import cv2, os, json, math, numpy as np

# ─────────────────────────────────────────────────────────────────────────────
CLS_PHOTOS_DIR  = "/Users/kavi/Downloads/Cls photos"
OUTPUT_LBPH     = "/Users/kavi/human_detection_project/face_model.yml"
OUTPUT_EIGEN    = "/Users/kavi/human_detection_project/eigen_model.yml"
LABEL_MAP_FILE  = "/Users/kavi/human_detection_project/roll_label_map.json"
FACE_SIZE       = (100, 100)
MIN_FACE_PX     = 18
DNN_PROTO       = "/Users/kavi/human_detection_project/deploy.prototxt"
DNN_MODEL_PATH  = "/Users/kavi/human_detection_project/res10_300x300_ssd_iter_140000_fp16.caffemodel"
# ─────────────────────────────────────────────────────────────────────────────

# ─── FACE ALIGNMENT ──────────────────────────────────────────────────────────
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

def align_face(face_gray: np.ndarray) -> np.ndarray:
    """
    Detect eyes and rotate/scale face so eyes are horizontally aligned.
    Falls back to plain resize if no eyes detected.
    """
    # Scale to working size first
    working = cv2.resize(face_gray, (150, 150))
    clahe   = cv2.createCLAHE(2.0, (8, 8))
    eq      = clahe.apply(working)

    eyes = _eye_cascade.detectMultiScale(eq, 1.1, 5, minSize=(15, 15))
    if len(eyes) < 2:
        return cv2.resize(face_gray, FACE_SIZE)

    # Sort eyes by x-coordinate (left → right)
    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
    cx1, cy1 = ex1 + ew1 // 2, ey1 + eh1 // 2
    cx2, cy2 = ex2 + ew2 // 2, ey2 + eh2 // 2

    # Angle of the eye line
    angle = math.degrees(math.atan2(cy2 - cy1, cx2 - cx1))
    h, w  = working.shape

    # Rotate
    M       = cv2.getRotationMatrix2D(((cx1 + cx2) / 2, (cy1 + cy2) / 2), angle, 1.0)
    aligned = cv2.warpAffine(working, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(aligned, FACE_SIZE)


# ─── AUGMENTATION ────────────────────────────────────────────────────────────
def gamma_correct(img, gamma):
    t = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, t)


def augment_face(face_gray: np.ndarray) -> list:
    """~44 augmented variants of an already-aligned 100×100 face."""
    base    = cv2.resize(face_gray, FACE_SIZE)
    h, w    = base.shape
    samples = []

    # Rotation × Brightness × Flip  →  5×3×2 = 30
    for angle in [-30, -15, 0, 15, 30]:
        M   = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        for alpha in [0.75, 1.0, 1.30]:
            bright = np.clip(rot * alpha, 0, 255).astype(np.uint8)
            samples.append(bright)
            samples.append(cv2.flip(bright, 1))

    # Blur levels  →  4
    for k in [3, 5, 7]:
        samples.append(cv2.GaussianBlur(base, (k, k), 0))
    samples.append(cv2.medianBlur(base, 3))

    # Downscale-upscale (simulate distance)  →  3
    for scale in [0.5, 0.3, 0.20]:
        small = cv2.resize(base, (max(1, int(w * scale)), max(1, int(h * scale))))
        samples.append(cv2.resize(small, FACE_SIZE, interpolation=cv2.INTER_LINEAR))

    # Gamma  →  2
    samples += [gamma_correct(base, 0.7), gamma_correct(base, 1.5)]

    # Noise  →  1
    noise = np.random.normal(0, 12.0, base.shape).astype(np.float32)
    samples.append(np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8))

    # Scale crops  →  3
    for mg in [8, 14, 20]:
        samples.append(cv2.resize(base[mg: h - mg, mg: w - mg], FACE_SIZE))

    return samples


# ─── DETECTION ───────────────────────────────────────────────────────────────
def detect_best_face(img_bgr, net, cascade_f, cascade_p):
    h, w = img_bgr.shape[:2]
    faces = []

    # DNN SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr,(300,300)),1.0,(300,300),(104,177,123))
    net.setInput(blob)
    dets = net.forward()
    for i in range(dets.shape[2]):
        c = dets[0,0,i,2]
        if c > 0.30:
            b = dets[0,0,i,3:7]*np.array([w,h,w,h])
            x1,y1,x2,y2 = b.astype(int)
            x1,y1 = max(0,x1),max(0,y1)
            x2,y2 = min(w,x2),min(h,y2)
            fw,fh = x2-x1,y2-y1
            if fw>MIN_FACE_PX and fh>MIN_FACE_PX:
                faces.append((x1,y1,fw,fh))

    if not faces:
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0,(8,8)); gray = clahe.apply(gray)
        for cas in [cascade_p, cascade_f]:
            det = cas.detectMultiScale(gray,1.1,3,minSize=(MIN_FACE_PX,MIN_FACE_PX))
            if len(det)>0: faces = list(det); break

    return max(faces, key=lambda f: f[2]*f[3]) if faces else None


# ─── COLLECT SAMPLES ─────────────────────────────────────────────────────────
def get_roll_number_folders(base_dir):
    return sorted(n for n in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir,n)) and not n.startswith("."))


def collect_samples(base_dir, roll_numbers, net, cascade_f, cascade_p):
    face_samples, label_list, label_map = [], [], {}
    total_ok = total_miss = 0

    for lbl, roll in enumerate(roll_numbers):
        label_map[lbl] = roll
        folder = os.path.join(base_dir, roll)
        imgs   = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        roll_count = 0
        for img_file in imgs:
            img = cv2.imread(os.path.join(folder, img_file))
            if img is None: continue
            hh,ww = img.shape[:2]
            if max(hh,ww)>1200:
                s=1200/max(hh,ww); img=cv2.resize(img,(int(ww*s),int(hh*s)))

            result = detect_best_face(img, net, cascade_f, cascade_p)
            if result is None:
                print(f"  ⚠ No face: {roll}/{img_file}"); total_miss+=1; continue

            x,y,fw,fh = result
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(2.0,(8,8)); gray = clahe.apply(gray)
            raw   = gray[y:y+fh, x:x+fw]

            # ★ EYE-ALIGN the face before augmentation ★
            aligned = align_face(raw)

            aug = augment_face(aligned)
            face_samples.extend(aug)
            label_list.extend([lbl]*len(aug))
            roll_count += len(aug)
            total_ok   += 1

        print(f"  ✅ {roll}: {roll_count} samples ({total_ok} photos)")

    print(f"\n📊 Total: {len(face_samples)} samples | {total_miss} missed")
    return face_samples, label_list, label_map


# ─── TRAIN & SAVE ─────────────────────────────────────────────────────────────
def train_and_save(face_samples, label_list, label_map):
    if not face_samples:
        print("❌ No samples."); return False

    arr_lbl = np.array(label_list, dtype=np.int32)

    # LBPH
    print(f"\n🏋 Training LBPH on {len(face_samples)} samples …")
    lbph = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
    lbph.train(face_samples, arr_lbl)
    lbph.write(OUTPUT_LBPH)
    print(f"✅ LBPH → {OUTPUT_LBPH}")

    # EigenFace (requires same-size images — already 100×100)
    print(f"🏋 Training EigenFace …")
    eigen = cv2.face.EigenFaceRecognizer_create(num_components=80)
    eigen.train(face_samples, arr_lbl)
    eigen.write(OUTPUT_EIGEN)
    print(f"✅ EigenFace → {OUTPUT_EIGEN}")

    with open(LABEL_MAP_FILE,"w") as f:
        json.dump({str(k):v for k,v in label_map.items()},f,indent=2)
    print(f"✅ Label map → {LABEL_MAP_FILE}")
    for lbl,roll in label_map.items():
        print(f"   {lbl} → {roll}")
    return True


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Maximum Accuracy Training (Align + Augment + Ensemble)")
    print("="*60)

    net = None
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL_PATH):
        net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL_PATH)
        print("✅ DNN loaded")
    else:
        class _D:
            def setInput(self,_): pass
            def forward(self): return np.zeros((1,1,0,7))
        net = _D()
        print("⚠  DNN missing")

    cascade_f = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    cascade_p = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_profileface.xml")

    roll_numbers = get_roll_number_folders(CLS_PHOTOS_DIR)
    if not roll_numbers: print("❌ No folders."); return

    print(f"\n📂 {len(roll_numbers)} students: {', '.join(roll_numbers)}\n")
    face_samples, label_list, label_map = collect_samples(
        CLS_PHOTOS_DIR, roll_numbers, net, cascade_f, cascade_p)
    if train_and_save(face_samples, label_list, label_map):
        print("\n🎉 Done! Run: streamlit run app.py")

if __name__ == "__main__":
    main()
