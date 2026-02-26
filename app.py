import streamlit as st
import cv2
import time
import os
import json
import numpy as np
from ultralytics import YOLO
from utils import count_people

LABEL_MAP_FILE  = "roll_label_map.json"
LBPH_THRESHOLD  = 150   # Classroom distance: blurry small faces need higher tolerance
EIGEN_THRESHOLD = 5000  # EigenFace distances are much larger numbers

# DNN face-detector model paths
DNN_PROTO  = "deploy.prototxt"
DNN_MODEL  = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
EIGEN_MODEL = "eigen_model.yml"

# Eye cascade for alignment
_eye_cascade_app = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

def load_roll_numbers() -> dict:
    """Load {int_label: roll_number} from JSON. Returns empty dict if file missing."""
    if not os.path.exists(LABEL_MAP_FILE):
        return {}
    with open(LABEL_MAP_FILE, "r") as f:
        raw = json.load(f)
    # keys are stored as strings in JSON – convert back to int
    return {int(k): v for k, v in raw.items()}

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="Accu AttenMarker AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- INJECT CUSTOM CSS ----------------
st.markdown("""
<style>
    /* Import Premium Space Grotesk Font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

    /* Global Typography & Hide Scrollbars */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f5f5f5; }
    ::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #2563eb; }

    /* Light Background */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: #1a202c;
    }
    
    /* Remove Streamlit Default Branding completely (Header, Footer, Menu) */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;} /* Hide 'Deploy' button */

    /* Custom App Headers */
    .main-header {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-top: -30px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #4b5563;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        letter-spacing: 2px;
    }

    /* Light Mode Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.15);
        border-color: #2563eb;
    }
    .metric-title {
        color: #64748b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .metric-value {
        color: #1a202c;
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .metric-value.highlight {
        color: #2563eb;
    }
    
    /* Status Banners */
    .status-banner {
        border-radius: 16px;
        padding: 18px 24px;
        text-align: center;
        font-weight: 700;
        font-size: 1.3rem;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.5s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-safe {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    .status-warning {
        background: #fef3c7;
        color: #854d0e;
        border: 1px solid #fcd34d;
    }
    .status-danger {
        background: #fee2e2;
        color: #7f1d1d;
        border: 2px solid #fca5a5;
        animation: dangerPulse 1.2s infinite;
    }

    @keyframes dangerPulse {
        0% { transform: scale(1); box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15); }
        50% { transform: scale(1.03); box-shadow: 0 0 40px rgba(220, 38, 38, 0.3); }
        100% { transform: scale(1); box-shadow: 0 0 20px rgba(220, 38, 38, 0.15); }
    }

    /* Custom Horizontal Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-bottom: 20px;
    }
    .grid-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .grid-title {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .grid-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1a202c;
    }

    /* Light Mode Sidebar */
    [data-testid="stSidebar"] {
        background: #f8f9fa !important;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    /* Light Mode Sliders & Radio Buttons */
    .stSlider > div > div > div > div { background-color: #2563eb !important; }
    .stRadio [role="radiogroup"] > label > div:first-of-type { background-color: #2563eb !important; border-color: #2563eb !important;}
    
    /* Custom Stylized Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 1px;
        transition: all 0.3s ease !important;
        border: none !important;
        position: relative;
        overflow: hidden;
    }
    
    /* Primary Button Override */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5) !important;
    }
    
    /* Secondary Button Override */
    .stButton > button[kind="secondary"] {
        background: #f3f4f6 !important;
        color: #1a202c !important;
        border: 1px solid #d1d5db !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #e5e7eb !important;
        border-color: #9ca3af !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-header'>🎓 EduVision AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Real-Time Classroom Crowd Analytics & Monitoring</div>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_face_recognizer():
    model_path = "classroom_ai/max_accuracy_run/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "yolov8x.pt"
    yolo_model = YOLO(model_path)

    # Load OpenCV Face Recognizer (LBPH)
    face_model_path = "face_model.yml"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if os.path.exists(face_model_path):
        recognizer.read(face_model_path)

    # Load EigenFace recognizer
    eigen_recognizer = cv2.face.EigenFaceRecognizer_create()
    if os.path.exists(EIGEN_MODEL):
        eigen_recognizer.read(EIGEN_MODEL)

    # Load DNN face detector (better at distance than Haar)
    dnn_net = None
    if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
        dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)

    # Load profile cascade for side-face detection
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml')

    # Dynamically load roll numbers from the trained label map
    label_map = load_roll_numbers()          # {int → roll_number}
    roll_list  = [label_map[i] for i in sorted(label_map.keys())]   # ordered list
    return yolo_model, recognizer, eigen_recognizer, face_cascade, dnn_net, profile_cascade, label_map, roll_list

model, face_recognizer, eigen_recognizer, face_cascade, dnn_net, profile_cascade, label_map, student_names_list = load_face_recognizer()

def detect_faces_live(frame_bgr):
    """
    Multi-detector pipeline:  DNN SSD  →  Profile Haar  →  Frontal Haar
    Returns list of (x, y, w, h). Eliminates duplicate detections.
    """
    h, w = frame_bgr.shape[:2]
    all_faces = []

    # 1. DNN SSD — best for frontal & ¾ faces, near & far
    if dnn_net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.40:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                fw, fh = x2 - x1, y2 - y1
                if fw > 15 and fh > 15:
                    all_faces.append((x1, y1, fw, fh))

    # 2. Profile Haar cascade — catches pure side faces
    gray_p = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_p = clahe.apply(gray_p)
    # Left profile
    left_p = profile_cascade.detectMultiScale(
        gray_p, 1.1, 3, minSize=(20, 20))
    if len(left_p) > 0:
        all_faces.extend(list(left_p))
    # Right profile (mirror)
    flipped = cv2.flip(gray_p, 1)
    right_raw = profile_cascade.detectMultiScale(
        flipped, 1.1, 3, minSize=(20, 20))
    for (x, y, fw, fh) in right_raw:
        all_faces.append((w - x - fw, y, fw, fh))

    # 3. Frontal Haar fallback if nothing found yet
    if not all_faces:
        det = face_cascade.detectMultiScale(
            gray_p, scaleFactor=1.1, minNeighbors=3, minSize=(18, 18))
        if len(det) > 0:
            all_faces.extend(list(det))

    # Deduplicate overlapping boxes (IoU > 0.4 → keep larger)
    def iou(a, b):
        ax1,ay1,aw,ah = a; ax2,ay2 = ax1+aw,ay1+ah
        bx1,by1,bw,bh = b; bx2,by2 = bx1+bw,by1+bh
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        inter = max(0,ix2-ix1) * max(0,iy2-iy1)
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0

    unique = []
    for face in sorted(all_faces, key=lambda f: f[2]*f[3], reverse=True):
        if not any(iou(face, u) > 0.4 for u in unique):
            unique.append(face)
    return unique

# ──────────────────────────────────────────────────────────────────────────────
# Face alignment helper (in-app, for prediction)
# ──────────────────────────────────────────────────────────────────────────────
import math

def align_face_app(face_gray: np.ndarray) -> np.ndarray:
    """Eye-detect and rotate so eyes are horizontal. Falls back to plain resize."""
    working = cv2.resize(face_gray, (150, 150))
    clahe   = cv2.createCLAHE(2.0, (8, 8))
    eq      = clahe.apply(working)
    eyes    = _eye_cascade_app.detectMultiScale(eq, 1.1, 5, minSize=(15, 15))
    if len(eyes) < 2:
        return cv2.resize(face_gray, (100, 100))
    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (ex1,ey1,ew1,eh1),(ex2,ey2,ew2,eh2) = eyes
    cx1,cy1 = ex1+ew1//2, ey1+eh1//2
    cx2,cy2 = ex2+ew2//2, ey2+eh2//2
    angle   = math.degrees(math.atan2(cy2-cy1, cx2-cx1))
    h, w    = working.shape
    M       = cv2.getRotationMatrix2D(((cx1+cx2)/2,(cy1+cy2)/2), angle, 1.0)
    aligned = cv2.warpAffine(working, M, (w,h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(aligned, (100, 100))


# ──────────────────────────────────────────────────────────────────────────────
# Smart Face Prediction: alignment + multi-scale + LBPH+EigenFace ensemble
# ──────────────────────────────────────────────────────────────────────────────
def _preprocess_crop(gray_crop: np.ndarray) -> np.ndarray:
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_crop)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return cv2.resize(denoised, (100, 100))


def predict_face(gray_crop: np.ndarray):
    """
    Full ensemble prediction:
    1. Eye-align the face crop
    2. Multi-scale voting with LBPH (3 crops)
    3. EigenFace vote
    4. Combine: roll that gets most votes from both models wins
    5. Ratio test for confidence
    Returns (roll_number, best_lbph_dist) or (None, dist).
    """
    h, w = gray_crop.shape[:2]
    if h < 5 or w < 5:
        return None, 9999

    # 1. Align
    aligned = align_face_app(gray_crop)   # 100×100 aligned face

    votes     = {}   # roll → vote count
    all_dists = {}   # roll → min LBPH dist

    # 2. Multi-scale LBPH voting (3 crops of aligned face)
    aH, aW = aligned.shape
    crops = [(0, 0, aW, aH)]
    for mg in [max(1,int(aH*0.08)), max(1,int(aH*0.14))]:
        if aH-2*mg>5 and aW-2*mg>5:
            crops.append((mg, mg, aW-mg, aH-mg))

    for (x1,y1,x2,y2) in crops:
        crop = aligned[y1:y2, x1:x2]
        face = _preprocess_crop(crop)
        lbl, dist = face_recognizer.predict(face)
        roll = label_map.get(lbl)
        if roll:
            votes[roll]     = votes.get(roll, 0) + 1
            all_dists[roll] = min(all_dists.get(roll, 9999), dist)

    # 3. EigenFace vote
    try:
        e_lbl, e_dist = eigen_recognizer.predict(aligned)
        e_roll = label_map.get(e_lbl)
        if e_roll and e_dist < EIGEN_THRESHOLD:
            votes[e_roll] = votes.get(e_roll, 0) + 1
    except Exception:
        pass

    if not votes:
        return None, 9999

    # 4. Winner by most votes, tie-break by min LBPH dist
    best_roll = max(votes, key=lambda r: (votes[r], -all_dists.get(r, 9999)))
    best_dist = all_dists.get(best_roll, 9999)

    # Hard LBPH threshold
    if best_dist >= LBPH_THRESHOLD:
        return None, best_dist

    # 5. Ratio test
    other_dists = [d for r,d in all_dists.items() if r != best_roll]
    if other_dists:
        second = min(other_dists)
        if best_dist / (second + 1e-5) > 0.82:
            return None, best_dist

    return best_roll, best_dist


with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.markdown("---")
    
    mode = st.radio(
        "Observation Mode",
        ["📷 Live Camera Feed", "🖼 Upload Snapshot Analytics"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ AI Parameters")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05, 
                           help="Minimum probability to consider a detection valid. Higher values reduce false positives.")
    overcrowd_limit = st.slider("Overcrowd Limit", 1, 100, 20, 
                                help="Trigger alerts when the student count exceeds this number.")

# ---------------- STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False

# Initialize attendance state early so modal can access it safely
# Reload roll numbers on each run (not cached so hot-reloads pick up new models)
roll_numbers = student_names_list   # alias for readability
if "attendance" not in st.session_state or set(st.session_state.attendance.keys()) != set(roll_numbers):
    st.session_state.attendance = {roll: "Absent" for roll in roll_numbers}

# ---------------- LAYOUT ----------------
# Main dividing columns: 70% Video Area, 30% Analytics
main_col, stats_col = st.columns([7, 3])

with main_col:
    # Dual Video Feeds
    vid_col1, vid_col2 = st.columns(2)
    with vid_col1:
        st.markdown("<h5 style='text-align: center; color: #4b5563;'>Face recognition</h5>", unsafe_allow_html=True)
        face_placeholder = st.empty()
    with vid_col2:
        st.markdown("<h5 style='text-align: center; color: #4b5563;'>Person Detection</h5>", unsafe_allow_html=True)
        yolo_placeholder = st.empty()
        
    st.markdown("<br>", unsafe_allow_html=True)
    # Control Buttons horizontally below videos
    btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])
    start = btn_col1.button("▶ Start Session", width="stretch", type="primary")
    stop = btn_col2.button("⏹ End Session", width="stretch", type="primary")
    show_res = btn_col3.button("📊 Attendance Results", width="stretch", type="primary")

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False
    st.rerun()

# ---------------- SUMMARY MODAL ----------------
@st.dialog("Attendance Summary")
def show_attendance_summary():
    from datetime import datetime
    session_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
    present_count = sum(1 for s in st.session_state.attendance.values() if s == "Present")
    total_count   = len(st.session_state.attendance)
    st.markdown(
        f"<h4 style='text-align:center;color:#1e293b;margin-top:0;'>"
        f"📋 Session: {session_time} &nbsp;|&nbsp; "
        f"✅ Present: {present_count}/{total_count}</h4>",
        unsafe_allow_html=True
    )

    html = "<table style='width:100%; border-collapse:collapse; margin-bottom:20px; font-family:sans-serif;'>"
    html += "<tr style='background:#0284c7; color:white; text-align:left;'>"
    html += "<th style='padding:10px; border:1px solid #ddd;'>#</th>"
    html += "<th style='padding:10px; border:1px solid #ddd;'>Roll Number</th>"
    html += "<th style='padding:10px; border:1px solid #ddd; text-align:center;'>Status</th>"
    html += "</tr>"

    for i, (roll, status) in enumerate(st.session_state.attendance.items(), 1):
        color = "#16a34a" if status == "Present" else "#dc2626"
        bg    = "#dcfce7" if status == "Present" else "#fee2e2"
        html += f"<tr><td style='padding:10px;border:1px solid #ddd;text-align:center;'>{i}</td>"
        html += f"<td style='padding:10px;border:1px solid #ddd;font-weight:bold;color:#333;'>{roll}</td>"
        html += f"<td style='padding:10px;border:1px solid #ddd;text-align:center;'>"
        html += f"<span style='background:{bg};color:{color};padding:4px 14px;border-radius:12px;font-weight:bold;'>{status}</span>"
        html += "</td></tr>"

    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    if col2.button("✅ Close", width="stretch", type="primary"):
        st.session_state.show_results = False
        st.rerun()

if show_res:
    show_attendance_summary()

with stats_col:
    st.markdown("### Attendance Status")
    
    # Render Attendance Table dynamically based on running state
    attendance_placeholder = st.empty()
    
    def render_attendance():
        if not st.session_state.attendance:
            attendance_placeholder.info("⚠️ No students found. Run `train_from_cls_photos.py` first.")
            return
        html = "<div style='display:flex; flex-direction:column; gap:8px;'>"
        for roll, status in st.session_state.attendance.items():
            if st.session_state.run:
                # While running: show roll numbers with scanning indicator
                html += (f"<div style='padding:10px; border-radius:8px; background:#f8f9fa; "
                         f"border:1px solid #e2e8f0; display:flex; align-items:center;'>"
                         f"<strong style='color:#4b5563; font-size:14px;'>🎓 {roll}</strong>"
                         f"</div>")
            else:
                # When stopped: reveal Present / Absent badge
                color    = "#22c55e" if status == "Present" else "#ef4444"
                html += (f"<div style='padding:10px; border-radius:8px; background:#f8f9fa; "
                         f"border-left:4px solid {color}; display:flex; justify-content:space-between; "
                         f"align-items:center; box-shadow:0 1px 3px rgba(0,0,0,0.1);'>"
                         f"<strong style='color:#1e293b; font-size:14px;'>🎓 {roll}</strong>"
                         f"<span style='color:white; font-weight:bold; background:{color}; "
                         f"padding:4px 14px; border-radius:12px; font-size:13px;'>{status}</span>"
                         f"</div>")
        html += "</div>"
        attendance_placeholder.markdown(html, unsafe_allow_html=True)
        
    render_attendance()
    
    # Top Metrics Grid Placeholder
    metrics_placeholder = st.empty()
    
    # Analytics Pie Chart — live data from attendance
    st.markdown("---")
    alert_placeholder = st.empty()
    st.markdown("### Analytics")
    pie_placeholder = st.empty()

def render_pie():
    present = sum(1 for s in st.session_state.attendance.values() if s == "Present")
    absent = len(st.session_state.attendance) - present
    pie_url = f"https://quickchart.io/chart?c={{type:%27doughnut%27,data:{{labels:[%27Absent%27,%27Present%27],datasets:[{{data:[{absent},{present}],backgroundColor:[%27%23ef4444%27,%27%2322c55e%27]}}]}}}}"
    pie_placeholder.markdown(f"<div style='text-align: center;'><img src='{pie_url}' width='150'></div>", unsafe_allow_html=True)

render_pie()

if not st.session_state.run and mode == "📷 Live Camera Feed":
    face_placeholder.info("👋 Setup Ready.")
    yolo_placeholder.info("Click **Start Session** below to initiate.")

# ---------------- UTILS FOR UI ----------------
# Setup CSV Logging
LOG_FILE = "detection_logs.csv"

def init_csv():
    # Create file and write headers if it doesn't exist
    if not os.path.exists(LOG_FILE):
        import csv
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Mode", "Person_Count", "Alert_Status"])

def log_to_csv(mode_name, count, limit):
    import csv
    from datetime import datetime
    status = "Safe"
    if count >= limit + 5:
        status = "Critical Overcrowding"
    elif count > limit:
        status = "Warning Near Capacity"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, mode_name, count, status])

# Initialize CSV on startup
init_csv()

def render_metrics(yolo_count, attendance_dict):
    present_count = sum(1 for status in attendance_dict.values() if status == "Present")
    absent_count = len(attendance_dict) - present_count
    
    html = f"""
    <div class="metrics-grid">
        <div class="grid-card">
            <div class="grid-title">Head Count</div>
            <div class="grid-value">{yolo_count}</div>
        </div>
        <div class="grid-card">
            <div class="grid-title">Present</div>
            <div class="grid-value" style="color: #22c55e;">{present_count}</div>
        </div>
        <div class="grid-card">
            <div class="grid-title">Absent</div>
            <div class="grid-value" style="color: #ef4444;">{absent_count}</div>
        </div>
    </div>
    """
    metrics_placeholder.markdown(html, unsafe_allow_html=True)

def render_alert(count, limit):
    if count >= limit + 5: # Severe Overcrowding
        html = f"<div class='status-banner status-danger'>🚨 CRITICAL: OVERCROWDED ({count}/{limit})</div>"
    elif count > limit: # Warning
        html = f"<div class='status-banner status-warning'>⚠️ WARNING: Near Capacity ({count}/{limit})</div>"
    else: # Safe
        html = f"<div class='status-banner status-safe'>✅ STATUS: NORMAL ({count}/{limit})</div>"
    
    alert_placeholder.markdown(html, unsafe_allow_html=True)

# ---------------- GROUP PHOTO FACE DETECTOR ----------------
def detect_faces_group(img_bgr):
    """Detect faces using DNN → fallback Haar, return sorted left-to-right."""
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
            if conf > 0.45:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                fw, fh = x2 - x1, y2 - y1
                if fw > 20 and fh > 20:
                    faces.append((x1, y1, fw, fh))
        if faces:
            faces.sort(key=lambda b: b[0])
            return faces
    # Fallback Haar cascade
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(25, 25))
    return sorted(list(faces), key=lambda b: b[0])


def annotate_group_photo(img_bgr, faces, names):
    """Draw colorful bounding boxes and name labels on the group photo."""
    annotated = img_bgr.copy()
    palette = [
        (0, 200, 80),   # green
        (255, 140, 0),  # orange
        (0, 180, 255),  # blue
        (200, 0, 200),  # purple
        (0, 80, 255),   # deep blue
    ]
    for idx, (x, y, w, h) in enumerate(faces):
        if idx >= len(names):
            break
        color = palette[idx % len(palette)]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        label = names[idx]
        font_scale = max(0.55, w / 220)
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (x, y - th - 14), (x + tw + 10, y), color, -1)
        cv2.putText(annotated, label, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness)
    return annotated


# ---------------- IMAGE MODE ----------------
if mode == "🖼 Upload Snapshot Analytics":
    with main_col:
        st.markdown("#### 📸 Upload Group Photo for Attendance")
        uploaded_file = st.file_uploader(
            "Upload the classroom / group photo to detect faces and mark attendance...",
            type=["jpg", "jpeg", "png"]
        )
        # Show registered roll numbers for reference
        st.markdown("##### 🎓 Registered Roll Numbers")
        st.caption("  |  ".join(roll_numbers) if roll_numbers else "⚠️ No students trained yet. Run `train_from_cls_photos.py` first.")

    if uploaded_file:
        with st.spinner("🔍 Detecting faces & marking attendance..."):
            try:
                file_bytes = uploaded_file.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

                # Resize if too large
                height, width = img.shape[:2]
                if max(height, width) > 1920:
                    scale = 1920 / max(height, width)
                    img = cv2.resize(img, (int(width * scale), int(height * scale)))

                # ── YOLO Person Count ──
                results = model(img, conf=confidence, iou=0.45)
                count, acc = count_people(results, confidence)
                log_to_csv("Image Upload", count, overcrowd_limit)
                annotated_yolo = results[0].plot()
                cv2.putText(annotated_yolo, f"Detected: {count}", (30, 70),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 170), 4)
                annotated_yolo_rgb = cv2.cvtColor(annotated_yolo, cv2.COLOR_BGR2RGB)

                # ── FACE RECOGNITION via full multi-detector pipeline ──
                # Reset attendance fresh for each uploaded photo
                st.session_state.attendance = {roll: "Absent" for roll in roll_numbers}

                # Use the same DNN + profile + frontal detector as live mode
                faces_up = detect_faces_live(img)

                # Build CLAHE grayscale for LBPH prediction
                gray_upload = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe_up    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_upload = clahe_up.apply(gray_upload)

                annotated_face = img.copy()
                recognized_rolls = []
                ih, iw = img.shape[:2]

                for (x, y, w_f, h_f) in faces_up:
                    # Clamp bounds
                    x, y = max(0, x), max(0, y)
                    w_f  = min(w_f, iw - x)
                    h_f  = min(h_f, ih - y)
                    if w_f < 5 or h_f < 5:
                        continue

                    face_gray = gray_upload[y:y+h_f, x:x+w_f]
                    matched_roll, dist = predict_face(face_gray)

                    if matched_roll is not None:
                        recognized_rolls.append(matched_roll)
                        st.session_state.attendance[matched_roll] = "Present"
                        color_box = (0, 200, 80)       # green
                        label_txt = f"{matched_roll}  ✓"
                    else:
                        color_box = (0, 0, 210)        # red
                        label_txt = "Unknown"

                    # Draw bold box
                    cv2.rectangle(annotated_face, (x, y), (x+w_f, y+h_f), color_box, 3)
                    # Label background pill
                    font_scale = max(0.45, w_f / 250)
                    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
                    cv2.rectangle(annotated_face,
                                  (x, max(0, y - th - 12)),
                                  (x + tw + 10, y),
                                  color_box, -1)
                    cv2.putText(annotated_face, label_txt,
                                (x + 5, max(th, y - 6)),
                                cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2)

                annotated_face_rgb = cv2.cvtColor(annotated_face, cv2.COLOR_BGR2RGB)

                face_placeholder.image(annotated_face_rgb, use_column_width=True,
                                       caption=f"Face Recognition — {len(recognized_rolls)} student(s) identified")
                yolo_placeholder.image(annotated_yolo_rgb, use_column_width=True,
                                       caption=f"Person Detection | {count} person(s) found")

                render_attendance()
                render_metrics(count, st.session_state.attendance)
                render_alert(count, overcrowd_limit)
                render_pie()

                present_n = sum(1 for s in st.session_state.attendance.values() if s == "Present")
                absent_n  = len(st.session_state.attendance) - present_n

                with main_col:
                    if len(faces_up) == 0:
                        st.warning("⚠️ No faces detected. Try a clearer / higher-resolution photo.")
                    else:
                        st.success(f"✅ {present_n} student(s) **Present** | {absent_n} **Absent**")
                        att_html = ("<table style='width:100%;border-collapse:collapse;"
                                    "margin-top:10px;font-family:sans-serif;'>"
                                    "<tr style='background:#2563eb;color:white;'>"
                                    "<th style='padding:8px;border:1px solid #ddd;'>#</th>"
                                    "<th style='padding:8px;border:1px solid #ddd;'>Roll Number</th>"
                                    "<th style='padding:8px;border:1px solid #ddd;text-align:center;'>Status</th></tr>")
                        for i, (roll, status) in enumerate(st.session_state.attendance.items(), 1):
                            color = "#16a34a" if status == "Present" else "#dc2626"
                            bg    = "#dcfce7" if status == "Present" else "#fee2e2"
                            icon  = "✅" if status == "Present" else "❌"
                            badge = (f"<span style='background:{bg};color:{color};"
                                     f"padding:4px 14px;border-radius:12px;font-weight:bold;'>"
                                     f"{icon} {status}</span>")
                            att_html += (f"<tr><td style='padding:8px;border:1px solid #ddd;"
                                         f"text-align:center;'>{i}</td>"
                                         f"<td style='padding:8px;border:1px solid #ddd;"
                                         f"font-weight:bold;'>{roll}</td>"
                                         f"<td style='padding:8px;border:1px solid #ddd;"
                                         f"text-align:center;'>{badge}</td></tr>")
                        att_html += "</table>"
                        st.markdown(att_html, unsafe_allow_html=True)

                log_to_csv("Image Upload", count, overcrowd_limit)
                st.success("✅ Log saved to `detection_logs.csv` successfully!")

            except Exception as e:
                st.error(f"Failed to process image. Error: {str(e)}")

# ---------------- LIVE MODE ----------------
if mode == "📷 Live Camera Feed" and st.session_state.run:
    cap = cv2.VideoCapture(0)
    prev_time = 0
    last_log_time = time.time() # Tracker for logging intervals

    try:
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to acquire camera feed.")
                break

            # AI Person Detection (YOLO)
            results = model(frame, conf=confidence, iou=0.45)
            count, acc = count_people(results, confidence)
            annotated_yolo = results[0].plot()

            # AI Face Recognition (OpenCV LBPH — DNN detector for any distance)
            annotated_face = frame.copy()
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_full = clahe.apply(gray_full)

            faces = detect_faces_live(frame)

            attendance_updated = False
            for (x, y, w, h) in faces:
                x, y = max(0, x), max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                if w < 5 or h < 5:
                    continue

                face_gray = gray_full[y: y+h, x: x+w]
                matched_roll, dist = predict_face(face_gray)

                if matched_roll is not None:
                    if st.session_state.attendance.get(matched_roll) != "Present":
                        st.session_state.attendance[matched_roll] = "Present"
                        attendance_updated = True
                    display_label = f"{matched_roll} ({dist:.0f})"
                    cv2.rectangle(annotated_face, (x, y), (x+w, y+h), (0, 220, 80), 2)
                    cv2.putText(annotated_face, display_label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 2)
                else:
                    cv2.rectangle(annotated_face, (x, y), (x+w, y+h), (0, 0, 220), 2)
                    cv2.putText(annotated_face, f"? ({dist:.0f})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

            # FPS calculation
            current_time = time.time()
            fps = 1 / (current_time - prev_time + 1e-5)
            prev_time = current_time

            # CSV Logging every 2 seconds to avoid excessive disk I/O
            if current_time - last_log_time >= 2.0:
                log_to_csv("Live Camera", count, overcrowd_limit)
                last_log_time = current_time

            # Convert BGR to RGB for Streamlit displaying
            annotated_yolo = cv2.cvtColor(annotated_yolo, cv2.COLOR_BGR2RGB)
            annotated_face = cv2.cvtColor(annotated_face, cv2.COLOR_BGR2RGB)
            
            # UI Updates (Separate Feeds)
            face_placeholder.image(annotated_face, use_column_width=True)
            yolo_placeholder.image(annotated_yolo, use_column_width=True)
            render_metrics(count, st.session_state.attendance)
            render_alert(count, overcrowd_limit)
            
            # Instantly update attendance list and pie chart if a new student was recognized
            if attendance_updated:
                render_attendance()
                render_pie()

            time.sleep(0.01) # Small sleep to prevent freezing Streamlit

    finally:
        # Guarantee camera resource is released when loop breaks or Stop is clicked
        cap.release()
        cv2.destroyAllWindows()