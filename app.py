import streamlit as st
import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from utils import count_people

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
    
    # Load OpenCV Face Recognizer
    face_model_path = "face_model.yml"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if os.path.exists(face_model_path):
        recognizer.read(face_model_path)
    
    names = ["Abhijeeth", "Kavishik", "Aryan", "Anand", "Shashank"]
    return yolo_model, recognizer, face_cascade, names

model, face_recognizer, face_cascade, student_names_list = load_face_recognizer()

# ---------------- SIDEBAR ----------------
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

    st.session_state.run = False

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
    st.markdown("<h4 style='text-align: center; color: #1e293b; margin-top: 0;'>Teacher: Najeeb | Session ID: 144</h4>", unsafe_allow_html=True)
    
    html = "<table style='width:100%; border-collapse: collapse; margin-bottom: 20px; font-family: sans-serif;'>"
    html += "<tr style='background-color: #0284c7; color: white; text-align: left;'>"
    html += "<th style='padding: 10px; border: 1px solid #ddd;'>Student Name</th>"
    html += "<th style='padding: 10px; border: 1px solid #ddd; text-align: center;'>Status</th>"
    html += "</tr>"

    for student, status in st.session_state.attendance.items():
        color = "#16a34a" if status == "Present" else "#dc2626"
        bg = "#dcfce7" if status == "Present" else "#fee2e2"
        html += "<tr>"
        html += f"<td style='padding: 10px; border: 1px solid #ddd; font-weight: bold; color: #333;'>{student}</td>"
        html += "<td style='padding: 10px; border: 1px solid #ddd; text-align: center;'>"
        html += f"<span style='background-color: {bg}; color: {color}; padding: 4px 12px; border-radius: 12px; font-weight: bold;'>{status}</span>"
        html += "</td></tr>"

    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,1])
    if col2.button("✅ Close", width="stretch", type="primary"):
        st.session_state.show_results = False
        st.rerun()

if show_res:
    show_attendance_summary()

# ---------------- LAYOUT ----------------
# Main dividing columns: 70% Video, 30% Analytics
main_col, stats_col = st.columns([7, 3])

with stats_col:
    st.markdown("### Attendance Status")
    
    # Initialize attendance state
    student_names = ["Abhijeeth", "Kavishik", "Aryan", "Anand", "Shashank"]
    if "attendance" not in st.session_state:
        st.session_state.attendance = {name: "Absent" for name in student_names}
    
    # Render Attendance Table dynamically based on running state
    attendance_placeholder = st.empty()
    
    def render_attendance():
        html = "<div style='display: flex; flex-direction: column; gap: 8px;'>"
        for student, status in st.session_state.attendance.items():
            if st.session_state.run:
                # While running: Only show names with neutral background
                html += f"<div style='padding: 10px; border-radius: 8px; background-color: #f8f9fa; border: 1px solid #e2e8f0; display: flex; align-items: center;'>"
                html += f"<strong style='color: #4b5563; font-size: 15px;'>{student}  —  B-15 S-A</strong>"
                html += "</div>"
            else:
                # When stopped: Reveal Red/Green Badges
                color = "#22c55e" if status == "Present" else "#ef4444"
                bg_color = "#f8f9fa"
                html += f"<div style='padding: 10px; border-radius: 8px; background-color: {bg_color}; border-left: 4px solid {color}; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>"
                html += f"<strong style='color: #1e293b; font-size: 15px;'>{student}  —  B-15 S-A</strong>"
                html += f"<span style='color: white; font-weight: bold; background: {color}; padding: 4px 14px; border-radius: 12px; font-size: 13px;'>{status}</span>"
                html += "</div>"
        html += "</div>"
        attendance_placeholder.markdown(html, unsafe_allow_html=True)
        
    render_attendance()
    
    # Top Metrics Grid Placeholder
    metrics_placeholder = st.empty()
    
    # Analytics Pie Chart Placeholder
    st.markdown("---")
    alert_placeholder = st.empty()
    st.markdown("### Analytics")
    st.markdown("<div style='text-align: center;'><img src='https://quickchart.io/chart?c={type:%27doughnut%27,data:{datasets:[{data:[50,50],backgroundColor:[%27%23ef4444%27,%27%2322c55e%27]}]}}' width='150'></div>", unsafe_allow_html=True)

with main_col:
    if not st.session_state.run and mode == "📷 Live Camera Feed":
        face_placeholder.info("👋 Setup Ready.")
        yolo_placeholder.info("Click **Start Session** below to initiate.")
    elif not st.session_state.run and mode == "📷 Live Camera Feed" and "run" in st.session_state and not st.session_state.run:
        face_placeholder.warning("SYSTEM STOPPED.")
        yolo_placeholder.warning("Review Attendance Status.")

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

# ---------------- IMAGE MODE ----------------
if mode == "🖼 Upload Snapshot Analytics":
    with main_col:
        uploaded_file = st.file_uploader("Drop a classroom image here to analyze...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with st.spinner("Analyzing image for EduVision metrics..."):
            try:
                # Read image
                file_bytes = uploaded_file.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

                # Optional Resize if image is huge (e.g. 4K) to prevent memory crashes
                height, width = img.shape[:2]
                max_dim = 1920
                if max(height, width) > max_dim:
                    scale = max_dim / max(height, width)
                    img = cv2.resize(img, (int(width * scale), int(height * scale)))

                # YOLO prediction (matching live camera logic)
                results = model(img, conf=confidence, iou=0.45)
                count, acc = count_people(results, confidence)

                # Write record to CSV
                log_to_csv("Image Upload", count, overcrowd_limit)

                # Annotation
                annotated = results[0].plot()
                
                # Add text overlay
                cv2.putText(annotated, f"Detected: {count}", (30, 70),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 170), 4)

                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                # Show Result
                frame_placeholder.image(annotated, use_column_width=True, caption=f"Processing Complete | {width}x{height} original -> {img.shape[1]}x{img.shape[0]} processed")
                render_metrics(count, {})
                render_alert(count, overcrowd_limit)
                st.success("Log saved to `detection_logs.csv` successfully!")

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

            # AI Face Recognition (OpenCV)
            annotated_face = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            
            attendance_updated = False
            for (x, y, w, h) in faces:
                # Predict the face
                id_, dist = face_recognizer.predict(gray[y:y+h, x:x+w])
                
                # Lower distance means higher confidence for LBPH (typically < 85 is a good match)
                if dist < 85:
                    matched_name = student_names_list[id_]
                    if st.session_state.attendance[matched_name] != "Present":
                        st.session_state.attendance[matched_name] = "Present"
                        attendance_updated = True
                    
                    # Draw a green box and name for recognized faces
                    cv2.rectangle(annotated_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(annotated_face, matched_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    # Draw red outline for unknown/unconfident matches
                    cv2.rectangle(annotated_face, (x, y), (x+w, y+h), (0, 0, 255), 2)

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
            
            # Instantly update attendance list if a new student was recognized
            if attendance_updated:
                render_attendance()

            time.sleep(0.01) # Small sleep to prevent freezing Streamlit

    finally:
        # Guarantee camera resource is released when loop breaks or Stop is clicked
        cap.release()
        cv2.destroyAllWindows()