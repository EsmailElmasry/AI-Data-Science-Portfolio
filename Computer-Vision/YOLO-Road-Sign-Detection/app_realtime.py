"""
RoadSign AI — Real-Time Road Sign Detection using YOLOv8
A presentation-ready Streamlit application for a university
AI / Computer Vision / Data Science graduation project.

Run with:
    streamlit run app.py
"""

import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "best.pt"

CLASS_NAMES = {
    0: "crosswalk",
    1: "speedlimit",
    2: "stop",
    3: "trafficlight",
}

DISPLAY_NAMES = {
    "crosswalk": "Crosswalk",
    "speedlimit": "Speed Limit",
    "stop": "Stop",
    "trafficlight": "Traffic Light",
}

# Distinct professional color per class (BGR for OpenCV drawing)
CLASS_COLORS_BGR = {
    "crosswalk": (219, 152, 52),     # blue-ish
    "speedlimit": (60, 200, 240),    # amber
    "stop": (60, 60, 235),           # red
    "trafficlight": (110, 220, 60),  # green
}

# Same colors in hex, used for charts / CSS
CLASS_COLORS_HEX = {
    "Crosswalk": "#3498DB",
    "Speed Limit": "#F0A82E",
    "Stop": "#EB3C3C",
    "Traffic Light": "#3CDC6E",
}

ACCENT = "#22D3EE"
ACCENT_2 = "#3B82F6"
BG_DARK = "#0B1220"
BG_CARD = "#111A2C"
BG_CARD_SOFT = "#0F1A2E"
BORDER = "#1E2A42"
TEXT_MAIN = "#E7EDF7"
TEXT_MUTED = "#8FA0BF"


# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="RoadSign AI | YOLOv8 Road Sign Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------------------------------

def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 15% 0%, rgba(34,211,238,0.06), transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(59,130,246,0.08), transparent 40%),
                {BG_DARK};
        }}

        #MainMenu, footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{background: transparent;}}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }}

        /* ---------- Hero header ---------- */
        .hero-wrap {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
            padding: 1.6rem 2rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(34,211,238,0.10), rgba(59,130,246,0.05));
            border: 1px solid {BORDER};
            margin-bottom: 1.6rem;
        }}
        .hero-left {{ display: flex; align-items: center; gap: 0.9rem; }}
        .hero-icon {{
            width: 54px; height: 54px;
            display: flex; align-items: center; justify-content: center;
            border-radius: 14px;
            background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
            font-size: 26px;
            box-shadow: 0 8px 24px rgba(34,211,238,0.35);
        }}
        .hero-title {{
            font-size: 1.7rem;
            font-weight: 800;
            color: {TEXT_MAIN};
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .hero-subtitle {{
            font-size: 0.92rem;
            color: {TEXT_MUTED};
            margin-top: 2px;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            color: {ACCENT};
            background: rgba(34,211,238,0.10);
            border: 1px solid rgba(34,211,238,0.30);
        }}

        /* ---------- Section headers ---------- */
        .section-header {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {TEXT_MAIN};
            margin: 1.8rem 0 0.8rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .section-sub {{
            font-size: 0.88rem;
            color: {TEXT_MUTED};
            margin-top: -0.5rem;
            margin-bottom: 1rem;
        }}

        /* ---------- Cards ---------- */
        .card {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }}

        .metric-card {{
            background: linear-gradient(160deg, {BG_CARD} 0%, {BG_CARD_SOFT} 100%);
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            text-align: left;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(34,211,238,0.4);
        }}
        .metric-label {{
            font-size: 0.78rem;
            color: {TEXT_MUTED};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 1.65rem;
            font-weight: 800;
            color: {TEXT_MAIN};
        }}
        .metric-icon {{
            float: right;
            font-size: 1.4rem;
            opacity: 0.85;
        }}

        /* ---------- Image panels ---------- */
        .img-panel-title {{
            font-size: 0.9rem;
            font-weight: 700;
            color: {TEXT_MAIN};
            margin-bottom: 0.6rem;
            padding-left: 2px;
        }}

        /* ---------- Empty state ---------- */
        .empty-state {{
            border: 1.5px dashed {BORDER};
            border-radius: 20px;
            padding: 3.2rem 2rem;
            text-align: center;
            background: rgba(255,255,255,0.015);
            margin-top: 0.5rem;
        }}
        .empty-state .icon {{
            font-size: 2.6rem;
            margin-bottom: 0.6rem;
        }}
        .empty-state .title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {TEXT_MAIN};
        }}
        .empty-state .desc {{
            font-size: 0.88rem;
            color: {TEXT_MUTED};
            margin-top: 4px;
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: #0A0F1C;
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.6rem;
        }}
        .sidebar-logo {{
            display: flex; align-items: center; gap: 0.6rem;
            margin-bottom: 1.4rem;
        }}
        .sidebar-logo-icon {{
            width: 38px; height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
            display: flex; align-items: center; justify-content: center;
            font-size: 19px;
        }}
        .sidebar-logo-text {{
            font-size: 1.05rem;
            font-weight: 800;
            color: {TEXT_MAIN};
        }}
        .sidebar-section-title {{
            font-size: 0.78rem;
            font-weight: 700;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 1.1rem 0 0.5rem 0;
        }}
        .info-row {{
            display: flex; justify-content: space-between;
            font-size: 0.85rem;
            padding: 6px 0;
            border-bottom: 1px solid {BORDER};
            color: {TEXT_MUTED};
        }}
        .info-row b {{ color: {TEXT_MAIN}; font-weight: 600; }}

        /* ---------- Table ---------- */
        .styled-table-wrap {{
            border: 1px solid {BORDER};
            border-radius: 14px;
            overflow: hidden;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
        }}

        /* ---------- Class chip ---------- */
        .chip {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 8px;
            font-size: 0.78rem;
            font-weight: 700;
            color: #0B1220;
        }}

        /* ---------- Footer ---------- */
        .app-footer {{
            text-align: center;
            margin-top: 3rem;
            padding-top: 1.4rem;
            border-top: 1px solid {BORDER};
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            line-height: 1.6;
        }}
        .app-footer b {{ color: {TEXT_MAIN}; }}

        /* Uploader */
        [data-testid="stFileUploader"] section {{
            background: {BG_CARD};
            border: 1.5px dashed {BORDER};
            border-radius: 16px;
        }}

        /* Slider label */
        .stSlider label p {{
            color: {TEXT_MAIN} !important;
            font-weight: 600;
        }}

        /* Alerts */
        div[data-testid="stAlert"] {{
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# MODEL LOADING (cached — loaded once)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load the YOLOv8 model once and cache it across reruns."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model


# --------------------------------------------------------------------------
# INFERENCE HELPERS
# --------------------------------------------------------------------------

def run_inference(model, image_bgr: np.ndarray, conf: float, iou: float):
    """Run YOLOv8 inference and return the raw ultralytics Results object
    plus the measured processing time in seconds."""
    start = time.perf_counter()
    results = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    return results[0], elapsed


def draw_detections(image_bgr: np.ndarray, result) -> np.ndarray:
    """Draw professional bounding boxes + labels on a copy of the image."""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    thickness = max(2, round(min(h, w) / 400))
    font_scale = max(0.5, min(h, w) / 1100)

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        raw_name = CLASS_NAMES.get(cls_id, str(cls_id))
        label_name = DISPLAY_NAMES.get(raw_name, raw_name)
        color = CLASS_COLORS_BGR.get(raw_name, (0, 200, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = f"{label_name} {conf * 100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        label_y1 = max(0, y1 - th - baseline - 6)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 10, y1), color, -1, cv2.LINE_AA)
        text_color = (10, 10, 10)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 5 if y1 - 5 > 5 else y1 + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    return annotated



def process_uploaded_video(model, uploaded_file, conf: float, iou: float):
    """Run YOLOv8 on an uploaded video frame-by-frame."""
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src_file:
        src_file.write(uploaded_file.getbuffer())
        src_path = Path(src_file.name)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        src_path.unlink(missing_ok=True)
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(tempfile.gettempdir()) / f"roadsign_result_{int(time.time() * 1000)}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        src_path.unlink(missing_ok=True)
        raise ValueError("Could not create the output video.")

    total_frames = 0
    total_detections = 0
    highest_conf = 0.0
    start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, _ = run_inference(model, frame, conf, iou)
        writer.write(draw_detections(frame, result))

        total_frames += 1
        if result.boxes is not None and len(result.boxes) > 0:
            total_detections += len(result.boxes)
            highest_conf = max(
                highest_conf,
                max(float(c.item()) for c in result.boxes.conf),
            )

    cap.release()
    writer.release()
    src_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - start
    return output_path, total_frames, total_detections, highest_conf, elapsed


def render_webcam(model, conf: float, iou: float):
    """Real-time webcam detection using streamlit-webrtc."""
    try:
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
        import av
    except ImportError:
        st.error(
            "Webcam support needs streamlit-webrtc. "
            "Install it with: python -m pip install streamlit-webrtc av"
        )
        return

    class RoadSignVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            result, _ = run_inference(model, image, conf, iou)
            annotated = draw_detections(image, result)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="road-sign-webcam",
        video_processor_factory=RoadSignVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def build_detection_dataframe(result) -> pd.DataFrame:
    """Turn ultralytics results into a clean pandas DataFrame."""
    rows = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            raw_name = CLASS_NAMES.get(cls_id, str(cls_id))
            label_name = DISPLAY_NAMES.get(raw_name, raw_name)
            rows.append({"Detected Sign": label_name, "Confidence": conf})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Confidence", ascending=False).reset_index(drop=True)
        df.insert(0, "#", range(1, len(df) + 1))
        df["Confidence"] = df["Confidence"].apply(lambda v: f"{v * 100:.1f}%")
    return df


# --------------------------------------------------------------------------
# UI COMPONENTS
# --------------------------------------------------------------------------

def render_hero():
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-left">
                <div class="hero-icon">🚦</div>
                <div>
                    <p class="hero-title">RoadSign AI</p>
                    <p class="hero-subtitle">Real-Time Road Sign Detection using YOLOv8</p>
                </div>
            </div>
            <div class="badge">YOLOv8 • Computer Vision • Object Detection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="sidebar-logo-icon">🚦</div>
                <div class="sidebar-logo-text">RoadSign AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Model Information</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card" style="padding: 0.9rem 1rem;">
                <div class="info-row"><span>Model</span><b>YOLOv8n</b></div>
                <div class="info-row"><span>Task</span><b>Object Detection</b></div>
                <div class="info-row"><span>Classes</span><b>4</b></div>
                <div class="info-row" style="border-bottom:none;"><span>Weights</span><b>best.pt</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Detection Settings</div>', unsafe_allow_html=True)
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.01,
            help="Minimum confidence score required to keep a detection.",
        )
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.01,
            help="Intersection-over-Union threshold used for Non-Max Suppression.",
        )

        st.markdown('<div class="sidebar-section-title">Detectable Classes</div>', unsafe_allow_html=True)
        chip_html = "".join(
            f'<span class="chip" style="background:{CLASS_COLORS_HEX[name]}; margin: 3px 4px 3px 0;">{name}</span>'
            for name in DISPLAY_NAMES.values()
        )
        st.markdown(f'<div style="line-height:2.2;">{chip_html}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.78rem; color:{TEXT_MUTED}; line-height:1.5;">
            Built with Ultralytics YOLOv8, OpenCV and Streamlit.<br>
            Graduation Project — Computer Vision.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return conf_threshold, iou_threshold


def render_metric_card(col, label, value, icon):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="metric-icon">{icon}</span>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_empty_state():
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">🛣️</div>
            <div class="title">Upload a road image to start detection</div>
            <div class="desc">Supported formats: JPG, JPEG, PNG &nbsp;•&nbsp; The model will detect crosswalks, speed limits, stop signs and traffic lights.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_class_bar_chart(df_counts: pd.DataFrame):
    colors = [CLASS_COLORS_HEX.get(c, ACCENT) for c in df_counts["Class"]]
    fig = go.Figure(
        go.Bar(
            x=df_counts["Class"],
            y=df_counts["Count"],
            marker_color=colors,
            text=df_counts["Count"],
            textposition="outside",
            width=0.5,
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, family="Inter"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        xaxis=dict(showgrid=False, color=TEXT_MAIN),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False, dtick=1),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <b>RoadSign AI</b> | YOLOv8 Road Sign Detection<br>
            Computer Vision Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------

def main():
    inject_css()
    render_hero()
    conf_threshold, iou_threshold = render_sidebar()

    st.markdown('<div class="section-header">📤 Road Sign Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Upload an image and let YOLOv8 detect road signs automatically.</div>',
        unsafe_allow_html=True,
    )

    # ---- Model loading with error handling ----
    if not MODEL_PATH.exists():
        st.error(
            f"⚠️ Model file not found at `{MODEL_PATH.name}`. "
            f"Please place your trained `best.pt` file in the same directory as `app.py`."
        )
        render_footer()
        return

    try:
        with st.spinner("Loading YOLOv8 model..."):
            model = load_model(str(MODEL_PATH))
    except Exception as e:
        st.error(f"⚠️ Failed to load the model. Details: {e}")
        render_footer()
        return

    input_mode = st.radio(
        "Input Source",
        ["Image", "Video", "Webcam"],
        horizontal=True,
        help="Choose an image, uploaded video, or live webcam stream.",
    )

    if input_mode == "Webcam":
        st.markdown(
            '<div class="section-sub">Start your camera and detect road signs in real time.</div>',
            unsafe_allow_html=True,
        )
        render_webcam(model, conf_threshold, iou_threshold)
        render_footer()
        return

    if input_mode == "Video":
        st.markdown(
            '<div class="section-sub">Upload a video and run YOLOv8 detection frame-by-frame.</div>',
            unsafe_allow_html=True,
        )

        uploaded_video = st.file_uploader(
            "Drag and drop a video here, or click to browse",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV and MKV.",
            key="video_uploader",
        )

        if uploaded_video is None:
            st.info("Upload a video to start detection.")
            render_footer()
            return

        st.video(uploaded_video)

        if st.button("🚀 Run Video Detection", use_container_width=True):
            try:
                with st.spinner("Processing video frame-by-frame..."):
                    (
                        output_path,
                        total_frames,
                        total_detections,
                        highest_conf,
                        elapsed,
                    ) = process_uploaded_video(
                        model, uploaded_video, conf_threshold, iou_threshold
                    )

                st.success("Video detection completed successfully.")

                m1, m2, m3, m4 = st.columns(4)
                render_metric_card(m1, "Frames Processed", total_frames, "🎞️")
                render_metric_card(m2, "Total Detections", total_detections, "🎯")
                render_metric_card(
                    m3, "Highest Confidence", f"{highest_conf * 100:.1f}%", "⭐"
                )
                render_metric_card(m4, "Processing Time", f"{elapsed:.2f}s", "⚡")

                st.markdown(
                    '<div class="section-header">🎬 Detection Result</div>',
                    unsafe_allow_html=True,
                )
                st.video(str(output_path))

                with open(output_path, "rb") as video_file:
                    st.download_button(
                        "⬇️ Download Detection Video",
                        data=video_file.read(),
                        file_name="roadsign_detection_result.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"⚠️ Video processing failed. Details: {e}")

        render_footer()
        return

    # ---------------- Image mode (original functionality) ----------------
    st.markdown(
        '<div class="section-sub">Upload an image and let YOLOv8 detect road signs automatically.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drag and drop an image here, or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
        key="image_uploader",
    )

    if uploaded_file is None:
        render_empty_state()
        render_footer()
        return

    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("⚠️ Invalid image file. Please upload a valid JPG, JPEG or PNG image.")
        render_empty_state()
        render_footer()
        return

    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        with st.spinner("Running YOLOv8 inference..."):
            result, elapsed = run_inference(
                model, image_bgr, conf_threshold, iou_threshold
            )
    except Exception as e:
        st.error(f"⚠️ An error occurred during inference. Details: {e}")
        render_footer()
        return

    annotated_bgr = draw_detections(image_bgr, result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    num_detections = 0 if result.boxes is None else len(result.boxes)
    confidences = (
        [] if result.boxes is None else [float(c.item()) for c in result.boxes.conf]
    )
    highest_conf = max(confidences) * 100 if confidences else 0.0
    detected_class_ids = (
        set() if result.boxes is None else {int(c.item()) for c in result.boxes.cls}
    )

    st.markdown(
        '<div class="section-header">📊 Results Dashboard</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    render_metric_card(c1, "Total Detections", num_detections, "🎯")
    render_metric_card(c2, "Highest Confidence", f"{highest_conf:.1f}%", "⭐")
    render_metric_card(c3, "Detected Classes", len(detected_class_ids), "🏷️")
    render_metric_card(c4, "Processing Time", f"{elapsed:.2f}s", "⚡")

    st.markdown(
        '<div class="section-header">🖼️ Image Comparison</div>',
        unsafe_allow_html=True,
    )
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            '<div class="img-panel-title">Original Image</div>',
            unsafe_allow_html=True,
        )
        st.image(image_rgb, use_container_width=True)
    with col_right:
        st.markdown(
            '<div class="img-panel-title">Detection Result</div>',
            unsafe_allow_html=True,
        )
        st.image(annotated_rgb, use_container_width=True)

    st.markdown(
        '<div class="section-header">📋 Detection Table</div>',
        unsafe_allow_html=True,
    )
    df = build_detection_dataframe(result)
    if df.empty:
        st.info("No road signs detected.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    if not df.empty:
        st.markdown(
            '<div class="section-header">📈 Detection Summary</div>',
            unsafe_allow_html=True,
        )
        raw_names = [
            CLASS_NAMES.get(int(c.item()), str(int(c.item())))
            for c in result.boxes.cls
        ]
        display_series = pd.Series([DISPLAY_NAMES.get(n, n) for n in raw_names])
        counts = display_series.value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        render_class_bar_chart(counts)

    render_footer()
    return



if __name__ == "__main__":
    main()
