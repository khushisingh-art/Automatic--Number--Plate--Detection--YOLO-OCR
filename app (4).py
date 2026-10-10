
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import easyocr
import re
import tempfile
import os
from collections import Counter, defaultdict

from PIL import Image
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Automatic Number Plate Detection and Recognition",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS DESIGN
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

.stApp {
    background: radial-gradient(circle at top left, #172f50, #08111f 48%, #050b14);
    color: #f4f7ff;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

.hero {
    padding: 35px;
    border-radius: 24px;
    background: linear-gradient(120deg, #153c62, #172347, #302452);
    border: 1px solid #34577e;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px #00000035;
}

.hero h1 {
    font-size: 38px;
    font-weight: 700;
    color: white;
}

.hero p {
    color: #c3d5ed;
    font-size: 16px;
}

.card {
    background: linear-gradient(145deg, #14263d, #0e1a2b);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #263e5c;
    min-height: 120px;
}

.card-title {
    color: #9bb2ce;
    font-size: 14px;
}

.card-value {
    color: #53d8ff;
    font-size: 27px;
    font-weight: bold;
    margin-top: 10px;
}

.section-title {
    font-size: 24px;
    font-weight: bold;
    margin-top: 25px;
    margin-bottom: 10px;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #087fae, #5368e8);
    color: white;
    border: none;
    padding: 12px;
    font-weight: bold;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #079bcf, #687cff);
    color: white;
    transform: translateY(-2px);
}

div[data-testid="stFileUploader"] {
    background: #102038;
    border-radius: 15px;
    padding: 12px;
}

div[data-testid="stSidebar"] {
    background: #0b1728;
}

div[data-testid="stSidebar"] * {
    color: #edf5ff;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "processed_count" not in st.session_state:
    st.session_state.processed_count = 0

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ==========================================
# LOAD YOLO AND EASYOCR MODELS
# ==========================================

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"


@st.cache_resource
def load_models():
    model = YOLO(str(MODEL_PATH))
    reader = easyocr.Reader(["en"], gpu=False)
    return model, reader


# ==========================================
# NUMBER PLATE DETECTION FUNCTION
# ==========================================

def detect_number_plates(image_rgb, confidence):
    model, reader = load_models()

    results = model.predict(
        source=image_rgb,
        conf=confidence,
        verbose=False
    )

    annotated_bgr = cv2.cvtColor(
        image_rgb.copy(),
        cv2.COLOR_RGB2BGR
    )

    detected_plates = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )

            yolo_confidence = float(box.conf[0].cpu())

            height, width = image_rgb.shape[:2]

            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = image_rgb[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            enlarged = cv2.resize(
                plate_crop,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            gray = cv2.cvtColor(
                enlarged, cv2.COLOR_RGB2GRAY
            )

            ocr_results = reader.readtext(
                gray,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            raw_text = "".join(
                item[1] for item in ocr_results
            ).upper()

            plate_text = re.sub(
                r"[^A-Z0-9]", "", raw_text
            )

            ocr_confidence = (
                float(np.mean([item[2] for item in ocr_results]))
                if ocr_results else 0.0
            )

            display_text = plate_text or "Plate detected"

            cv2.rectangle(
                annotated_bgr,
                (x1, y1),
                (x2, y2),
                (0, 220, 255),
                3
            )

            cv2.putText(
                annotated_bgr,
                display_text[:30],
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
                cv2.LINE_AA
            )

            detected_plates.append({
                "plate_text": plate_text or "Not recognized",
                "yolo_confidence": yolo_confidence,
                "ocr_confidence": ocr_confidence,
                "crop": plate_crop
            })

    annotated_rgb = cv2.cvtColor(
        annotated_bgr, cv2.COLOR_BGR2RGB
    )

    return annotated_rgb, detected_plates


# ==========================================
# VIDEO PROCESSING FUNCTION
# ==========================================

def process_video(video_bytes, confidence):
    input_path = None
    output_path = None
    cap = None
    writer = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(video_bytes)
            input_path = temp_input.name

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Video open nahi hui. Another MP4/AVI video try karein.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 25.0
        if original_width <= 0 or original_height <= 0:
            raise ValueError("Invalid video dimensions.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            output_path = temp_output.name

        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (original_width, original_height)
        )
        if not writer.isOpened():
            raise ValueError("Output video create nahi ho paayi.")

        model, reader = load_models()
        # Keep recent OCR readings for each tracked vehicle/plate.
        track_readings = defaultdict(list)
        track_locked_text = {}
        unique_plates = {}
        frame_count = 0

        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_placeholder = st.empty()

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            scale = min(1.0, 1280 / max(1, frame_rgb.shape[1]))
            if scale < 1.0:
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (int(frame_rgb.shape[1] * scale), int(frame_rgb.shape[0] * scale))
                )

            # ByteTrack maintains IDs between nearby frames; OCR voting reduces flicker.
            results = model.track(
                source=frame_rgb, conf=confidence, imgsz=1280,
                persist=True, tracker="bytetrack.yaml", verbose=False
            )

            annotated_bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
                    det_conf = float(box.conf[0].cpu())
                    h, w = frame_rgb.shape[:2]
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame_rgb[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # OCR is run on an enlarged, contrast-enhanced plate crop.
                    enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(enlarged, cv2.COLOR_RGB2GRAY)
                    gray = cv2.equalizeHist(gray)
                    ocr_results = reader.readtext(
                        gray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        detail=1, paragraph=False
                    )
                    raw_text = "".join(item[1] for item in ocr_results).upper()
                    plate_text = re.sub(r"[^A-Z0-9]", "", raw_text)
                    ocr_conf = float(np.mean([item[2] for item in ocr_results])) if ocr_results else 0.0

                    # A stable track ID is preferred; use a temporary per-frame key if unavailable.
                    key = str(track_id) if track_id is not None else f"frame{frame_count}_{x1}_{y1}"
                    if plate_text and len(plate_text) >= 4:
                        track_readings[key].append(plate_text)
                        track_readings[key] = track_readings[key][-9:]
                        counts = Counter(track_readings[key])
                        candidate, votes = counts.most_common(1)[0]
                        # Lock only after repeated agreement; otherwise keep showing verification.
                        if votes >= 3:
                            track_locked_text[key] = candidate

                        shown_text = track_locked_text.get(key, "Verifying...")
                        if key in track_locked_text:
                            stable = track_locked_text[key]
                            if stable not in unique_plates:
                                unique_plates[stable] = {
                                    "yolo_confidence": det_conf,
                                    "ocr_confidence": ocr_conf
                                }
                            else:
                                unique_plates[stable]["yolo_confidence"] = max(
                                    unique_plates[stable]["yolo_confidence"], det_conf
                                )
                                unique_plates[stable]["ocr_confidence"] = max(
                                    unique_plates[stable]["ocr_confidence"], ocr_conf
                                )
                    else:
                        shown_text = track_locked_text.get(key, "Verifying...")

                    cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 220, 255), 2)
                    cv2.putText(
                        annotated_bgr, shown_text[:30], (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA
                    )

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            if annotated_rgb.shape[1] != original_width or annotated_rgb.shape[0] != original_height:
                annotated_rgb = cv2.resize(annotated_rgb, (original_width, original_height))
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

            if frame_count == 1 or frame_count % 10 == 0:
                preview_placeholder.image(
                    annotated_rgb,
                    caption=f"Video processing preview — Frame {frame_count}",
                    use_container_width=True
                )
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.write(
                f"Processing frame {frame_count} / {total_frames if total_frames > 0 else 'unknown'}"
            )

        progress_bar.empty()
        status_text.empty()
        preview_placeholder.empty()

        if frame_count == 0:
            raise ValueError("Video mein readable frames nahi mile.")

        return output_path, unique_plates, frame_count

    except Exception:
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if input_path and os.path.exists(input_path):
            os.remove(input_path)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("## 🚘 ANPR STUDIO")
    st.caption("Automatic Number Plate Detection")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📋 Recognition History",
            "ℹ️ About Project"
        ]
    )

    st.divider()
    st.markdown("### ⚙️ Settings")

    show_preview = st.toggle(
        "Show image preview",
        value=True
    )

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.99,
        value=0.50,
        step=0.05
    )

    st.divider()
    st.caption(
        "Built with Python, OpenCV, EasyOCR and Streamlit."
    )


# ==========================================
# MAIN HEADER
# ==========================================
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="ANPR Studio",
    page_icon="🚘",
    layout="wide"
)

banner = Path("assets/anpr_banner.png")

if banner.exists():
    st.image(str(banner), use_container_width=True)

st.title("🚘 Automatic Number Plate Detection and Recognition")
st.caption(
    "AI-powered vehicle detection and number plate recognition"
)

st.info(
    "Upload a vehicle image or video to begin detection."
)


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">📊 Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📁 Files Processed</div>
            <div class="card-value">{st.session_state.processed_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🧾 Saved Results</div>
            <div class="card-value">{len(st.session_state.history)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🎯 Confidence Setting</div>
            <div class="card-value">{confidence:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🖼️ Upload Vehicle Image or Video</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a vehicle image or video",
        type=["jpg", "jpeg", "png", "jfif", "mp4", "avi", "mov", "mkv"],
        help="Upload a vehicle image or video for number plate recognition."
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()
        upload_id = uploaded_file.name + str(len(file_bytes))

        extension = Path(uploaded_file.name).suffix.lower()

        image_extensions = {".jpg", ".jpeg", ".png",".jfif"}
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

        # --------------------------------------
        # IMAGE WORKFLOW
        # --------------------------------------

        if extension in image_extensions:

            try:
                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                image_array = np.array(image)

                left, right = st.columns(
                    [1.2, 0.8], gap="large"
                )

                with left:
                    if show_preview:
                        st.image(
                            image,
                            caption="Uploaded Vehicle Image",
                            use_container_width=True
                        )

                    st.success("Image uploaded successfully!")

                with right:
                    st.markdown("### 🔍 Image Information")
                    st.write(f"File name: {uploaded_file.name}")
                    st.write(f"Image width: {image.width} px")
                    st.write(f"Image height: {image.height} px")
                    st.write(f"Detection threshold: {confidence:.0%}")

                    analyse = st.button(
                        "🚀 Analyse Number Plate",
                        key="analyse_image",
                        use_container_width=True
                    )

                    if analyse:
                        if not MODEL_PATH.exists():
                            st.error(
                                f"Model file not found: {MODEL_PATH}. "
                                "Place best.pt in the same folder as app.py."
                            )
                        else:
                            try:
                                with st.spinner(
                                    "YOLO is detecting plates and EasyOCR "
                                    "is reading the text..."
                                ):
                                    result_image, plates = detect_number_plates(
                                        image_array, confidence
                                    )

                                st.session_state.processed_count += 1

                                st.session_state.last_result = {
                                    "upload_id": upload_id,
                                    "image": result_image,
                                    "plates": plates
                                }

                                timestamp = datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )

                                st.session_state.history.append({
                                    "Date and Time": timestamp,
                                    "Image": uploaded_file.name,
                                    "Plates Detected": len(plates),
                                    "Recognized Text": ", ".join(
                                        p["plate_text"] for p in plates
                                    ) if plates else "No plate detected",
                                    "Best YOLO Confidence": f"{max(
                                        (p['yolo_confidence'] for p in plates),
                                        default=0
                                    ):.1%}"
                                })

                            except Exception as e:
                                st.error(f"Detection failed: {e}")

                # Display saved image result
                last = st.session_state.last_result

                if (
                    last is not None
                    and last.get("upload_id") == upload_id
                    and "image" in last
                ):
                    st.markdown("### 📝 Recognition Results")

                    result_col, details_col = st.columns([1.2, 0.8])

                    with result_col:
                        st.image(
                            last["image"],
                            caption="Detected Number Plate",
                            use_container_width=True
                        )

                        success, encoded = cv2.imencode(
                            ".jpg",
                            cv2.cvtColor(
                                last["image"], cv2.COLOR_RGB2BGR
                            )
                        )

                        if success:
                            st.download_button(
                                "⬇️ Download Detection Image",
                                data=encoded.tobytes(),
                                file_name="number_plate_result.jpg",
                                mime="image/jpeg"
                            )

                    with details_col:
                        st.metric(
                            "Plates Detected", len(last["plates"])
                        )

                        if last["plates"]:
                            for i, plate in enumerate(
                                last["plates"], start=1
                            ):
                                with st.container(border=True):
                                    st.markdown(f"#### 🚘 Plate {i}")
                                    st.code(plate["plate_text"])
                                    st.write(
                                        f"YOLO confidence: "
                                        f"{plate['yolo_confidence']:.1%}"
                                    )
                                    st.write(
                                        f"OCR confidence: "
                                        f"{plate['ocr_confidence']:.1%}"
                                    )
                                    st.image(
                                        plate["crop"],
                                        caption="Plate crop",
                                        use_container_width=True
                                    )
                        else:
                            st.warning(
                                "No plate detected. Try a clearer image "
                                "or adjust the confidence slider."
                            )

                    st.caption(
                        "OCR can confuse similar characters such as O/0 "
                        "and I/1. Verify the result manually."
                    )

            except Exception as e:
                st.error(f"Could not open image: {e}")

        # --------------------------------------
        # VIDEO WORKFLOW
        # --------------------------------------

        elif extension in video_extensions:

            st.markdown("### 🎬 Video Information")
            st.write(f"File name: {uploaded_file.name}")
            st.write(f"File size: {len(file_bytes) / (1024 * 1024):.2f} MB")

            if show_preview:
                st.video(file_bytes)

            st.info(
                "Upload complete. Click Analyse Video to detect number "
                "plates frame by frame and recognise their text."
            )

            analyse_video = st.button(
                "🚀 Analyse Video",
                key="analyse_video",
                use_container_width=True
            )

            if analyse_video:

                if not MODEL_PATH.exists():
                    st.error(
                        f"Model file not found: {MODEL_PATH}. "
                        "Place best.pt in the same folder as app.py."
                    )

                else:
                    output_path = None

                    try:
                        with st.spinner(
                            "Processing video frames with YOLO + EasyOCR. "
                            "Please wait..."
                        ):
                            output_path, unique_plates, frame_count = (
                                process_video(file_bytes, confidence)
                            )

                        # Read output into memory before removing temp file
                        with open(output_path, "rb") as video_file:
                            output_bytes = video_file.read()

                        st.session_state.processed_count += 1

                        timestamp = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                        recognised_text = (
                            ", ".join(unique_plates.keys())
                            if unique_plates
                            else "No plate detected"
                        )

                        best_confidence = max(
                            (
                                info["yolo_confidence"]
                                for info in unique_plates.values()
                            ),
                            default=0.0
                        )

                        st.session_state.history.append({
                            "Date and Time": timestamp,
                            "Image": uploaded_file.name,
                            "Plates Detected": len(unique_plates),
                            "Recognized Text": recognised_text,
                            "Best YOLO Confidence": f"{best_confidence:.1%}"
                        })

                        st.session_state.last_result = {
                            "upload_id": upload_id,
                            "video_bytes": video_bytes if False else None
                        }

                        st.success(
                            f"Video processed successfully! "
                            f"Frames processed: {frame_count}"
                        )

                        st.markdown("### 🎥 Processed Video")

                        st.video(output_bytes)

                        st.download_button(
                            "⬇️ Download Processed Video",
                            data=output_bytes,
                            file_name="number_plate_detection.mp4",
                            mime="video/mp4",
                            key="download_processed_video"
                        )

                        st.markdown("### 🧾 Recognised Number Plates")

                        if unique_plates:
                            for i, (text, info) in enumerate(
                                unique_plates.items(), start=1
                            ):
                                with st.container(border=True):
                                    st.markdown(f"#### 🚘 Plate {i}")
                                    st.code(text)
                                    st.write(
                                        "Best YOLO confidence: "
                                        f"{info['yolo_confidence']:.1%}"
                                    )
                                    st.write(
                                        "OCR confidence: "
                                        f"{info['ocr_confidence']:.1%}"
                                    )
                        else:
                            st.warning(
                                "No readable number plate was found. "
                                "Try a clearer video or adjust confidence."
                            )

                    except Exception as e:
                        st.error(f"Video processing failed: {e}")

                    finally:
                        if output_path and os.path.exists(output_path):
                            os.remove(output_path)

        else:
            st.error("Unsupported file format.")

    else:
        st.markdown("""
        <div class="card">
            <h3>🚗 Ready to Get Started?</h3>
            <p>Upload a vehicle image or video to begin recognition.</p>
            <p style="color:#9bb2ce;">
                Supported: JPG, JPEG, JFIF, PNG, MP4, AVI, MOV and MKV.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# HISTORY PAGE
# ==========================================

elif page == "📋 Recognition History":

    st.markdown("## 📋 Recognition History")
    st.write("Review results saved during this session.")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download History CSV",
            data=csv_data,
            file_name="recognition_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.success("History cleared.")
            st.rerun()

    else:
        st.info("No recognition results have been saved yet.")


# ==========================================
# ABOUT PAGE
# ==========================================

elif page == "ℹ️ About Project":

    st.markdown("## ℹ️ About the Project")

    st.write("""
    Automatic Number Plate Detection and Recognition is a
    computer-vision application designed to locate vehicle
    registration plates and read their characters from images
    and videos.
    """)

    st.markdown("### 🧠 Technologies")

    st.markdown("""
    - **Python:** Core programming language
    - **YOLO:** Number plate detection
    - **OpenCV:** Image and video processing
    - **EasyOCR:** Character recognition
    - **Streamlit:** Interactive web interface
    - **Pandas:** Tabular history and CSV export
    """)

    st.markdown("### 🔄 Workflow")

    st.markdown("""
    1. Upload a vehicle image or video.
    2. Detect the number plate.
    3. Extract the plate region.
    4. Recognise characters using OCR.
    5. Review or download the processed result.
    """)

    st.warning(
        "Recognition can be incorrect in blurry, dark or angled "
        "images and videos. Verify results manually."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center;color:#8198b5;padding:10px;">
    🚘 Automatic Number Plate Detection and Recognition
    <br>
    Smart Vision • Intelligent Recognition • Python
</div>
""", unsafe_allow_html=True)
