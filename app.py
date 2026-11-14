"""SPAZE — Phase 1: Foundation & Floor-Plan Image Processing.

Streamlit Application providing local, deterministic computer vision floor-plan processing,
candidate room extraction, interactive scale calibration, manual human corrections,
and structured JSON schema export.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np

from config.settings import AppConfig, ProcessingConfig, RoomConfig
from models.floorplan_schema import (
    DetectionStatus,
    FloorPlan,
    MeasurementSource,
    Room,
    ScaleInfo,
)
from modules.floorplan_parser import FloorPlanParser
from modules.image_loader import ImageLoadError, LoadedImage, load_image_from_bytes, load_image_from_path
from modules.image_processor import ImageProcessor, ProcessedStages
from modules.room_editor import RoomEditor
from modules.room_extractor import RoomExtractor
from modules.wall_detector import WallDetector
from utils.export import export_floorplan_to_json, generate_export_summary
from utils.validation import ValidationError
from utils.visualization import (
    convert_single_channel_to_rgb,
    overlay_candidate_rooms,
    overlay_wall_candidates,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{AppConfig.APP_NAME} — Architectural Layout Processing",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Architectural Theme Styles
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark charcoal and deep slate styling */
    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Container */
    .spaze-header-container {
        padding: 1.5rem 1.8rem;
        background: linear-gradient(135deg, rgba(22, 28, 42, 0.95) 0%, rgba(13, 17, 26, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }
    .spaze-title {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: 2.5px;
        background: linear-gradient(90deg, #00d2ff 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: inline-block;
    }
    .spaze-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        border-radius: 9999px;
        background-color: rgba(99, 102, 241, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        margin-left: 1rem;
        vertical-align: middle;
    }
    .spaze-tagline {
        font-size: 0.95rem;
        font-weight: 500;
        color: #94a3b8;
        letter-spacing: 1.5px;
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .spaze-desc {
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.45;
        margin: 0;
    }

    /* Metric Card Styling */
    .metric-card {
        background-color: #161c28;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: left;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Status Badges */
    .badge-confirmed {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-candidate {
        background-color: rgba(52, 152, 219, 0.15);
        color: #3498db;
        border: 1px solid rgba(52, 152, 219, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-review {
        background-color: rgba(243, 156, 18, 0.15);
        color: #f39c12;
        border: 1px solid rgba(243, 156, 18, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-rejected {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header() -> None:
    """Render top branding banner."""
    st.markdown(
        f"""
        <div class="spaze-header-container">
            <div>
                <span class="spaze-title">{AppConfig.APP_NAME}</span>
                <span class="spaze-badge">{AppConfig.PHASE}</span>
            </div>
            <div class="spaze-tagline">{AppConfig.TAGLINE}</div>
            <p class="spaze-desc">
                High-precision architectural floor-plan processing powered by deterministic local computer vision.
                Extracts wall candidates and candidate room boundaries, provides interactive scale calibration,
                enables human-in-the-loop geometric corrections, and produces structured floor-plan schemas ready for
                generative AI and 3D modeling.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Initialize state containers if not present."""
    if "loaded_image" not in st.session_state:
        st.session_state.loaded_image = None
    if "processed_stages" not in st.session_state:
        st.session_state.processed_stages = None
    if "wall_candidates" not in st.session_state:
        st.session_state.wall_candidates = []
    if "candidate_rooms" not in st.session_state:
        st.session_state.candidate_rooms = []
    if "floorplan" not in st.session_state:
        st.session_state.floorplan = None
    if "scale_info" not in st.session_state:
        st.session_state.scale_info = ScaleInfo()
    if "sample_loaded" not in st.session_state:
        st.session_state.sample_loaded = False


def trigger_cv_pipeline(
    loaded: LoadedImage,
    adaptive_method: str,
    canny_th1: float,
    canny_th2: float,
    use_bilateral: bool,
    hough_min_len: int,
    hough_gap: int,
    hough_thresh: int,
) -> None:
    """Execute OpenCV pipeline and populate session state."""
    with st.spinner("Processing floor-plan with local computer vision pipeline..."):
        # 1. Preprocessing stages
        processor = ImageProcessor()
        stages = processor.process(
            loaded.image_rgb,
            use_bilateral=use_bilateral,
            adaptive_method=adaptive_method,
            canny_th1=canny_th1,
            canny_th2=canny_th2,
        )

        # 2. Wall Line Candidates
        wall_detector = WallDetector()
        walls = wall_detector.detect_walls(
            stages.morphology,
            min_line_length=hough_min_len,
            max_line_gap=hough_gap,
            hough_threshold=hough_thresh,
        )

        # 3. Candidate Room Extraction
        room_extractor = RoomExtractor()
        extracted_candidates = room_extractor.extract_rooms(
            stages.morphology,
            image_width=loaded.width,
            image_height=loaded.height,
        )

        # 4. Structured FloorPlan assembly
        floorplan = FloorPlanParser.build_initial_floorplan(
            loaded_image=loaded,
            processed_stages=stages,
            wall_candidates=walls,
            candidate_rooms=extracted_candidates,
            scale_info=st.session_state.scale_info,
        )

        # Update Session State
        st.session_state.loaded_image = loaded
        st.session_state.processed_stages = stages
        st.session_state.wall_candidates = walls
        st.session_state.candidate_rooms = extracted_candidates
        st.session_state.floorplan = floorplan


def main() -> None:
    initialize_session_state()
    render_header()

    # -------------------------------------------------------------------------
    # Sidebar Controls: CV Parameters & Scale Calibration
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### ⚙️ Pipeline Parameters")

    with st.sidebar.expander("Binarization & Edges", expanded=False):
        adaptive_method = st.selectbox(
            "Thresholding Method",
            options=["gaussian", "mean", "otsu"],
            index=0,
            help="Adaptive Gaussian or Mean accommodates lighting/scan unevenness. Otsu computes global bimodal threshold.",
        )
        use_bilateral = st.checkbox(
            "Use Bilateral Denoising",
            value=False,
            help="Edge-preserving filter. Useful for noisy paper textures.",
        )
        canny_th1 = st.slider("Canny Low Threshold", 10.0, 200.0, 50.0, 5.0)
        canny_th2 = st.slider("Canny High Threshold", 50.0, 300.0, 150.0, 5.0)

    with st.sidebar.expander("Wall Line Parameters", expanded=False):
        hough_min_len = st.slider("Min Wall Length (px)", 10, 200, 40, 5)
        hough_gap = st.slider("Max Wall Gap (px)", 2, 40, 10, 2)
        hough_thresh = st.slider("Hough Accumulator Thresh", 20, 150, 60, 5)

    # -------------------------------------------------------------------------
    # Section A & B: Image Upload & Sample Selector
    # -------------------------------------------------------------------------
    upload_col, sample_col = st.columns([3, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload 2D Architectural Floor Plan (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Images are decoded and processed locally in memory. No data is sent to external services.",
        )

    with sample_col:
        st.write("")
        st.write("")
        sample_btn = st.button("📂 Load Sample Test Fixture", use_container_width=True)

    loaded_image: Optional[LoadedImage] = None

    if sample_btn:
        sample_path = Path("sample_data/sample_floorplan.png")
        if sample_path.exists():
            try:
                loaded_image = load_image_from_path(sample_path)
                st.session_state.sample_loaded = True
                trigger_cv_pipeline(
                    loaded=loaded_image,
                    adaptive_method=adaptive_method,
                    canny_th1=canny_th1,
                    canny_th2=canny_th2,
                    use_bilateral=use_bilateral,
                    hough_min_len=hough_min_len,
                    hough_gap=hough_gap,
                    hough_thresh=hough_thresh,
                )
                st.success("Loaded and processed sample floor-plan fixture successfully!")
            except ImageLoadError as err:
                st.error(f"Failed to load sample fixture: {err}")
        else:
            st.error("Sample fixture not found at sample_data/sample_floorplan.png")

    elif uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            loaded_image = load_image_from_bytes(
                raw_bytes=file_bytes,
                filename=uploaded_file.name,
            )
            # Re-run pipeline if image changed
            current_sha = st.session_state.loaded_image.sha256 if st.session_state.loaded_image else None
            if current_sha != loaded_image.sha256:
                st.session_state.sample_loaded = False
                trigger_cv_pipeline(
                    loaded=loaded_image,
                    adaptive_method=adaptive_method,
                    canny_th1=canny_th1,
                    canny_th2=canny_th2,
                    use_bilateral=use_bilateral,
                    hough_min_len=hough_min_len,
                    hough_gap=hough_gap,
                    hough_thresh=hough_thresh,
                )
        except ImageLoadError as load_err:
            st.error(f"⚠️ Image Load Error: {load_err}")
            return
        except Exception as exc:
            st.error(f"Unexpected error validating image: {exc}")
            return

    # If no image is active yet, display guidance instructions
    if st.session_state.loaded_image is None:
        st.info("👆 Please upload an architectural floor-plan image or click 'Load Sample Test Fixture' to begin.")
        return

    loaded = st.session_state.loaded_image
    stages = st.session_state.processed_stages
    floorplan: FloorPlan = st.session_state.floorplan

    # -------------------------------------------------------------------------
    # Section C & E: Status Dashboard & High-Level Metrics
    # -------------------------------------------------------------------------
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Candidates</div>
                <div class="metric-value">{len(floorplan.candidate_rooms)}</div>
                <div class="metric-sub">Auto-detected regions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Confirmed</div>
                <div class="metric-value" style="color: #2ecc71;">{len(floorplan.confirmed_rooms)}</div>
                <div class="metric-sub">User verified rooms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        review_count = sum(1 for r in floorplan.candidate_rooms if r.detection_status == DetectionStatus.NEEDS_REVIEW.value)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Needs Review</div>
                <div class="metric-value" style="color: #f39c12;">{review_count}</div>
                <div class="metric-sub">Irregular / borderline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        scale_label = f"{floorplan.scale_information.pixels_per_meter:.1f} px/m" if floorplan.scale_information.is_calibrated else "Uncalibrated"
        scale_color = "#38bdf8" if floorplan.scale_information.is_calibrated else "#94a3b8"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Scale Calibration</div>
                <div class="metric-value" style="color: {scale_color}; font-size: 1.25rem;">{scale_label}</div>
                <div class="metric-sub">{"Meters enabled" if floorplan.scale_information.is_calibrated else "Pixels only"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">CV Latency</div>
                <div class="metric-value">{stages.processing_time_ms:.1f} <span style="font-size: 0.9rem;">ms</span></div>
                <div class="metric-sub">Resolution: {loaded.width}x{loaded.height}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # -------------------------------------------------------------------------
    # Scale Calibration Tool
    # -------------------------------------------------------------------------
    with st.expander("📏 Physical Scale Calibration (Pixels to Real-World Meters)", expanded=not floorplan.scale_information.is_calibrated):
        cal_col1, cal_col2, cal_col3 = st.columns([2, 2, 1.5])
        with cal_col1:
            cal_px = st.number_input(
                "Known Pixel Distance",
                min_value=1.0,
                max_value=float(max(loaded.width, loaded.height)),
                value=680.0 if st.session_state.sample_loaded else 200.0,
                step=10.0,
                help="Pixel span of a known wall segment or dimension line.",
            )
        with cal_col2:
            cal_m = st.number_input(
                "Real-World Length (meters)",
                min_value=0.1,
                max_value=200.0,
                value=13.6 if st.session_state.sample_loaded else 4.0,
                step=0.5,
                help="Real-world length in meters corresponding to the pixel distance above.",
            )
        with cal_col3:
            st.write("")
            st.write("")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Apply Scale", use_container_width=True):
                    try:
                        floorplan.update_scale(pixel_distance=cal_px, real_meters=cal_m)
                        st.session_state.scale_info = floorplan.scale_information
                        st.success(f"Scale calibrated: {floorplan.scale_information.pixels_per_meter:.2f} px/m")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Calibration failed: {err}")
            with btn_col2:
                if st.button("Reset Scale", use_container_width=True):
                    floorplan.scale_information.reset()
                    for r in floorplan.candidate_rooms + floorplan.confirmed_rooms:
                        r.apply_scale(floorplan.scale_information)
                    st.session_state.scale_info = floorplan.scale_information
                    st.info("Scale reset to uncalibrated pixel mode.")
                    st.rerun()

        if floorplan.scale_information.is_calibrated:
            st.caption(f"ℹ️ {floorplan.scale_information.calibration_notes}")
        else:
            st.caption("ℹ️ Uncalibrated. All coordinates and dimensions are strictly recorded in image pixels.")

    # -------------------------------------------------------------------------
    # Section D: Computer Vision Pipeline Stages
    # -------------------------------------------------------------------------
    st.markdown("### 🔍 Computer Vision Inspection & Intermediate Stages")
    tab_annotated, tab_walls, tab_thresh, tab_edges, tab_morph, tab_gray, tab_orig = st.tabs([
        "🏷️ Annotated Rooms",
        "🧱 Wall Candidates",
        "🔲 Threshold / Binary",
        "⚡ Canny Edges",
        "🔬 Morphology",
        "🔘 Grayscale",
        "🖼️ Source Original",
    ])

    all_current_rooms = floorplan.candidate_rooms + floorplan.confirmed_rooms

    with tab_annotated:
        annotated_img = overlay_candidate_rooms(
            base_image_rgb=stages.original_rgb,
            rooms=all_current_rooms,
            scale_info=floorplan.scale_information,
        )
        st.image(
            annotated_img,
            caption=f"Detected Room Boundaries (Confirmed: {len(floorplan.confirmed_rooms)}, Candidates: {len(floorplan.candidate_rooms)})",
            use_container_width=True,
        )

    with tab_walls:
        walls_overlay = overlay_wall_candidates(
            base_image_rgb=stages.original_rgb,
            walls=floorplan.wall_candidates,
        )
        st.image(
            walls_overlay,
            caption=f"Hough Wall Candidates ({len(floorplan.wall_candidates)} detected: Cyan=Horizontal, Magenta=Vertical, Gold=Diagonal)",
            use_container_width=True,
        )

    with tab_thresh:
        thresh_rgb = convert_single_channel_to_rgb(stages.threshold)
        st.image(thresh_rgb, caption="Inverted Binary Wall Mask (Walls=255, Rooms=0)", use_container_width=True)

    with tab_edges:
        edges_rgb = convert_single_channel_to_rgb(stages.edges)
        st.image(edges_rgb, caption="Canny Edge Map", use_container_width=True)

    with tab_morph:
        morph_rgb = convert_single_channel_to_rgb(stages.morphology)
        st.image(morph_rgb, caption="Morphologically Closed Wall Mask (Bridges gaps between wall lines)", use_container_width=True)

    with tab_gray:
        gray_rgb = convert_single_channel_to_rgb(stages.grayscale)
        st.image(gray_rgb, caption="Grayscale Conversion", use_container_width=True)

    with tab_orig:
        st.image(
            stages.original_rgb,
            caption=f"Source Image ({loaded.width}x{loaded.height} px, {loaded.format}, SHA: {loaded.sha256[:12]}...)",
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # Section F: Human-in-the-Loop Room Editor & Table
    # -------------------------------------------------------------------------
    st.markdown("### ✍️ Room Correction & Human Verification")

    st.write(
        "Review extracted candidate rooms below. You can edit names, change room types, "
        "adjust coordinates or dimensions, accept/reject candidates, or add missing rooms manually."
    )

    edit_col, table_col = st.columns([1.2, 2.0])

    with edit_col:
        st.markdown("#### Room Edit Panel")
        room_options = {r.room_id: f"{r.label} ({r.room_id}) — {r.detection_status}" for r in all_current_rooms}

        if not room_options:
            st.warning("No rooms detected yet. Use the manual addition form below.")
        else:
            selected_id = st.selectbox(
                "Select Room to Edit",
                options=list(room_options.keys()),
                format_func=lambda rid: room_options[rid],
            )

            # Find selected room
            active_room = next((r for r in all_current_rooms if r.room_id == selected_id), None)
            if active_room:
                with st.form(key=f"edit_form_{active_room.room_id}"):
                    new_label = st.text_input("Label", value=active_room.label)
                    new_type = st.selectbox(
                        "Room Type",
                        options=list(RoomConfig.ROOM_TYPES),
                        index=RoomConfig.ROOM_TYPES.index(active_room.room_type)
                        if active_room.room_type in RoomConfig.ROOM_TYPES
                        else RoomConfig.ROOM_TYPES.index("Unclassified"),
                    )

                    c_x, c_y = st.columns(2)
                    with c_x:
                        new_x = st.number_input("X (px)", value=float(active_room.x), min_value=0.0, step=5.0)
                    with c_y:
                        new_y = st.number_input("Y (px)", value=float(active_room.y), min_value=0.0, step=5.0)

                    c_w, c_h = st.columns(2)
                    with c_w:
                        new_w = st.number_input("Width (px)", value=float(active_room.width), min_value=1.0, step=5.0)
                    with c_h:
                        new_h = st.number_input("Height (px)", value=float(active_room.height), min_value=1.0, step=5.0)

                    new_status = st.selectbox(
                        "Status",
                        options=list(RoomConfig.DETECTION_STATUSES),
                        index=RoomConfig.DETECTION_STATUSES.index(active_room.detection_status)
                        if active_room.detection_status in RoomConfig.DETECTION_STATUSES
                        else 0,
                    )

                    save_btn = st.form_submit_button("💾 Save Room Changes", use_container_width=True)
                    if save_btn:
                        try:
                            RoomEditor.update_room(
                                room=active_room,
                                label=new_label,
                                room_type=new_type,
                                x=new_x,
                                y=new_y,
                                width=new_w,
                                height=new_h,
                                detection_status=new_status,
                                image_width=loaded.width,
                                image_height=loaded.height,
                                scale_info=floorplan.scale_information,
                            )
                            FloorPlanParser.update_floorplan_with_rooms(floorplan, all_current_rooms)
                            st.success(f"Updated {active_room.room_id} successfully.")
                            st.rerun()
                        except ValidationError as v_err:
                            st.error(f"Validation failed: {v_err}")

                # Quick action buttons outside form
                btn_accept, btn_reject = st.columns(2)
                with btn_accept:
                    if st.button("✅ Confirm Room", key=f"conf_{active_room.room_id}", use_container_width=True):
                        floorplan.confirm_room(active_room.room_id)
                        st.success(f"Confirmed {active_room.room_id}")
                        st.rerun()
                with btn_reject:
                    if st.button("❌ Reject Room", key=f"rej_{active_room.room_id}", use_container_width=True):
                        floorplan.reject_room(active_room.room_id)
                        st.warning(f"Rejected {active_room.room_id}")
                        st.rerun()

        # Manual Room Addition
        with st.expander("➕ Add Room Manually"):
            with st.form("manual_room_form"):
                m_label = st.text_input("Room Label", value="Balcony")
                m_type = st.selectbox("Room Type", options=list(RoomConfig.ROOM_TYPES), index=6)
                m_x = st.number_input("X (px)", min_value=0.0, max_value=float(loaded.width - 10), value=50.0)
                m_y = st.number_input("Y (px)", min_value=0.0, max_value=float(loaded.height - 10), value=50.0)
                m_w = st.number_input("Width (px)", min_value=10.0, max_value=float(loaded.width), value=120.0)
                m_h = st.number_input("Height (px)", min_value=10.0, max_value=float(loaded.height), value=90.0)

                add_submit = st.form_submit_button("Create Manual Room", use_container_width=True)
                if add_submit:
                    new_id = f"room_manual_{len(all_current_rooms) + 1:02d}"
                    try:
                        new_manual_room = RoomEditor.create_manual_room(
                            room_id=new_id,
                            label=m_label,
                            room_type=m_type,
                            x=m_x,
                            y=m_y,
                            width=m_w,
                            height=m_h,
                            image_width=loaded.width,
                            image_height=loaded.height,
                            scale_info=floorplan.scale_information,
                        )
                        floorplan.confirmed_rooms.append(new_manual_room)
                        st.success(f"Created user-confirmed room '{m_label}' ({new_id}).")
                        st.rerun()
                    except ValidationError as v_err:
                        st.error(f"Failed to create room: {v_err}")

    with table_col:
        st.markdown("#### Structured Room Inventory")
        if not all_current_rooms:
            st.info("No rooms extracted or created yet.")
        else:
            # Build table data dictionary
            rows = []
            for r in all_current_rooms:
                if floorplan.scale_information.is_calibrated and r.physical_width and r.physical_height:
                    dim_str = f"{r.physical_width:.2f} x {r.physical_height:.2f} m ({r.physical_area:.2f} m²)"
                else:
                    dim_str = f"{int(r.width)} x {int(r.height)} px ({int(r.pixel_area):,} px²)"

                rows.append({
                    "ID": r.room_id,
                    "Label": r.label,
                    "Type": r.room_type,
                    "Status": r.detection_status,
                    "Bounds (X, Y)": f"({int(r.x)}, {int(r.y)})",
                    "Dimensions": dim_str,
                    "Quality": f"{r.confidence_score:.2f}",
                })

            st.dataframe(rows, use_container_width=True, hide_index=True)

            # Display explanation of confidence scores
            with st.expander("ℹ️ Quality Score Breakdown & Methodology"):
                for r in all_current_rooms:
                    st.markdown(f"**{r.label} ({r.room_id})**: `{r.confidence_explanation}`")

    # -------------------------------------------------------------------------
    # Section G & H: Structured Data & JSON Export
    # -------------------------------------------------------------------------
    st.markdown("### 💾 Structured Floor-Plan Data & Export")

    json_export_str = export_floorplan_to_json(floorplan)
    summary = generate_export_summary(floorplan)

    col_meta, col_download = st.columns([2, 1])
    with col_meta:
        st.json(summary)
    with col_download:
        st.write("")
        st.write("")
        export_filename = f"spaze_floorplan_{Path(loaded.filename).stem}.json"
        st.download_button(
            label="📥 Download Structured Floor-Plan (JSON)",
            data=json_export_str,
            file_name=export_filename,
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("View Full Structured JSON Payload"):
        st.code(json_export_str, language="json")


if __name__ == "__main__":
    main()
