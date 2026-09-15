import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Protein Synthesis Analysis",
    page_icon="🔬",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    font-size: 38px;
    font-weight: 700;
    color: #172033;
}

.subtitle {
    font-size: 17px;
    color: #5f6b7a;
    margin-bottom: 25px;
}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e9f0;
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #687386;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #172033;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">🔬 Protein Synthesis Analysis Using YOLO</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered microscopy image analysis using YOLO-based object detection.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_PATH = (
     Path("models/best.pt")
)

model = YOLO(MODEL_PATH)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.divider()

    st.subheader("🧬 Model Information")

    st.write("**Model:** YOLO11n")
    st.write("**Detection Class:** Nucleus")
    st.write("**Framework:** Ultralytics")
    st.write("**Device:** CPU")

    st.divider()

    st.caption(
        "This system detects visually observable cellular "
        "structures in microscopy images."
    )

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("📤 Upload Microscopy Image")

uploaded_file = st.file_uploader(
    "Choose a microscopy image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # Original image
    with col1:

        st.markdown("### 🖼️ Original Image")

        st.image(
            image,
            use_container_width=True
        )

    # Analyze button
    with col2:

        st.markdown("### 🔍 Analysis")

        st.write(
            "Click the button below to detect nuclei "
            "using the trained YOLO model."
        )

        analyze = st.button(
            "🔍 Analyze Image",
            use_container_width=True
        )

        if analyze:

            with st.spinner("Analyzing microscopy image..."):

                image_array = np.array(image)

                results = model.predict(
                    source=image_array,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]

                # Count detections
                nucleus_count = len(result.boxes)

                # Confidence values
                if nucleus_count > 0:
                    confidence_values = (
                        result.boxes.conf.cpu().numpy()
                    )

                    average_confidence = (
                        float(np.mean(confidence_values)) * 100
                    )

                    highest_confidence = (
                        float(np.max(confidence_values)) * 100
                    )

                else:
                    average_confidence = 0
                    highest_confidence = 0

                # Annotated image
                annotated_image = result.plot(
                    labels=False,
                    conf=False
                )

            # --------------------------------------------------
            # RESULTS
            # --------------------------------------------------

            st.divider()

            st.subheader("📊 Detection Summary")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:

                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-title">
                            Detected Nuclei
                        </div>
                        <div class="metric-value">
                            {nucleus_count}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with metric2:

                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-title">
                            Average Confidence
                        </div>
                        <div class="metric-value">
                            {average_confidence:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with metric3:

                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-title">
                            Highest Confidence
                        </div>
                        <div class="metric-value">
                            {highest_confidence:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # --------------------------------------------------
            # DETECTION IMAGE
            # --------------------------------------------------

            st.divider()

            st.subheader("🧬 Detection Result")

            st.image(
                annotated_image,
                caption="Detected cellular nuclei",
                use_container_width=True
            )

            # --------------------------------------------------
            # INTERPRETATION
            # --------------------------------------------------

            st.divider()

            st.subheader("📋 Analysis Interpretation")

            if nucleus_count > 0:

                st.success(
                    f"The YOLO model detected {nucleus_count} "
                    "nuclei in the uploaded microscopy image."
                )

                st.info(
                    "The bounding boxes represent the regions "
                    "identified by the trained object-detection model."
                )

            else:

                st.warning(
                    "No nuclei were detected at the selected "
                    "confidence threshold."
                )

else:

    st.info(
        "👆 Upload a microscopy image above to begin analysis."
    )

   # --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.subheader("📈 Model Performance")

st.write(
    "Performance of the trained YOLO model on the independent test dataset."
)

# Performance values
precision = 98.2
recall = 88.6
map50 = 92.0
map5095 = 77.1

# Metric cards
perf1, perf2, perf3, perf4 = st.columns(4)

with perf1:
    st.metric("Precision", f"{precision}%")

with perf2:
    st.metric("Recall", f"{recall}%")

with perf3:
    st.metric("mAP@50", f"{map50}%")

with perf4:
    st.metric("mAP@50–95", f"{map5095}%")

# Performance chart
st.markdown("### 📊 Performance Comparison")

chart_data = {
    "Metric": [
        "Precision",
        "Recall",
        "mAP@50",
        "mAP@50–95"
    ],
    "Score": [
        precision,
        recall,
        map50,
        map5095
    ]
}

st.bar_chart(
    chart_data,
    x="Metric",
    y="Score"
)

st.caption(
    "Test dataset: 20 microscopy images containing 2,279 annotated nucleus instances."
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Protein Synthesis Analysis Using YOLO | "
    "B.Tech Project | Computer Vision & Deep Learning"
)