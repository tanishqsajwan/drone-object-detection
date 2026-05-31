import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image

st.set_page_config(page_title="Drone Object Detection", layout="wide")
st.title("🚁 Drone Object Detection")
st.markdown("---")

model = YOLO(r"C:\Users\tanis\OneDrive\Desktop\yolo-project\runs\detect\detects\train\weights\best.pt")
model.names[0] = "person"
model.names[1] = "person"

tab1, tab2 = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])

with tab1:
    st.subheader("Upload a Drone Image")
    image_file = st.file_uploader(
        "Choose an image", 
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    run_image = st.button("Run Detection", key="run_image")

    if image_file and run_image:
        image = Image.open(image_file)
        image_np = np.array(image)

        with st.spinner("Detecting objects..."):
            results = model(image_np)
            annotated = results[0].plot()

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)
        with col2:
            st.image(annotated, caption="Output Image", use_column_width=True)

        boxes = results[0].boxes
        if len(boxes) > 0:
            st.markdown("### 📊 Detection Summary")
            from collections import Counter
            labels = [model.names[int(c)] for c in boxes.cls]
            counts = Counter(labels)
            cols = st.columns(len(counts))
            for i, (label, count) in enumerate(counts.items()):
                cols[i].metric(label.capitalize(), count)
        else:
            st.info("No objects detected.")

    elif image_file and not run_image:
        st.image(image_file, caption="Preview", use_column_width=True)

with tab2:
    st.subheader("Upload a Drone Video")
    video_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )
    run_video = st.button("Run Detection", key="run_video")

    if video_file and run_video:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(video_file.read())
        temp.flush()

        cap = cv2.VideoCapture(temp.name)
        if not cap.isOpened():
            st.error("Could not open video file.")
        else:
            st.info("Processing video frame by frame...")
            stframe = st.empty()
            stop = st.button("⏹ Stop", key="stop_video")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop:
                    break
                results = model(frame)
                annotated = results[0].plot()
                stframe.image(annotated, channels="BGR", use_column_width=True)

            cap.release()
            st.success("Video processing complete!")

    elif video_file and not run_video:
        st.video(video_file)