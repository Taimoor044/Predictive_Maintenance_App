import streamlit as st
import os
import cv2
import numpy as np
import subprocess
import glob

# Streamlit app
st.title("Fire Door and Floor Predictive Maintenance")
st.write("Upload an image to detect defects and get maintenance recommendations.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image
    image_filename = uploaded_file.name
    with open(image_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Verify the image and weights file exist
    weights_path = 'yolov5/runs/train/exp2/weights/best.pt'
    if not os.path.exists(image_filename):
        st.error(f"Error: Image file {image_filename} not found after saving.")
        st.stop()
    if not os.path.exists(weights_path):
        st.error(f"Error: Weights file {weights_path} not found.")
        st.stop()

    # Run YOLOv5 inference with error capturing
    try:
        result = subprocess.run([
            'python', 'yolov5/detect.py',
            '--weights', weights_path,
            '--img', '640',
            '--conf', '0.01',
            '--source', image_filename,
            '--save-txt',
            '--save-conf',
            '--augment'
        ], capture_output=True, text=True, check=True)
        st.write("Inference Output:")
        st.write(result.stdout)
        if result.stderr:
            st.write("Inference Errors:")
            st.write(result.stderr)
    except subprocess.CalledProcessError as e:
        st.error("Error running detect.py:")
        st.write(e.stdout)
        st.write(e.stderr)
        st.stop()

    # Find the latest exp folder
    exp_folders = glob.glob('yolov5/runs/detect/exp*')
    if not exp_folders:
        st.error("No detection results found. The inference may have failed.")
        st.write("Current directory contents:")
        st.write(os.listdir('.'))
        st.write("yolov5/runs directory contents:")
        st.write(os.listdir('yolov5/runs') if os.path.exists('yolov5/runs') else "yolov5/runs does not exist")
        st.stop()

    latest_exp = max(exp_folders, key=os.path.getctime)
    label_filename = image_filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
    label_path = os.path.join(latest_exp, 'labels', label_filename)

    # Load and display the output image
    output_image_path = os.path.join(latest_exp, image_filename)
    if os.path.exists(output_image_path):
        output_image = cv2.imread(output_image_path)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        st.image(output_image, caption="Detected Defects", use_column_width=True)
    else:
        st.write("Output image not found at:", output_image_path)

    # Parse detection results and calculate dimensions
    image_width, image_height = 640, 640

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            detections = f.readlines()

        st.write("### Detected Defects and Maintenance Recommendations:")
        for detection in detections:
            values = detection.strip().split()
            if len(values) == 6:
                class_id, x_center, y_center, bbox_width, bbox_height, conf = map(float, values)
            elif len(values) == 5:
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, values)
                conf = 0.0
            else:
                st.write(f"Skipping invalid detection line: {detection.strip()}")
                continue

            class_id = int(class_id)
            class_name = ['crack', 'dent', 'scratch'][class_id]

            # Calculate dimensions in pixels
            defect_length_px = bbox_width * image_width
            defect_width_px = bbox_height * image_height

            # Determine severity for cracks
            severity = 'Unknown'
            priority = 'Low'
            if class_name == 'crack':
                if defect_length_px > 100 and defect_width_px < 20:
                    severity = 'Minor'
                    priority = 'Medium'
                elif abs(defect_length_px - defect_width_px) < 50:
                    severity = 'Serious'
                    priority = 'High'
                elif defect_length_px > 200 and defect_width_px > 100:
                    severity = 'Gaping'
                    priority = 'Urgent'
                else:
                    severity = 'Moderate'
                    priority = 'Medium'
            elif class_name == 'dent':
                severity = 'Moderate'
                priority = 'Medium'
            elif class_name == 'scratch':
                severity = 'Minor'
                priority = 'Low'

            st.write(f"- **{class_name}** (Confidence: {conf:.2f})")
            if class_name == 'crack':
                st.write(f"  - **Dimensions**: Length = {defect_length_px:.1f}px, Width = {defect_width_px:.1f}px")
                st.write(f"  - **Severity**: {severity} crack")
                st.write(f"  - **Priority**: {priority}")
                if severity == 'Minor':
                    st.write("  - **Recommendation**: Monitor the crack; schedule an inspection to assess potential growth.")
                elif severity == 'Serious':
                    st.write("  - **Recommendation**: Immediate repair required. This crack may indicate deeper structural damage.")
                elif severity == 'Gaping':
                    st.write("  - **Recommendation**: Urgent action needed. Replace or reinforce the affected area immediately.")
                else:
                    st.write("  - **Recommendation**: Schedule repair to prevent further deterioration.")
            elif class_name == 'dent':
                st.write(f"  - **Severity**: {severity}")
                st.write(f"  - **Priority**: {priority}")
                st.write("  - **Recommendation**: Schedule repair to prevent further damage.")
            elif class_name == 'scratch':
                st.write(f"  - **Severity**: {severity}")
                st.write(f"  - **Priority**: {priority}")
                st.write("  - **Recommendation**: Monitor; consider repainting to prevent rust.")
    else:
        st.write("No defects detected. Label file not found at:", label_path)
