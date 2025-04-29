import streamlit as st
import os
import cv2
import numpy as np
import subprocess
import glob
import sys
import shutil

python_path = "/home/adminuser/venv/bin/python3"

# Streamlit app
st.title("Fire Door and Floor Predictive Maintenance")
st.write("Upload an image to detect defects and get maintenance predictions.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# User input for time since last maintenance
months_since_maintenance = st.number_input("Months since last maintenance", min_value=0, value=0, step=1)

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

    # Run YOLOv5 inference without displaying output
    try:
        result = subprocess.run([
            python_path, 'yolov5/detect.py',
            '--weights', weights_path,
            '--img', '640',
            '--conf', '0.5',
            '--source', image_filename,
            '--save-txt',
            '--save-conf',
            '--augment'
        ], capture_output=True, text=True, check=True)
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
        st.image(output_image, caption="Processed Image", use_column_width=True)
    else:
        st.write("Output image not found at:", output_image_path)

    # Parse detection results and calculate dimensions
    image_width, image_height = 640, 640

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            detections = f.readlines()

        st.write("### Detected Defects and Maintenance Predictions:")
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

            # Simulate deterioration rate and predict maintenance timeline
            deterioration_rate = {'crack': 0.1, 'dent': 0.05, 'scratch': 0.02}  # Monthly growth rate
            severity_score = {'Minor': 1, 'Moderate': 2, 'Serious': 3, 'Gaping': 4, 'Unknown': 1}
            current_score = severity_score.get(severity, 1)
            rate = deterioration_rate[class_name]
            # Adjust score based on time since last maintenance
            adjusted_score = current_score + (rate * months_since_maintenance)
            # Estimate months until critical (severity score reaches 4)
            months_to_critical = (4 - adjusted_score) / rate if rate > 0 else float('inf')
            if months_to_critical == float('inf') or months_to_critical < 0:
                prediction = "No immediate deterioration predicted."
            else:
                prediction = f"Predicted to become critical in approximately {int(months_to_critical)} months."
            # Adjust priority if time since maintenance is high
            if months_since_maintenance > 12 and priority != 'Urgent':
                priority = 'High'
                prediction += " (Priority increased due to long time since last maintenance)"

            st.write(f"- **{class_name}** (Confidence: {conf:.2f})")
            if class_name == 'crack':
                st.write(f"  - **Dimensions**: Length = {defect_length_px:.1f}px, Width = {defect_width_px:.1f}px")
                st.write(f"  - **Severity**: {severity} crack")
                st.write(f"  - **Priority**: {priority}")
                st.write(f"  - **Prediction**: {prediction}")
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
                st.write(f"  - **Prediction**: {prediction}")
                st.write("  - **Recommendation**: Schedule repair to prevent further damage.")
            elif class_name == 'scratch':
                st.write(f"  - **Severity**: {severity}")
                st.write(f"  - **Priority**: {priority}")
                st.write(f"  - **Prediction**: {prediction}")
                st.write("  - **Recommendation**: Monitor; consider repainting to prevent rust.")
    else:
        st.write("No defects detected in the image.")
        st.write("### Let's make an educated prediction based on the condition of the door/floor.")
        
        # Questions for the user
        age_years = st.number_input("How old is the door/floor (in years)?", min_value=0, value=0, step=1)
        usage_frequency = st.selectbox("How often is the door/floor used?", ["Rarely (e.g., a few times a month)", "Moderately (e.g., daily)", "Heavily (e.g., multiple times a day)"])
        moisture_exposure = st.selectbox("Is the door/floor exposed to moisture (e.g., rain, humidity)?", ["No", "Sometimes", "Frequently"])

        # Calculate a risk score based on answers
        risk_score = 0
        # Age factor
        if age_years > 10:
            risk_score += 2
        elif age_years > 5:
            risk_score += 1
        # Usage factor
        if usage_frequency == "Heavily (e.g., multiple times a day)":
            risk_score += 2
        elif usage_frequency == "Moderately (e.g., daily)":
            risk_score += 1
        # Moisture factor
        if moisture_exposure == "Frequently":
            risk_score += 2
        elif moisture_exposure == "Sometimes":
            risk_score += 1
        # Adjust risk based on time since last maintenance
        if months_since_maintenance > 12:
            risk_score += 1

        # Make an educated prediction
        st.write("### Maintenance Prediction Based on Conditions:")
        if risk_score >= 4:
            st.write("- **Risk Level**: High")
            st.write("- **Prediction**: Even though no defects are currently visible, the door/floor is at high risk of developing issues (e.g., cracks, wear) soon due to its age, usage, and environmental conditions.")
            st.write("- **Recommendation**: Schedule a thorough inspection within the next month and consider preventive maintenance, such as sealing or reinforcing the door/floor.")
        elif risk_score >= 2:
            st.write("- **Risk Level**: Moderate")
            st.write("- **Prediction**: The door/floor may develop defects in the near future, especially if conditions like usage or moisture exposure increase.")
            st.write("- **Recommendation**: Monitor the door/floor regularly and schedule an inspection within the next 3 months.")
        else:
            st.write("- **Risk Level**: Low")
            st.write("- **Prediction**: The door/floor appears to be in good condition with low risk of defects in the near term.")
            st.write("- **Recommendation**: Continue regular maintenance and monitor for any changes.")
