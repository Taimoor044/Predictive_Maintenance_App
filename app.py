import streamlit as st
st.write("Debug: Starting app.py")
import os
st.write("Debug: Imported os")
import cv2
st.write("Debug: Imported cv2")
import numpy as np
st.write("Debug: Imported numpy")
import subprocess
st.write("Debug: Imported subprocess")
import glob
st.write("Debug: Imported glob")
import sys
st.write("Debug: Imported sys")
import shutil
st.write("Debug: Imported shutil")

python_path = "/home/adminuser/venv/bin/python3"
st.write(f"Debug: Set python_path to {python_path}")

# Simulated dataset for defect probability based on conditions
simulated_dataset = {
    # (age_range, usage, temp_range, moisture_level): defect_probability
    ("old", "heavy", "high", "high"): 0.80,  # 80% chance of defect
    ("old", "heavy", "high", "medium"): 0.70,
    ("old", "heavy", "low", "high"): 0.65,
    ("old", "moderate", "high", "high"): 0.60,
    ("old", "light", "low", "low"): 0.30,
    ("medium", "heavy", "high", "high"): 0.50,
    ("medium", "moderate", "medium", "medium"): 0.40,
    ("medium", "light", "low", "low"): 0.20,
    ("new", "heavy", "high", "high"): 0.30,
    ("new", "moderate", "medium", "medium"): 0.15,
    ("new", "light", "low", "low"): 0.10,
}

# Streamlit app
st.title("Fire Door and Floor Predictive Maintenance")
st.write("Upload an image to detect defects and get maintenance predictions.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image
    image_filename = uploaded_file.name
    st.write(f"Debug: Saving image as {image_filename}")
    with open(image_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Verify the image and weights file exist
    weights_path = 'yolov5/runs/train/exp2/weights/best.pt'
    st.write(f"Debug: Checking if image exists at {image_filename}: {os.path.exists(image_filename)}")
    if not os.path.exists(image_filename):
        st.error(f"Error: Image file {image_filename} not found after saving.")
        st.stop()
    st.write(f"Debug: Checking if weights exist at {weights_path}: {os.path.exists(weights_path)}")
    if not os.path.exists(weights_path):
        st.error(f"Error: Weights file {weights_path} not found.")
        st.stop()

    # Run YOLOv5 inference without displaying output
    st.write("Debug: Running YOLOv5 inference...")
    try:
        result = subprocess.run([
            python_path, 'yolov5/detect.py',
            '--weights', weights_path,
            '--img', '640',
            '--conf', '0.2',
            '--source', image_filename,
            '--save-txt',
            '--save-conf',
            '--augment'
        ], capture_output=True, text=True, check=True, timeout=300)
        st.write("Debug: Inference completed successfully.")
        if result.stdout:
            st.write("Debug: Inference stdout:", result.stdout)
        if result.stderr:
            st.write("Debug: Inference stderr:", result.stderr)
    except subprocess.TimeoutExpired:
        st.error("Inference timed out after 5 minutes. The image may be too large or the server may be under heavy load.")
        st.stop()
    except subprocess.CalledProcessError as e:
        st.error("Error running detect.py:")
        st.write("Debug: Inference stdout:", e.stdout)
        st.write("Debug: Inference stderr:", e.stderr)
        st.stop()
    except Exception as e:
        st.error("Unexpected error during inference:")
        st.write(f"Debug: Error message: {str(e)}")
        st.stop()

    # Find the latest exp folder
    st.write("Debug: Looking for exp folders in yolov5/runs/detect...")
    exp_folders = glob.glob('yolov5/runs/detect/exp*')
    st.write(f"Debug: Found exp folders: {exp_folders}")
    if not exp_folders:
        st.error("No detection results found. The inference may have failed.")
        st.write("Current directory contents:")
        st.write(os.listdir('.'))
        st.write("yolov5/runs directory contents:")
        st.write(os.listdir('yolov5/runs') if os.path.exists('yolov5/runs') else "yolov5/runs does not exist")
        st.stop()

    latest_exp = max(exp_folders, key=os.path.getctime)
    st.write(f"Debug: Latest exp folder: {latest_exp}")
    label_filename = image_filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
    label_path = os.path.join(latest_exp, 'labels', label_filename)
    st.write(f"Debug: Looking for label file at {label_path}")

    # Load and display the output image
    output_image_path = os.path.join(latest_exp, image_filename)
    st.write(f"Debug: Checking for output image at {output_image_path}: {os.path.exists(output_image_path)}")
    if os.path.exists(output_image_path):
        output_image = cv2.imread(output_image_path)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        st.image(output_image, caption="Processed Image", use_column_width=True)
    else:
        st.write("Output image not found at:", output_image_path)

    # Parse detection results and calculate dimensions
    image_width, image_height = 640, 640

    if os.path.exists(label_path):
        st.write("Debug: Label file found, processing detections...")
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

            # Calculate severity score (based on dimensions and defect type)
            base_score = {'crack': 3, 'dent': 2, 'scratch': 1}  # Base severity by type
            size_factor = (defect_length_px * defect_width_px) / (image_width * image_height)  # Normalized area
            severity_score = base_score[class_name] + (size_factor * 10)  # Scale size impact
            severity_score = min(severity_score, 10)  # Cap at 10

            # Determine severity and priority based on score
            if severity_score >= 8:
                severity = 'Severe'
                priority = 'Urgent'
            elif severity_score >= 5:
                severity = 'Moderate'
                priority = 'High'
            elif severity_score >= 2:
                severity = 'Minor'
                priority = 'Medium'
            else:
                severity = 'Negligible'
                priority = 'Low'

            st.write(f"- **{class_name}** (Confidence: {conf:.2f})")
            st.write(f"  - **Dimensions**: Length = {defect_length_px:.1f}px, Width = {defect_width_px:.1f}px")
            st.write(f"  - **Severity Score**: {severity_score:.1f}/10")
            st.write(f"  - **Severity**: {severity}")
            st.write(f"  - **Priority**: {priority}")
            if class_name == 'crack':
                if severity == 'Severe':
                    st.write("  - **Recommendation**: Urgent action needed. Replace or reinforce the affected area immediately.")
                elif severity == 'Moderate':
                    st.write("  - **Recommendation**: Schedule repair to prevent further deterioration.")
                elif severity == 'Minor':
                    st.write("  - **Recommendation**: Monitor the crack; schedule an inspection to assess potential growth.")
                else:
                    st.write("  - **Recommendation**: Monitor periodically.")
            elif class_name == 'dent':
                if severity in ['Severe', 'Moderate']:
                    st.write("  - **Recommendation**: Schedule repair to prevent further damage.")
                else:
                    st.write("  - **Recommendation**: Monitor periodically.")
            elif class_name == 'scratch':
                if severity == 'Severe':
                    st.write("  - **Recommendation**: Schedule repair to prevent rust.")
                else:
                    st.write("  - **Recommendation**: Monitor; consider repainting to prevent rust.")
    else:
        st.write("Debug: No label file found, proceeding to questions section...")
        st.write("No defects detected in the image.")
        st.write("### Let's make an educated prediction based on the condition of the door/floor.")
        
        # Questions for the user
        age_years = st.number_input("How old is the door/floor (in years)?", min_value=0, value=0, step=1)
        usage_frequency = st.selectbox("How often is the door/floor used?", ["Rarely (e.g., a few times a month)", "Moderately (e.g., daily)", "Heavily (e.g., multiple times a day)"])
        temperature = st.number_input("What is the average temperature the door/floor is exposed to (in °C)?", min_value=-50.0, max_value=100.0, value=20.0, step=1.0)
        moisture_level = st.selectbox("What is the typical moisture level the door/floor is exposed to?", ["Low (e.g., dry environment)", "Medium (e.g., occasional humidity)", "High (e.g., frequent rain or humidity)"])

        # Map user input to dataset categories
        age_range = "new" if age_years < 5 else "medium" if age_years < 10 else "old"
        usage = "light" if usage_frequency == "Rarely (e.g., a few times a month)" else "moderate" if usage_frequency == "Moderately (e.g., daily)" else "heavy"
        temp_range = "low" if temperature < 20 else "medium" if temperature < 30 else "high"
        moisture = "low" if moisture_level == "Low (e.g., dry environment)" else "medium" if moisture_level == "Medium (e.g., occasional humidity)" else "high"

        # Look up defect probability in the simulated dataset
        condition_key = (age_range, usage, temp_range, moisture)
        defect_probability = simulated_dataset.get(condition_key, 0.40)  # Default to 40% if exact match not found

        # Calculate a risk score based on conditions
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
        # Temperature factor
        if temperature > 30:
            risk_score += 2
        elif temperature > 20:
            risk_score += 1
        # Moisture factor
        if moisture_level == "High (e.g., frequent rain or humidity)":
            risk_score += 2
        elif moisture_level == "Medium (e.g., occasional humidity)":
            risk_score += 1

        # Adjust risk score based on defect probability
        risk_score += (defect_probability * 5)  # Scale probability impact (max 4 points)

        # Make an educated prediction
        st.write("### Maintenance Prediction Based on Conditions:")
        st.write(f"- **Estimated Defect Probability**: {defect_probability * 100:.0f}% (based on simulated dataset)")
        if risk_score >= 7:
            st.write("- **Risk Level**: High")
            st.write("- **Prediction**: The door/floor is at high risk of developing defects soon due to its age, usage, temperature, and moisture conditions.")
            st.write("- **Recommendation**: Schedule a thorough inspection within the next month and consider preventive maintenance, such as sealing or reinforcing the door/floor.")
        elif risk_score >= 4:
            st.write("- **Risk Level**: Moderate")
            st.write("- **Prediction**: The door/floor may develop defects in the near future, especially if conditions worsen.")
            st.write("- **Recommendation**: Monitor the door/floor regularly and schedule an inspection within the next 3 months.")
        else:
            st.write("- **Risk Level**: Low")
            st.write("- **Prediction**: The door/floor appears to be in good condition with low risk of defects in the near term.")
            st.write("- **Recommendation**: Continue regular maintenance and monitor for any changes.")
