import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Add this near the top after the DATA_CSV setup:
CALIBRATION_CSV = "calibration_data.csv"

# Function to load calibration
def load_calibration():
    if os.path.exists(CALIBRATION_CSV):
        try:
            cal_df = pd.read_csv(CALIBRATION_CSV)
            if not cal_df.empty and 'cm_per_pixel' in cal_df.columns:
                return cal_df.iloc[-1]['cm_per_pixel']  # Get most recent calibration
        except:
            pass
    return None

# Function to save calibration
def save_calibration(cm_per_pixel, ref_object, ref_cm):
    cal_data = pd.DataFrame({
        "timestamp": [datetime.now().isoformat()],
        "cm_per_pixel": [cm_per_pixel],
        "reference_object": [ref_object],
        "reference_cm": [ref_cm]
    })
    if os.path.exists(CALIBRATION_CSV):
        cal_data.to_csv(CALIBRATION_CSV, mode='a', header=False, index=False)
    else:
        cal_data.to_csv(CALIBRATION_CSV, index=False)

st.set_page_config(page_title="Seedling Tracker", layout="centered")
st.title("🌱 Seedling Growth Tracker")

# Load or create data CSV
DATA_CSV = "growth_data.csv"
if not os.path.exists(DATA_CSV):
    pd.DataFrame(columns=["timestamp", "avg_leaf_length_cm", "plant_count"]).to_csv(DATA_CSV, index=False)

# Take photo from phone camera
image_data = st.camera_input("Take a photo of your seedling tray (with ruler visible)")

# calibrate measurements    
st.sidebar.subheader("📏 Calibration")
# Check for existing calibration
saved_calibration = load_calibration()
if saved_calibration:
    use_saved = st.sidebar.checkbox("Use saved calibration", value=True)
    if use_saved:
        st.sidebar.success(f"Using saved calibration: {saved_calibration:.6f} cm/pixel")
        cm_per_pixel = saved_calibration
        skip_calibration = True
    else:
        skip_calibration = False
else:
    skip_calibration = False
    st.sidebar.info("No saved calibration found")

if not skip_calibration:
    st.sidebar.markdown("**Instructions:** Place a common object in your photo, then select it below and tap two points on that object.")

    ref_options = st.sidebar.selectbox(
        "What reference object are you using?",
        [
            "Custom length",
            "Credit card (8.5cm long)",
            "US Quarter (2.4cm diameter)", 
            "US Penny (1.9cm diameter)",
            "Standard ruler (1cm between marks)",
            "Paperclip (5cm long)",
            "AA Battery (5cm long)",
            "Post-it note (7.6cm wide)",
            "Business card (8.5cm long)"
        ]
    )

    # Set reference length based on selection
    if ref_options == "Custom length":
        ref_cm = st.sidebar.number_input("Actual length between selected points (cm)", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    else:
        ref_lengths = {
            "Credit card (8.5cm long)": 8.5,
            "US Quarter (2.4cm diameter)": 2.4,
            "US Penny (1.9cm diameter)": 1.9,
            "Standard ruler (1cm between marks)": 1.0,
            "Paperclip (5cm long)": 5.0,
            "AA Battery (5cm long)": 5.0,
            "Post-it note (7.6cm wide)": 7.6,
            "Business card (8.5cm long)": 8.5
        }
        ref_cm = ref_lengths[ref_options]
        st.sidebar.info(f"Using {ref_cm}cm as reference length")

# resizing and calibrating
if image_data is not None:
    # Convert to OpenCV format
    img = Image.open(image_data)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Calculate proportional resize to fit width of 400 while maintaining aspect ratio
    original_height, original_width = img_cv.shape[:2]
    target_width = 400
    target_height = int((target_width / original_width) * original_height)
    
    img_resized = cv2.resize(img_cv, (target_width, target_height))

    if not skip_calibration:
        st.subheader("📐 Select Reference Points")
        st.markdown(f"**Instructions:** Use the sliders to position the crosshairs over the two endpoints of your reference object")
        
        # Show image with overlay
        img_display = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Point selection with number inputs and adjustment buttons
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First Point (Red):**")
            x1 = st.number_input("X position", 0, target_width, target_width//4, key="x1_input")
            y1 = st.number_input("Y position", 0, target_height, target_height//2, key="y1_input")

        with col2:
            st.write("**Second Point (Blue):**")
            x2 = st.number_input("X position", 0, target_width, 3*target_width//4, key="x2_input")
            y2 = st.number_input("Y position", 0, target_height, target_height//2, key="y2_input")    # Draw crosshairs on image (same as before)
            img_with_points = img_display.copy()
            
            # Draw crosshairs for point 1 (red)
            cv2.line(img_with_points, (x1-10, y1), (x1+10, y1), (255, 0, 0), 2)
            cv2.line(img_with_points, (x1, y1-10), (x1, y1+10), (255, 0, 0), 2)
            
            # Draw crosshairs for point 2 (blue)
            cv2.line(img_with_points, (x2-10, y2), (x2+10, y2), (0, 0, 255), 2)
            cv2.line(img_with_points, (x2, y2-10), (x2, y2+10), (0, 0, 255), 2)
            
            # Draw line between points
            cv2.line(img_with_points, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display without stretching
            st.image(img_with_points, caption="Position the crosshairs on your reference object endpoints")   
            # Calculate distance
            ref_pixels = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            
            if ref_pixels > 0:
                st.success(f"✅ Distance: {ref_pixels:.1f} pixels = {ref_cm}cm")
            else:
                st.warning("⚠️ Please set different positions for the two points")
                ref_pixels = None

        if ref_pixels is not None:
            cm_per_pixel = ref_cm / ref_pixels
            
            # Save the calibration
            save_calibration(cm_per_pixel, ref_options, ref_cm)
            st.success(f"✅ Distance: {ref_pixels:.1f} pixels = {ref_cm}cm (Calibration saved!)")
    else:
        cm_per_pixel = load_calibration()

    # Convert to HSV and mask green
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological clean-up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaf_lengths = []
    plant_count = 0

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 30:  # Filter out small noise
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 3.0:  # Filter out very thin/wide shapes
                # Calculate diagonal of bounding box in pixels
                diagonal_pixels = ((w ** 2 + h ** 2) ** 0.5)
                
                # Convert to cm and divide by 2 for leaf length estimate
                leaf_length_cm = (diagonal_pixels * cm_per_pixel) / 2
                leaf_lengths.append(leaf_length_cm)
                plant_count += 1
                
                # Draw bounding box and diagonal
                cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Draw diagonal line
                cv2.line(img_resized, (x, y), (x + w, y + h), (255, 0, 255), 2)
                
                # Label with length
                cv2.putText(img_resized, f"{leaf_length_cm:.1f}cm", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Calculate average
    avg_leaf_length = sum(leaf_lengths) / len(leaf_lengths) if leaf_lengths else 0

    # Update the data saving
    new_entry = pd.DataFrame({
        "timestamp": [datetime.now().isoformat()], 
        "avg_leaf_length_cm": [avg_leaf_length],
        "plant_count": [plant_count]
    })
    # Check if file exists to determine if we need headers
    if os.path.exists(DATA_CSV):
        new_entry.to_csv(DATA_CSV, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(DATA_CSV, mode='w', header=True, index=False)

    # Show results
    st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), caption="Detected Plants", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Plants Detected", plant_count)
    with col2:
        st.metric("Average Leaf Length", f"{avg_leaf_length:.2f} cm")

    # Add first transplant readiness indicator (from seed tray to individual pots)
    if avg_leaf_length >= 0.5 and plant_count >= 1:
        if avg_leaf_length >= 0.5 and avg_leaf_length < 1.5:
            first_transplant_status = "🌱 Ready for first transplant (to individual pots)"
            st.info(first_transplant_status)
        elif avg_leaf_length >= 1.5 and avg_leaf_length < 2.5:
            first_transplant_status = "🪴 Growing well in individual pots"
            st.success(first_transplant_status)
        elif avg_leaf_length >= 2.5 and avg_leaf_length <= 4.0:
            first_transplant_status = "🌿 Ready for outdoor transplanting!"
            st.success(first_transplant_status)
        elif avg_leaf_length > 4.0:
            first_transplant_status = "🌳 Past optimal transplant size - transplant soon"
            st.warning(first_transplant_status)
        else:
            first_transplant_status = "🌰 Still developing first true leaves"
            st.info(first_transplant_status)

# CSV initialization:
if not os.path.exists(DATA_CSV):
    pd.DataFrame(columns=["timestamp", "avg_leaf_length_cm", "plant_count"]).to_csv(DATA_CSV, index=False)

# plotting section:
if os.path.exists(DATA_CSV):
    df = pd.read_csv(DATA_CSV)
    if not df.empty:
        required_cols = {"timestamp", "avg_leaf_length_cm", "plant_count"}
        if required_cols.issubset(df.columns):
            df = df.dropna(subset=["timestamp", "avg_leaf_length_cm"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp")

            st.subheader("📈 Growth Over Time")
            
            # Create subplot for both metrics
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            
            # Average leaf length plot
            ax1.plot(df["timestamp"], df["avg_leaf_length_cm"], marker='o', color='green')
            ax1.set_ylabel("Average Leaf Length (cm)")
            ax1.grid(True)
            ax1.set_title("Average Leaf Length")
            
            # Plant count plot
            ax2.plot(df["timestamp"], df["plant_count"], marker='s', color='blue')
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Number of Plants")
            ax2.grid(True)
            ax2.set_title("Plant Count")
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("❌ Data file is missing required columns.")
