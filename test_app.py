import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="OpenCV Plant Detection Playground", layout="wide")

st.title("🔬 OpenCV Plant Detection Parameter Playground")
st.markdown("Upload one image and experiment with OpenCV parameters to understand plant detection!")

# Upload image
uploaded_file = st.file_uploader(
    "📷 Upload a plant image to experiment with", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a typical seedling photo you want to use for testing"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = opencv_image.shape[:2]
    
    # Create three columns for layout
    col2, col3 = st.columns([1, 1])
    
    #with col1:
    #    st.subheader("Original Image")
    #    st.image(image, caption="Original", use_container_width=True)
    
    # Parameter controls in sidebar
    st.sidebar.header("🎛️ OpenCV Parameters")
    
    # HSV Color Range Controls
    st.sidebar.subheader("🌈 HSV Color Detection")
    hue_min = st.sidebar.slider("Hue Min", 0, 179, 25, help="Lower hue bound (green = 35-85)")
    hue_max = st.sidebar.slider("Hue Max", 0, 179, 85, help="Upper hue bound")
    
    sat_min = st.sidebar.slider("Saturation Min", 0, 255, 40, help="Lower saturation (higher = more vivid colors)")
    sat_max = st.sidebar.slider("Saturation Max", 0, 255, 255, help="Upper saturation")
    
    val_min = st.sidebar.slider("Value Min", 0, 255, 40, help="Lower brightness (higher = brighter pixels only)")
    val_max = st.sidebar.slider("Value Max", 0, 255, 255, help="Upper brightness")
    
    # Morphological Operations
    st.sidebar.subheader("🔧 Morphological Operations")
    morph_open_size = st.sidebar.slider("Opening Kernel Size", 1, 15, 3, help="Removes small noise (higher = more aggressive)")
    morph_close_size = st.sidebar.slider("Closing Kernel Size", 1, 15, 7, help="Fills gaps (higher = fills bigger gaps)")
    
    # Size Filtering
    st.sidebar.subheader("📏 Size Filtering")
    min_area_percent = st.sidebar.slider("Min Area (%)", 0.01, 5.0, 0.1, help="Minimum area as % of image")
    max_area_percent = st.sidebar.slider("Max Area (%)", 1.0, 50.0, 15.0, help="Maximum area as % of image")
    
    # Shape Filtering
    st.sidebar.subheader("📐 Shape Filtering")
    min_aspect_ratio = st.sidebar.slider("Min Aspect Ratio", 0.1, 2.0, 0.1, help="Width/Height minimum (lower = allow thinner stems)")
    max_aspect_ratio = st.sidebar.slider("Max Aspect Ratio", 1.0, 10.0, 8.0, help="Width/Height maximum (higher = allow very thin stems)")
    min_solidity = st.sidebar.slider("Min Solidity", 0.1, 1.0, 0.2, help="How 'filled in' the shape is (lower = allow irregular stems)")
    
    # Stem Detection
    st.sidebar.subheader("🌿 Stem Detection")
    detect_stems = st.sidebar.checkbox("Enable Stem Detection", value=True, help="Use line detection for thin stems")
    stem_length_threshold = st.sidebar.slider("Min Stem Length", 10, 100, 30, help="Minimum length for detected lines")
    stem_thickness = st.sidebar.slider("Max Stem Thickness", 1, 20, 8, help="Maximum thickness for stem lines")
    
    # Dilation for thin features
    st.sidebar.subheader("🔍 Thin Feature Enhancement")
    enhance_thin_features = st.sidebar.checkbox("Enhance Thin Features", value=True, help="Dilate mask to capture thin stems")
    dilation_size = st.sidebar.slider("Dilation Size", 1, 10, 3, help="How much to expand thin features")
    
    # Alternative color detection
    st.sidebar.subheader("🎨 Alternative Detection")
    use_lab_channel = st.sidebar.checkbox("Use LAB Color Space", value=False, help="Better for some lighting conditions")
    lab_threshold = st.sidebar.slider("LAB Green Threshold", 0, 50, 15, help="LAB A-channel threshold (lower = more green)")
    
    # Edge Filtering
    st.sidebar.subheader("🖼️ Edge Filtering")
    edge_margin_percent = st.sidebar.slider("Edge Margin (%)", 0.0, 20.0, 5.0, help="Ignore objects near image edges")
    
    # Apply parameters and process image
    def process_image_with_parameters():
        # Convert to HSV
        hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        
        # Create color mask
        lower_bound = np.array([hue_min, sat_min, val_min])
        upper_bound = np.array([hue_max, sat_max, val_max])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Alternative LAB color space detection
        if use_lab_channel:
            lab = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2LAB)
            a_channel = lab[:,:,1]
            # A channel: green is negative, red is positive
            _, lab_mask = cv2.threshold(a_channel, 127 - lab_threshold, 255, cv2.THRESH_BINARY_INV)
            mask = cv2.bitwise_or(mask, lab_mask)
        
        # Enhance thin features before morphological operations
        if enhance_thin_features:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
            mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # Morphological operations
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_size, morph_open_size))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Stem detection using line detection
        stem_mask = np.zeros_like(mask)
        if detect_stems:
            # Use HoughLinesP to detect thin lines (stems)
            edges = cv2.Canny(mask, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                   minLineLength=stem_length_threshold, 
                                   maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Draw thick lines to create stem mask
                    cv2.line(stem_mask, (x1, y1), (x2, y2), 255, stem_thickness)
        
        # Combine original mask with stem mask
        combined_mask = cv2.bitwise_or(mask, stem_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Size filtering
        total_pixels = width * height
        min_area = total_pixels * (min_area_percent / 100)
        max_area = total_pixels * (max_area_percent / 100)
        
        # Edge margin
        margin_x = width * (edge_margin_percent / 100)
        margin_y = height * (edge_margin_percent / 100)
        
        # Filter contours
        valid_contours = []
        rejected_contours = []
        stem_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size check
            if not (min_area <= area <= max_area):
                rejected_contours.append((contour, "Size"))
                continue
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio check
            aspect_ratio = w / h if h > 0 else 0
            if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                # Check if it might be a stem (very thin)
                if aspect_ratio > max_aspect_ratio and h > stem_length_threshold:
                    stem_contours.append(contour)
                    continue
                rejected_contours.append((contour, "Aspect Ratio"))
                continue
            
            # Solidity check
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < min_solidity:
                # Stems can have low solidity, so be more lenient
                if aspect_ratio > 3.0:  # Likely a stem
                    stem_contours.append(contour)
                    continue
                rejected_contours.append((contour, "Solidity"))
                continue
            
            # Edge check
            if (x < margin_x or y < margin_y or 
                x + w > width - margin_x or y + h > height - margin_y):
                rejected_contours.append((contour, "Edge"))
                continue
            
            valid_contours.append(contour)
        
        return combined_mask, valid_contours, rejected_contours, stem_contours
    
    # Process the image
    mask, valid_contours, rejected_contours, stem_contours = process_image_with_parameters()
    
    # Create annotated image
    annotated_image = opencv_image.copy()
    
    # Draw valid contours in red
    for i, contour in enumerate(valid_contours):
        cv2.drawContours(annotated_image, [contour], -1, (0, 0, 255), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(annotated_image, f"Plant {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw stem contours in yellow
    for i, contour in enumerate(stem_contours):
        cv2.drawContours(annotated_image, [contour], -1, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(annotated_image, f"Stem {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw rejected contours in blue
    for contour, reason in rejected_contours:
        cv2.drawContours(annotated_image, [contour], -1, (255, 0, 0), 1)
    
    # Convert back to RGB for display
    annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.subheader("Color Mask")
        st.image(mask, caption="Green Detection Mask", use_container_width=True)
    
    with col3:
        st.subheader("Detection Results")
        st.image(annotated_rgb, caption="Plants (Green), Stems (Yellow), Rejected (Red)", use_container_width=True)
    
    # Results summary
    st.subheader("📊 Detection Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Valid Plants", len(valid_contours))
    with col2:
        st.metric("Detected Stems", len(stem_contours))
    with col3:
        st.metric("Rejected Objects", len(rejected_contours))
    with col4:
        if valid_contours:
            largest_area = max(cv2.contourArea(c) for c in valid_contours)
            st.metric("Largest Plant (px²)", f"{largest_area:.0f}")
        else:
            st.metric("Largest Plant (px²)", "0")
    with col5:
        total_plant_area = sum(cv2.contourArea(c) for c in valid_contours)
        total_stem_area = sum(cv2.contourArea(c) for c in stem_contours)
        st.metric("Total Area", f"{total_plant_area + total_stem_area:.0f}")
    
    # Detailed analysis
    if valid_contours or rejected_contours:
        st.subheader("🔍 Detailed Analysis")
        
        # Valid contours table
        if valid_contours:
            st.write("**✅ Detected Plants:**")
            plant_data = []
            for i, contour in enumerate(valid_contours):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                plant_data.append({
                    "Plant": f"Plant {i+1}",
                    "Area (px²)": f"{area:.0f}",
                    "Width": w,
                    "Height": h,
                    "Aspect Ratio": f"{aspect_ratio:.2f}",
                    "Solidity": f"{solidity:.2f}",
                    "Position": f"({x}, {y})"
                })
            
            st.dataframe(plant_data)
        
        # Rejected contours summary
        if rejected_contours:
            st.write("**❌ Rejected Objects by Reason:**")
            rejection_reasons = {}
            for _, reason in rejected_contours:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
            for reason, count in rejection_reasons.items():
                st.write(f"- {reason}: {count} objects")
    
    # Parameter summary for copying
    st.subheader("📋 Current Parameters")
    st.code(f"""
# HSV Color Range
lower_green = np.array([{hue_min}, {sat_min}, {val_min}])
upper_green = np.array([{hue_max}, {sat_max}, {val_max}])

# Morphological Operations  
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ({morph_open_size}, {morph_open_size}))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ({morph_close_size}, {morph_close_size}))

# Size Filtering
min_area = total_pixels * {min_area_percent/100}
max_area = total_pixels * {max_area_percent/100}

# Shape Filtering
min_aspect_ratio = {min_aspect_ratio}
max_aspect_ratio = {max_aspect_ratio}
min_solidity = {min_solidity}

# Edge Filtering
edge_margin = {edge_margin_percent}% of image dimensions
""")

else:
    st.info("👆 Upload an image to start experimenting with OpenCV parameters!")
    
    st.subheader("🎓 How to Use This Tool")
    st.markdown("""
    This playground helps you understand how each OpenCV parameter affects plant detection:
    
    **Color Detection (HSV):**
    - **Hue**: Controls what colors are detected (35-85 captures most greens)
    - **Saturation**: How vivid the color must be (higher = ignore faded colors)  
    - **Value**: How bright the color must be (higher = ignore dark shadows)
    
    **Morphological Operations:**
    - **Opening**: Removes small noise and breaks thin connections
    - **Closing**: Fills gaps and holes in detected areas
    
    **Size Filtering:**
    - **Min/Max Area**: Reject objects that are too small (noise) or too large (background)
    
    **Shape Filtering:**
    - **Aspect Ratio**: Width/Height ratio (plants are usually not extremely elongated)
    - **Solidity**: How "filled in" the shape is (plants aren't usually very spiky)
    
    **Edge Filtering:**
    - Ignores objects near image edges (often partial background objects)
    
    **Workflow:**
    1. Upload a typical seedling photo
    2. Adjust HSV ranges until the mask captures your plants well
    3. Use morphological operations to clean up the mask
    4. Apply size and shape filters to remove false positives
    5. Copy the final parameters to use in your main app!
    """)