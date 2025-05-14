import streamlit as st
import pandas as pd
import cv2
import os
import numpy as np
from Config import Config

config = Config()

# Set page to wide mode
st.set_page_config(layout="wide")

def convert_to_yolo_format(frame_data, img_width, img_height):
    """Convert CSV data to YOLO format"""
    yolo_data = frame_data.copy()
    
    # Calculate center coordinates and normalize
    yolo_data['x_center'] = (yolo_data['x'] + yolo_data['w']/2) / img_width
    yolo_data['y_center'] = (yolo_data['y'] + yolo_data['h']/2) / img_height
    yolo_data['width'] = yolo_data['w'] / img_width
    yolo_data['height'] = yolo_data['h'] / img_height
    
    # Ensure normalized values are between 0 and 1
    yolo_data['x_center'] = yolo_data['x_center'].clip(0, 1)
    yolo_data['y_center'] = yolo_data['y_center'].clip(0, 1)
    yolo_data['width'] = yolo_data['width'].clip(0, 1)
    yolo_data['height'] = yolo_data['height'].clip(0, 1)
    
    return yolo_data

def draw_bounding_boxes(image, yolo_data, img_width, img_height):
    # Create a copy of the image to draw on
    image_with_boxes = image.copy()
    
    # Define colors for different classes (you can modify these colors)
    colors = {
        1: (0, 255, 0),    # Green for pedestrians
        2: (255, 0, 0),    # Blue for other objects
        3: (0, 0, 255),    # Red for vehicles
    }
    
    # Draw each bounding box
    for _, row in yolo_data.iterrows():
        # Convert YOLO format to pixel coordinates
        x = int((row['x_center'] - row['width']/2) * img_width)
        y = int((row['y_center'] - row['height']/2) * img_height)
        w = int(row['width'] * img_width)
        h = int(row['height'] * img_height)
        
        class_id = int(row['class'])
        track_id = int(row['id'])
        conf = float(row['conf'])
        
        # Get color based on class, default to white if class not in colors
        color = colors.get(class_id, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
        
        # Add label with ID, class and confidence
        label = f"ID: {track_id} Class: {class_id} ({conf:.2%})"
        cv2.putText(image_with_boxes, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image_with_boxes

def load_frame_data(frame_number):
    # Load the frame's ground truth data
    frame_gt_path = os.path.join(os.path.dirname(config.gt_mot_path), 'frame_gt_high_conf', f'frame_{frame_number:06d}.csv')
    if os.path.exists(frame_gt_path):
        return pd.read_csv(frame_gt_path)
    return None

def load_frame_image(frame_number):
    # Load the corresponding image
    image_path = os.path.join(config.image_path, f'{frame_number:06d}.jpg')
    if os.path.exists(image_path):
        return cv2.imread(image_path)
    return None

def save_yolo_format(frame_data, frame_number, img_width, img_height):
    """Save data in YOLO format"""
    yolo_dir = os.path.join(os.path.dirname(config.gt_mot_path), 'yolo_gt_high_conf')
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)
    
    # Convert to YOLO format
    yolo_data = convert_to_yolo_format(frame_data, img_width, img_height)
    
    # Save in YOLO format
    output_path = os.path.join(yolo_dir, f'frame_{frame_number:06d}.txt')
    with open(output_path, 'w') as f:
        for _, row in yolo_data.iterrows():
            f.write(f"{int(row['class'])} {row['x_center']:.6f} {row['y_center']:.6f} {row['width']:.6f} {row['height']:.6f}\n")

def main():
    st.title("YOLO Ground Truth Label Visualization (High Confidence)")
    
    # Get the list of available frames
    frame_gt_dir = os.path.join(os.path.dirname(config.gt_mot_path), 'frame_gt_high_conf')
    if not os.path.exists(frame_gt_dir):
        st.error("High confidence ground truth directory not found!")
        return
    
    frame_files = [f for f in os.listdir(frame_gt_dir) if f.endswith('.csv')]
    frame_numbers = [int(f.split('_')[1].split('.')[0]) for f in frame_files]
    frame_numbers.sort()
    
    # Initialize session state for current frame index if it doesn't exist
    if 'current_frame_index' not in st.session_state:
        st.session_state.current_frame_index = 0
    
    # Navigation buttons and frame counter
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⏮️ Previous Frame", use_container_width=True):
            st.session_state.current_frame_index = max(0, st.session_state.current_frame_index - 1)
    
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>Frame {frame_numbers[st.session_state.current_frame_index]}</h3>", 
                   unsafe_allow_html=True)
    
    with col3:
        if st.button("Next Frame ⏭️", use_container_width=True):
            st.session_state.current_frame_index = min(len(frame_numbers) - 1, 
                                                     st.session_state.current_frame_index + 1)
    
    # Get current frame number
    selected_frame = frame_numbers[st.session_state.current_frame_index]
    
    # Load and display the image
    image = load_frame_image(selected_frame)
    if image is not None:
        img_height, img_width = image.shape[:2]
        
        # Load ground truth data
        frame_data = load_frame_data(selected_frame)
        if frame_data is not None:
            # Convert to YOLO format
            yolo_data = convert_to_yolo_format(frame_data, img_width, img_height)
            
            # Save YOLO format if not already saved
            save_yolo_format(frame_data, selected_frame, img_width, img_height)
            
            # Draw bounding boxes
            image_with_boxes = draw_bounding_boxes(image, yolo_data, img_width, img_height)
            
            # Display both original and annotated images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, channels="BGR", caption="Original Image", use_column_width=True)
            with col2:
                st.image(image_with_boxes, channels="BGR", caption="Image with YOLO Bounding Boxes", use_column_width=True)
            
            # Create tabs for data and statistics
            tab1, tab2 = st.tabs(["YOLO Format Data", "Statistics"])
            
            with tab1:
                # Display YOLO format data
                yolo_display = yolo_data[['class', 'x_center', 'y_center', 'width', 'height', 'conf']]
                st.dataframe(yolo_display, use_container_width=True)
            
            with tab2:
                # Display statistics
                st.subheader("Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Objects", len(yolo_data))
                with col2:
                    st.metric("Unique IDs", yolo_data['id'].nunique())
                with col3:
                    st.metric("Average Confidence", f"{yolo_data['conf'].mean():.2%}")
                
                # Class distribution
                st.subheader("Class Distribution")
                class_counts = yolo_data['class'].value_counts()
                st.bar_chart(class_counts)
        else:
            st.error(f"Ground truth data not found for frame {selected_frame}")
    else:
        st.error(f"Image not found for frame {selected_frame}")

if __name__ == "__main__":
    main() 