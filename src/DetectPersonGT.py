import os
import cv2
import numpy as np
from GroupDetection import compute_foot_points, zone_grouping

def visualize_groups(image, foot_points, group_labels, boxes,zone_threshold):
    # Generate random colors for each group
    unique_labels = np.unique(group_labels[group_labels != -1])
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3), dtype=np.uint8)
    
    # Create a copy of the image
    vis_image = image.copy()
    
    # First draw individual bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    
    # Sort foot points by y-coordinate to find zone boundaries
    sorted_indices = np.argsort(foot_points[:, 1])
    foot_points_sorted = foot_points[sorted_indices]
    
    # Draw zones
    zones = []
    current_zone = [foot_points_sorted[0]]
    zone_boundaries = []
    
    for i in range(1, len(foot_points_sorted)):
        if abs(foot_points_sorted[i][1] - foot_points_sorted[i-1][1]) > zone_threshold:  # y_thresh
            zones.append(np.array(current_zone))
            # Calculate zone boundary (average y-coordinate between zones)
            boundary_y = (foot_points_sorted[i-1][1] + foot_points_sorted[i][1]) / 2
            zone_boundaries.append(boundary_y)
            current_zone = []
        current_zone.append(foot_points_sorted[i])
    zones.append(np.array(current_zone))
    
    # Draw zone boundaries
    for boundary_y in zone_boundaries:
        cv2.line(vis_image, (0, int(boundary_y)), (vis_image.shape[1], int(boundary_y)), (255, 0, 0), 2)
    
    # Then draw group bounding boxes
    for label in unique_labels:
        group_points = foot_points[group_labels == label]
        color = tuple(map(int, colors[label % len(colors)]))
        
        if len(group_points) > 0:
            # Calculate group bounding box
            min_x = np.min(group_points[:, 0])
            min_y = np.min(group_points[:, 1])
            max_x = np.max(group_points[:, 0])
            max_y = np.max(group_points[:, 1])
            
            # Add some padding to the bounding box
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(vis_image.shape[1], max_x + padding)
            max_y = min(vis_image.shape[0], max_y + padding)
            
            # Draw the bounding box
            cv2.rectangle(vis_image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color, 2)
            
            # Draw group label
            label_text = f"Group {label + 1}"
            cv2.putText(vis_image, label_text, (int(min_x), int(min_y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw points
            for point in group_points:
                cv2.circle(vis_image, (int(point[0]), int(point[1])), 5, color, -1)
    
    return vis_image

def read_yolo_labels(label_file, img_width, img_height):
    """Read YOLO format labels and convert to (x1,y1,x2,y2) format."""
    boxes = []
    with open(label_file, 'r') as f:
        for line in f:
            # YOLO format: <class> <x_center> <y_center> <width> <height>
            values = line.strip().split()
            if len(values) == 5:  # Only process if it's a valid line
                class_id = int(values[0])
                if class_id == 0:  # Only process person class (class 0)
                    x_center = float(values[1]) * img_width
                    y_center = float(values[2]) * img_height
                    width = float(values[3]) * img_width
                    height = float(values[4]) * img_height
                    
                    # Convert to (x1,y1,x2,y2) format
                    x1 = x_center - width/2
                    y1 = y_center - height/2
                    x2 = x_center + width/2
                    y2 = y_center + height/2
                    
                    boxes.append([x1, y1, x2, y2])
    return np.array(boxes)

def main():
    # Path to validation data
    base_path = r'C:\Users\MSI\Documents\GitHub\Pedestrian-Group-Detection-and-Tracking\dataset\MOT17\train\MOT17-09-DPM\yolo_dataset'
    images_path = os.path.join(base_path, 'images', 'val')
    labels_path = os.path.join(base_path, 'labels', 'val')
    
    # Create output directory for annotated images
    output_dir = 'output_detections_val_09'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the first image to determine video dimensions
    first_image = cv2.imread(os.path.join(images_path, sorted(os.listdir(images_path))[0]))
    height, width = first_image.shape[:2]
    
    # Initialize video writer
    video_path = os.path.join(output_dir, 'group_detection_val_09_results.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Process each image
    for filename in sorted(os.listdir(images_path)):
        if filename.endswith(('.jpg', '.png')):
            # Read image
            image_path = os.path.join(images_path, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            
            # Read corresponding label file
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(labels_path, label_filename)
            
            if os.path.exists(label_path):
                # Read and process YOLO format labels
                boxes = read_yolo_labels(label_path, width, height)
                
                if len(boxes) > 0:
                    # Compute foot points and group them
                    foot_points = compute_foot_points(boxes)
                    zone_labels = zone_grouping(foot_points, y_thresh=70)
                    
                    # Visualize groups and individual boxes
                    annotated_image = visualize_groups(image, foot_points, zone_labels, boxes, zone_threshold=70)
                else:
                    # No detections for this frame
                    annotated_image = image.copy()
            else:
                # No label file found
                annotated_image = image.copy()
            
            # Save annotated image
            output_path = os.path.join(output_dir, f'group_detected_val_09_{filename}')
            cv2.imwrite(output_path, annotated_image)
            
            # Write frame to video
            video_writer.write(annotated_image)
            
            # Display image
            cv2.imshow('Group Detection (Validation)', annotated_image)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release video writer
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Validation detection results saved in: {output_dir}")
    print(f"Video saved as: {video_path}")

if __name__ == "__main__":
    main() 