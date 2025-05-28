import os
import cv2
import argparse
import numpy as np
from YoloObjectDetector import YoloObjectDetector
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
import time
from collections import defaultdict, deque

# Dictionary to store trajectories for each track ID
trajectories = defaultdict(lambda: deque(maxlen=30))  # Store last 30 positions

def visualize_tracking(image, boxes, track_ids, fps=None):
    # Create a copy of the image
    vis_image = image.copy()
    
    # Draw trajectories first (so they appear behind the boxes)
    for box, track_id in zip(boxes, track_ids):
        # Get center point of the box
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Add current position to trajectory
        trajectories[track_id].append((center_x, center_y))
        
        # Draw trajectory
        if len(trajectories[track_id]) > 1:
            # Generate a unique color based on track_id
            color = tuple(map(int, np.random.RandomState(track_id).randint(0, 255, size=3)))
            
            # Draw lines connecting previous positions
            points = list(trajectories[track_id])
            for i in range(1, len(points)):
                cv2.line(vis_image, points[i-1], points[i], color, 2)
    
    # Draw bounding boxes and track IDs for each person
    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = box
        # Generate a unique color based on track_id
        color = tuple(map(int, np.random.RandomState(track_id).randint(0, 255, size=3)))
        
        # Draw bounding box
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw track ID
        cv2.putText(vis_image, f"ID: {track_id}", (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw FPS if provided
    if fps is not None:
        cv2.putText(vis_image, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return vis_image

class Args:
    def __init__(self):
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.min_box_area = 10
        self.mot20 = False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Person Detection and Tracking')
    parser.add_argument('--track', action='store_true', help='Enable tracking with ByteTrack')
    args = parser.parse_args()
    
    # Initialize detector
    detector = YoloObjectDetector()
    
    # Initialize ByteTrack tracker with proper configuration
    tracker_args = Args()
    tracker = BYTETracker(tracker_args)
    timer = Timer()
    
    # Load the pre-trained model
    model_path = 'best.pt'
    detector.load_model(model_path)
    
    # Path to test images
    test_images_path = r'C:\Users\MSI\Documents\GitHub\Pedestrian-Group-Detection-and-Tracking\dataset\MOT17\test\MOT17-07-SDP\img1'
    
    # Create output directory for annotated images
    output_dir = 'output_detections'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the first image to determine video dimensions
    first_image = cv2.imread(os.path.join(test_images_path, sorted(os.listdir(test_images_path))[0]))
    height, width = first_image.shape[:2]
    
    # Initialize video writer
    video_path = os.path.join(output_dir, 'detection_results.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # MOT17 dataset typically uses 30 fps
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Process each image in the directory
    for filename in sorted(os.listdir(test_images_path)):
        if filename.endswith(('.jpg', '.png')):
            # Read image
            image_path = os.path.join(test_images_path, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update FPS every 30 frames
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Perform detection with confidence threshold
            results = detector.detect(image, conf=0.5)
            
            if results is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                
                if len(boxes) > 0:  # Only process if there are detections
                    # Prepare detections for tracking
                    detections = np.concatenate([boxes, scores[:, None]], axis=1)
                    
                    # Update tracker
                    timer.tic()
                    online_targets = tracker.update(detections, [height, width], [height, width])
                    timer.toc()
                    
                    # Get tracking results
                    online_tlwhs = []
                    online_ids = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                    
                    # Convert tlwh to xyxy format
                    online_boxes = []
                    for tlwh in online_tlwhs:
                        x1, y1, w, h = tlwh
                        online_boxes.append([x1, y1, x1 + w, y1 + h])
                    
                    # Visualize tracking results with FPS
                    annotated_image = visualize_tracking(image, online_boxes, online_ids, fps)
                else:
                    # No detections, just use the original image
                    annotated_image = visualize_tracking(image, [], [], fps)
                
                # Save annotated image
                output_path = os.path.join(output_dir, f'detected_{filename}')
                cv2.imwrite(output_path, annotated_image)
                
                # Write frame to video
                video_writer.write(annotated_image)
                
                # Display image
                cv2.imshow('Person Detection and Tracking', annotated_image)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Release video writer
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Detection results saved in: {output_dir}")
    print(f"Video saved as: {video_path}")

if __name__ == "__main__":
    main() 