import cv2
import numpy as np
from Config import Config
import os
import pandas as pd
import tqdm 
import shutil
import traceback

config = Config()

def load_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        raise

def load_gt_mot(gt_mot_path):
    try:
        columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']
        gt_mot = pd.read_csv(gt_mot_path, header=None, names=columns)
        return gt_mot
    except Exception as e:
        print(f"Error loading ground truth file {gt_mot_path}: {str(e)}")
        raise

def convert_to_yolo_format(frame_gt, image_width, image_height):
    # Convert to YOLO format (normalized coordinates)
    yolo_gt = frame_gt.copy()
    
    # Ensure coordinates are within image bounds
    yolo_gt['x'] = yolo_gt['x'].clip(0, image_width)
    yolo_gt['y'] = yolo_gt['y'].clip(0, image_height)
    yolo_gt['w'] = yolo_gt['w'].clip(0, image_width - yolo_gt['x'])
    yolo_gt['h'] = yolo_gt['h'].clip(0, image_height - yolo_gt['y'])
    
    # Calculate center coordinates and normalize
    yolo_gt['x_center'] = (yolo_gt['x'] + yolo_gt['w']/2) / image_width
    yolo_gt['y_center'] = (yolo_gt['y'] + yolo_gt['h']/2) / image_height
    yolo_gt['width'] = yolo_gt['w'] / image_width
    yolo_gt['height'] = yolo_gt['h'] / image_height
    
    # Ensure normalized values are between 0 and 1
    yolo_gt['x_center'] = yolo_gt['x_center'].clip(0, 1)
    yolo_gt['y_center'] = yolo_gt['y_center'].clip(0, 1)
    yolo_gt['width'] = yolo_gt['width'].clip(0, 1)
    yolo_gt['height'] = yolo_gt['height'].clip(0, 1)
    
    # Adjust class indices to start from 0
    yolo_gt['class'] = yolo_gt['class'] - 1
    
    # Create YOLO format string (class_id x_center y_center width height)
    yolo_gt['yolo_format'] = yolo_gt.apply(
        lambda row: f"{int(row['class'])} {row['x_center']:.6f} {row['y_center']:.6f} {row['width']:.6f} {row['height']:.6f}",
        axis=1
    )
    
    return yolo_gt['yolo_format'].tolist()

def process_sequence(sequence_path):
    """Process a single MOT17 sequence"""
    try:
        print(f"\nProcessing sequence: {sequence_path}")
        
        # Get paths for the sequence
        image_path = os.path.join(sequence_path, 'img1')
        gt_mot_path = os.path.join(sequence_path, 'gt', 'gt.txt')
        
        print(f"Image path: {image_path}")
        print(f"Ground truth path: {gt_mot_path}")
        
        # Verify paths exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image directory not found: {image_path}")
        if not os.path.exists(gt_mot_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_mot_path}")
        
        # Load ground truth
        gt_mot = load_gt_mot(gt_mot_path)
        print(f"Loaded ground truth with {len(gt_mot)} entries")
        
        # Create output directories for this sequence
        sequence_name = os.path.basename(sequence_path)
        output_dir = os.path.join(os.path.dirname(sequence_path), 'processed', sequence_name)
        frame_gt_dir = os.path.join(output_dir, 'frame_gt_high_conf')
        yolo_gt_dir = os.path.join(output_dir, 'yolo_gt_high_conf')
        
        print(f"Creating output directories:")
        print(f"Output dir: {output_dir}")
        print(f"Frame GT dir: {frame_gt_dir}")
        print(f"YOLO GT dir: {yolo_gt_dir}")
        
        os.makedirs(frame_gt_dir, exist_ok=True)
        os.makedirs(yolo_gt_dir, exist_ok=True)
        
        # Get image dimensions from the first image
        first_image = os.path.join(image_path, os.listdir(image_path)[0])
        print(f"Loading first image: {first_image}")
        image = load_image(first_image)
        image_height, image_width = image.shape[:2]
        print(f"Image dimensions: {image_width}x{image_height}")
        
        # Process each frame
        unique_frames = gt_mot['frame'].unique()
        print(f"Processing {len(unique_frames)} unique frames")
        
        processed_frames = 0
        for frame in tqdm.tqdm(unique_frames, desc=f"Processing {sequence_name}"):
            # Filter high confidence data (confidence > 0.6)
            frame_gt = gt_mot[gt_mot['frame'] == frame].reset_index(drop=True)
            frame_gt_high_conf = frame_gt[frame_gt['conf'] > 0.6].reset_index(drop=True)
            
            if len(frame_gt_high_conf) > 0:
                # Save frame ground truth
                output_file = os.path.join(frame_gt_dir, f'frame_{frame:06d}.csv')
                frame_gt_high_conf.to_csv(output_file, index=False)
                
                # Convert to YOLO format and save
                yolo_gt = convert_to_yolo_format(frame_gt_high_conf, image_width, image_height)
                yolo_output_file = os.path.join(yolo_gt_dir, f'frame_{frame:06d}.txt')
                with open(yolo_output_file, 'w') as f:
                    f.write('\n'.join(yolo_gt))
                
                processed_frames += 1
        
        print(f"Processed {processed_frames} frames with high confidence detections")
        return output_dir
        
    except Exception as e:
        print(f"Error processing sequence {sequence_path}:")
        print(traceback.format_exc())
        raise

def organize_yolo_dataset(processed_dirs):
    """Organize YOLO labels and images in the same directory"""
    try:
        # Create a new directory for the organized dataset
        dataset_dir = os.path.join(os.path.dirname(processed_dirs[0]), 'yolo_dataset')
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            print(f"Created dataset directory: {dataset_dir}")
        
        # Create directory structure
        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')
        
        # Create train and val directories for both images and labels
        train_images_dir = os.path.join(images_dir, 'train')
        val_images_dir = os.path.join(images_dir, 'val')
        train_labels_dir = os.path.join(labels_dir, 'train')
        val_labels_dir = os.path.join(labels_dir, 'val')
        
        # Create all directories
        for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
        
        # Create labelmap.txt
        labelmap = {
            0: "pedestrian"
        }
        
        labelmap_path = os.path.join(dataset_dir, 'labelmap.txt')
        with open(labelmap_path, 'w') as f:
            for class_id, class_name in labelmap.items():
                f.write(f"{class_id} {class_name}\n")
        print(f"Created labelmap at: {labelmap_path}")
        
        # Process each sequence
        all_yolo_files = []
        sequence_paths = {}  # Store original sequence paths
        
        for processed_dir in processed_dirs:
            yolo_dir = os.path.join(processed_dir, 'yolo_gt_high_conf')
            if not os.path.exists(yolo_dir):
                print(f"Warning: YOLO labels directory not found: {yolo_dir}")
                continue
            
            # Get the original sequence path
            sequence_name = os.path.basename(processed_dir)
            original_sequence_path = os.path.join(os.path.dirname(os.path.dirname(processed_dir)), sequence_name)
            sequence_paths[sequence_name] = original_sequence_path
            
            yolo_files = [f for f in os.listdir(yolo_dir) if f.endswith('.txt')]
            all_yolo_files.extend([(yolo_dir, f, sequence_name) for f in yolo_files])
        
        if not all_yolo_files:
            raise FileNotFoundError("No label files found in any sequence")
        
        # Sort files to ensure consistent splitting
        all_yolo_files.sort(key=lambda x: x[1])
        print(f"Found {len(all_yolo_files)} total label files")
        
        # Calculate split index (80% train, 20% validation)
        split_idx = int(len(all_yolo_files) * 0.8)
        train_files = all_yolo_files[:split_idx]
        val_files = all_yolo_files[split_idx:]
        
        print(f"\nSplitting dataset:")
        print(f"Total files: {len(all_yolo_files)}")
        print(f"Train files: {len(train_files)}")
        print(f"Validation files: {len(val_files)}")
        
        def copy_file_safely(src, dst):
            """Safely copy a file with error handling"""
            try:
                if not os.path.exists(src):
                    print(f"Source file not found: {src}")
                    return False
                
                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                
                # Check if destination file exists and is different
                if os.path.exists(dst):
                    if os.path.getsize(src) == os.path.getsize(dst):
                        print(f"File already exists and is identical: {dst}")
                        return True
                    else:
                        print(f"Overwriting existing file: {dst}")
                
                shutil.copy2(src, dst)
                return True
            except Exception as e:
                print(f"Error copying {src} to {dst}: {str(e)}")
                return False
        
        # Process train files
        train_copied = 0
        train_errors = 0
        for yolo_dir, yolo_file, sequence_name in tqdm.tqdm(train_files, desc="Processing train files"):
            try:
                # Get frame number from filename
                frame_number = yolo_file.split('_')[1].split('.')[0]
                
                # Source paths
                yolo_src = os.path.join(yolo_dir, yolo_file)
                image_src = os.path.join(sequence_paths[sequence_name], 'img1', f'{frame_number}.jpg')
                
                # Destination paths
                yolo_dst = os.path.join(train_labels_dir, f'{sequence_name}_frame{frame_number}.txt')
                image_dst = os.path.join(train_images_dir, f'{sequence_name}_frame{frame_number}.jpg')
                
                # Copy both files
                yolo_success = copy_file_safely(yolo_src, yolo_dst)
                image_success = copy_file_safely(image_src, image_dst)
                
                if yolo_success and image_success:
                    train_copied += 1
                else:
                    train_errors += 1
                    
            except Exception as e:
                print(f"Error processing train file {yolo_file}: {str(e)}")
                train_errors += 1
        
        # Process validation files
        val_copied = 0
        val_errors = 0
        for yolo_dir, yolo_file, sequence_name in tqdm.tqdm(val_files, desc="Processing validation files"):
            try:
                # Get frame number from filename
                frame_number = yolo_file.split('_')[1].split('.')[0]
                
                # Source paths
                yolo_src = os.path.join(yolo_dir, yolo_file)
                image_src = os.path.join(sequence_paths[sequence_name], 'img1', f'{frame_number}.jpg')
                
                # Destination paths
                yolo_dst = os.path.join(val_labels_dir, f'{sequence_name}_frame{frame_number}.txt')
                image_dst = os.path.join(val_images_dir, f'{sequence_name}_frame{frame_number}.jpg')
                
                # Copy both files
                yolo_success = copy_file_safely(yolo_src, yolo_dst)
                image_success = copy_file_safely(image_src, image_dst)
                
                if yolo_success and image_success:
                    val_copied += 1
                else:
                    val_errors += 1
                    
            except Exception as e:
                print(f"Error processing validation file {yolo_file}: {str(e)}")
                val_errors += 1
        
        print("\nDataset verification:")
        print(f"Train images copied: {train_copied}")
        print(f"Train errors: {train_errors}")
        print(f"Validation images copied: {val_copied}")
        print(f"Validation errors: {val_errors}")
        
        # Verify the dataset structure
        train_images = len([f for f in os.listdir(train_images_dir) if f.endswith('.jpg')])
        train_labels = len([f for f in os.listdir(train_labels_dir) if f.endswith('.txt')])
        val_images = len([f for f in os.listdir(val_images_dir) if f.endswith('.jpg')])
        val_labels = len([f for f in os.listdir(val_labels_dir) if f.endswith('.txt')])
        
        print("\nFinal dataset structure:")
        print(f"Train images: {train_images}")
        print(f"Train labels: {train_labels}")
        print(f"Validation images: {val_images}")
        print(f"Validation labels: {val_labels}")
        
        if train_images != train_labels or val_images != val_labels:
            print("\nWarning: Mismatch between number of images and labels!")
            print("Please check the dataset manually.")
        
        print(f"\nDataset successfully organized in: {dataset_dir}")
        print(f"Train images in: {train_images_dir}")
        print(f"Train labels in: {train_labels_dir}")
        print(f"Validation images in: {val_images_dir}")
        print(f"Validation labels in: {val_labels_dir}")
        print(f"Labelmap at: {labelmap_path}")
        
    except Exception as e:
        print(f"Error organizing dataset: {str(e)}")
        raise

def main():
    try:
        # Define the sequences to process
        sequences = [
            r"C:\Users\MSI\Documents\GitHub\Pedestrian-Group-Detection-and-Tracking\dataset\MOT17\train\MOT17-02-SDP",
            r"C:\Users\MSI\Documents\GitHub\Pedestrian-Group-Detection-and-Tracking\dataset\MOT17\train\MOT17-09-SDP",
            r"C:\Users\MSI\Documents\GitHub\Pedestrian-Group-Detection-and-Tracking\dataset\MOT17\train\MOT17-11-SDP"
        ]
        
        # Verify sequences exist
        for seq in sequences:
            if not os.path.exists(seq):
                raise FileNotFoundError(f"Sequence directory not found: {seq}")
        
        # Process each sequence
        processed_dirs = []
        for sequence_path in sequences:
            processed_dir = process_sequence(sequence_path)
            processed_dirs.append(processed_dir)
        
        # Organize the combined dataset
        organize_yolo_dataset(processed_dirs)
        
    except Exception as e:
        print("Error in main:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()

