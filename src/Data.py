import cv2
import numpy as np
from Config import Config
import os
import pandas as pd
import tqdm 
import shutil

config = Config()

def load_image(image_path):
    image = cv2.imread(image_path)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def load_gt_mot(gt_mot_path):
    columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']
    gt_mot = pd.read_csv(gt_mot_path, header=None, names=columns)
    return gt_mot

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

def split_gt_by_frame(gt_mot):
    # Create a new directory for frame-wise ground truth files
    output_dir = os.path.join(os.path.dirname(config.gt_mot_path), 'frame_gt_high_conf')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a new directory for YOLO format ground truth files
    yolo_output_dir = os.path.join(os.path.dirname(config.gt_mot_path), 'yolo_gt_high_conf')
    if not os.path.exists(yolo_output_dir):
        os.makedirs(yolo_output_dir)
    
    gt_mot_list = []
    unique_frames = gt_mot['frame'].unique()
    
    # Get image dimensions from the first image
    image_path = os.path.join(config.image_path, os.listdir(config.image_path)[0])
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    
    for frame in tqdm.tqdm(unique_frames, desc="Processing frames"):
        # Filter high confidence data (confidence > 0.6)
        frame_gt = gt_mot[gt_mot['frame'] == frame].reset_index(drop=True)
        frame_gt_high_conf = frame_gt[frame_gt['conf'] > 0.6].reset_index(drop=True)
        
        if len(frame_gt_high_conf) > 0:  # Only process frames with high confidence detections
            gt_mot_list.append(frame_gt_high_conf)
            
            # Save each frame's ground truth as CSV
            output_file = os.path.join(output_dir, f'frame_{frame:06d}.csv')
            frame_gt_high_conf.to_csv(output_file, index=False)
            
            # Convert to YOLO format and save
            yolo_gt = convert_to_yolo_format(frame_gt_high_conf, image_width, image_height)
            yolo_output_file = os.path.join(yolo_output_dir, f'frame_{frame:06d}.txt')
            with open(yolo_output_file, 'w') as f:
                f.write('\n'.join(yolo_gt))
    
    return gt_mot_list

def organize_yolo_dataset():
    """Organize YOLO labels and images in the same directory"""
    try:
        # Create a new directory for the organized dataset
        dataset_dir = os.path.join(os.path.dirname(config.gt_mot_path), 'yolo_dataset')
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
        
        # Get list of YOLO label files
        yolo_dir = os.path.join(os.path.dirname(config.gt_mot_path), 'yolo_gt_high_conf')
        if not os.path.exists(yolo_dir):
            raise FileNotFoundError(f"YOLO labels directory not found: {yolo_dir}")
        
        yolo_files = [f for f in os.listdir(yolo_dir) if f.endswith('.txt')]
        if not yolo_files:
            raise FileNotFoundError(f"No label files found in {yolo_dir}")
        
        yolo_files.sort()  # Sort to ensure consistent splitting
        print(f"Found {len(yolo_files)} label files in {yolo_dir}")
        
        # Calculate split index (80% train, 20% validation)
        split_idx = int(len(yolo_files) * 0.8)
        train_files = yolo_files[:split_idx]
        val_files = yolo_files[split_idx:]
        
        print(f"\nSplitting dataset:")
        print(f"Total files: {len(yolo_files)}")
        print(f"Train files: {len(train_files)}")
        print(f"Validation files: {len(val_files)}")
        
        # Process train files
        train_copied = 0
        for yolo_file in tqdm.tqdm(train_files, desc="Processing train files"):
            # Get frame number from filename
            frame_number = yolo_file.split('_')[1].split('.')[0]
            
            # Source paths
            yolo_src = os.path.join(yolo_dir, yolo_file)
            image_src = os.path.join(config.image_path, f'{frame_number}.jpg')
            
            # Destination paths
            yolo_dst = os.path.join(train_labels_dir, f'frame{frame_number}.txt')
            image_dst = os.path.join(train_images_dir, f'frame{frame_number}.jpg')
            
            # Copy files if they exist
            if os.path.exists(yolo_src) and os.path.exists(image_src):
                shutil.copy2(yolo_src, yolo_dst)
                shutil.copy2(image_src, image_dst)
                train_copied += 1
            else:
                print(f"Warning: Missing files for frame {frame_number}")
                if not os.path.exists(yolo_src):
                    print(f"  Label file not found: {yolo_src}")
                if not os.path.exists(image_src):
                    print(f"  Image file not found: {image_src}")
        
        # Process validation files
        val_copied = 0
        for yolo_file in tqdm.tqdm(val_files, desc="Processing validation files"):
            # Get frame number from filename
            frame_number = yolo_file.split('_')[1].split('.')[0]
            
            # Source paths
            yolo_src = os.path.join(yolo_dir, yolo_file)
            image_src = os.path.join(config.image_path, f'{frame_number}.jpg')
            
            # Destination paths
            yolo_dst = os.path.join(val_labels_dir, f'frame{frame_number}.txt')
            image_dst = os.path.join(val_images_dir, f'frame{frame_number}.jpg')
            
            # Copy files if they exist
            if os.path.exists(yolo_src) and os.path.exists(image_src):
                shutil.copy2(yolo_src, yolo_dst)
                shutil.copy2(image_src, image_dst)
                val_copied += 1
            else:
                print(f"Warning: Missing files for frame {frame_number}")
                if not os.path.exists(yolo_src):
                    print(f"  Label file not found: {yolo_src}")
                if not os.path.exists(image_src):
                    print(f"  Image file not found: {image_src}")
        
        # Verify the dataset structure
        train_images = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
        train_labels = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
        val_images = [f for f in os.listdir(val_images_dir) if f.endswith('.jpg')]
        val_labels = [f for f in os.listdir(val_labels_dir) if f.endswith('.txt')]
        
        print("\nDataset verification:")
        print(f"Train images copied: {train_copied}")
        print(f"Train images found: {len(train_images)}")
        print(f"Train labels found: {len(train_labels)}")
        print(f"Validation images copied: {val_copied}")
        print(f"Validation images found: {len(val_images)}")
        print(f"Validation labels found: {len(val_labels)}")
        
        if not train_images or not train_labels:
            raise RuntimeError("No training data found after organization")
        if not val_images or not val_labels:
            raise RuntimeError("No validation data found after organization")
        
        print(f"\nDataset successfully organized in: {dataset_dir}")
        print(f"Train images in: {train_images_dir}")
        print(f"Train labels in: {train_labels_dir}")
        print(f"Validation images in: {val_images_dir}")
        print(f"Validation labels in: {val_labels_dir}")
        print(f"Labelmap at: {labelmap_path}")
        
    except Exception as e:
        print(f"Error organizing dataset: {str(e)}")
        raise

def split_dataset():
    """Split the dataset into train and validation sets"""
    # Source directory
    dataset_dir = os.path.join(os.path.dirname(config.gt_mot_path), 'yolo_dataset')
    
    # Create train and val directories
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # Sort to ensure consistent splitting
    
    # Calculate split index (80% train, 20% validation)
    split_idx = int(len(image_files) * 0.8)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Copy files to respective directories
    for img_file in train_images:
        # Copy image
        src_img = os.path.join(dataset_dir, img_file)
        dst_img = os.path.join(train_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        # Copy corresponding label
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        src_label = os.path.join(dataset_dir, label_file)
        dst_label = os.path.join(train_dir, label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
    
    for img_file in val_images:
        # Copy image
        src_img = os.path.join(dataset_dir, img_file)
        dst_img = os.path.join(val_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        # Copy corresponding label
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        src_label = os.path.join(dataset_dir, label_file)
        dst_label = os.path.join(val_dir, label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
    
    print(f"Dataset split completed. Train set in {train_dir}, Validation set in {val_dir}")

def cleanup_dataset():
    """Clean up old dataset files"""
    base_dir = os.path.dirname(config.gt_mot_path)
    dirs_to_clean = [
        'frame_gt_high_conf',
        'yolo_gt_high_conf',
        'yolo_dataset'
    ]
    
    for dir_name in dirs_to_clean:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"Removing {dir_path}")
            shutil.rmtree(dir_path)

def main():
    # Clean up old dataset files
    cleanup_dataset()
    
    image_path = config.image_path
    image_list = os.listdir(image_path)
    gt_mot_path = config.gt_mot_path
    gt_mot_list = os.listdir(gt_mot_path)
    image_path = os.path.join(image_path, image_list[0])
    gt_mot_path = os.path.join(gt_mot_path, gt_mot_list[0])
    image = load_image(image_path)
    gt_mot = load_gt_mot(gt_mot_path)
    gt_mot_list = split_gt_by_frame(gt_mot)
    
    # Organize YOLO dataset
    organize_yolo_dataset()
    
    # Split dataset into train and validation sets
    split_dataset()

if __name__ == "__main__":
    main()

