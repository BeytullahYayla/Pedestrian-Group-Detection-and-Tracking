from ultralytics import YOLO
from Config import Config
import os
import cv2
from datetime import datetime
import yaml
import torch
from argparse import ArgumentParser

def argparser():
    parser = ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--video', type=str, default='', help='Path to the video file')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Model name')
    parser.add_argument('--train',type=bool, default=False, help='Train the model')
    return parser.parse_args()

class YoloObjectDetector:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.available_models = ['yolo11n.pt']
        
    def load_model(self, model_name='yolo11n.pt'):
        """Load a YOLO model"""
        try:
            # Get the absolute path to the data.yaml file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_yaml_path = os.path.join(current_dir, 'data.yaml')
            
            # Verify data.yaml exists
            if not os.path.exists(data_yaml_path):
                raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")
            
            # Load the model
            self.model = YOLO(model_name)
            print(f"Model {model_name} loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def train(self, epochs=70, batch_size=1, img_size=640):
        """Train the YOLO model"""
        try:
            if self.model is None:
                print("Please load a model first using load_model()")
                return False
            
            # Get the absolute path to the data.yaml file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_yaml_path = os.path.join(current_dir, 'data.yaml')
            
            # Verify data.yaml exists
            if not os.path.exists(data_yaml_path):
                raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")
            
            # Print dataset information
            print("\nDataset Information:")
            print(f"data.yaml path: {data_yaml_path}")
            print(f"Dataset root: {os.path.dirname(data_yaml_path)}")
            
            # Start training
            print("\nStarting training...")
            results = self.model.train(
                data=data_yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device='cuda'
            )
            
            print("Training completed successfully")
            return True
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False

    def detect(self, image, conf=0.25):
        """Detect objects in an image"""
        if self.model is None:
            print("Please load a model first using load_model()")
            return None
        
        try:
            results = self.model(image, conf=conf)
            return results
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return None

    def list_available_models(self):
        """List available YOLO models"""
        print("Available models:")
        for model in self.available_models:
            print(f"- {model}")
            
    def inference_from_video(self, video_path="C:/Users/MSI/Documents/GitHub/Pedestrian-Group-Detection-and-Tracking/src/runs/detect/train7/weights/best.pt"):
        """Inference from a video"""
        if self.model is None:
            print("Please load a model first using load_model()")
            return None

        try:
            cap = cv2.VideoCapture(video_path)
            
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            output_path = os.path.splitext(video_path)[0] + '_detected.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                results = self.model(frame)
                
                annotated_frame = results[0].plot()
                
                out.write(annotated_frame)
                
                cv2.imshow('YOLO Object Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            print(f"Output video saved to: {output_path}")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False

def main():
    args = argparser()
    detector = YoloObjectDetector()
    detector.list_available_models()
    
    print("\nStarting training...")
    detector.load_model()
    if args.train:
        detector.train()
    if args.video:
        detector.inference_from_video(args.video)

if __name__ == "__main__":
    main()


    
    
    
