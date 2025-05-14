from ultralytics import YOLO
from Config import Config
import os
import cv2
from datetime import datetime
import yaml
import torch

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

    def train(self, epochs=100, batch_size=4, img_size=640):
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
                device='cpu'
            )
            
            print("Training completed successfully")
            return True
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False

    def detect(self, image):
        """Detect objects in an image"""
        if self.model is None:
            print("Please load a model first using load_model()")
            return None
        
        try:
            results = self.model(image)
            return results
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return None

    def list_available_models(self):
        """List available YOLO models"""
        print("Available models:")
        for model in self.available_models:
            print(f"- {model}")

def main():
    detector = YoloObjectDetector()
    detector.list_available_models()
    
    print("\nStarting training...")
    detector.load_model()
    detector.train()

if __name__ == "__main__":
    main()


    
    
    
