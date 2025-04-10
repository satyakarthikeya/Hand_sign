import cv2
from ultralytics import YOLO
import os
import numpy as np
import argparse
import torch

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Real-time hand detection with YOLOv8")
    parser.add_argument('--model', type=str, default='best.pt', help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='', help='Device to use (cpu, cuda, etc.)')
    args = parser.parse_args()
    
    # Set device for detection
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load the YOLOv8 model
    model_path = args.model
    
    # If the model doesn't exist, use the last trained model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, looking for alternatives...")
        if os.path.exists('runs/detect'):
            model_dirs = [d for d in os.listdir('runs/detect') if d.startswith('hand_detection_model')]
            if model_dirs:
                latest_model = sorted(model_dirs)[-1]
                model_path = os.path.join('runs', 'detect', latest_model, 'weights', 'best.pt')
                print(f"Using alternative model from: {model_path}")
    
    if not os.path.exists(model_path):
        print("No trained model found. Please train the model first or specify a valid model path.")
        return
        
    print(f"Loading model from {model_path}")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
        
        # Test model with a dummy image
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = model(dummy_img, verbose=False)  # This will verify the model works
        print("Model test inference successful")
    except Exception as e:
        print(f"Error loading or testing model: {e}")
        print("Trying alternative loading method...")
        try:
            model = YOLO(model_path, task='detect')
            print("Model loaded successfully with explicit task parameter.")
            _ = model(dummy_img, verbose=False)
            print("Model test inference successful")
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            return

    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print(f"Starting real-time hand detection with {os.path.basename(model_path)}... Press 'q' to quit")
    print(f"Confidence threshold: {args.conf}")
    
    # Define colors for different confidence levels
    HIGH_CONF_COLOR = (0, 255, 0)     # Green for high confidence (>0.7)
    MED_CONF_COLOR = (0, 255, 255)    # Yellow for medium confidence (0.5-0.7)
    LOW_CONF_COLOR = (0, 0, 255)      # Red for low confidence (<0.5)
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        try:
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=args.conf, verbose=False)

            # Visualize the results on the frame
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                for box in boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    
                    # Determine color based on confidence
                    if conf > 0.7:
                        box_color = HIGH_CONF_COLOR
                    elif conf > 0.5:
                        box_color = MED_CONF_COLOR
                    else:
                        box_color = LOW_CONF_COLOR
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Add confidence score
                    cv2.putText(frame, f'Hand: {conf:.2f}', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Add model info overlay
            cv2.putText(frame, f"Model: {os.path.basename(model_path)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add device info overlay
            cv2.putText(frame, f"Device: {device}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        except Exception as e:
            print(f"Error during inference: {e}")
            cv2.putText(frame, "Detection Error!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Real-time Hand Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()