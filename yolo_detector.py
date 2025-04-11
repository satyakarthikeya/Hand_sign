import cv2
from ultralytics import YOLO
import os

def main():
    # Load the YOLOv8 model
    model_path = 'best.pt'  # Default model path
    
    # If the model doesn't exist, use the last trained model
    if not os.path.exists(model_path):
        model_dirs = [d for d in os.listdir('runs/detect') if d.startswith('hand_detection_model')]
        if model_dirs:
            latest_model = sorted(model_dirs)[-1]
            model_path = os.path.join('runs', 'detect', latest_model, 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        print("No trained model found. Please train the model first using train.py")
        return
        
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting real-time hand detection... Press 'q' to quit")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.3)  # Lower confidence threshold for better detection

        # Visualize the results on the frame
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence score
                cv2.putText(frame, f'Hand: {conf:.2f}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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