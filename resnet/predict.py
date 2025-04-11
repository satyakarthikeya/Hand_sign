import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# Constants
IMG_SIZE = 64  # Must match the size used during training
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'hand_sign_resnet_model.pth'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define ResNet Model
class ResNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetModel, self).__init__()
        
        # Load ResNet18 model
        self.resnet = models.resnet18(weights=None)
        
        # Replace the final fully connected layer for our number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

def preprocess_image(image_path=None, image=None):
    """
    Preprocess an image for prediction
    """
    if image_path is not None:
        # Load image from path
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
    elif image is not None:
        # Use provided image
        img = image.copy()
    else:
        print("Error: No image provided")
        return None
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to standard size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    
    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def predict_digit(image_path=None, image=None):
    """
    Predict digit from image
    """
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        return -1, 0.0
    
    # Load model
    model = ResNetModel(num_classes=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Preprocess image
    processed_image = preprocess_image(image_path, image)
    if processed_image is None:
        return -1, 0.0
    
    # Move to device
    processed_image = processed_image.to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get the predicted class and confidence
    predictions = probabilities.cpu().numpy()
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    return predicted_class, confidence

def start_webcam_prediction():
    """
    Start webcam and predict digits from live video feed
    """
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Train the model first.")
        return
        
    # Load model
    model = ResNetModel(num_classes=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print("Model loaded successfully. Starting webcam...")
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam started. Press 'q' to quit, 'c' to capture and predict.")
    print("Show your hand sign to the camera and press 'c' to predict.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        # Display original frame
        cv2.putText(frame, "Press 'c' to predict", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hand Sign Recognition (ResNet)", frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture current frame and predict
            print("Predicting...")
            
            # Preprocess the image
            processed_image = preprocess_image(image=frame)
            
            # Make prediction
            with torch.no_grad():
                processed_image = processed_image.to(DEVICE)
                outputs = model(processed_image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            predictions = probabilities.cpu().numpy()
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Display result
            result_text = f"Predicted: {predicted_class}, Confidence: {confidence:.2f}"
            print(result_text)
            
            # Show result on image
            cv2.putText(frame, result_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Prediction Result", frame)
            cv2.waitKey(2000)  # Show for 2 seconds
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict hand sign digits using ResNet')
    parser.add_argument('--image', type=str, help='Path to image file (optional)')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for prediction')
    
    args = parser.parse_args()
    
    if args.image:
        # Predict from image file
        predicted_class, confidence = predict_digit(image_path=args.image)
        print(f"Predicted digit: {predicted_class}, Confidence: {confidence:.4f}")
    elif args.webcam:
        # Use webcam
        start_webcam_prediction()
    else:
        print("Please specify either --image or --webcam")