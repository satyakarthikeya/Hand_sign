import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import torchvision.transforms as transforms

# Constants
IMG_SIZE = 64  # Must match the size used during training
# Use absolute path to model file in the models directory
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'hand_sign_vgg_model.pth'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define VGG Model - need same architecture as training
class VGGModel(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGModel, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classification block
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*128, 512),  # 8x8 comes from IMG_SIZE/2^3
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

def preprocess_image(image):
    """
    Preprocess a single image for prediction
    
    Args:
        image: Input image (numpy array or PIL Image)
        
    Returns:
        Preprocessed image ready for model prediction (PyTorch tensor)
    """
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Debug info
    print(f"Image shape before resize: {image.shape}")
        
    # Resize to expected input size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    print(f"Image shape after resize: {image.shape}")
    
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    
    # Convert to PyTorch tensor and add batch dimension
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
    print(f"Tensor shape: {image.shape}")
    
    return image

def predict_digit(image_path=None, image=None):
    """
    Predict the digit from an image
    
    Args:
        image_path: Path to image file (optional)
        image: Numpy array or PIL Image (optional)
        
    Returns:
        Predicted digit and confidence score
    """
    if image_path is not None and os.path.exists(image_path):
        print(f"Reading image from {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image from {image_path}")
            return -1, 0.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image is None:
        raise ValueError("Either image_path or image must be provided")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Train the model first.")
        alternative_path = 'hand_sign_vgg_model.pth'
        if os.path.exists(alternative_path):
            print(f"Found model at alternative path: {alternative_path}")
            model_path = alternative_path
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH} or {alternative_path}")
    else:
        model_path = MODEL_PATH
    
    # Load model
    try:
        print(f"Loading model from {model_path}")
        model = VGGModel(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Preprocess the image
    try:
        processed_image = preprocess_image(image)
        processed_image = processed_image.to(DEVICE)
        print(f"Image processed and moved to {DEVICE}")
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise
    
    # Make prediction
    try:
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the predicted class and confidence
        predictions = probabilities.cpu().numpy()
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        print(f"Prediction successful: class={predicted_class}, confidence={confidence:.4f}")
        return predicted_class, confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        raise

def start_webcam_prediction():
    """
    Start webcam and predict digits from live video feed
    """
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        alternative_path = 'hand_sign_vgg_model.pth'
        if os.path.exists(alternative_path):
            print(f"Found model at alternative path: {alternative_path}")
            model_path = alternative_path
        else:
            print(f"Model file not found at {MODEL_PATH} or alternative path. Train the model first.")
            return
    else:
        model_path = MODEL_PATH
        
    # Load model
    try:
        print(f"Loading model from {model_path}")
        model = VGGModel(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
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
            print("Error: Failed to capture image from webcam.")
            break
        
        # Display instructions
        cv2.putText(frame, "Press 'c' to capture and predict. Press 'q' to quit.", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Hand Sign Prediction", frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
        elif key == ord('c'):
            # Capture and predict
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                processed_image = preprocess_image(rgb_frame)
                processed_image = processed_image.to(DEVICE)
                
                # Make prediction
                with torch.no_grad():
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
            except Exception as e:
                print(f"Error during prediction: {e}")
                cv2.putText(frame, f"Error: {str(e)[:30]}...", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Prediction Result", frame)
                cv2.waitKey(2000)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict hand sign digits')
    parser.add_argument('--image', type=str, help='Path to image file (optional)')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for prediction')
    
    args = parser.parse_args()
    
    try:
        print(f"Running in directory: {os.getcwd()}")
        print(f"Looking for model at: {os.path.abspath(MODEL_PATH)}")
        
        if args.webcam:
            start_webcam_prediction()
        elif args.image:
            if not os.path.exists(args.image):
                print(f"Error: Image file not found at {args.image}")
            else:
                try:
                    digit, conf = predict_digit(image_path=args.image)
                    print(f"Predicted digit: {digit}")
                    print(f"Confidence: {conf:.2f}")
                except Exception as e:
                    print(f"Error predicting from image: {e}")
        else:
            print("Please specify either --image or --webcam")
            print("Example: python predict.py --webcam")
            print("Example: python predict.py --image path/to/image.jpg")
    except Exception as e:
        print(f"Unexpected error: {e}")