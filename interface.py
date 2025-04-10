import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import os
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageTk
import threading
import time

# Constants
IMG_SIZE = 64
VGG_MODEL_PATH = os.path.join('vgg', 'hand_sign_vgg_model.pth')
RESNET_MODEL_PATH = os.path.join('resnet', 'hand_sign_resnet_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Theme colors
DARK_BG = '#212121'
LIGHT_TEXT = '#FFFFFF'
ACCENT_COLOR = '#BB86FC'
SECONDARY_COLOR = '#03DAC6'
ERROR_COLOR = '#CF6679'
SURFACE_COLOR = '#424242'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Define VGG Model
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

class HandSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Sign Recognition")
        self.root.configure(bg=DARK_BG)
        self.root.geometry("1280x720")
        
        # Set application state
        self.running = True
        self.drawing = False
        self.active_model = "VGG"  # Default model
        self.points = deque(maxlen=1024)
        self.canvas = None
        self.last_prediction = None
        self.prediction_confidence = 0.0
        
        # Load models
        self.load_models()
        
        # Create UI
        self.create_ui()
        
        # Start camera
        self.cap = cv2.VideoCapture(0)
        self.update()
        
        # Set cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_models(self):
        """Load both VGG and ResNet models"""
        self.models = {}
        
        try:
            # Load VGG model
            if os.path.exists(VGG_MODEL_PATH):
                print("Loading VGG model...")
                vgg_model = VGGModel(num_classes=10)
                vgg_model.load_state_dict(torch.load(VGG_MODEL_PATH, map_location=DEVICE))
                vgg_model.to(DEVICE)
                vgg_model.eval()
                self.models["VGG"] = vgg_model
                print("VGG model loaded successfully.")
            else:
                print(f"VGG model not found at {VGG_MODEL_PATH}")
                
            # Load ResNet model
            if os.path.exists(RESNET_MODEL_PATH):
                print("Loading ResNet model...")
                resnet_model = ResNetModel(num_classes=10)
                resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=DEVICE))
                resnet_model.to(DEVICE)
                resnet_model.eval()
                self.models["ResNet"] = resnet_model
                print("ResNet model loaded successfully.")
            else:
                print(f"ResNet model not found at {RESNET_MODEL_PATH}")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def create_ui(self):
        """Create the user interface"""
        # Create main frame
        self.main_frame = tk.Frame(self.root, bg=DARK_BG)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for camera feed and controls
        self.left_panel = tk.Frame(self.main_frame, bg=DARK_BG, width=640)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Camera feed
        self.camera_label = tk.Label(self.left_panel, bg=DARK_BG)
        self.camera_label.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Controls section
        self.controls_frame = tk.Frame(self.left_panel, bg=SURFACE_COLOR, padx=10, pady=10)
        self.controls_frame.pack(fill=tk.X, pady=10)
        
        # Drawing controls
        self.drawing_frame = tk.LabelFrame(self.controls_frame, text="Drawing", fg=LIGHT_TEXT, bg=SURFACE_COLOR)
        self.drawing_frame.pack(fill=tk.X, pady=5)
        
        self.drawing_btn = ttk.Button(self.drawing_frame, text="Toggle Drawing", command=self.toggle_drawing)
        self.drawing_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.clear_btn = ttk.Button(self.drawing_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_btn = ttk.Button(self.drawing_frame, text="Save Drawing", command=self.save_drawing)
        self.save_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Model selection
        self.model_frame = tk.LabelFrame(self.controls_frame, text="Model", fg=LIGHT_TEXT, bg=SURFACE_COLOR)
        self.model_frame.pack(fill=tk.X, pady=5)
        
        self.model_var = tk.StringVar(value=self.active_model)
        
        self.vgg_radio = ttk.Radiobutton(self.model_frame, text="VGG", variable=self.model_var, 
                                         value="VGG", command=self.set_model)
        self.vgg_radio.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.resnet_radio = ttk.Radiobutton(self.model_frame, text="ResNet", variable=self.model_var, 
                                           value="ResNet", command=self.set_model)
        self.resnet_radio.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.predict_btn = ttk.Button(self.model_frame, text="Predict", command=self.predict_from_canvas)
        self.predict_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Status bar
        self.status_frame = tk.Frame(self.left_panel, bg=DARK_BG, pady=5)
        self.status_frame.pack(fill=tk.X)
        
        self.status_text = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.status_frame, textvariable=self.status_text, 
                                    fg=LIGHT_TEXT, bg=DARK_BG, anchor="w")
        self.status_label.pack(fill=tk.X)
        
        # Right panel for canvas and prediction
        self.right_panel = tk.Frame(self.main_frame, bg=DARK_BG, width=640)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Canvas
        self.canvas_label_frame = tk.LabelFrame(self.right_panel, text="Drawing Canvas", 
                                            fg=LIGHT_TEXT, bg=SURFACE_COLOR)
        self.canvas_label_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.canvas_label = tk.Label(self.canvas_label_frame, bg="black")
        self.canvas_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Prediction
        self.prediction_frame = tk.LabelFrame(self.right_panel, text="Prediction", 
                                           fg=LIGHT_TEXT, bg=SURFACE_COLOR)
        self.prediction_frame.pack(fill=tk.X, pady=10)
        
        self.result_var = tk.StringVar(value="No prediction yet")
        self.result_label = tk.Label(self.prediction_frame, textvariable=self.result_var, 
                                   font=("Arial", 24, "bold"), fg=ACCENT_COLOR, bg=SURFACE_COLOR)
        self.result_label.pack(pady=10)
        
        self.confidence_var = tk.StringVar(value="")
        self.confidence_label = tk.Label(self.prediction_frame, textvariable=self.confidence_var, 
                                      fg=LIGHT_TEXT, bg=SURFACE_COLOR)
        self.confidence_label.pack(pady=5)
        
        # Style the UI
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TRadiobutton", font=("Arial", 10))
    
    def update(self):
        """Update function for capturing webcam frames and processing"""
        if self.running:
            ret, frame = self.cap.read()
            
            if ret:
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                
                # Initialize canvas if not already
                if self.canvas is None:
                    self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Process hand landmarks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Drawing status overlay
                status_text = f"Drawing: {'ON' if self.drawing else 'OFF'}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0) if self.drawing else (0, 0, 255), 2)
                
                # Model status overlay
                model_text = f"Active Model: {self.active_model}"
                cv2.putText(frame, model_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 200, 0), 2)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        
                        # Get index finger tip coordinates
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        x = int(index_tip.x * width)
                        y = int(index_tip.y * height)
                        
                        # Draw circle at fingertip
                        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                        
                        # Draw on canvas when drawing is active
                        if self.drawing:
                            self.points.append((x, y))
                            for i in range(1, len(self.points)):
                                if self.points[i-1] is None or self.points[i] is None:
                                    continue
                                cv2.line(self.canvas, self.points[i-1], self.points[i], (255, 255, 255), 5)
                                
                # Convert the frame for display in tkinter
                cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv_img)
                tk_img = ImageTk.PhotoImage(image=pil_img)
                self.camera_label.config(image=tk_img)
                self.camera_label.image = tk_img
                
                # Convert the canvas for display
                pil_canvas = Image.fromarray(self.canvas)
                tk_canvas = ImageTk.PhotoImage(image=pil_canvas)
                self.canvas_label.config(image=tk_canvas)
                self.canvas_label.image = tk_canvas
                
            self.root.after(10, self.update)
    
    def toggle_drawing(self):
        """Toggle drawing mode on/off"""
        self.drawing = not self.drawing
        status = "ON" if self.drawing else "OFF"
        self.status_text.set(f"Drawing: {status}")
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
            self.points.clear()
            self.status_text.set("Canvas cleared")
    
    def save_drawing(self):
        """Save the drawing"""
        if self.canvas is None or np.sum(self.canvas) == 0:
            messagebox.showwarning("Warning", "Canvas is empty. Nothing to save.")
            return
        
        # Ask if the image is OK
        is_ok = messagebox.askyesno("Confirmation", "Is the drawn digit OK?")
        
        if is_ok:
            # Ask for digit and filename
            digit = simpledialog.askstring("Save", "Which digit is this (0-9)?:")
            if digit and digit.isdigit() and 0 <= int(digit) <= 9:
                digit = int(digit)
                
                # Determine folder path
                folder_map = {
                    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
                    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
                }
                
                folder_path = os.path.join("Data", folder_map[digit])
                
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                # Count existing files and create filename
                existing_files = len([f for f in os.listdir(folder_path) if f.startswith(f"digit_{digit}_")])
                file_num = existing_files + 1
                filename = f"digit_{digit}_{file_num:03d}.png"
                filepath = os.path.join(folder_path, filename)
                
                # Save the image
                cv2.imwrite(filepath, self.canvas)
                messagebox.showinfo("Success", f"Drawing saved as {filepath}")
                self.status_text.set(f"Saved as {filename}")
                
                # Clear canvas after saving
                self.clear_canvas()
            else:
                messagebox.showwarning("Error", "Invalid digit. Please enter a number between 0-9.")
        else:
            messagebox.showinfo("Info", "Drawing discarded. You can try again.")
    
    def set_model(self):
        """Set the active model"""
        selected = self.model_var.get()
        if selected in self.models:
            self.active_model = selected
            self.status_text.set(f"Active model set to {self.active_model}")
        else:
            messagebox.showerror("Error", f"{selected} model is not loaded")
            # Revert to previously selected model
            self.model_var.set(self.active_model)
    
    def preprocess_canvas(self):
        """Preprocess canvas image for prediction"""
        if self.canvas is None:
            return None
            
        # Convert to RGB if needed
        if len(self.canvas.shape) == 2:
            img = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2RGB)
        else:
            img = self.canvas.copy()
            
        # Resize to expected input size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def predict_from_canvas(self):
        """Make prediction using the active model"""
        if self.active_model not in self.models:
            messagebox.showerror("Error", f"{self.active_model} model not loaded")
            return
            
        # Check if canvas is empty
        if self.canvas is None or np.sum(self.canvas) == 0:
            messagebox.showwarning("Warning", "Canvas is empty. Nothing to predict.")
            return
            
        # Preprocess canvas image
        processed_image = self.preprocess_canvas()
        if processed_image is None:
            messagebox.showerror("Error", "Could not process canvas image for prediction")
            return
            
        # Move to device
        processed_image = processed_image.to(DEVICE)
        
        # Get active model
        model = self.models[self.active_model]
        
        # Make prediction
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the predicted class and confidence
        predictions = probabilities.cpu().numpy()
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Update UI
        self.last_prediction = predicted_class
        self.prediction_confidence = confidence
        self.result_var.set(f"Predicted: {predicted_class}")
        self.confidence_var.set(f"Confidence: {confidence:.4f}")
        
        # Update status
        self.status_text.set(f"Prediction made: {predicted_class} (Confidence: {confidence:.4f})")
    
    def on_closing(self):
        """Handle application closing"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        hands.close()
        self.root.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create root window with dark theme
    root = tk.Tk()
    app = HandSignApp(root)
    root.mainloop()