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

# Import for YOLOv8 support
try:
    from ultralytics import YOLO
except ImportError:
    print("WARNING: Ultralytics package not found. YOLO detection won't work.")
    print("Please install it with: pip install ultralytics")

# Constants
IMG_SIZE = 64
VGG_MODEL_PATH = os.path.join('vgg', 'hand_sign_vgg_model.pth')
RESNET_MODEL_PATH = os.path.join('resnet', 'hand_sign_resnet_model.pth')
YOLO_MODEL_PATH = 'best.pt'  # Path to the YOLO model for hand detection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# YOLO Configuration
YOLO_CONF_THRESHOLD = 0.3  # Confidence threshold for YOLO detections
YOLO_IOU_THRESHOLD = 0.45  # IOU threshold for NMS
YOLO_BOX_COLOR = (0, 255, 255)  # Yellow color for bounding boxes (BGR)
YOLO_TEXT_COLOR = (0, 255, 255)  # Yellow color for text (BGR)

# Bounding Box Colors - BGR format
HIGH_CONF_COLOR = (0, 255, 0)     # Green for high confidence (>0.7)
MED_CONF_COLOR = (0, 255, 255)    # Yellow for medium confidence (0.5-0.7)
LOW_CONF_COLOR = (0, 0, 255)      # Red for low confidence (<0.5)
BBOX_THICKNESS = 2                # Thickness of bounding box lines
BBOX_FONT_SIZE = 0.5              # Font size for bounding box text
BBOX_PADDING = 10                 # Padding for prediction box inside bounding box

# Theme colors - Modern dark theme
DARK_BG = '#1E1E1E'               # Darker background
LIGHT_TEXT = '#FFFFFF'            # White text
ACCENT_COLOR = '#61DAFB'          # Bright blue accent (React blue)
SECONDARY_COLOR = '#10B981'       # Green secondary color
ERROR_COLOR = '#EF4444'           # Red for errors/warnings
SURFACE_COLOR = '#2D2D2D'         # Slightly lighter surface color
HEADER_COLOR = '#111111'          # Very dark header color
BUTTON_HOVER = '#3D3D3D'          # Button hover color
SUCCESS_COLOR = '#34D399'         # Success color

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
        
        # Center window on screen
        self.center_window()
        
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
        
    def center_window(self):
        """Center the window on the screen"""
        # Get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate position coordinates
        width = 1280
        height = 720
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # Set the window position
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
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
            
            # Load YOLOv8 model for hand detection - improved implementation
            print("Loading YOLOv8 model for hand detection...")
            model_path = YOLO_MODEL_PATH
            
            # If the model doesn't exist, use the last trained model
            if not os.path.exists(model_path):
                if os.path.exists('runs/detect'):
                    model_dirs = [d for d in os.listdir('runs/detect') if d.startswith('hand_detection_model')]
                    if model_dirs:
                        latest_model = sorted(model_dirs)[-1]
                        model_path = os.path.join('runs', 'detect', latest_model, 'weights', 'best.pt')
                        print(f"Main model not found, using alternative from: {model_path}")
            
            if os.path.exists(model_path):
                try:
                    # Using force_reload=True to prevent cache-related issues
                    self.yolo_model = YOLO(model_path, verbose=True)
                    print(f"YOLOv8 model loaded successfully from {model_path}")
                    
                    # Test inference to ensure the model is working
                    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                    try:
                        _ = self.yolo_model(dummy_img)
                        print("YOLOv8 model test inference successful")
                    except Exception as test_err:
                        print(f"YOLOv8 test inference failed: {test_err}")
                        # Try reloading with task parameter explicitly set
                        try:
                            self.yolo_model = YOLO(model_path, task='detect')
                            print("Reloaded YOLOv8 model with explicit task parameter")
                            _ = self.yolo_model(dummy_img)
                            print("Second test inference successful")
                        except Exception as reload_err:
                            print(f"Reload attempt failed: {reload_err}")
                            self.yolo_model = None
                except Exception as e:
                    print(f"Error loading YOLOv8 model: {e}")
                    self.yolo_model = None
            else:
                print(f"YOLO model not found at {model_path}")
                self.yolo_model = None
                
        except Exception as e:
            print(f"Error loading models: {e}")
            self.yolo_model = None
    
    def create_ui(self):
        """Create the user interface"""
        # Create main frame with rounded corners and padding
        self.main_frame = tk.Frame(self.root, bg=DARK_BG, padx=15, pady=15)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Application header
        self.header_frame = tk.Frame(self.main_frame, bg=HEADER_COLOR, height=60)
        self.header_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.app_title = tk.Label(self.header_frame, text="Hand Sign Recognition", 
                                 font=("Segoe UI", 18, "bold"), fg=ACCENT_COLOR, bg=HEADER_COLOR)
        self.app_title.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Device info
        device_text = f"Using: {DEVICE}"
        self.device_label = tk.Label(self.header_frame, text=device_text,
                                     font=("Segoe UI", 10), fg=LIGHT_TEXT, bg=HEADER_COLOR)
        self.device_label.pack(side=tk.RIGHT, padx=15, pady=15)
        
        # Content area with scrollable panels
        self.content_frame = tk.Frame(self.main_frame, bg=DARK_BG)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel with scrollbar for camera feed and controls
        self.left_container = tk.Frame(self.content_frame, bg=DARK_BG)
        self.left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 7.5))
        
        # Add scrollbar to left panel
        self.left_scrollbar = ttk.Scrollbar(self.left_container)
        self.left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas for scrolling
        self.left_canvas = tk.Canvas(self.left_container, bg=DARK_BG, 
                                   yscrollcommand=self.left_scrollbar.set,
                                   highlightthickness=0)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        self.left_scrollbar.config(command=self.left_canvas.yview)
        
        # Create frame inside canvas for content
        self.left_panel = tk.Frame(self.left_canvas, bg=DARK_BG)
        self.left_canvas_window = self.left_canvas.create_window((0, 0), window=self.left_panel, anchor="nw", tags="self.left_panel")
        
        # Camera feed with title and user instructions
        self.camera_frame = tk.Frame(self.left_panel, bg=SURFACE_COLOR, padx=2, pady=2)
        self.camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.camera_title = tk.Label(self.camera_frame, text="Camera Feed", 
                                    font=("Segoe UI", 12, "bold"), fg=ACCENT_COLOR, bg=SURFACE_COLOR)
        self.camera_title.pack(pady=5)
        
        # User instructions
        self.instructions_label = tk.Label(self.camera_frame, 
                                         text="Show hand signs in the camera to detect digits",
                                         font=("Segoe UI", 9), fg=LIGHT_TEXT, bg=SURFACE_COLOR)
        self.instructions_label.pack(pady=(0, 5))
        
        self.camera_label = tk.Label(self.camera_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls section
        self.controls_frame = tk.Frame(self.left_panel, bg=SURFACE_COLOR, padx=15, pady=15)
        self.controls_frame.pack(fill=tk.X)
        
        # Drawing controls
        self.drawing_frame = tk.LabelFrame(self.controls_frame, text="Drawing Controls", 
                                         fg=ACCENT_COLOR, bg=SURFACE_COLOR, padx=10, pady=10,
                                         font=("Segoe UI", 10, "bold"))
        self.drawing_frame.pack(fill=tk.X, pady=7)
        
        self.drawing_btn = ttk.Button(self.drawing_frame, text="Toggle Drawing", command=self.toggle_drawing)
        self.drawing_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.clear_btn = ttk.Button(self.drawing_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_btn = ttk.Button(self.drawing_frame, text="Save Drawing", command=self.save_drawing)
        self.save_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # YOLO Controls with modern styling
        self.yolo_frame = tk.LabelFrame(self.controls_frame, text="YOLO Detection Settings", 
                                      fg=ACCENT_COLOR, bg=SURFACE_COLOR, padx=10, pady=10,
                                      font=("Segoe UI", 10, "bold"))
        self.yolo_frame.pack(fill=tk.X, pady=7)
        
        # YOLO Confidence slider with better visual
        self.yolo_conf_frame = tk.Frame(self.yolo_frame, bg=SURFACE_COLOR)
        self.yolo_conf_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(self.yolo_conf_frame, text="Detection Confidence:", 
                fg=LIGHT_TEXT, bg=SURFACE_COLOR, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=5)
        
        self.yolo_conf_var = tk.DoubleVar(value=YOLO_CONF_THRESHOLD)
        self.yolo_conf_slider = ttk.Scale(self.yolo_conf_frame, from_=0.1, to=0.9, length=200, 
                                         variable=self.yolo_conf_var, orient=tk.HORIZONTAL)
        self.yolo_conf_slider.pack(side=tk.LEFT, padx=5)
        
        self.yolo_conf_label = tk.Label(self.yolo_conf_frame, text=f"{YOLO_CONF_THRESHOLD:.1f}", 
                                      fg=ACCENT_COLOR, bg=SURFACE_COLOR, width=3, 
                                      font=("Segoe UI", 9, "bold"))
        self.yolo_conf_label.pack(side=tk.LEFT, padx=5)
        
        # Update label when slider changes
        self.yolo_conf_var.trace_add("write", lambda *args: self.yolo_conf_label.config(
            text=f"{self.yolo_conf_var.get():.1f}"))
        
        # YOLO Enable/Disable
        self.yolo_enabled_var = tk.BooleanVar(value=True)
        self.yolo_enabled_check = ttk.Checkbutton(self.yolo_frame, text="Enable YOLO Hand Detection", 
                                                variable=self.yolo_enabled_var)
        self.yolo_enabled_check.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Model selection - modern radio button frame
        self.model_frame = tk.LabelFrame(self.controls_frame, text="Model Selection", 
                                       fg=ACCENT_COLOR, bg=SURFACE_COLOR, padx=10, pady=10,
                                       font=("Segoe UI", 10, "bold"))
        self.model_frame.pack(fill=tk.X, pady=7)
        
        self.model_var = tk.StringVar(value=self.active_model)
        
        self.model_selection_frame = tk.Frame(self.model_frame, bg=SURFACE_COLOR)
        self.model_selection_frame.pack(fill=tk.X, pady=5)
        
        self.vgg_radio = ttk.Radiobutton(self.model_selection_frame, text="VGG Model", variable=self.model_var, 
                                        value="VGG", command=self.set_model)
        self.vgg_radio.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.resnet_radio = ttk.Radiobutton(self.model_selection_frame, text="ResNet Model", variable=self.model_var, 
                                          value="ResNet", command=self.set_model)
        self.resnet_radio.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.predict_btn = ttk.Button(self.model_selection_frame, text="Predict Digit", command=self.predict_from_canvas)
        self.predict_btn.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Status bar with modern styling
        self.status_frame = tk.Frame(self.left_panel, bg=HEADER_COLOR, height=30)
        self.status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_text = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.status_frame, textvariable=self.status_text, 
                                   font=("Segoe UI", 9), fg=LIGHT_TEXT, bg=HEADER_COLOR, anchor="w")
        self.status_label.pack(fill=tk.X, padx=10, pady=5)
        
        # Right panel with scrollbar for canvas and prediction
        self.right_container = tk.Frame(self.content_frame, bg=DARK_BG)
        self.right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(7.5, 0))
        
        # Add scrollbar to right panel
        self.right_scrollbar = ttk.Scrollbar(self.right_container)
        self.right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas for scrolling
        self.right_canvas = tk.Canvas(self.right_container, bg=DARK_BG, 
                                    yscrollcommand=self.right_scrollbar.set,
                                    highlightthickness=0)
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        self.right_scrollbar.config(command=self.right_canvas.yview)
        
        # Create frame inside canvas for content
        self.right_panel = tk.Frame(self.right_canvas, bg=DARK_BG)
        self.right_canvas_window = self.right_canvas.create_window((0, 0), window=self.right_panel, anchor="nw", tags="self.right_panel")
        
        # Canvas with better styling
        self.canvas_label_frame = tk.LabelFrame(self.right_panel, text="Drawing Canvas", 
                                             fg=ACCENT_COLOR, bg=SURFACE_COLOR, 
                                             font=("Segoe UI", 12, "bold"),
                                             padx=10, pady=10)
        self.canvas_label_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.canvas_label = tk.Label(self.canvas_label_frame, bg=HEADER_COLOR, bd=2, relief="groove")
        self.canvas_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Prediction panel with modern styling
        self.prediction_frame = tk.LabelFrame(self.right_panel, text="Prediction Result", 
                                           fg=ACCENT_COLOR, bg=SURFACE_COLOR,
                                           font=("Segoe UI", 12, "bold"),
                                           padx=15, pady=15)
        self.prediction_frame.pack(fill=tk.X)
        
        # Add prediction visualization
        self.digit_visual_frame = tk.Frame(self.prediction_frame, bg=HEADER_COLOR, padx=10, pady=10)
        self.digit_visual_frame.pack(fill=tk.X, pady=10)
        
        self.digit_canvas = tk.Canvas(self.digit_visual_frame, width=150, height=150, 
                                     bg=HEADER_COLOR, highlightthickness=0)
        self.digit_canvas.pack(pady=5)
        
        # Add prediction text display
        self.digit_frame = tk.Frame(self.prediction_frame, bg=SURFACE_COLOR)
        self.digit_frame.pack(fill=tk.X, pady=10)
        
        self.result_var = tk.StringVar(value="No prediction yet")
        self.result_label = tk.Label(self.digit_frame, textvariable=self.result_var, 
                                   font=("Segoe UI", 28, "bold"), fg=ACCENT_COLOR, bg=SURFACE_COLOR)
        self.result_label.pack(pady=5)
        
        self.confidence_var = tk.StringVar(value="")
        self.confidence_label = tk.Label(self.digit_frame, textvariable=self.confidence_var, 
                                       font=("Segoe UI", 10), fg=LIGHT_TEXT, bg=SURFACE_COLOR)
        self.confidence_label.pack(pady=5)
        
        # Additional info section
        self.info_frame = tk.Frame(self.prediction_frame, bg=SURFACE_COLOR)
        self.info_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.model_info_label = tk.Label(self.info_frame, text=f"Using {self.active_model} model",
                                        font=("Segoe UI", 9), fg=SECONDARY_COLOR, bg=SURFACE_COLOR)
        self.model_info_label.pack(pady=2)
        
        # Style the UI components
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TButton", font=("Segoe UI", 10), background=SURFACE_COLOR)
        self.style.map("TButton", 
                       background=[('active', BUTTON_HOVER)],
                       foreground=[('active', LIGHT_TEXT)])
        self.style.configure("TCheckbutton", font=("Segoe UI", 10))
        self.style.configure("TRadiobutton", font=("Segoe UI", 10))
        self.style.map("TCheckbutton", 
                       background=[('active', SURFACE_COLOR)],
                       foreground=[('active', ACCENT_COLOR)])
        self.style.map("TRadiobutton", 
                      background=[('active', SURFACE_COLOR)],
                      foreground=[('active', ACCENT_COLOR)])
        
        # Configure scale style
        self.style.configure("TScale", background=SURFACE_COLOR, troughcolor=BUTTON_HOVER)
        
        # Make window resizable
        self.root.resizable(True, True)
        self.root.minsize(1024, 768)
        
        # Initialize prediction visual with placeholder
        self.update_prediction_visual()
        
        # Configure the scroll regions when frames change size
        self.left_panel.bind("<Configure>", lambda e: self.left_canvas.configure(
            scrollregion=self.left_canvas.bbox("all")))
            
        self.right_panel.bind("<Configure>", lambda e: self.right_canvas.configure(
            scrollregion=self.right_canvas.bbox("all")))
        
        # Bind mousewheel to scrollbars
        self.left_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Add help box for bounding box explanation
        self.add_bounding_box_help()
    
    def add_bounding_box_help(self):
        """Add help information about the bounding box display"""
        # Create help frame in the left panel
        self.bbox_help_frame = tk.LabelFrame(self.left_panel, text="Bounding Box Guide", 
                                          fg=ACCENT_COLOR, bg=SURFACE_COLOR, 
                                          font=("Segoe UI", 10, "bold"),
                                          padx=10, pady=10)
        self.bbox_help_frame.pack(fill=tk.X, pady=10)
        
        # Help text content
        help_text = (
            "The camera feed shows bounding boxes around detected hands:\n\n"
            "• GREEN box: High confidence detection (>70%)\n"
            "• YELLOW box: Medium confidence detection (50-70%)\n"
            "• RED box: Low confidence detection (<50%)\n\n"
            "Each box displays the hand detection confidence and\n"
            "the predicted digit with its confidence score when available."
        )
        
        help_label = tk.Label(self.bbox_help_frame, text=help_text, 
                             justify=tk.LEFT, fg=LIGHT_TEXT, bg=SURFACE_COLOR,
                             font=("Segoe UI", 9), padx=5, pady=5)
        help_label.pack(fill=tk.X)
        
        # Add color examples
        color_frame = tk.Frame(self.bbox_help_frame, bg=SURFACE_COLOR, pady=5)
        color_frame.pack(fill=tk.X)
        
        # High confidence example
        high_frame = tk.Frame(color_frame, bg=SURFACE_COLOR)
        high_frame.pack(side=tk.LEFT, padx=10)
        high_sample = tk.Canvas(high_frame, width=20, height=20, bg=SURFACE_COLOR, highlightthickness=0)
        high_sample.create_rectangle(2, 2, 18, 18, outline="#00FF00", width=2)
        high_sample.pack(side=tk.LEFT)
        tk.Label(high_frame, text="High", fg=SUCCESS_COLOR, bg=SURFACE_COLOR).pack(side=tk.LEFT, padx=5)
        
        # Medium confidence example
        med_frame = tk.Frame(color_frame, bg=SURFACE_COLOR)
        med_frame.pack(side=tk.LEFT, padx=10)
        med_sample = tk.Canvas(med_frame, width=20, height=20, bg=SURFACE_COLOR, highlightthickness=0)
        med_sample.create_rectangle(2, 2, 18, 18, outline="#FFFF00", width=2)
        med_sample.pack(side=tk.LEFT)
        tk.Label(med_frame, text="Medium", fg=ACCENT_COLOR, bg=SURFACE_COLOR).pack(side=tk.LEFT, padx=5)
        
        # Low confidence example
        low_frame = tk.Frame(color_frame, bg=SURFACE_COLOR)
        low_frame.pack(side=tk.LEFT, padx=10)
        low_sample = tk.Canvas(low_frame, width=20, height=20, bg=SURFACE_COLOR, highlightthickness=0)
        low_sample.create_rectangle(2, 2, 18, 18, outline="#FF0000", width=2)
        low_sample.pack(side=tk.LEFT)
        tk.Label(low_frame, text="Low", fg=ERROR_COLOR, bg=SURFACE_COLOR).pack(side=tk.LEFT, padx=5)
    
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling on canvases"""
        # Determine which canvas is under the mouse
        x, y = self.root.winfo_pointerxy()
        widget = self.root.winfo_containing(x, y)
        
        # Scroll amount (consider different OS scroll directions)
        scroll_amount = -1 * (event.delta // 120)
        
        # Scroll the canvas that contains the mouse
        if widget and (self.left_canvas in widget.winfo_toplevel().winfo_children() or 
                      widget is self.left_canvas):
            self.left_canvas.yview_scroll(scroll_amount, "units")
        elif widget and (self.right_canvas in widget.winfo_toplevel().winfo_children() or 
                        widget is self.right_canvas):
            self.right_canvas.yview_scroll(scroll_amount, "units")
    
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
                
                # Process hand landmarks with MediaPipe
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
                
                # Use YOLO model for hand detection if available
                if self.yolo_model is not None and self.yolo_enabled_var.get():
                    try:
                        # Run YOLOv8 inference on the frame
                        conf_threshold = self.yolo_conf_var.get()
                        # Add try/except specifically around the prediction part
                        try:
                            results_yolo = self.yolo_model(frame, conf=conf_threshold, verbose=False)
                            
                            # Show detection count 
                            detection_count = len(results_yolo)
                            cv2.putText(frame, f"Detections: {detection_count}", (10, 90), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            
                            # Draw YOLO model info on frame
                            cv2.putText(frame, f"YOLO model: {os.path.basename(YOLO_MODEL_PATH)}", (10, 120), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                            
                            # Visualize the results on the frame - using YOLOv8's result format
                            for result in results_yolo:
                                boxes = result.boxes  # Boxes object for bbox outputs
                                if len(boxes) == 0:
                                    continue
                                    
                                for box in boxes:
                                    try:
                                        # Get box coordinates and confidence - with error handling
                                        xyxy = box.xyxy[0].cpu().numpy()
                                        x1, y1, x2, y2 = xyxy.astype(int)
                                        conf = float(box.conf[0])
                                        
                                        # Only draw if confidence is high enough
                                        if conf > conf_threshold:  
                                            # Determine color based on confidence
                                            if conf > 0.7:
                                                box_color = HIGH_CONF_COLOR  # Green (BGR)
                                            elif conf > 0.5:
                                                box_color = MED_CONF_COLOR   # Yellow (BGR)
                                            else:
                                                box_color = LOW_CONF_COLOR   # Red (BGR)
                                        
                                            # Draw bounding box with enhanced visibility
                                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, BBOX_THICKNESS)
                                            
                                            # Add a label background for better text visibility
                                            label_bg_pt1 = (int(x1), int(y1) - 25)
                                            label_bg_pt2 = (int(x1) + 130, int(y1))
                                            cv2.rectangle(frame, label_bg_pt1, label_bg_pt2, box_color, -1)  # -1 to fill
                                            
                                            # Display hand detection confidence
                                            info_text = f"Hand: {conf:.2f}"
                                            cv2.putText(frame, info_text, (int(x1) + 5, int(y1) - 8),
                                                      cv2.FONT_HERSHEY_SIMPLEX, BBOX_FONT_SIZE, (255, 255, 255), 1)
                                            
                                            # Only try to predict digit if confidence is high enough
                                            try:
                                                digit, digit_conf = self.predict_from_image_region(rgb_frame, x1, y1, x2, y2)
                                                
                                                # Only show digit prediction if confidence is high enough
                                                if digit_conf > 0.5:
                                                    # Draw filled rectangle for prediction with better visibility
                                                    pred_bg_pt1 = (int(x1), int(y2))
                                                    pred_bg_pt2 = (int(x1) + 130, int(y2) + 30)
                                                    cv2.rectangle(frame, pred_bg_pt1, pred_bg_pt2, box_color, -1)  # -1 to fill
                                                    
                                                    # Draw border around prediction box for better visibility
                                                    cv2.rectangle(frame, pred_bg_pt1, pred_bg_pt2, (255, 255, 255), 1)
                                                    
                                                    # Show prediction with confidence
                                                    pred_text = f"Digit: {digit} ({digit_conf:.2f})"
                                                    cv2.putText(frame, pred_text, (int(x1) + 5, int(y2) + 20),
                                                              cv2.FONT_HERSHEY_SIMPLEX, BBOX_FONT_SIZE, (255, 255, 255), 1)
                                                    
                                                    # Update the UI prediction panel with latest detection
                                                    self.last_prediction = digit
                                                    self.prediction_confidence = digit_conf
                                                    self.result_var.set(f"Predicted: {digit}")
                                                    self.confidence_var.set(f"Confidence: {digit_conf:.4f}")
                                                    self.update_prediction_visual(digit, digit_conf)
                                                    self.model_info_label.config(text=f"Using {self.active_model} model")
                                                    
                                                    # Change color of confidence text based on confidence level
                                                    if digit_conf > 0.8:
                                                        self.confidence_label.config(fg=SUCCESS_COLOR)
                                                    elif digit_conf > 0.5:
                                                        self.confidence_label.config(fg=ACCENT_COLOR)
                                                    else:
                                                        self.confidence_label.config(fg=ERROR_COLOR)
                                            except Exception as digit_err:
                                                # Show error in status bar but continue execution
                                                self.status_text.set(f"Digit prediction error: {str(digit_err)}")
                                    except Exception as box_err:
                                        # Handle errors in box processing
                                        print(f"Error processing box: {box_err}")
                                        continue
                                    
                        except Exception as detect_err:
                            print(f"YOLO detection error: {detect_err}")
                            cv2.putText(frame, "YOLO Detection Failed", (10, 120),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                    except Exception as e:
                        # Display error message in status bar
                        self.status_text.set(f"YOLO error: {str(e)}")
                        # Also display error on frame
                        cv2.putText(frame, "YOLO Error - See status", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Process MediaPipe landmarks for drawing
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
        
        # Update UI text
        self.last_prediction = predicted_class
        self.prediction_confidence = confidence
        self.result_var.set(f"Predicted: {predicted_class}")
        self.confidence_var.set(f"Confidence: {confidence:.4f}")
        
        # Update the visual elements
        self.update_prediction_visual(predicted_class, confidence)
        self.model_info_label.config(text=f"Using {self.active_model} model")
        
        # Change color of confidence text based on confidence level
        if confidence > 0.8:
            self.confidence_label.config(fg=SUCCESS_COLOR)
        elif confidence > 0.5:
            self.confidence_label.config(fg=ACCENT_COLOR)
        else:
            self.confidence_label.config(fg=ERROR_COLOR)
        
        # Update status
        self.status_text.set(f"Prediction made: {predicted_class} (Confidence: {confidence:.4f})")
    
    def predict_from_image_region(self, image, x1, y1, x2, y2):
        """
        Predict digit from a specific region of an image (detected hand)
        
        Args:
            image: Input image (numpy array)
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            predicted_class: The predicted digit
            confidence: Confidence score for the prediction
        """
        if self.active_model not in self.models:
            return -1, 0.0
            
        # Extract the hand region
        # Add padding to ensure we get the full hand
        padding = 20
        h, w = image.shape[:2]
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        
        hand_region = image[y1:y2, x1:x2]
        
        # Skip very small regions
        if hand_region.shape[0] < 10 or hand_region.shape[1] < 10:
            return -1, 0.0
            
        # Preprocess the hand region
        # Resize to square while maintaining aspect ratio
        h, w = hand_region.shape[:2]
        max_dim = max(h, w)
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        offset_h = (max_dim - h) // 2
        offset_w = (max_dim - w) // 2
        square_img[offset_h:offset_h+h, offset_w:offset_w+w] = hand_region
        
        # Resize to expected input size
        resized_img = cv2.resize(square_img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize pixel values to [0, 1]
        normalized_img = resized_img / 255.0
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).float()
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        # Move to device
        img_tensor = img_tensor.to(DEVICE)
        
        # Get active model
        model = self.models[self.active_model]
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the predicted class and confidence
        predictions = probabilities.cpu().numpy()
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return predicted_class, confidence
    
    def update_prediction_visual(self, digit=None, confidence=0.0):
        """Update the prediction visualization canvas with the detected digit"""
        # Clear previous drawing
        self.digit_canvas.delete("all")
        
        # Default case - no prediction
        if digit is None:
            self.digit_canvas.create_text(75, 75, text="?", 
                                         font=("Segoe UI", 60, "bold"), fill=ACCENT_COLOR)
            return
        
        # Draw the digit with size based on confidence
        size = min(60 + int(confidence * 20), 90)  # Size increases with confidence
        color = ACCENT_COLOR
        if confidence > 0.8:
            color = SUCCESS_COLOR
        elif confidence < 0.4:
            color = ERROR_COLOR
            
        # Draw the digit
        self.digit_canvas.create_text(75, 75, text=str(digit), 
                                     font=("Segoe UI", size, "bold"), fill=color)
        
        # Draw confidence meter
        meter_width = 120
        meter_height = 10
        x_start = 15
        y_pos = 130
        
        # Background meter (empty)
        self.digit_canvas.create_rectangle(
            x_start, y_pos, 
            x_start + meter_width, y_pos + meter_height,
            fill=HEADER_COLOR, outline=LIGHT_TEXT)
        
        # Filled portion based on confidence
        fill_width = int(confidence * meter_width)
        if confidence > 0.7:
            meter_color = SUCCESS_COLOR
        elif confidence > 0.4:
            meter_color = ACCENT_COLOR
        else:
            meter_color = ERROR_COLOR
            
        self.digit_canvas.create_rectangle(
            x_start, y_pos,
            x_start + fill_width, y_pos + meter_height,
            fill=meter_color, outline="")
    
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
