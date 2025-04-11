import sys
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                           QPushButton, QVBoxLayout, QHBoxLayout, QFrame,
                           QRadioButton, QButtonGroup, QSlider, QCheckBox,
                           QMessageBox, QInputDialog, QProgressBar, QToolTip,
                           QShortcut, QGroupBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QSize
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QPainter, QColor, QPalette
import threading
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("WARNING: Ultralytics package not found. YOLO detection won't work.")
    print("Please install it with: pip install ultralytics")

# Constants
IMG_SIZE = 64
VGG_MODEL_PATH = os.path.join('models', 'hand_sign_vgg_model.pth')
RESNET_MODEL_PATH = os.path.join('models', 'hand_sign_resnet_model.pth')
CNN_MODEL_PATH = os.path.join('models', 'cnn_model.pt')
YOLO_MODEL_PATH = 'models\best.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# YOLO Configuration
YOLO_CONF_THRESHOLD = 0.3
BBOX_THICKNESS = 2
BBOX_FONT_SIZE = 0.5

# Colors (BGR format)
HIGH_CONF_COLOR = (0, 255, 0)     # Green
MED_CONF_COLOR = (0, 255, 255)    # Yellow
LOW_CONF_COLOR = (0, 0, 255)      # Red

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Model definitions
class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        # Initialize with pre-trained weights using the newer PyTorch pattern
        try:
            # For PyTorch 1.13+
            self.efficientnet = models.efficientnet_b0(weights="DEFAULT")
        except:
            # Fallback for older PyTorch versions
            self.efficientnet = models.efficientnet_b0(pretrained=True)
        # Modify the classifier for our number of classes
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.efficientnet(x)

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
            nn.Linear(8*8*128, 512),
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

class ResNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

class StyledGroupBox(QGroupBox):
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                background-color: #2D2D2D;
                border: 2px solid #3D3D3D;
                border-radius: 5px;
                margin-top: 1em;
                font-weight: bold;
                color: #61DAFB;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
        """)

class AnimatedButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._animation = QPropertyAnimation(self, b"size")
        self._animation.setDuration(100)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def enterEvent(self, event):
        self._animation.setStartValue(self.size())
        self._animation.setEndValue(QSize(int(self.width() * 1.1), int(self.height() * 1.1)))
        self._animation.start()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self._animation.setStartValue(self.size())
        self._animation.setEndValue(QSize(int(self.width() / 1.1), int(self.height() / 1.1)))
        self._animation.start()
        super().leaveEvent(event)

class HandSignWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setupModels()
        self.setupCamera()
        self.setupShortcuts()
        
    def setupShortcuts(self):
        """Set up keyboard shortcuts"""
        QShortcut(QKeySequence("D"), self, self.toggleDrawing)
        QShortcut(QKeySequence("C"), self, self.clearCanvas)
        QShortcut(QKeySequence("S"), self, self.saveDrawing)
        QShortcut(QKeySequence("P"), self, self.predictFromCanvas)
        QShortcut(QKeySequence("Q"), self, self.close)
        
    def initUI(self):
        self.setWindowTitle('Hand Sign Recognition')
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1E1E1E;
            }
            QLabel {
                color: #FFFFFF;
                padding: 5px;
            }
            QPushButton {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3D3D3D;
            }
            QPushButton:pressed {
                background-color: #4D4D4D;
            }
            QFrame {
                background-color: #2D2D2D;
                border-radius: 8px;
                padding: 10px;
            }
            QRadioButton, QCheckBox {
                color: #FFFFFF;
                spacing: 8px;
            }
            QRadioButton::indicator, QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #4D4D4D;
                height: 8px;
                background: #2D2D2D;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #61DAFB;
                border: 1px solid #2D2D2D;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 2px solid #2D2D2D;
                border-radius: 5px;
                text-align: center;
                padding: 2px;
                background-color: #1E1E1E;
            }
            QProgressBar::chunk {
                border-radius: 3px;
            }
            QSplitter::handle {
                background-color: #3D3D3D;
            }
        """)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (Camera and Controls)
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        # Camera feed with title
        camera_group = StyledGroupBox("Camera Feed")
        camera_layout = QVBoxLayout(camera_group)
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("border: 2px solid #3D3D3D; border-radius: 4px;")
        camera_layout.addWidget(self.camera_label)
        
        # Status overlay
        self.status_overlay = QLabel()
        self.status_overlay.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.status_overlay.setStyleSheet("color: #61DAFB; background: none;")
        camera_layout.addWidget(self.status_overlay)
        
        left_layout.addWidget(camera_group)
        
        # Controls in a scrollable area
        controls_group = StyledGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Drawing controls
        drawing_group = StyledGroupBox("Drawing Tools")
        drawing_layout = QHBoxLayout(drawing_group)
        
        self.drawing_btn = AnimatedButton('Toggle Drawing (D)')
        self.drawing_btn.setToolTip('Toggle drawing mode on/off (Press D)')
        self.drawing_btn.clicked.connect(self.toggleDrawing)
        drawing_layout.addWidget(self.drawing_btn)
        
        self.clear_btn = AnimatedButton('Clear Canvas (C)')
        self.clear_btn.setToolTip('Clear the drawing canvas (Press C)')
        self.clear_btn.clicked.connect(self.clearCanvas)
        drawing_layout.addWidget(self.clear_btn)
        
        self.save_btn = AnimatedButton('Save Drawing (S)')
        self.save_btn.setToolTip('Save the current drawing (Press S)')
        self.save_btn.clicked.connect(self.saveDrawing)
        drawing_layout.addWidget(self.save_btn)
        
        controls_layout.addWidget(drawing_group)
        
        # YOLO controls
        yolo_group = StyledGroupBox("Hand Detection Settings")
        yolo_layout = QVBoxLayout(yolo_group)
        
        self.yolo_enabled = QCheckBox('Enable YOLO Hand Detection')
        self.yolo_enabled.setChecked(True)
        self.yolo_enabled.setToolTip('Toggle YOLO hand detection')
        yolo_layout.addWidget(self.yolo_enabled)
        
        conf_layout = QHBoxLayout()
        conf_label = QLabel('Detection Confidence:')
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(30)
        self.conf_slider.setToolTip('Adjust confidence threshold for hand detection')
        
        self.conf_value_label = QLabel('0.30')
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value_label.setText(f'{v/100:.2f}'))
        
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        
        yolo_layout.addLayout(conf_layout)
        controls_layout.addWidget(yolo_group)
        
        # Model selection
        model_group = StyledGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        self.model_group = QButtonGroup()
        models_layout = QHBoxLayout()
        
        self.vgg_radio = QRadioButton('VGG Model')
        self.vgg_radio.setToolTip('Use VGG model for prediction')
        self.resnet_radio = QRadioButton('ResNet Model')
        self.resnet_radio.setToolTip('Use ResNet model for prediction')
        self.cnn_radio = QRadioButton('CNN Model')
        self.cnn_radio.setToolTip('Use CNN (EfficientNet) model for prediction')
        self.vgg_radio.setChecked(True)
        
        self.model_group.addButton(self.vgg_radio)
        self.model_group.addButton(self.resnet_radio)
        self.model_group.addButton(self.cnn_radio)
        models_layout.addWidget(self.vgg_radio)
        models_layout.addWidget(self.resnet_radio)
        models_layout.addWidget(self.cnn_radio)
        model_layout.addLayout(models_layout)
        
        self.predict_btn = AnimatedButton('Predict Digit (P)')
        self.predict_btn.setToolTip('Make prediction on current drawing (Press P)')
        self.predict_btn.clicked.connect(self.predictFromCanvas)
        model_layout.addWidget(self.predict_btn)
        
        controls_layout.addWidget(model_group)
        
        # Status bar
        status_group = StyledGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel('Ready')
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        controls_layout.addWidget(status_group)
        
        left_layout.addWidget(controls_group)
        splitter.addWidget(left_panel)
        
        # Right panel (Canvas and Prediction)
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        
        # Canvas
        canvas_group = StyledGroupBox("Drawing Canvas")
        canvas_layout = QVBoxLayout(canvas_group)
        
        self.canvas_label = QLabel()
        self.canvas_label.setMinimumSize(640, 480)
        self.canvas_label.setAlignment(Qt.AlignCenter)
        self.canvas_label.setStyleSheet("border: 2px solid #3D3D3D; border-radius: 4px;")
        canvas_layout.addWidget(self.canvas_label)
        
        right_layout.addWidget(canvas_group)
        
        # Prediction display
        prediction_group = StyledGroupBox("Prediction Results")
        prediction_layout = QVBoxLayout(prediction_group)
        
        self.prediction_label = QLabel('No prediction yet')
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet('font-size: 24px; font-weight: bold;')
        prediction_layout.addWidget(self.prediction_label)
        
        self.confidence_label = QLabel('')
        self.confidence_label.setAlignment(Qt.AlignCenter)
        prediction_layout.addWidget(self.confidence_label)
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMaximum(100)
        prediction_layout.addWidget(self.confidence_bar)
        
        right_layout.addWidget(prediction_group)
        splitter.addWidget(right_panel)
        
        # Initialize state variables
        self.drawing = False
        self.canvas = None
        self.points = deque(maxlen=1024)
        self.active_model = "VGG"
        
        # Set window size and show
        self.setMinimumSize(1280, 720)
        self.show()
        
        # Set initial splitter sizes
        splitter.setSizes([640, 640])
        
        # Set tool tips
        QToolTip.setFont(QApplication.font())
        
    def setupModels(self):
        """Load ML models"""
        self.models = {}
        
        try:
            # Load VGG model
            if os.path.exists(VGG_MODEL_PATH):
                vgg_model = VGGModel(num_classes=10)
                vgg_model.load_state_dict(torch.load(VGG_MODEL_PATH, map_location=DEVICE))
                vgg_model.to(DEVICE)
                vgg_model.eval()
                self.models["VGG"] = vgg_model
            
            # Load ResNet model
            if os.path.exists(RESNET_MODEL_PATH):
                resnet_model = ResNetModel(num_classes=10)
                resnet_model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=DEVICE))
                resnet_model.to(DEVICE)
                resnet_model.eval()
                self.models["ResNet"] = resnet_model
            
            # Load CNN model
            if os.path.exists(CNN_MODEL_PATH):
                cnn_model = CNNModel(num_classes=10)
                cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
                cnn_model.to(DEVICE)
                cnn_model.eval()
                self.models["CNN"] = cnn_model
            
            # Load YOLO model
            if os.path.exists(YOLO_MODEL_PATH):
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
            else:
                self.yolo_model = None
                
        except Exception as e:
            print(f"Error loading models: {e}")
            self.yolo_model = None
    
    def setupCamera(self):
        """Initialize camera and timer for updates"""
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)  # Update every 10ms
    
    def updateStatusOverlay(self, frame):
        """Update status information overlay on camera feed"""
        status_text = []
        status_text.append(f"Drawing: {'ON' if self.drawing else 'OFF'}")
        status_text.append(f"Model: {self.active_model}")
        if self.yolo_enabled.isChecked():
            status_text.append(f"YOLO Conf: {self.conf_slider.value()/100:.2f}")
        
        y_pos = 30
        for text in status_text:
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)
            y_pos += 30
    
    def update(self):
        """Update camera feed and processing"""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            if self.canvas is None:
                self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # YOLO detection
            if self.yolo_model and self.yolo_enabled.isChecked():
                try:
                    conf_threshold = self.conf_slider.value() / 100
                    results_yolo = self.yolo_model(frame, conf=conf_threshold, verbose=False)
                    
                    for result in results_yolo:
                        boxes = result.boxes
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy.astype(int)
                            conf = float(box.conf[0])
                            
                            if conf > conf_threshold:
                                color = HIGH_CONF_COLOR if conf > 0.7 else \
                                       MED_CONF_COLOR if conf > 0.5 else LOW_CONF_COLOR
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
                                cv2.putText(frame, f"Hand: {conf:.2f}", (x1 + 5, y1 - 8),
                                          cv2.FONT_HERSHEY_SIMPLEX, BBOX_FONT_SIZE, (255, 255, 255), 1)
                                
                except Exception as e:
                    print(f"YOLO error: {e}")
            
            # Process MediaPipe landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x = int(index_tip.x * width)
                    y = int(index_tip.y * height)
                    
                    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                    
                    if self.drawing:
                        self.points.append((x, y))
                        for i in range(1, len(self.points)):
                            if self.points[i-1] is None or self.points[i] is None:
                                continue
                            cv2.line(self.canvas, self.points[i-1], self.points[i], (255, 255, 255), 5)
            
            # Add status overlay
            self.updateStatusOverlay(frame)
            
            # Convert frame to QImage for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            frame_qt = QImage(frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(frame_qt))
            
            # Convert canvas to QImage
            canvas_rgb = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
            canvas_qt = QImage(canvas_rgb.data, w, h, w * ch, QImage.Format_RGB888)
            self.canvas_label.setPixmap(QPixmap.fromImage(canvas_qt))
    
    def toggleDrawing(self):
        """Toggle drawing mode"""
        self.drawing = not self.drawing
        self.status_label.setText(f"Drawing: {'ON' if self.drawing else 'OFF'}")
        self.drawing_btn.setStyleSheet(
            'background-color: #34D399;' if self.drawing else 'background-color: #2D2D2D;'
        )
    
    def clearCanvas(self):
        """Clear the drawing canvas"""
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
            self.points.clear()
            self.status_label.setText("Canvas cleared")
    
    def saveDrawing(self):
        """Save the current drawing"""
        if self.canvas is None or np.sum(self.canvas) == 0:
            QMessageBox.warning(self, "Warning", "Canvas is empty. Nothing to save.")
            return
        
        reply = QMessageBox.question(self, "Confirmation", "Is the drawn digit OK?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            digit, ok = QInputDialog.getInt(self, "Save Drawing",
                                          "Which digit is this (0-9)?:",
                                          0, 0, 9)
            
            if ok:
                folder_map = {
                    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
                    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
                }
                
                folder_path = os.path.join("Data", folder_map[digit])
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                existing_files = len([f for f in os.listdir(folder_path) 
                                    if f.startswith(f"digit_{digit}_")])
                file_num = existing_files + 1
                filename = f"digit_{digit}_{file_num:03d}.png"
                filepath = os.path.join(folder_path, filename)
                
                cv2.imwrite(filepath, self.canvas)
                self.status_label.setText(f"Saved as {filename}")
                self.clearCanvas()
    
    def predictFromCanvas(self):
        """Make prediction using the active model"""
        if self.canvas is None or np.sum(self.canvas) == 0:
            QMessageBox.warning(self, "Warning", "Canvas is empty. Nothing to predict.")
            return
        
        # Get active model
        if self.vgg_radio.isChecked():
            model_name = "VGG"
            input_size = IMG_SIZE
        elif self.resnet_radio.isChecked():
            model_name = "ResNet"
            input_size = IMG_SIZE
        else:
            model_name = "CNN"
            input_size = 224  # EfficientNet expects 224x224 input
            
        if model_name not in self.models:
            QMessageBox.critical(self, "Error", f"{model_name} model not loaded")
            return
        
        # Show prediction in progress
        self.prediction_label.setText("Predicting...")
        self.confidence_label.setText("Please wait...")
        self.confidence_bar.setValue(0)
        self.predict_btn.setEnabled(False)
        QApplication.processEvents()  # Force UI update
        
        try:
            # Preprocess canvas
            img = cv2.resize(self.canvas, (input_size, input_size))
            img = img / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
            
            # Make prediction
            model = self.models[model_name]
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            predictions = probabilities.cpu().numpy()
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Update UI
            self.prediction_label.setText(f"Predicted: {predicted_class}")
            self.confidence_label.setText(f"Confidence: {confidence:.4f}")
            self.confidence_bar.setValue(int(confidence * 100))
            
            # Set color based on confidence
            if confidence > 0.8:
                color = "#34D399"  # Green
            elif confidence > 0.5:
                color = "#61DAFB"  # Blue
            else:
                color = "#EF4444"  # Red
            
            self.confidence_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #2D2D2D;
                    border-radius: 2px;
                    text-align: center;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                }}
            """)
            
            self.status_label.setText(f"Prediction made: {predicted_class} "
                                    f"(Confidence: {confidence:.4f})")
            
        except Exception as e:
            self.prediction_label.setText("Prediction failed")
            self.confidence_label.setText(f"Error: {str(e)}")
            self.status_label.setText("Error during prediction")
        
        finally:
            self.predict_btn.setEnabled(True)
            QApplication.processEvents()  # Force UI update
    
    def closeEvent(self, event):
        """Handle application closing"""
        self.cap.release()
        hands.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HandSignWindow()
    sys.exit(app.exec_())