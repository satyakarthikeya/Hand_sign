# Hand Sign Recognition

A sophisticated application for real-time hand sign and digit recognition using computer vision techniques.

## Features

- **Multi-model Support**: Uses both VGG and ResNet models for digit classification
- **YOLOv8 Hand Detection**: Advanced hand detection using YOLOv8 object detection
- **MediaPipe Integration**: Uses MediaPipe for hand landmark detection
- **Drawing Canvas**: Draw digits directly using finger tracking 
- **Modern UI**: Dark-themed, modern interface with responsive design
- **Real-time Processing**: Performs detection and recognition in real-time

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics (YOLOv8)
- MediaPipe
- tkinter
- PIL (Pillow)
- NumPy

## Installation

1. Clone or download this repository
2. Install required packages:
```
pip install torch torchvision opencv-python ultralytics mediapipe pillow numpy
```
3. Ensure you have tkinter installed:
   - Windows: Typically included with Python
   - Linux: `sudo apt-get install python3-tk`
   - macOS: `brew install python-tk@3.x`

## Usage

### Main Application

Run the main application with:

```
python interface.py
```

This launches the full interface with both drawing capabilities and real-time hand sign recognition.

### Standalone YOLO Detection

For testing just the hand detection functionality:

```
python yolo_detector.py
```

Optional arguments:
- `--model`: Path to YOLO model (default: 'best.pt')
- `--conf`: Confidence threshold (default: 0.3)
- `--device`: Device to use for inference (default: auto-detect)

Example:
```
python yolo_detector.py --model best.pt --conf 0.4 --device cpu
```

## Controls and Features

- **Toggle Drawing**: Turn on/off the drawing mode
- **Clear Canvas**: Clear the drawing canvas
- **Save Drawing**: Save the drawn digit to the dataset
- **YOLO Settings**: Adjust confidence threshold and enable/disable detection
- **Model Selection**: Choose between VGG and ResNet models
- **Prediction**: Get real-time predictions from the selected model

## Dataset Structure

The application uses and can add to a dataset with the following structure:

```
Data/
  zero/
    digit_0_001.png
    ...
  one/
    digit_1_001.png
    ...
  ...
  nine/
    digit_9_001.png
    ...
```

## Models

- **VGG**: Custom VGG-like architecture for digit recognition
- **ResNet**: ResNet18-based model for digit recognition
- **YOLOv8**: Pre-trained model for hand detection (best.pt)

## Bounding Box Color Guide

- **Green**: High confidence detection (>70%)
- **Yellow**: Medium confidence detection (50-70%)
- **Red**: Low confidence detection (<50%)

## Troubleshooting

If you encounter issues:

1. **YOLO detection not working**: Make sure the best.pt model exists and ultralytics is properly installed
2. **Webcam not detected**: Check if your webcam is properly connected and permissions are granted
3. **Model loading errors**: Ensure all model files are in the correct locations

## License

This project is available for educational and personal use.

## Acknowledgements

- YOLOv8 by Ultralytics
- MediaPipe by Google
- PyTorch and the computer vision community