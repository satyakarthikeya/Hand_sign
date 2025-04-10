# Hand Sign Digit Recognition

This project uses a VGG-style convolutional neural network to recognize hand sign digits (0-9) from images or webcam, using PyTorch.

## Setup

### Requirements

Install the required Python packages:

```bash
pip install torch torchvision opencv-python numpy matplotlib scikit-learn pillow
```

### Dataset Structure

The dataset should be organized in the following structure:

```
Data/
    zero/
        digit_0_001.png
        digit_0_002.png
        ...
    one/
        digit_1_001.png
        digit_1_002.png
        ...
    # and so on for digits 2-9
```

## Usage

### Training the Model

To train the model on your dataset, run:

```bash
python Vgg_train.py
```

This will:
1. Load images from the Data directory
2. Preprocess the images
3. Train a VGG-style CNN model using PyTorch
4. Save the trained model as `hand_sign_vgg_model.pth`
5. Also save a TorchScript version for mobile deployment as `hand_sign_model_mobile.pt`
6. Generate a training history graph as `training_history.png`

### Making Predictions

After training, you can use the model to predict digits from:

1. **Webcam**:
   ```bash
   python predict.py --webcam
   ```
   - Press 'c' to capture and predict
   - Press 'q' to quit

2. **Image file**:
   ```bash
   python predict.py --image path/to/your/image.png
   ```

## Model Architecture

The model uses a simplified VGG-style architecture:
- 3 convolutional blocks with increasing filter sizes (32, 64, 128)
- Each block has 2 convolutional layers followed by max pooling
- Fully connected layers with dropout for classification

## PyTorch Implementation

The implementation uses PyTorch's:
- `nn.Module` for model definition
- `DataLoader` for efficient batch loading
- `torch.optim` for optimization
- TorchScript for mobile deployment

## Customization

You can adjust parameters in the scripts:
- `IMG_SIZE`: Change image resolution (default: 64x64)
- `BATCH_SIZE`: Adjust training batch size
- `EPOCHS`: Change number of training epochs
- `DEVICE`: The code automatically uses GPU if available