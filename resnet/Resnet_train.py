import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Constants
IMG_SIZE = 64  # Target size for image resizing
BATCH_SIZE = 32
EPOCHS = 50  # Reduced to 50 epochs for ResNet
NUM_CLASSES = 10  # 0-9 digits
DATASET_PATH = "Data"
MODEL_SAVE_PATH = "hand_sign_resnet_model.pth"
MOBILE_MODEL_PATH = "hand_sign_resnet_mobile.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset class
class HandSignDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx]
        
        # Convert numpy image (H,W,C) to PyTorch tensor (C,H,W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        if self.transform:
            img = self.transform(img)
            
        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        return img

def load_and_preprocess_data():
    """
    Load images from dataset directory and preprocess them for training
    """
    X = []
    y = []
    
    # Map folder names to class indices
    class_mapping = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
    }
    
    print("Loading dataset from:", DATASET_PATH)
    
    # Iterate through each class folder
    for class_name, class_idx in class_mapping.items():
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Path {class_path} does not exist. Skipping.")
            continue
        
        img_count = 0
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Normalize pixel values to [0, 1]
            img = img / 255.0
            
            X.append(img)
            y.append(class_idx)
            img_count += 1
            
        print(f"Loaded {img_count} images for class {class_name}")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    return X, y

# Create ResNet model with transfer learning
class ResNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetModel, self).__init__()
        
        # Load pre-trained ResNet18 model
        self.resnet = models.resnet18(weights=None)  # Don't use pre-trained weights since our data is specific
        
        # Replace the final fully connected layer for our number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Train the model for the full number of epochs
    """
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc)
        
        # Update learning rate based on validation accuracy
        scheduler.step(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}')
        
        # Save best model
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'Saved model with validation accuracy: {best_val_acc:.4f}')
    
    print(f'Training completed for full {num_epochs} epochs.')
    print(f'Best model was at epoch {best_epoch+1} with validation accuracy: {best_val_acc:.4f}')
    
    # Load the best model before returning
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'best_epoch': best_epoch
    }

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history['train_acc'], label='Training Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('ResNet Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Mark the best epoch
    if 'best_epoch' in history:
        best_epoch = history['best_epoch']
        ax1.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch+1}')
    
    # Plot loss
    ax2.plot(history['train_loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('ResNet Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    # Mark the best epoch
    if 'best_epoch' in history:
        ax2.axvline(x=best_epoch, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('resnet_training_history.png')
    plt.show()

def train_resnet_model():
    """
    Main function to train the ResNet model
    """
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    
    if len(X) == 0:
        print("No data found. Please check your dataset path.")
        return
        
    print(f"Dataset loaded: {len(X)} images, {NUM_CLASSES} classes")
    
    # Enhanced data augmentation techniques for more robust model
    transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.15, 0.15)),
        transforms.RandomAffine(0, scale=(0.85, 1.15)),
        transforms.RandomAffine(0, shear=10),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    
    # Create dataset
    dataset = HandSignDataset(X, y, transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    print(f"Training set: {train_size} images")
    print(f"Validation set: {val_size} images")
    print(f"Device being used: {DEVICE}")
    
    # Create model
    print("Creating ResNet model...")
    model = ResNetModel(NUM_CLASSES).to(DEVICE)
    print(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    print("Training model...")
    print(f"Will train for {EPOCHS} epochs")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    
    # Plot training history
    plot_training_history(history)
    
    print(f"Model training completed. Model saved as '{MODEL_SAVE_PATH}'")
    
    # Save model for mobile deployment
    example = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(MOBILE_MODEL_PATH)
    
    print(f"Mobile-optimized model saved as '{MOBILE_MODEL_PATH}'")

if __name__ == "__main__":
    train_resnet_model()