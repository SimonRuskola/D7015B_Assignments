import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ['L0', 'L1', 'L2', 'L3']
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}

TARGET_SECTION = 'S1S2'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
TEST_SIZE = 0.40
NUM_WORKERS = 0 if os.name == 'nt' else 2

# Device selection (CUDA / MPS / CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# STEP 1: DATA LOADING & DATASET CLASS
# ============================================================================
def collect_image_paths_and_labels(base_dir, section=TARGET_SECTION):
    """Scan spectrogram images in target class directories."""
    image_paths = []
    labels = []
    
    for label in LABELS:
        folder = base_dir / label
        if not folder.exists():
            print(f"Warning: Directory {folder} not found. Skipping...")
            continue
            
        files = [
            filepath for filepath in sorted(folder.iterdir())
            if filepath.is_file()
            and filepath.suffix.lower() in IMAGE_EXTENSIONS
            and section in filepath.name
        ]
        print(f"Found {len(files)} {section} images for class {label}")
        
        for filepath in files:
            image_paths.append(str(filepath))
            labels.append(LABEL_MAP[label])
            
    return np.array(image_paths), np.array(labels)


class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


# Image transformations standard for pretrained ResNet-18
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ============================================================================
# STEP 2: BUILD / LOAD RESNET-18
# ============================================================================
def build_resnet18(num_classes=4):
    """Load pretrained ResNet-18 and replace top classification head."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Replace final FC layer for 4 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def plot_training_curves(history):
    """Plot training and test history for loss and accuracy."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# STEP 3: TRAINING LOOP
# ============================================================================
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=25):
    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # --- Evaluation Phase ---
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
        epoch_test_loss = running_test_loss / total_test
        epoch_test_acc = correct_test / total_test
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc*100:.2f}%")
        
    return history

# ============================================================================
# STEP 4: EVALUATION & PLOTTING
# ============================================================================
def evaluate_and_plot(model, test_loader, test_paths, test_labels):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS,
                cbar_kws={'label': 'Count'})
    plt.title(f'ResNet-18 Confusion Matrix (Overall Accuracy: {acc*100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'resnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS")
    print("="*50)
    print(f"Overall Test Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=LABELS))
    
    # 2. Plot Sample Spectrograms with Predictions
    indices = np.random.choice(len(test_paths), size=min(6, len(test_paths)), replace=False)
    plt.figure(figsize=(12, 8))
    
    for i, idx in enumerate(indices):
        img_path = test_paths[idx]
        true_lbl = LABELS[test_labels[idx]]
        pred_lbl = LABELS[all_preds[idx]]
        
        img = Image.open(img_path)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        
        color = 'green' if true_lbl == pred_lbl else 'red'
        plt.title(f"True: {true_lbl} | Pred: {pred_lbl}", color=color, fontweight='bold')
        plt.axis('off')
        
    plt.suptitle("Spectrogram Sample Predictions (ResNet-18)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nOutputs saved successfully to '{OUTPUT_DIR}'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("Loading image paths...")
    X_paths, y = collect_image_paths_and_labels(DATA_DIR)
    
    if len(X_paths) == 0:
        raise FileNotFoundError(
            f"No image data found. Check that {DATA_DIR} contains L0-L3 folders with {TARGET_SECTION} spectrograms."
        )
        
    print(f"Total images collected: {len(X_paths)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train / Test split (60% / 40%)
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        X_paths, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    
    # Create Datasets and DataLoaders
    train_dataset = SpectrogramDataset(X_train_paths, y_train, transform=train_transforms)
    test_dataset = SpectrogramDataset(X_test_paths, y_test, transform=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Initialize Model, Loss, and Optimizer
    model = build_resnet18(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    print("\nStarting ResNet-18 Fine-Tuning...")
    history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=EPOCHS)
    plot_training_curves(history)
    
    # Save Model Weights
    torch.save(model.state_dict(), OUTPUT_DIR / 'resnet18_wear_model.pth')
    
    # Evaluate
    evaluate_and_plot(model, test_loader, X_test_paths, y_test)