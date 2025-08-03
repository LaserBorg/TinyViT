'''
TinyViT Training with Pretrained Weights and Two-Stage Fine-tuning

Features:
- Load pretrained weights from checkpoint
- Two-stage training: frozen backbone + full fine-tuning
- Early stopping
- TensorBoard logging
- Mixed precision training
- Confusion matrix
'''

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="models.tiny_vit")

import os
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from models.tiny_vit import TinyViT
from tqdm import tqdm
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast


def load_model_config(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_model(cfg, num_classes, img_size, device):
    model_cfg = cfg["MODEL"]
    tiny_cfg = model_cfg["TINY_VIT"]

    model = TinyViT(
        img_size=img_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dims=tiny_cfg["EMBED_DIMS"],
        depths=tiny_cfg["DEPTHS"],
        num_heads=tiny_cfg["NUM_HEADS"],
        window_sizes=tiny_cfg["WINDOW_SIZES"],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=model_cfg.get("DROP_PATH_RATE", 0.1),
        use_checkpoint=False
    ).to(device)

    return model


def load_pretrained_weights(model, pretrained_path, num_classes, device):
    """
    Load pretrained weights and handle classifier head mismatch
    """
    print(f"Loading pretrained weights from: {pretrained_path}")
    
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained checkpoint not found at {pretrained_path}")
        return model
    
    # Load pretrained state dict
    pretrained_state = torch.load(pretrained_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in pretrained_state:
        pretrained_state = pretrained_state['model']
    elif 'state_dict' in pretrained_state:
        pretrained_state = pretrained_state['state_dict']
    
    # Get model state dict
    model_state = model.state_dict()
    
    # Filter out classifier head if number of classes doesn't match
    filtered_state = {}
    classifier_keys = ['head.weight', 'head.bias', 'classifier.weight', 'classifier.bias']
    
    for key, value in pretrained_state.items():
        if any(cls_key in key for cls_key in classifier_keys):
            # Check if classifier dimensions match
            if key in model_state and value.shape != model_state[key].shape:
                print(f"Skipping {key} due to shape mismatch: {value.shape} vs {model_state[key].shape}")
                continue
        
        if key in model_state:
            if value.shape == model_state[key].shape:
                filtered_state[key] = value
            else:
                print(f"Skipping {key} due to shape mismatch: {value.shape} vs {model_state[key].shape}")
    
    # Load filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    
    print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    if missing_keys:
        print(f"Missing keys: {missing_keys[:10]}...")  # Show first 10
    
    return model


def freeze_backbone(model):
    """
    Freeze all parameters except the classifier head
    """
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        # Keep head/classifier unfrozen
        if 'head' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen_params += 1
    
    print(f"Frozen {frozen_params}/{total_params} parameters")
    return model


def unfreeze_all(model):
    """
    Unfreeze all parameters
    """
    for param in model.parameters():
        param.requires_grad = True
    print("Unfrozen all parameters")
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, use_amp=True):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
    
    return train_loss / len(train_loader), train_correct / train_total


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return val_loss / len(val_loader), val_correct / val_total, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_class_labels(class_names, save_path):
    """
    Save class labels in ImageNet-style JSON format
    """
    class_labels = {}
    for idx, class_name in enumerate(class_names):
        # For custom datasets, we typically only have the folder name
        # You can extend this to include additional synonyms or descriptions
        class_labels[str(idx)] = [class_name]
    
    with open(save_path, 'w') as f:
        json.dump(class_labels, f, indent=2)
    
    print(f"Class labels saved to {save_path}")
    return class_labels


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        return self.transform(image=img)["image"]


# Settings
cfg_path = "configs/higher_resolution/tiny_vit_21m_224to384.yaml"
pretrained_path = "checkpoints/tiny_vit_21m_22kto1k_384_distill.pth"

data_dir = "dataset"
checkpoints_dir = "checkpoints"
logs_dir = "logs"

checkpoint_name = "tiny_vit_21m_384_finetuned.pth"
labels_name = "finetuned_classes.json"

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
val_dir = os.path.join(data_dir, "val")

# Create directories
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Count classes
num_classes = sum(os.path.isdir(os.path.join(train_dir, entry)) for entry in os.listdir(train_dir))
print("num_classes:", num_classes)

# Training parameters
img_size = 384  # 224, 512
batch_size = 6

# Two-stage training parameters
stage1_epochs = 0  # Frozen backbone training
stage2_epochs = 1  # Full fine-tuning

stage1_lr = 1e-3    # Higher learning rate for head training
stage2_lr = 1e-5    # Lower learning rate for full fine-tuning
step_size = 3
gamma = 0.5

# Mixed precision training
use_amp = False

# Early stopping
patience = 5

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TensorBoard writer
writer = SummaryWriter(logs_dir)

# Transforms
train_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

eval_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# Datasets and Dataloaders
train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "train"),
    transform=AlbumentationsTransform(train_transform)
)

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "val"),
    transform=AlbumentationsTransform(eval_transform)
)

test_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "test"),
    transform=AlbumentationsTransform(eval_transform)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Get class names and create class labels mapping
class_names = train_dataset.classes
print(f"Classes found: {class_names}")

# Save class labels in JSON format
labels_path = os.path.join(checkpoints_dir, labels_name)
class_labels_dict = save_class_labels(class_names, labels_path)

# Load config and build model
cfg = load_model_config(cfg_path)
img_size = cfg.get("DATA", {}).get("IMG_SIZE", img_size)
model = get_model(cfg, num_classes, img_size, device)

# Load pretrained weights
model = load_pretrained_weights(model, pretrained_path, num_classes, device)

# Initialize mixed precision scaler
scaler = GradScaler() if use_amp else None

# Loss function
criterion = nn.CrossEntropyLoss()

print("=" * 60)
print("STAGE 1: Training classifier head with frozen backbone")
print("=" * 60)

# Stage 1: Freeze backbone and train only the head
model = freeze_backbone(model)

# Optimizer and scheduler for stage 1
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=stage1_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Early stopping for stage 1
early_stopping = EarlyStopping(patience=patience)

# Stage 1 training loop
global_step = 0
for epoch in range(stage1_epochs):
    print(f"\nEpoch {epoch+1}/{stage1_epochs}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
    
    # Validate
    val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
    
    # Log to TensorBoard
    writer.add_scalar('Stage1/Train_Loss', train_loss, epoch)
    writer.add_scalar('Stage1/Train_Acc', train_acc, epoch)
    writer.add_scalar('Stage1/Val_Loss', val_loss, epoch)
    writer.add_scalar('Stage1/Val_Acc', val_acc, epoch)
    writer.add_scalar('Stage1/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Step scheduler
    scheduler.step()
    
    # Early stopping check
    if early_stopping(val_loss, model):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    
    global_step += 1

print("\n" + "=" * 60)
print("STAGE 2: Fine-tuning entire network")
print("=" * 60)

# Stage 2: Unfreeze all parameters and fine-tune
model = unfreeze_all(model)

# New optimizer and scheduler for stage 2 with lower learning rate
optimizer = optim.AdamW(model.parameters(), lr=stage2_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Reset early stopping for stage 2
early_stopping = EarlyStopping(patience=patience)

# Stage 2 training loop
for epoch in range(stage2_epochs):
    print(f"\nEpoch {epoch+1}/{stage2_epochs}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
    
    # Validate
    val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
    
    # Log to TensorBoard
    writer.add_scalar('Stage2/Train_Loss', train_loss, global_step + epoch)
    writer.add_scalar('Stage2/Train_Acc', train_acc, global_step + epoch)
    writer.add_scalar('Stage2/Val_Loss', val_loss, global_step + epoch)
    writer.add_scalar('Stage2/Val_Acc', val_acc, global_step + epoch)
    writer.add_scalar('Stage2/Learning_Rate', optimizer.param_groups[0]['lr'], global_step + epoch)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Step scheduler
    scheduler.step()
    
    # Early stopping check
    if early_stopping(val_loss, model):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Save final model with metadata
model_save_dict = {
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'class_names': class_names,
    'img_size': img_size,
    'config_path': cfg_path,
    'pretrained_path': pretrained_path,
    'labels_file': labels_name
}

torch.save(model_save_dict, os.path.join(checkpoints_dir, checkpoint_name))

print(f"\nModel saved to {os.path.join(checkpoints_dir, checkpoint_name)}")
print(f"Class labels saved to {labels_path}")

# Final evaluation on test set
print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)

model.eval()
test_correct, test_total = 0, 0
test_preds, test_labels = [], []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
        
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc:.4f}")

# Log final test accuracy
writer.add_scalar('Final/Test_Acc', test_acc, 0)

# Generate confusion matrix
cm_path = os.path.join(logs_dir, 'confusion_matrix.png')
plot_confusion_matrix(test_labels, test_preds, class_names, cm_path)
print(f"Confusion matrix saved to {cm_path}")

# Generate classification report
report = classification_report(test_labels, test_preds, target_names=class_names)
print("\nClassification Report:")
print(report)

# Save classification report
report_path = os.path.join(logs_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)

# Also save a summary file with training information
summary_info = {
    'final_test_accuracy': float(test_acc),
    'num_classes': num_classes,
    'class_names': class_names,
    'model_architecture': 'TinyViT',
    'config_used': cfg_path,
    'pretrained_checkpoint': pretrained_path,
    'training_stages': {
        'stage1_epochs': stage1_epochs,
        'stage1_lr': stage1_lr,
        'stage2_epochs': stage2_epochs,
        'stage2_lr': stage2_lr
    },
    'image_size': img_size,
    'batch_size': batch_size
}

summary_path = os.path.join(logs_dir, 'training_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary_info, f, indent=2)

print(f"Training summary saved to {summary_path}")

writer.close()
print(f"\nTensorBoard logs saved to {logs_dir}")
print("Training complete!")