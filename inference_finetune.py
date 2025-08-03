'''
TinyViT Inference Script

Load a fine-tuned TinyViT model and perform inference on images
'''

import os
import json
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.tiny_vit import TinyViT


def load_model_config(cfg_path):
    """Load model configuration from YAML file"""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_model(cfg, num_classes, img_size, device):
    """Create TinyViT model with specified configuration"""
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


def load_class_labels(labels_path):
    """Load class labels from JSON file"""
    with open(labels_path, 'r') as f:
        class_labels = json.load(f)
    return class_labels


def load_trained_model(checkpoint_path, labels_path, config_path, device):
    """
    Load a trained TinyViT model with its class labels
    
    Returns:
        model: Loaded PyTorch model
        class_labels: Dictionary mapping class indices to names
        transform: Image preprocessing transform
        img_size: Expected input image size
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model parameters from checkpoint
    num_classes = checkpoint['num_classes']
    img_size = checkpoint['img_size']
    
    # Load model configuration
    cfg = load_model_config(config_path)
    
    # Create model
    model = get_model(cfg, num_classes, img_size, device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load class labels
    class_labels = load_class_labels(labels_path)
    
    # Create preprocessing transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    print(f"Model loaded successfully!")
    print(f"Number of classes: {num_classes}")
    print(f"Image size: {img_size}")
    print(f"Classes: {[labels[0] for labels in class_labels.values()]}")
    
    return model, class_labels, transform, img_size


def predict_image(model, image_path, class_labels, transform, device, top_k=5):
    """
    Predict the class of a single image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to input image
        class_labels: Dictionary mapping class indices to names
        transform: Image preprocessing transform
        device: PyTorch device
        top_k: Number of top predictions to return
    
    Returns:
        predictions: List of (class_name, confidence) tuples
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    # Apply transforms
    transformed = transform(image=image_array)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for i in range(top_k):
            class_idx = str(top_indices[0][i].item())
            confidence = top_probs[0][i].item()
            class_name = class_labels[class_idx][0]  # Get primary class name
            predictions.append((class_name, confidence))
    
    return predictions


def predict_batch(model, image_paths, class_labels, transform, device):
    """
    Predict classes for a batch of images
    
    Args:
        model: Trained PyTorch model
        image_paths: List of paths to input images
        class_labels: Dictionary mapping class indices to names
        transform: Image preprocessing transform
        device: PyTorch device
    
    Returns:
        batch_predictions: List of prediction lists for each image
    """
    batch_tensors = []
    
    # Load and preprocess all images
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        transformed = transform(image=image_array)
        batch_tensors.append(transformed['image'])
    
    # Stack into batch
    batch_tensor = torch.stack(batch_tensors).to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_indices = torch.max(probabilities, 1)
        
        batch_predictions = []
        for i, pred_idx in enumerate(predicted_indices):
            class_idx = str(pred_idx.item())
            confidence = probabilities[i][pred_idx].item()
            class_name = class_labels[class_idx][0]
            batch_predictions.append((class_name, confidence))
    
    return batch_predictions


def main():
    """Example usage of the inference functions"""
    
    # Paths to saved model files
    checkpoint_path = "checkpoints/tiny_vit_21m_384_finetuned.pth"
    labels_path = "checkpoints/finetuned_classes.json"
    config_path = "configs/higher_resolution/tiny_vit_21m_224to384.yaml"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load trained model
        model, class_labels, transform, img_size = load_trained_model(
            checkpoint_path, labels_path, config_path, device
        )
        
        # Example: Single image prediction
        image_path = "path/to/your/test_image.jpg"  # Replace with actual image path
        
        if os.path.exists(image_path):
            print(f"\nPredicting image: {image_path}")
            predictions = predict_image(model, image_path, class_labels, transform, device, top_k=3)
            
            print("Top 3 predictions:")
            for i, (class_name, confidence) in enumerate(predictions, 1):
                print(f"{i}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)")
        else:
            print(f"Image not found: {image_path}")
            print("Please provide a valid image path to test the model.")
        
        # Example: Batch prediction
        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]  # Replace with actual paths
        existing_paths = [path for path in image_paths if os.path.exists(path)]
        
        if existing_paths:
            print(f"\nBatch prediction for {len(existing_paths)} images:")
            batch_predictions = predict_batch(model, existing_paths, class_labels, transform, device)
            
            for path, (class_name, confidence) in zip(existing_paths, batch_predictions):
                print(f"{os.path.basename(path)}: {class_name} ({confidence:.4f})")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Make sure you have:")
        print("1. Trained model checkpoint: checkpoints/tinyvit_finetuned.pth")
        print("2. Class labels file: checkpoints/class_labels.json")
        print("3. Model config file: configs/22kto1k/tiny_vit_21m_22kto1k.yaml")
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    main()