# %%
"""
CS 4774 Machine Learning - Homework 3
Knowledge Distillation for AI Dermatologist
Team: NeuralNexus
Optimized implementation for maximum F1 score with <25MB model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# %%
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# %%
# Hyperparameters - Optimized for best performance
CONFIG = {
    'num_classes': 10,
    'img_size': 224,
    'batch_size': 64,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'temperature': 4.0,  # Temperature for distillation
    'alpha': 0.7,  # Weight for distillation loss (0.7 distillation, 0.3 hard labels)
    'label_smoothing': 0.1,
    'num_workers': 4,
    'patience': 15,  # Early stopping patience
}

# %%
# Dataset class with advanced augmentation
class SkinDiseaseDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# %%
# Advanced data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(CONFIG['img_size']),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform for teacher model (MedSigLIP might have different requirements)
teacher_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Efficient Student Model - ShuffleNetV2 with custom head
class EfficientStudentModel(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super(EfficientStudentModel, self).__init__()
        # Use ShuffleNetV2 as backbone (very efficient, ~2-5 MB)
        self.backbone = models.shufflenet_v2_x1_0(weights='DEFAULT')
        
        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features
        
        # Replace classifier with custom head
        self.backbone.fc = nn.Identity()
        
        # Custom classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

# Alternative: Even more efficient model
class UltraEfficientStudent(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super(UltraEfficientStudent, self).__init__()
        # MobileNetV3 Small - extremely efficient
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        num_features = self.backbone.classifier[0].in_features
        
        # Custom classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# %%
# Teacher Model - MedSigLIP from Google
class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        try:
            # Try to load MedSigLIP from HuggingFace
            from transformers import AutoModel, AutoProcessor
            model_name = "google/siglip-so400m-patch14-384"  # MedSigLIP variant
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Add classification head
            hidden_size = self.model.config.vision_config.hidden_size
            self.classifier = nn.Linear(hidden_size, num_classes)
            
        except Exception as e:
            print(f"Could not load MedSigLIP, using ResNet50 as fallback teacher: {e}")
            # Fallback to ResNet50 pretrained on ImageNet
            self.model = models.resnet50(weights='DEFAULT')
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
            self.processor = None
            
    def forward(self, x):
        if self.processor is not None:
            # MedSigLIP path
            outputs = self.model.vision_model(x)
            pooled_output = outputs.pooler_output
            return self.classifier(pooled_output)
        else:
            # ResNet path
            return self.model(x)

# %%
# Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7, num_classes=10):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distillation_loss = self.kl_loss(soft_predictions, soft_targets) * (self.temperature ** 2)
        
        # Hard label loss
        student_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss, distillation_loss, student_loss

# %%
# Training function with knowledge distillation
def train_epoch(student_model, teacher_model, train_loader, criterion, optimizer, device):
    student_model.train()
    teacher_model.eval()
    
    running_loss = 0.0
    running_distill_loss = 0.0
    running_student_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass through teacher (no gradient)
        with torch.no_grad():
            teacher_logits = teacher_model(images)
        
        # Forward pass through student
        student_logits = student_model(images)
        
        # Calculate distillation loss
        loss, distill_loss, student_loss = criterion(student_logits, teacher_logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        running_distill_loss += distill_loss.item()
        running_student_loss += student_loss.item()
        
        _, predicted = torch.max(student_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    epoch_distill = running_distill_loss / len(train_loader)
    epoch_student = running_student_loss / len(train_loader)
    
    return epoch_loss, epoch_acc, epoch_distill, epoch_student

# %%
# Validation function
def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return accuracy, macro_f1, all_preds, all_labels

# %%
# Main training loop
def train_distillation(student_model, teacher_model, train_loader, val_loader, 
                      num_epochs, device, save_path='student_model.pth'):
    
    criterion = DistillationLoss(
        temperature=CONFIG['temperature'],
        alpha=CONFIG['alpha'],
        num_classes=CONFIG['num_classes']
    )
    
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Train
        train_loss, train_acc, distill_loss, student_loss = train_epoch(
            student_model, teacher_model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_acc, val_f1, _, _ = validate(student_model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Distill Loss: {distill_loss:.4f} | Student Loss: {student_loss:.4f}')
        print(f'Val Acc: {val_acc:.2f}% | Val Macro F1: {val_f1:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'lr': current_lr
        })
        
        # Save best model based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': val_f1,
                'accuracy': val_acc,
            }, save_path)
            print(f'✓ Saved best model with F1: {val_f1:.4f}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            print(f'Best F1: {best_f1:.4f} at epoch {best_epoch}')
            break
    
    return history, best_f1

# %%
# Model size calculation
def get_model_size(model, filepath):
    torch.save(model.state_dict(), filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    return size_mb

# %%
# Model compression techniques
def compress_model(model, quantize=True):
    """Apply post-training optimizations to reduce model size"""
    if quantize:
        # Dynamic quantization for linear layers
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
    return model

# %%
# Initialize models
def initialize_models():
    print("Initializing models...")
    
    # Student model (choose one based on size requirements)
    student = EfficientStudentModel(num_classes=CONFIG['num_classes']).to(device)
    # Alternative: student = UltraEfficientStudent(num_classes=CONFIG['num_classes']).to(device)
    
    # Teacher model
    teacher = TeacherModel(num_classes=CONFIG['num_classes']).to(device)
    
    # Calculate initial model sizes
    temp_path = 'temp_student.pth'
    student_size = get_model_size(student, temp_path)
    print(f"Student model size: {student_size:.2f} MB")
    
    if student_size >= 25:
        print("⚠ Warning: Student model exceeds 25 MB limit!")
        print("Consider using UltraEfficientStudent or further optimization")
    
    os.remove(temp_path)
    
    return student, teacher

# %%
# Main execution
if __name__ == "__main__":
    # Data paths (adjust these to your dataset location)
    TRAIN_CSV = 'train.csv'  # CSV with columns: image_filename, label
    TRAIN_IMG_DIR = 'train_images/'
    VAL_CSV = 'val.csv'  # If you have a validation set
    VAL_IMG_DIR = 'val_images/'
    
    # Create datasets and dataloaders
    print("Loading datasets...")
    try:
        train_dataset = SkinDiseaseDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=train_transform)
        val_dataset = SkinDiseaseDataset(VAL_CSV, VAL_IMG_DIR, transform=val_transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True, 
            num_workers=CONFIG['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False, 
            num_workers=CONFIG['num_workers'],
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
    except FileNotFoundError:
        print("⚠ Dataset files not found. Please adjust paths in the script.")
        print("Expected files:")
        print(f"  - {TRAIN_CSV}")
        print(f"  - {TRAIN_IMG_DIR}")
        print(f"  - {VAL_CSV} (optional, can use train/val split)")
        print(f"  - {VAL_IMG_DIR}")
        exit()
    
    # Initialize models
    student_model, teacher_model = initialize_models()
    
    # Load pretrained teacher if available
    # teacher_checkpoint = 'teacher_model.pth'
    # if os.path.exists(teacher_checkpoint):
    #     teacher_model.load_state_dict(torch.load(teacher_checkpoint))
    #     print(f"Loaded pretrained teacher from {teacher_checkpoint}")
    
    # Train with knowledge distillation
    print("\n" + "="*60)
    print("Starting knowledge distillation training")
    print("="*60)
    
    history, best_f1 = train_distillation(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CONFIG['num_epochs'],
        device=device,
        save_path='best_student_model.pth'
    )
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_student_model.pth')
    student_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    val_acc, val_f1, preds, labels = validate(student_model, val_loader, device)
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print(f"Final Macro F1 Score: {val_f1:.4f}")
    
    # Detailed classification report
    class_names = [
        'Eczema', 'Melanoma', 'Atopic Dermatitis', 'Basal Cell Carcinoma',
        'Melanocytic Nevi', 'Benign Keratosis-like Lesions', 
        'Psoriasis/Lichen Planus', 'Seborrheic Keratoses',
        'Tinea/Fungal infections', 'Warts/Viral Infections'
    ]
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))
    
    # Model size
    final_size = get_model_size(student_model, 'final_student_model.pth')
    print(f"\nFinal Model Size: {final_size:.2f} MB")
    
    # Calculate weighted score
    weighted_score = (10 * val_f1) - final_size
    print(f"Weighted Score: {weighted_score:.4f}")
    
    # Optional: Apply model compression
    print("\nApplying model compression...")
    compressed_model = compress_model(student_model, quantize=False)
    compressed_size = get_model_size(compressed_model, 'compressed_student_model.pth')
    print(f"Compressed Model Size: {compressed_size:.2f} MB")
    
    print("\n✓ Training completed successfully!")
    print(f"Best model saved as: best_student_model.pth")
    print(f"Final model saved as: final_student_model.pth")

# %%
# Export model for submission
def export_for_submission(model, filepath='submission_model.pth'):
    """
    Export model in the format required for submission
    """
    # Save only the state dict for smaller file size
    torch.save(model.state_dict(), filepath)
    size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Submission model saved: {filepath} ({size:.2f} MB)")
    return size

# %%
# Inference function for test set
def predict_test_set(model, test_loader, device):
    """
    Generate predictions for test set
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    
    return predictions

# %%
# Tips for optimization:
"""
OPTIMIZATION STRATEGIES:

1. Model Architecture:
   - ShuffleNetV2 (2-5 MB): Good balance of size and accuracy
   - MobileNetV3-Small (1-3 MB): Extremely efficient
   - EfficientNet-B0 (5-10 MB): Higher accuracy, larger size
   
2. Training Techniques:
   - Higher temperature (3-5): Better knowledge transfer
   - Alpha tuning (0.6-0.8): Balance between distillation and labels
   - Mixed precision training: Faster training
   - Gradient accumulation: Effective larger batch size
   
3. Data Augmentation:
   - Advanced augmentations (Cutout, Mixup, CutMix)
   - Test-time augmentation for inference
   
4. Model Compression:
   - Pruning: Remove low-importance weights
   - Quantization: Use int8 instead of float32
   - Knowledge distillation (already doing this!)
   
5. Hyperparameter Tuning:
   - Learning rate: Try 1e-4 to 1e-3
   - Batch size: 32-128 depending on GPU memory
   - Temperature: 3-5 for distillation
   - Alpha: 0.6-0.8 for loss weighting
   
6. Ensemble Techniques:
   - Train multiple students, use voting
   - Use different augmentations for diversity
"""

print("\n" + "="*60)
print("Knowledge Distillation Implementation Complete!")
print("="*60)
