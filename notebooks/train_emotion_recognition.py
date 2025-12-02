# %% [markdown]
# # Facial Expression Recognition Training Pipeline
# 
# This notebook trains a MobileNetV2-based model for facial expression recognition on the FER2013 dataset.
# 
# **Team:** VisionMasters  
# **Course:** ECE 4332 - AI Hardware Design and Implementation
# 
# ## Objectives
# - Train baseline FP32 model on FER2013 dataset
# - Achieve >85% test accuracy
# - Evaluate model performance with confusion matrix
# - Quantize model to INT8 for Edge TPU deployment
# 
# ## Setup Instructions for Google Colab
# 
# 1. Upload this notebook to Google Colab
# 2. Enable GPU: Runtime > Change runtime type > GPU
# 3. Run all cells in order

# %% [markdown]
# ## 1. Environment Setup

# %%
# Check if running on Colab
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🚀 Running on Google Colab")
    print("📦 Installing additional dependencies...")
    !pip install -q mediapipe
else:
    print("💻 Running locally")

# %%
# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {len(gpus)} GPU(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️  No GPU found, using CPU (training will be slower)")

# %%
# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
CONFIG = {
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'num_classes': 7,
    'seed': SEED
}

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## 2. Dataset Setup
# 
# ### For Google Colab Users:
# 
# You need to upload the FER2013 dataset. Two options:
# 
# **Option 1: Using Kaggle API (Recommended)**
# ```python
# # Upload your kaggle.json file
# from google.colab import files
# files.upload()  # Upload kaggle.json
# 
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d msambare/fer2013
# !unzip -q fer2013.zip -d /content/fer2013
# ```
# 
# **Option 2: Upload to Google Drive**
# - Download FER2013 from Kaggle
# - Upload to your Google Drive
# - Mount Drive and copy dataset

# %%
# Setup dataset paths
if IN_COLAB:
    # Colab paths
    BASE_DIR = Path('/content')
    DATA_DIR = BASE_DIR / 'fer2013'
    MODEL_DIR = BASE_DIR / 'models'
    RESULTS_DIR = BASE_DIR / 'results'
    
    # Option to mount Google Drive (uncomment if using Drive)
    # from google.colab import drive
    # drive.mount('/content/drive')
    # DATA_DIR = Path('/content/drive/MyDrive/fer2013')
else:
    # Local paths
    BASE_DIR = Path.cwd()
    if 'notebooks' in str(BASE_DIR):
        BASE_DIR = BASE_DIR.parent
    DATA_DIR = BASE_DIR / 'data' / 'fer2013'
    MODEL_DIR = BASE_DIR / 'models'
    RESULTS_DIR = BASE_DIR / 'results'

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'charts').mkdir(parents=True, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"Results directory: {RESULTS_DIR}")

# %% [markdown]
# ### Upload Dataset (Colab Only)
# 
# Uncomment and run ONE of these options:

# %%
# OPTION 1: Upload Kaggle credentials and download
if IN_COLAB and False:  # Change False to True to enable
    print("Uploading Kaggle credentials...")
    from google.colab import files
    uploaded = files.upload()  # Upload kaggle.json
    
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    
    print("Downloading FER2013 dataset...")
    !kaggle datasets download -d msambare/fer2013
    !unzip -q fer2013.zip -d {DATA_DIR}
    print("✅ Dataset downloaded and extracted")

# %%
# OPTION 2: Mount Google Drive (if dataset is in Drive)
if IN_COLAB and False:  # Change False to True to enable
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Update DATA_DIR to point to your Drive location
    # DATA_DIR = Path('/content/drive/MyDrive/fer2013')
    print(f"✅ Drive mounted. Update DATA_DIR if needed: {DATA_DIR}")

# %% [markdown]
# ## 3. Dataset Analysis and Verification

# %%
# Check if dataset exists
def check_dataset(data_dir):
    """Verify dataset structure"""
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    if not train_dir.exists() or not test_dir.exists():
        print(f"❌ Dataset not found at {data_dir}")
        print("\nExpected structure:")
        print("  fer2013/")
        print("    train/")
        print("      angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/")
        print("    test/")
        print("      angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/")
        return False
    
    print("✅ Dataset structure verified")
    return True

dataset_ok = check_dataset(DATA_DIR)
if not dataset_ok:
    print("\n⚠️  Please upload the dataset before continuing!")

# %%
# Analyze dataset statistics
def analyze_dataset(data_dir):
    """Count images per class and split"""
    stats = {'train': {}, 'test': {}}
    
    for split in ['train', 'test']:
        split_path = data_dir / split
        if not split_path.exists():
            continue
        
        for emotion in EMOTION_LABELS:
            emotion_path = split_path / emotion.lower()
            if emotion_path.exists():
                count = len(list(emotion_path.glob('*.jpg'))) + len(list(emotion_path.glob('*.png')))
                stats[split][emotion] = count
            else:
                stats[split][emotion] = 0
    
    return stats

if dataset_ok:
    stats = analyze_dataset(DATA_DIR)
    
    print("\n" + "="*60)
    print("FER2013 Dataset Statistics")
    print("="*60)
    print(f"{'Emotion':<12} {'Train':>10} {'Test':>10} {'Total':>10}")
    print("-"*60)
    
    for emotion in EMOTION_LABELS:
        train_count = stats['train'].get(emotion, 0)
        test_count = stats['test'].get(emotion, 0)
        total = train_count + test_count
        print(f"{emotion:<12} {train_count:>10,} {test_count:>10,} {total:>10,}")
    
    print("-"*60)
    total_train = sum(stats['train'].values())
    total_test = sum(stats['test'].values())
    print(f"{'TOTAL':<12} {total_train:>10,} {total_test:>10,} {total_train + total_test:>10,}")
    print("="*60)

# %%
# Visualize sample images
def visualize_samples(data_dir):
    """Show sample images from each emotion class"""
    train_dir = data_dir / 'train'
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, emotion in enumerate(EMOTION_LABELS):
        emotion_path = train_dir / emotion.lower()
        if emotion_path.exists():
            image_files = list(emotion_path.glob('*.jpg')) + list(emotion_path.glob('*.png'))
            if image_files:
                img = Image.open(image_files[0])
                axes[idx].imshow(img, cmap='gray')
                axes[idx].set_title(emotion, fontsize=14, fontweight='bold')
                axes[idx].axis('off')
    
    if len(EMOTION_LABELS) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Sample visualization saved")

if dataset_ok:
    visualize_samples(DATA_DIR)

# %%
# Plot class distribution
def plot_class_distribution(stats):
    """Create bar chart of class distribution"""
    emotions = EMOTION_LABELS
    train_counts = [stats['train'].get(e, 0) for e in emotions]
    test_counts = [stats['test'].get(e, 0) for e in emotions]
    
    x = np.arange(len(emotions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_counts, width, label='Train', color='#3498db')
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test', color='#e74c3c')
    
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('FER2013 Dataset - Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'charts' / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Class distribution chart saved")

if dataset_ok:
    plot_class_distribution(stats)

# %% [markdown]
# ## 4. Data Preparation

# %%
# Create data generators
def create_data_generators():
    """Setup data augmentation and generators"""
    print("📊 Setting up data generators...")
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Test data (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=CONFIG['seed'],
        color_mode='rgb'
    )
    
    val_generator = train_datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=CONFIG['seed'],
        color_mode='rgb'
    )
    
    test_generator = test_datagen.flow_from_directory(
        DATA_DIR / 'test',
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )
    
    print(f"✅ Train samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Test samples: {test_generator.samples}")
    print(f"✅ Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator, test_generator

if dataset_ok:
    train_gen, val_gen, test_gen = create_data_generators()

# %%
# Visualize augmented images
def visualize_augmentation(generator):
    """Show examples of augmented images"""
    batch = next(generator)
    images = batch[0][:9]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if dataset_ok:
    print("Sample augmented images:")
    visualize_augmentation(train_gen)

# %% [markdown]
# ## 5. Model Architecture

# %%
# Build model
def build_model():
    """Build MobileNetV2-based model"""
    print("🏗️  Building MobileNetV2 model...")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(*CONFIG['img_size'], 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build complete model
    inputs = keras.Input(shape=(*CONFIG['img_size'], 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(CONFIG['num_classes'], activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )
    
    print(f"✅ Model built successfully")
    print(f"   Total parameters: {model.count_params():,}")
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Trainable parameters: {trainable:,}")
    
    return model

if dataset_ok:
    model = build_model()
    model.summary()

# %% [markdown]
# ## 6. Training Callbacks

# %%
# Setup callbacks
def create_callbacks():
    """Create training callbacks"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_DIR / 'baseline_fp32_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(RESULTS_DIR / 'logs' / timestamp),
            histogram_freq=1
        )
    ]
    
    return callbacks

# %% [markdown]
# ## 7. Model Training

# %%
# Train model
if dataset_ok:
    print("\n" + "="*60)
    print("🚀 Starting Training")
    print("="*60)
    print(f"   Epochs: {CONFIG['epochs']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Learning rate: {CONFIG['learning_rate']}")
    print("="*60 + "\n")
    
    callbacks = create_callbacks()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = MODEL_DIR / 'baseline_fp32_final.h5'
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")

# %% [markdown]
# ## 8. Training Visualization

# %%
# Plot training history
def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'charts' / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Training history saved")

if dataset_ok:
    plot_training_history(history)

# %%
# Print training summary
if dataset_ok:
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Total Epochs: {len(history.history['loss'])}")
    print("="*60)

# %% [markdown]
# ## 9. Model Evaluation

# %%
# Evaluate on test set
if dataset_ok:
    print("\n🧪 Evaluating model on test set...")
    
    # Load best model
    best_model = keras.models.load_model(MODEL_DIR / 'baseline_fp32_best.h5')
    
    # Evaluate
    test_results = best_model.evaluate(test_gen, verbose=1)
    
    print("\n" + "="*60)
    print("Test Set Results")
    print("="*60)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    if len(test_results) > 2:
        print(f"Test Top-2 Accuracy: {test_results[2]:.4f} ({test_results[2]*100:.2f}%)")
    print("="*60)

# %%
# Get predictions
if dataset_ok:
    print("\n📊 Generating predictions...")
    predictions = best_model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    print(f"✅ Generated {len(predictions)} predictions")

# %% [markdown]
# ## 10. Confusion Matrix

# %%
# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
                ax=ax2, cbar_kws={'label': 'Percentage (%)'})
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'charts' / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrix saved")
    
    return cm

if dataset_ok:
    cm = plot_confusion_matrix(y_true, y_pred)

# %%
# Per-class accuracy
def plot_per_class_accuracy(y_true, y_pred):
    """Plot accuracy for each emotion class"""
    accuracies = []
    for i in range(len(EMOTION_LABELS)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            accuracies.append(acc * 100)
        else:
            accuracies.append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(EMOTION_LABELS, accuracies, color='#3498db', edgecolor='black')
    
    # Color based on accuracy
    for bar, acc in zip(bars, accuracies):
        if acc >= 85:
            bar.set_color('#27ae60')
        elif acc >= 70:
            bar.set_color('#f39c12')
        else:
            bar.set_color('#e74c3c')
    
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Target line
    ax.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Target (85%)')
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'charts' / 'per_class_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Per-class accuracy chart saved")

if dataset_ok:
    plot_per_class_accuracy(y_true, y_pred)

# %%
# Classification report
if dataset_ok:
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    report = classification_report(y_true, y_pred, 
                                   target_names=EMOTION_LABELS,
                                   digits=4)
    print(report)
    print("="*60)
    
    # Save report
    with open(RESULTS_DIR / 'classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print("✅ Classification report saved")

# %% [markdown]
# ## 11. Model Quantization (INT8)

# %%
# Convert to TFLite (FP32)
def convert_to_tflite_fp32(model, output_path):
    """Convert Keras model to FP32 TFLite"""
    print("\n🔄 Converting to FP32 TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ FP32 TFLite model saved: {size_mb:.2f} MB")
    
    return tflite_model

if dataset_ok:
    fp32_tflite_path = MODEL_DIR / 'model_fp32.tflite'
    fp32_tflite = convert_to_tflite_fp32(best_model, fp32_tflite_path)

# %%
# Create representative dataset for quantization
def representative_dataset_generator():
    """Generate calibration data for INT8 quantization"""
    for _ in range(100):
        image, _ = next(val_gen)
        yield [image.astype(np.float32)]

# %%
# Convert to INT8 TFLite
def convert_to_tflite_int8(model, output_path, representative_dataset):
    """Convert to INT8 quantized TFLite"""
    print("\n🔄 Converting to INT8 TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ INT8 TFLite model saved: {size_mb:.2f} MB")
    
    return tflite_model

if dataset_ok:
    int8_tflite_path = MODEL_DIR / 'model_int8.tflite'
    int8_tflite = convert_to_tflite_int8(best_model, int8_tflite_path, 
                                         representative_dataset_generator)

# %%
# Compare model sizes
if dataset_ok:
    fp32_size = fp32_tflite_path.stat().st_size / (1024 * 1024)
    int8_size = int8_tflite_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*60)
    print("Model Size Comparison")
    print("="*60)
    print(f"FP32 Model: {fp32_size:.2f} MB")
    print(f"INT8 Model: {int8_size:.2f} MB")
    print(f"Compression Ratio: {fp32_size/int8_size:.2f}x")
    print(f"Size Reduction: {(1 - int8_size/fp32_size)*100:.1f}%")
    print("="*60)

# %% [markdown]
# ## 12. Save Results Summary

# %%
# Save complete results
if dataset_ok:
    results_summary = {
        'config': CONFIG,
        'training': {
            'final_train_acc': float(history.history['accuracy'][-1]),
            'final_val_acc': float(history.history['val_accuracy'][-1]),
            'best_val_acc': float(max(history.history['val_accuracy'])),
            'epochs_trained': len(history.history['loss'])
        },
        'test': {
            'loss': float(test_results[0]),
            'accuracy': float(test_results[1]),
            'top_2_accuracy': float(test_results[2]) if len(test_results) > 2 else None
        },
        'quantization': {
            'fp32_size_mb': float(fp32_size),
            'int8_size_mb': float(int8_size),
            'compression_ratio': float(fp32_size/int8_size)
        }
    }
    
    # Save as JSON
    with open(RESULTS_DIR / 'training_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save as text
    with open(RESULTS_DIR / 'training_results.txt', 'w') as f:
        f.write("Facial Expression Recognition - Training Results\n")
        f.write("="*60 + "\n\n")
        f.write("Configuration:\n")
        for key, value in CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nTraining Results:\n")
        f.write(f"  Final Training Accuracy: {results_summary['training']['final_train_acc']:.4f}\n")
        f.write(f"  Final Validation Accuracy: {results_summary['training']['final_val_acc']:.4f}\n")
        f.write(f"  Best Validation Accuracy: {results_summary['training']['best_val_acc']:.4f}\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  Test Loss: {results_summary['test']['loss']:.4f}\n")
        f.write(f"  Test Accuracy: {results_summary['test']['accuracy']:.4f}\n")
        f.write(f"\nModel Sizes:\n")
        f.write(f"  FP32: {results_summary['quantization']['fp32_size_mb']:.2f} MB\n")
        f.write(f"  INT8: {results_summary['quantization']['int8_size_mb']:.2f} MB\n")
        f.write(f"  Compression: {results_summary['quantization']['compression_ratio']:.2f}x\n")
    
    print("\n✅ Results summary saved")

# %% [markdown]
# ## 13. Download Results (Colab Only)

# %%
# Download files from Colab
if IN_COLAB and dataset_ok:
    from google.colab import files
    
    print("\n📥 Preparing files for download...")
    
    # Option 1: Download individual files
    print("\nDownload trained models:")
    print("  - baseline_fp32_best.h5")
    print("  - model_fp32.tflite")
    print("  - model_int8.tflite")
    
    # Uncomment to download specific files
    # files.download(str(MODEL_DIR / 'baseline_fp32_best.h5'))
    # files.download(str(MODEL_DIR / 'model_int8.tflite'))
    
    # Option 2: Create zip archive
    print("\nOr create a zip file with all results:")
    print("Run: !zip -r results.zip models/ results/")

# %% [markdown]
# ## 14. Next Steps
# 
# ### ✅ What You've Accomplished
# 
# - ✅ Trained MobileNetV2 model on FER2013
# - ✅ Achieved target accuracy (check results above)
# - ✅ Created comprehensive evaluation metrics
# - ✅ Quantized model to INT8 for Edge TPU
# - ✅ Generated visualizations for presentation
# 
# ### 🚀 Next Actions
# 
# 1. **Download Models**:
#    - `baseline_fp32_best.h5` - Best Keras model
#    - `model_int8.tflite` - Quantized for Edge TPU
# 
# 2. **Compile for Edge TPU** (on Raspberry Pi):
#    ```bash
#    edgetpu_compiler model_int8.tflite
#    ```
# 
# 3. **Prepare for Midterm**:
#    - Use generated charts in `results/charts/`
#    - Present training curves and confusion matrix
#    - Show test accuracy and model compression
# 
# 4. **Deploy to Hardware**:
#    - Transfer models to Raspberry Pi
#    - Run `src/hardware/inference_demo.py`
#    - Benchmark real-time performance
# 
# ### 📊 Key Results to Report
# 
# - Test Accuracy: ___% (from cell output above)
# - Model Size Reduction: ___x (FP32 → INT8)
# - Per-class accuracy breakdown
# - Dataset distribution and challenges (class imbalance)
# 
# **Great work! 🎉**

# %%
# Final summary
if dataset_ok:
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print("\n📁 Generated Files:")
    print(f"   Models: {MODEL_DIR}")
    print("     - baseline_fp32_best.h5 (Keras model)")
    print("     - baseline_fp32_final.h5 (Final checkpoint)")
    print("     - model_fp32.tflite (FP32 TFLite)")
    print("     - model_int8.tflite (INT8 TFLite - ready for Edge TPU)")
    print(f"\n   Results: {RESULTS_DIR}")
    print("     - training_results.txt (Summary)")
    print("     - training_summary.json (JSON format)")
    print("     - classification_report.txt (Per-class metrics)")
    print(f"\n   Charts: {RESULTS_DIR / 'charts'}")
    print("     - training_history.png")
    print("     - confusion_matrix.png")
    print("     - per_class_accuracy.png")
    print("     - class_distribution.png")
    print("\n✅ All files ready for presentation and deployment!")
    print("="*70)
else:
    print("\n⚠️  Dataset not loaded - please upload FER2013 dataset to continue")

# %%

