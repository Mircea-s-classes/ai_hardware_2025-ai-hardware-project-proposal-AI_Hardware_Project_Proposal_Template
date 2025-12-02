# Training Notebooks

This directory contains Jupyter notebooks for training the emotion recognition model.

## 📓 Available Notebooks

### `train_emotion_recognition.py`

Complete training pipeline in Python script format with `# %%` cell markers for Jupytext conversion.

**Includes:**
- Dataset loading and verification
- Data augmentation and visualization
- Model training (MobileNetV2)
- Evaluation with confusion matrix
- INT8 quantization for Edge TPU
- Results visualization

## 🚀 Quick Start with Google Colab

### Option 1: Convert to Jupyter Notebook (Recommended)

1. **Install Jupytext** (if not already installed):
```bash
pip install jupytext
```

2. **Convert to .ipynb**:
```bash
cd notebooks
jupytext --to ipynb train_emotion_recognition.py
```

This creates `train_emotion_recognition.ipynb`.

3. **Upload to Google Colab**:
   - Go to https://colab.research.google.com/
   - File > Upload notebook
   - Select `train_emotion_recognition.ipynb`
   - Change runtime: Runtime > Change runtime type > GPU

4. **Upload Dataset** (in Colab):
   - Follow instructions in the notebook (Section 2)
   - Use Kaggle API or upload from Google Drive

5. **Run All Cells**:
   - Runtime > Run all

### Option 2: Use Python Script Directly

You can also run the script directly (though notebook format is better for visualization):

```bash
python train_emotion_recognition.py
```

## 📦 What the Notebook Does

### 1. Environment Setup
- Detects if running on Colab or locally
- Installs required dependencies
- Configures GPU if available

### 2. Dataset Handling
- Provides two methods to load FER2013:
  - **Kaggle API** (recommended for Colab)
  - **Google Drive mount** (if dataset already uploaded)
- Verifies dataset structure
- Analyzes class distribution

### 3. Data Preparation
- Creates train/validation/test generators
- Applies data augmentation
- Visualizes samples and augmentations

### 4. Model Training
- Builds MobileNetV2-based model
- Trains with callbacks (early stopping, LR scheduling)
- Saves best model checkpoint
- Generates training curves

### 5. Evaluation
- Tests on FER2013 test set
- Creates confusion matrix
- Calculates per-class accuracy
- Generates classification report

### 6. Quantization
- Converts to FP32 TFLite
- Quantizes to INT8 for Edge TPU
- Compares model sizes
- Shows compression ratio

### 7. Results Export
- Saves all charts and metrics
- Creates summary JSON
- Downloads results (on Colab)

## 📊 Expected Outputs

After running the notebook, you'll have:

### Models (in `models/` or `/content/models/`)
```
baseline_fp32_best.h5      # Best Keras model (~14 MB)
baseline_fp32_final.h5     # Final checkpoint
model_fp32.tflite          # FP32 TFLite (~14 MB)
model_int8.tflite          # INT8 TFLite (~3.5 MB) ✅ For Edge TPU
```

### Results (in `results/` or `/content/results/`)
```
training_results.txt       # Human-readable summary
training_summary.json      # Machine-readable results
classification_report.txt  # Per-class metrics

charts/
├── training_history.png        # Training curves
├── confusion_matrix.png        # Confusion matrix (counts & %)
├── per_class_accuracy.png      # Bar chart per emotion
├── class_distribution.png      # Dataset distribution
└── dataset_samples.png         # Sample images
```

## 🎮 Google Colab Tips

### Enable GPU
```
Runtime > Change runtime type > Hardware accelerator > GPU
```

### Check GPU
```python
!nvidia-smi
```

### Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Download Files
```python
from google.colab import files
files.download('models/model_int8.tflite')
```

### Create Results Archive
```bash
!zip -r results.zip models/ results/
!zip -r charts.zip results/charts/
```

## ⏱️ Training Time

**With GPU (T4 on Colab):**
- Training: 2-3 hours
- Evaluation: 5-10 minutes
- Total: ~3 hours

**With CPU:**
- Training: 8-12 hours
- Not recommended unless no GPU available

## 💾 Storage Requirements

### Google Colab
- **Dataset**: ~300 MB (FER2013)
- **Models**: ~30 MB (all formats)
- **Results**: ~5 MB (charts + reports)
- **Total**: ~350 MB

Colab gives you ~100 GB, so plenty of space!

### Local
Same requirements as above.

## 🐛 Troubleshooting

### "Dataset not found"
- Make sure you uploaded FER2013 to correct location
- Check expected structure in notebook Section 2
- Verify paths: `/content/fer2013/train/` and `/content/fer2013/test/`

### Out of Memory
- Reduce batch size in CONFIG: `'batch_size': 16` or `8`
- Clear previous outputs: Edit > Clear all outputs
- Restart runtime: Runtime > Restart runtime

### Training Too Slow
- Enable GPU: Runtime > Change runtime type > GPU
- Reduce epochs: `'epochs': 30`
- Check GPU is detected: Cell should show "GPU available"

### Can't Download Files
- Use Google Drive instead:
  ```python
  !cp -r models/ /content/drive/MyDrive/fer2013_models/
  ```

## 📚 Additional Resources

### Jupytext Documentation
- https://jupytext.readthedocs.io/

### Google Colab Guides
- https://colab.research.google.com/notebooks/intro.ipynb

### FER2013 Dataset
- https://www.kaggle.com/datasets/msambare/fer2013

## 🎯 Next Steps After Training

1. **Download Models**:
   - `model_int8.tflite` (most important for Edge TPU)
   - `baseline_fp32_best.h5` (for further training/analysis)

2. **Download Results**:
   - All charts from `results/charts/`
   - `training_results.txt` and `classification_report.txt`

3. **Prepare Presentation**:
   - Use charts in your slides
   - Report test accuracy and model compression
   - Show confusion matrix

4. **Deploy to Hardware** (Week 5+):
   - Transfer `model_int8.tflite` to Raspberry Pi
   - Compile with Edge TPU Compiler
   - Run real-time demo

## 📝 Notes

- The notebook is self-contained and fully documented
- Each section has markdown explanations
- All visualizations are saved automatically
- Progress is printed throughout training
- Safe to interrupt and resume (checkpoints saved)

**Happy Training! 🚀**

