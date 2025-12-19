# Data Directory

## 📊 Dataset Location

The training data for this project is located in the source code directory for easier access during development and deployment:

**Primary Data Location:**
- **`../src/emote_detector/pose_data/`** - Collected pose training data
  - `pose_features_*.npy` - Extracted feature vectors (18 geometric features)
  - `pose_labels_*.npy` - Corresponding pose labels (0-3)
  - `pose_metadata_*.json` - Collection metadata (timestamps, sample counts)

## 🎯 Data Description

### Training Dataset

**Source:** Custom collected data using the data collector tool

**Collection Method:**
```bash
cd ../src/emote_detector
python data_collector.py
```

**Dataset Statistics:**
- **Total Samples:** ~300-400 samples
- **Classes:** 4 pose types
  - 0: Laughing (hands raised, celebratory)
  - 1: Yawning (hands near mouth)
  - 2: Crying (hands covering face)
  - 3: Taunting (arms crossed)
- **Samples per Class:** 50-100 each
- **Train/Test Split:** 80/20
- **Features:** 18 geometric features extracted from MediaPipe pose landmarks

### Features Extracted

From MediaPipe's 33 pose landmarks, we extract 18 geometric features:

**Distance Features:**
- Shoulder width
- Hip width
- Torso height
- Arm lengths (left, right)
- Hand-to-shoulder distances

**Angle Features:**
- Elbow angles (left, right)
- Shoulder angles
- Hip angles

**Position Features:**
- Hand heights relative to shoulders
- Hand positions relative to face
- Body center alignment

### Data Format

**Feature Array (pose_features_*.npy):**
```python
# Shape: (num_samples, 18)
# Each row is one pose sample with 18 extracted features
# Normalized to [0, 1] range
```

**Label Array (pose_labels_*.npy):**
```python
# Shape: (num_samples,)
# Integer labels: 0, 1, 2, 3
```

**Metadata (pose_metadata_*.json):**
```json
{
  "timestamp": "2025-12-17T12:34:56",
  "num_samples": 320,
  "class_distribution": {
    "0": 80,
    "1": 80,
    "2": 80,
    "3": 80
  },
  "feature_dim": 18
}
```

## 🔄 Data Collection Workflow

1. **Collect Raw Data**
   ```bash
   cd ../src/emote_detector
   python data_collector.py
   ```
   - Position yourself in front of webcam
   - Press `0-3` to select pose to record
   - Press `SPACE` or `a` to capture samples
   - Aim for 50-100 samples per pose
   - Press `t` to auto-train or `s` to save

2. **Train Model**
   ```bash
   python train_model.py
   ```
   - Loads data from `pose_data/`
   - Trains Random Forest classifier
   - Generates evaluation charts
   - Saves model to `pose_classifier_model.pkl`

3. **Deploy**
   - Copy trained model to Raspberry Pi
   - Run inference with `python main.py`

## 📁 Additional Data Locations

**Emote Assets:**
- `../src/emote_detector/emotes/images/` - Emote overlay images (PNG)
- `../src/emote_detector/emotes/sounds/` - Sound effects (MP3)

**Results Data:**
- `../results/macbook_results/` - MacBook performance metrics
- `../results/rpi_results/` - Raspberry Pi 4 performance metrics
- `../results/comparison_charts/` - Generated comparison visualizations

## 🔒 Data Privacy

**Note:** This project uses custom-collected pose data. Each user collects their own training data, so:
- No pre-existing dataset is required
- Data is specific to each user's poses and environment
- Model is trained on user-specific data for best accuracy
- No personal identifying information is collected (only pose landmarks)

## 📝 Dataset Citation

If using a pre-collected dataset, cite it here. For this project:

**Custom Dataset:**
- **Name:** Custom Pose-Based Emote Dataset
- **Collection Date:** December 2025
- **Collection Tool:** Custom data collector (`data_collector.py`)
- **Collection Environment:** Indoor, standard lighting, USB webcam
- **Annotator:** Self-labeled during collection
- **License:** Project use only (educational)

## 🚀 Quick Start

To collect your own data:

```bash
# 1. Set up environment
cd ../src/emote_detector
source ../../venv/bin/activate

# 2. Collect data
python data_collector.py
# Follow on-screen instructions
# Press 't' when done to auto-train

# 3. Verify training results
ls results/charts/  # Check generated evaluation charts
cat results/training_results.txt  # View accuracy metrics
```

## 📊 Expected Data Quality

For good model performance, collect:
- ✅ **Balanced classes:** Similar number of samples per pose (~50-100 each)
- ✅ **Variety:** Different angles, distances, lighting conditions
- ✅ **Clear poses:** Ensure pose is clearly visible to webcam
- ✅ **Consistent timing:** Hold each pose for 1-2 seconds during capture
- ✅ **Full body visible:** Ensure full body (or at least upper body) is in frame

With good quality data:
- Expected accuracy: **85-90%**
- Inference time: **<1ms on MacBook, ~12ms on RPi4**
- Real-time performance: **10-15 FPS**

## 🔗 Related Documentation

- **[Data Collection Tool](../src/emote_detector/data_collector.py)** - Source code
- **[Feature Extraction](../src/emote_detector/pose_classifier.py)** - Feature engineering implementation
- **[Training Pipeline](../src/emote_detector/train_model.py)** - Model training script
- **[Main README](../README.md)** - Project overview
- **[Deployment Guide](../docs/DEPLOYMENT_GUIDE.md)** - Full deployment instructions

