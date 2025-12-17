# Real-Time Pose-Based Emote Detection on Raspberry Pi 4

**Team:** VisionMasters  
**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Platform:** Raspberry Pi 4 Model B (CPU-only inference)

## 🎯 Project Overview

This project implements real-time pose-based gesture recognition that maps body poses to Clash Royale emotes. The system demonstrates edge AI deployment on resource-constrained hardware, measuring performance metrics relevant to AI hardware design.

### Key Features
- **MediaPipe Holistic**: Pre-trained pose detection (33 body landmarks)
- **Feature Engineering**: 45 geometric features extracted from pose landmarks
- **Random Forest Classifier**: Fast, interpretable ML model (<1ms inference)
- **Custom Data Collection**: Train on your own poses
- **Performance Metrics**: Detailed latency, FPS, CPU/memory, and temperature monitoring
- **Real-time Performance**: 10-15 FPS on Raspberry Pi 4 with optimizations

## 📁 Project Structure

```
.
├── src/emote_detector/           # Main application
│   ├── main.py                   # Demo application
│   ├── data_collector.py         # Collect training data
│   ├── train_model.py            # Train & generate evaluation charts
│   ├── holistic_detector.py      # MediaPipe wrapper
│   ├── pose_classifier.py        # Random Forest classifier
│   ├── performance_metrics.py    # Performance profiling
│   ├── pose_classifier_model.pkl # Trained model
│   ├── pose_data/                # Collected training data
│   ├── emotes/                   # Emote images and sounds
│   └── results/                  # Training results and charts
├── docs/
│   └── Project_Proposal.md       # Project proposal
├── presentations/                # Presentation slides
├── report/                       # Final report
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start (Development Machine)

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect Training Data

```bash
cd src/emote_detector
python data_collector.py
```

**Controls:**
- `0-4`: Select pose to record
- `SPACE`: Capture single sample
- `a`: Auto-collect samples
- `s`: Save data
- `t`: Train model
- `q`: Quit

### 3. Train Model & Generate Charts

```bash
python train_model.py
```

This generates evaluation charts in `results/charts/`:
- Confusion matrix
- Feature importance
- Per-class accuracy
- Data distribution

### 4. Run Demo

```bash
python main.py
```

## 🍓 Raspberry Pi 4 Deployment

### Setup on RPi

```bash
# Install pyenv and Python 3.11 (MediaPipe requires Python <3.12)
curl https://pyenv.run | bash
# Add pyenv to ~/.bashrc, then:
pyenv install 3.11.9
pyenv local 3.11.9

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy opencv-python-headless mediapipe scikit-learn pygame matplotlib seaborn psutil
```

### Copy files to RPi

```bash
scp -r src/emote_detector pi@<RPI_IP>:~/emote_detector/
```

### Run on RPi (via SSH with X11 forwarding)

```bash
ssh -Y pi@<RPI_IP>
cd ~/emote_detector
source venv/bin/activate

# Fast mode (optimized for RPi)
python main.py --fast

# With performance metrics collection
python main.py --fast --metrics
```

### Command Line Options

| Flag | Description |
|------|-------------|
| `--fast` | Optimized mode: 320x240 processing, 640x480 display, skip frames |
| `--metrics` | Enable performance metrics collection |
| `--complexity 0` | Use lightest MediaPipe model |
| `--resolution low` | Process at 320x240 |
| `--skip 2` | Process every 2nd frame |
| `--scale 2` | 2x display scaling |

## 🎮 Supported Poses (4 Classes)

| ID | Pose | Gesture | Emote |
|----|------|---------|-------|
| 0 | **Laughing** | Hands raised, celebratory | 😂 Laughing King |
| 1 | **Yawning** | Hands near mouth | 🥱 Yawning |
| 2 | **Crying** | Hands covering face | 😢 Crying |
| 3 | **Taunting** | Arms crossed | 😏 Taunting |

## 📊 Performance Metrics

### Collected Metrics

| Category | Measurements |
|----------|--------------|
| **Timing** | Frame time, MediaPipe inference, Classifier inference |
| **Throughput** | FPS (mean, min, max) |
| **Latency** | P50, P95, P99 percentiles |
| **System** | CPU %, Memory %, Temperature °C |

### Expected Performance (RPi 4)

| Metric | Fast Mode | Standard Mode |
|--------|-----------|---------------|
| **FPS** | 10-15 | 5-8 |
| **Frame Latency** | 70-100ms | 125-200ms |
| **MediaPipe** | 50-80ms | 100-150ms |
| **Classifier** | <1ms | <1ms |
| **CPU Usage** | 70-90% | 80-100% |

### Output Files (with `--metrics`)

```
results/metrics/
├── metrics_TIMESTAMP.json    # Raw data
├── metrics_TIMESTAMP.csv     # For plotting
└── metrics_TIMESTAMP.md      # Report for presentation
```

## 🏗️ System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────┐
│ USB Webcam  │────▶│  MediaPipe       │────▶│  Feature        │────▶│  Random    │
│ (320x240)   │     │  Holistic        │     │  Extraction     │     │  Forest    │
│             │     │  (TFLite CPU)    │     │  (45 features)  │     │  Classifier│
└─────────────┘     └──────────────────┘     └─────────────────┘     └────────────┘
      │                    │                        │                      │
      │              33 Pose                   Geometric              Pose Label +
      │              Landmarks                 Features               Confidence
      ▼                    ▼                        ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Display (640x480) + Emote Overlay + Sound                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔬 AI Hardware Insights

This project demonstrates key AI hardware concepts:

1. **Model Selection Trade-offs**: MediaPipe (TFLite) enables real-time pose detection on CPU
2. **Feature Engineering vs. Deep Learning**: Geometric features + Random Forest is more efficient than end-to-end CNN on CPU
3. **Pipeline Optimization**: Resolution reduction, frame skipping, and display scaling improve throughput
4. **Resource Profiling**: Metrics collection reveals bottlenecks (MediaPipe dominates inference time)
5. **Thermal Management**: Sustained operation requires performance trade-offs to avoid throttling

## 👥 Team Members

- **Allen Chen** (wmm7wr@virginia.edu) - Hardware Integration
- **Marvin Rivera** (tkk9wg@virginia.edu) - Team Lead, Documentation
- **Sami Kang** (ajp3cx@virginia.edu) - Model Training, Inference

## 📚 References

- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
- [Raspberry Pi 4 Documentation](https://www.raspberrypi.com/documentation/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)

## 📝 License

This project is for educational purposes as part of ECE 4332/6332 at the University of Virginia.
