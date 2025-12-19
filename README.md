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

## 📚 Documentation

- **[Quick Start Guide](#-quick-start-development-machine)** - Get started on your development machine
- **[How To Use with Hardware Platform](#-how-to-use-the-software-with-hardware-platform)** - Complete deployment guide
- **[Project Proposal](docs/Project_Proposal.md)** - Original project proposal and timeline
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Detailed setup for MacBook and RPi4
- **[Presentation Materials](presentations/)** - Slide deck and speaker notes
- **[Platform Comparison](results/PLATFORM_COMPARISON.md)** - MacBook vs RPi4 analysis

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
│   ├── Project_Proposal.md       # Project proposal
│   └── DEPLOYMENT_GUIDE.md       # Complete deployment guide
├── presentations/                # Presentation slides and materials
│   ├── PRESENTATION_SLIDES.md    # Full slide deck with speaker notes
│   └── PRESENTATION_SUMMARY.md   # Quick reference guide
├── results/                      # All results and metrics
│   ├── comparison_charts/        # MacBook vs RPi4 comparison graphs
│   ├── macbook_results/          # MacBook performance metrics
│   └── rpi_results/              # RPi4 performance metrics & training charts
├── report/                       # Final report templates
├── requirements.txt              # Python dependencies
└── generate_comparison_charts.py # Generate platform comparison charts
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

---

## 📘 How To Use the Software with the Hardware Platform

This section provides comprehensive instructions for deploying and running the pose-based emote detection system on the Raspberry Pi 4 hardware platform.

### Prerequisites

**Hardware Required:**
- Raspberry Pi 4 Model B (4GB RAM recommended)
- USB webcam (or Raspberry Pi Camera Module)
- microSD card (32GB+ recommended)
- Power supply (5V/3A USB-C)
- (Optional) Monitor, keyboard, mouse for initial setup
- (Optional) Mac/PC for remote SSH access

**Software Required:**
- Raspberry Pi OS (64-bit recommended)
- Python 3.11.x (MediaPipe requires Python < 3.12)
- Internet connection for initial setup

### Step-by-Step Deployment Guide

#### Phase 1: Prepare Development Machine (Mac/PC)

**1. Clone Repository**
```bash
git clone <repository-url>
cd ai-hardware-project-proposal-visionmasters
```

**2. Set Up Python Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Collect Training Data**
```bash
cd src/emote_detector
python data_collector.py
```

- Press `0-3` to select a pose (Laughing, Yawning, Crying, Taunting)
- Press `SPACE` or `a` to capture samples (aim for 50-100 per pose)
- Press `t` to train the model automatically
- Or press `s` to save and train manually later

**4. Train Model (if not auto-trained)**
```bash
python train_model.py
```

This generates:
- `pose_classifier_model.pkl` - The trained Random Forest model
- `results/charts/` - Evaluation charts (confusion matrix, accuracy, feature importance)
- `results/training_results.txt` - Classification report

**5. Test Locally**
```bash
python main.py --fast --metrics
```

Verify the system works before deploying to RPi.

#### Phase 2: Set Up Raspberry Pi 4

**1. Initial RPi Setup**
```bash
# Connect monitor, keyboard to RPi, boot up
# Enable SSH: sudo raspi-config → Interface Options → SSH → Enable
# Find IP address
hostname -I
```

**2. Connect via SSH from Development Machine**
```bash
# From your Mac/PC
ssh pi@<RPI_IP_ADDRESS>
# Default password: raspberry (change this!)
```

**3. Install Python 3.11 on RPi**

MediaPipe doesn't support Python 3.12+ yet, so we need 3.11:

```bash
# On RPi terminal
curl https://pyenv.run | bash

# Add to ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.11.9
pyenv install 3.11.9
pyenv local 3.11.9
```

**4. Install System Dependencies (on RPi)**
```bash
sudo apt update
sudo apt install -y libcap-dev build-essential libssl-dev zlib1g-dev \
                    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
                    libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
                    liblzma-dev python3-openssl git
```

**5. Create Virtual Environment (on RPi)**
```bash
cd ~
python -m venv venv
source venv/bin/activate
```

**6. Install Python Dependencies (on RPi)**
```bash
pip install --upgrade pip
pip install numpy opencv-python mediapipe scikit-learn pygame matplotlib seaborn psutil
```

#### Phase 3: Deploy Code to Raspberry Pi

**From your development machine:**

```bash
# Copy entire emote_detector directory to RPi
cd /path/to/ai-hardware-project-proposal-visionmasters
scp -r src/emote_detector pi@<RPI_IP>:~/emote_detector/
```

This transfers:
- All Python scripts
- Trained model (`pose_classifier_model.pkl`)
- Training data (`pose_data/`)
- Emote images and sounds

#### Phase 4: Run on Raspberry Pi

**Option A: With Display (HDMI Monitor)**

Connect a monitor directly to RPi via HDMI:

```bash
ssh pi@<RPI_IP>
cd ~/emote_detector
source venv/bin/activate
python main.py --fast
```

**Option B: Remote Display (X11 Forwarding)**

Run with display forwarding over SSH:

```bash
# 1. Install XQuartz on Mac (or Xming on Windows)
# 2. Connect with X11 forwarding
ssh -Y pi@<RPI_IP>

# 3. On RPi, run the demo
cd ~/emote_detector
source venv/bin/activate
python main.py --fast
```

**Note:** X11 forwarding is slower (~2-4 FPS) due to network overhead.

**Option C: Headless Mode (Performance Testing)**

For pure performance metrics without display overhead:

```bash
ssh pi@<RPI_IP>
cd ~/emote_detector
source venv/bin/activate
python main.py --ultra-fast --headless --metrics
```

#### Phase 5: Collect Performance Metrics

**On Raspberry Pi:**
```bash
# Run with metrics collection (headless for best performance)
python main.py --ultra-fast --headless --metrics

# Metrics saved to results/metrics/
ls results/metrics/
```

**Copy Results Back to Development Machine:**
```bash
# From your Mac/PC terminal
mkdir -p results/rpi_results
scp -r pi@<RPI_IP>:~/emote_detector/results/metrics results/rpi_results/
```

**Generate Comparison Charts:**
```bash
# On development machine
python generate_comparison_charts.py
```

Charts saved to `results/comparison_charts/`:
- `summary_dashboard.png` - Comprehensive comparison
- `inference_comparison.png` - Latency breakdown
- `cost_performance.png` - FPS per dollar analysis
- `fps_comparison.png` - FPS comparison
- `speedup_comparison.png` - Relative performance

### Command Line Options Reference

| Flag | Description | When to Use |
|------|-------------|-------------|
| `--fast` | 320x240 processing, skip frames | Recommended for RPi with display |
| `--ultra-fast` | 160x120 processing, aggressive skipping | Best for headless RPi |
| `--metrics` | Enable performance profiling | For collecting benchmark data |
| `--headless` | No display (auto-quit after 60s) | Pure performance testing |
| `--complexity 0` | Lightest MediaPipe model | Already default |
| `--resolution low` | Process at 320x240 | Included in `--fast` |
| `--skip N` | Process every Nth frame | Adjust based on needs |

### Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'cv2'`
- **Solution:** Activate virtual environment: `source venv/bin/activate`

**Problem:** Cannot open camera
- **Solution:** Check USB webcam is connected: `ls /dev/video*`
- Try camera index 0, 1, or 2

**Problem:** X11 display errors
- **Solution:** Switch RPi to X11 (not Wayland): `sudo raspi-config` → Advanced → Compositor → X11

**Problem:** Low FPS (< 5)
- **Solution:** Use `--fast` or `--ultra-fast` mode
- Ensure model was retrained on RPi: `python train_model.py`
- Check CPU temperature: `vcgencmd measure_temp` (should be < 80°C)

**Problem:** Model compatibility errors
- **Solution:** Retrain model on the target platform to match scikit-learn versions

### Performance Optimization Tips

1. **Use headless mode** for benchmarking (no display overhead)
2. **Lower resolution** to 160x120 for fastest performance
3. **Retrain model on RPi** to ensure library compatibility
4. **Close background apps** to free CPU resources
5. **Consider USB webcam** (easier than Pi Camera Module setup)

### Complete Workflow Summary

```bash
# === ON DEVELOPMENT MACHINE ===
# 1. Collect data and train
cd src/emote_detector
python data_collector.py  # Collect samples, press 't' to train
python main.py --fast --metrics  # Test locally

# 2. Deploy to RPi
scp -r . pi@<RPI_IP>:~/emote_detector/

# === ON RASPBERRY PI ===
# 3. Run and collect metrics
ssh pi@<RPI_IP>
cd ~/emote_detector
source venv/bin/activate
python main.py --ultra-fast --headless --metrics

# === BACK ON DEVELOPMENT MACHINE ===
# 4. Retrieve results and generate comparison
scp -r pi@<RPI_IP>:~/emote_detector/results/metrics results/rpi_results/
python generate_comparison_charts.py
```

For more detailed deployment instructions, see **[docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)**.

---

## 🍓 Raspberry Pi 4 Quick Reference

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

## 🏆 Results and Achievements

### Project Goals vs. Actual Results

| Goal | Target | Achieved | Status |
|------|--------|----------|--------|
| Real-time FPS on RPi4 | ≥ 10 FPS | 9.7 FPS | ✅ Near target |
| Total Inference Latency | < 250ms | 201ms | ✅ Excellent |
| Model Accuracy | > 85% | 85-88% | ✅ Target met |
| Edge Deployment | Working system | Success | ✅ Complete |
| Full ML Pipeline | End-to-end | Complete | ✅ Done |

### Performance Comparison: MacBook vs Raspberry Pi 4

#### Inference Latency Breakdown

| Component | MacBook M1 | RPi4 (Headless) | Slowdown Factor |
|-----------|------------|-----------------|-----------------|
| **MediaPipe (Pose Detection)** | 23.5 ms | 189 ms | 8.0x |
| **Random Forest Classifier** | 1.3 ms | 12 ms | 9.1x |
| **Total Frame Time** | 53 ms | 102 ms | 1.9x |
| **Overall FPS** | 14.7 | 9.7 | 1.5x |

**Key Insight:** Despite 8-9x slower individual components, the overall system is only 1.5x slower due to smart pipeline optimization (frame skipping, resolution reduction).

#### Cost-Performance Analysis

| Platform | Cost | FPS | FPS per $100 | Value Ratio |
|----------|------|-----|--------------|-------------|
| MacBook M1 | $1,500 | 14.7 | 0.98 | Baseline |
| **Raspberry Pi 4** | **$50** | **9.7** | **19.4** | **🏆 20x better** |

**Bottom Line:** The RPi4 delivers **20 times better value per dollar** for this edge AI application.

#### System Resource Utilization

| Metric | MacBook | RPi4 (Headless) |
|--------|---------|-----------------|
| CPU Usage | ~15% | ~50-70% |
| Memory Usage | Low | Moderate |
| Power Consumption | ~30W | ~5W (6x more efficient) |
| Temperature | Cool | 52.6°C (well under 80°C limit) |

### Model Training Results

**Dataset:** 300-400 samples across 4 pose classes
- Laughing, Yawning, Crying, Taunting
- 80/20 train-test split

**Random Forest Configuration:**
- 10 trees (reduced from 100 for speed)
- Max depth: 4 (reduced from 10)
- Single-core inference (n_jobs=1)
- Model size: ~25 KB

**Performance Metrics:**
- Overall Accuracy: **85-88%**
- Training Time: **< 5 seconds**
- Inference Time: **12ms on RPi4, 1.3ms on MacBook**
- Well-balanced across all 4 classes

**Feature Engineering:**
- 18 geometric features extracted from 33 pose landmarks
- Most important: hand height, arm angles, hand-to-face distances

### Optimization Journey: 20x Speedup

Through systematic profiling and optimization, we achieved a **20x speedup** in the Random Forest classifier:

| Iteration | Configuration | RPi4 Latency | Speedup |
|-----------|---------------|--------------|---------|
| 1. Initial | 100 trees, n_jobs=-1 | 239 ms | Baseline |
| 2. Threading fix | n_jobs=1 | 160 ms | 1.5x |
| 3. Model reduction | 20 trees | 30 ms | 8x |
| 4. **Final** | **10 trees, depth=4** | **12 ms** | **20x** ✅ |

**Key Discovery:** Multi-threading (n_jobs=-1) actually **hurt performance** on the RPi's 4-core ARM CPU due to context switching overhead. Single-core inference proved optimal for lightweight models.

### Deployment Environment Impact

We measured performance across different deployment scenarios:

| Mode | Display | FPS | Use Case |
|------|---------|-----|----------|
| **Headless** | None | 9.7 | Performance benchmarking |
| **HDMI Direct** | Local monitor | ~7-8 (est) | Production deployment |
| **X11 over WiFi** | Remote SSH | 2.5 | Development/debugging |

**Lesson:** Display overhead matters! X11 forwarding adds ~4x latency due to network transmission.

### Technical Achievements

1. ✅ **End-to-end ML pipeline** from data collection to deployment
2. ✅ **Custom training on commodity hardware** (no cloud GPUs needed)
3. ✅ **Real-time inference on $50 edge device** (9.7 FPS)
4. ✅ **Comprehensive performance profiling** (FPS, latency, CPU, temp)
5. ✅ **Platform comparison** with actionable insights
6. ✅ **20x model optimization** through systematic profiling
7. ✅ **Cost-effective deployment** (20x better value than laptop)

### Comparison Charts

All performance comparison visualizations are available in `results/comparison_charts/`:

- **Summary Dashboard** - Comprehensive overview of all metrics
- **Inference Latency** - MediaPipe vs Random Forest breakdown
- **FPS Comparison** - Overall throughput comparison
- **Cost-Performance** - Value per dollar analysis
- **Training Evaluation** - Confusion matrix, accuracy, feature importance

See [results/PLATFORM_COMPARISON.md](results/PLATFORM_COMPARISON.md) for detailed analysis.

---

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
