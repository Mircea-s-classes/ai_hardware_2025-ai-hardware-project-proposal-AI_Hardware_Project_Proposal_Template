# Deployment Guide - Emote Detection System

This guide covers the complete setup and deployment process for both development (MacBook) and edge deployment (Raspberry Pi 4).

---

## 📋 Table of Contents

- [MacBook Setup](#-macbook-setup-development)
- [Raspberry Pi 4 Setup](#-raspberry-pi-4-setup-edge-deployment)
- [Data Collection](#-data-collection-both-platforms)
- [Model Training](#-model-training-both-platforms)
- [Running the Demo](#-running-the-demo)
- [Performance Metrics](#-collecting-performance-metrics)
- [Troubleshooting](#-troubleshooting)

---

## 💻 MacBook Setup (Development)

### Prerequisites

- macOS (tested on macOS Sonoma)
- Python 3.9 - 3.11 (MediaPipe requirement)
- Webcam
- Terminal

### Step 1: Clone the Repository

```bash
cd ~/Workspace
git clone https://github.com/Mircea-s-classes/ai-hardware-project-proposal-visionmasters.git
cd ai-hardware-project-proposal-visionmasters
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies installed:**
- opencv-python
- mediapipe
- numpy
- scikit-learn
- pygame
- matplotlib
- seaborn
- psutil

### Step 4: Verify Installation

```bash
cd src/emote_detector
python -c "import cv2, mediapipe, sklearn; print('✅ All dependencies installed')"
```

---

## 🍓 Raspberry Pi 4 Setup (Edge Deployment)

### Prerequisites

- Raspberry Pi 4 Model B (4GB RAM recommended)
- Raspberry Pi OS (64-bit)
- USB Webcam (e.g., Logitech Brio 100)
- SSH access or monitor/keyboard

### Step 1: Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Install System Dependencies

```bash
sudo apt install -y python3-dev python3-pip libcap-dev
```

### Step 3: Install pyenv (for Python 3.11)

MediaPipe requires Python <3.12. Install pyenv to manage Python versions:

```bash
curl https://pyenv.run | bash
```

Add to `~/.bashrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reload shell:

```bash
source ~/.bashrc
```

### Step 4: Install Python 3.11

```bash
pyenv install 3.11.9
pyenv local 3.11.9
python --version  # Should show Python 3.11.9
```

### Step 5: Create Virtual Environment

```bash
mkdir ~/emote_detector
cd ~/emote_detector
python -m venv venv
source venv/bin/activate
```

### Step 6: Install Python Dependencies

```bash
pip install --upgrade pip
pip install numpy opencv-python-headless mediapipe scikit-learn pygame matplotlib seaborn psutil
```

**Note:** Use `opencv-python-headless` on RPi for better performance.

### Step 7: Copy Project Files from MacBook

On your **MacBook**:

```bash
# Replace <RPI_IP> with your Raspberry Pi's IP address
# Get IP with: ssh pi@<RPI_IP> "hostname -I"

cd ~/Workspace/ai-hardware-project-proposal-visionmasters/src/emote_detector
scp -r *.py emotes/ pose_data/ pose_classifier_model.pkl pi@<RPI_IP>:~/emote_detector/
```

### Step 8: Configure X11 (for Display)

#### Option A: VNC (Recommended for Remote Demo)

```bash
sudo raspi-config
# Navigate to: Interface Options → VNC → Enable
```

Download VNC Viewer on your Mac and connect to `<RPI_IP>`.

#### Option B: SSH with X11 Forwarding

On your **Mac**, ensure XQuartz is installed:

```bash
brew install --cask xquartz
```

Then SSH with X11:

```bash
ssh -Y pi@<RPI_IP>
```

---

## 📸 Data Collection (Both Platforms)

### Step 1: Run Data Collector

```bash
cd src/emote_detector  # or ~/emote_detector on RPi
source venv/bin/activate  # if not already activated
python data_collector.py
```

### Step 2: Collect Samples

**Pose Classes:**
- **0:** Laughing (hands raised)
- **1:** Yawning (hands near mouth)
- **2:** Crying (hands covering face)
- **3:** Taunting (arms crossed)

**Controls:**
- Press `0-3` to select pose
- Press `SPACE` to capture a sample
- Press `a` to auto-collect 30 samples
- Press `s` to save data
- Press `t` to train model
- Press `q` to quit

**Recommendation:** Collect 50-100 samples per pose for best results.

### Step 3: Save Data

Press `s` to save your collected data to `pose_data/`.

---

## 🧠 Model Training (Both Platforms)

### Option 1: Train During Collection

Press `t` in the data collector to train immediately.

### Option 2: Train Separately with Evaluation

```bash
python train_model.py
```

**Output:**
```
results/
├── charts/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── per_class_accuracy.png
│   └── data_distribution.png
├── classification_report.txt
└── training_results.txt
```

---

## 🎮 Running the Demo

### MacBook

#### Standard Mode

```bash
cd src/emote_detector
source ../../venv/bin/activate
python main.py
```

#### Fast Mode (for testing RPi-like performance)

```bash
python main.py --fast
```

#### With Performance Metrics

```bash
python main.py --fast --metrics
```

### Raspberry Pi 4

#### Via VNC

1. Open VNC Viewer on Mac
2. Connect to `<RPI_IP>`
3. Open terminal on RPi desktop
4. Run:

```bash
cd ~/emote_detector
source venv/bin/activate
python main.py --fast
```

#### Via SSH with X11

On **Mac**:

```bash
ssh -Y pi@<RPI_IP>
cd ~/emote_detector
source venv/bin/activate
python main.py --fast
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--fast` | Enable all optimizations (320x240, complexity=0, skip=2, scale=2x) | `python main.py --fast` |
| `--metrics` | Collect performance metrics | `python main.py --metrics` |
| `--complexity 0` | Use lightest MediaPipe model (0=fast, 1=balanced, 2=accurate) | `python main.py --complexity 0` |
| `--resolution low` | Process at 320x240 (low/medium/high) | `python main.py --resolution low` |
| `--skip 2` | Process every 2nd frame | `python main.py --skip 2` |
| `--scale 2` | Display at 2x size | `python main.py --scale 2` |
| `--camera 1` | Use camera index 1 | `python main.py --camera 1` |

### Runtime Controls

- **`q`** - Quit
- **`s`** - Save screenshot
- **`m`** - Print live metrics (if `--metrics` enabled)

---

## 📊 Collecting Performance Metrics

### MacBook Metrics

```bash
cd src/emote_detector
source ../../venv/bin/activate
python main.py --fast --metrics
```

Run for 30-60 seconds, perform various poses, then press `q`.

### Raspberry Pi 4 Metrics

```bash
cd ~/emote_detector
source venv/bin/activate
python main.py --fast --metrics
```

### Output Files

```
results/metrics/
├── metrics_YYYYMMDD_HHMMSS.json    # Raw data (for analysis)
├── metrics_YYYYMMDD_HHMMSS.csv     # For plotting in Excel/Python
└── metrics_YYYYMMDD_HHMMSS.md      # Formatted report
```

### Metrics Collected

| Metric | Description |
|--------|-------------|
| **FPS** | Frames per second (mean, min, max) |
| **Frame Time** | Total processing time per frame (P50, P95, P99) |
| **MediaPipe Inference** | Pose detection latency |
| **Classifier Inference** | Random Forest classification time |
| **CPU Usage** | Processor utilization % |
| **Memory Usage** | RAM utilization % |
| **Temperature** | CPU temperature (RPi only) |
| **Confidence** | Classification confidence scores |

---

## 🔧 Troubleshooting

### MacBook Issues

#### Camera Not Found

```bash
# List available cameras
ls /dev/video*

# Try different camera indices
python main.py --camera 1
```

#### ModuleNotFoundError

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Raspberry Pi 4 Issues

#### MediaPipe Installation Fails

**Problem:** Python version too new (≥3.12)

**Solution:**
```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

#### Camera Not Detected

**Check cameras:**
```bash
v4l2-ctl --list-devices
```

**Try different camera index:**
```bash
python main.py --camera 1 --fast
```

#### Qt Platform Plugin Error

**Error:** `Could not load Qt platform plugin`

**Solution:** Use X11 instead of Wayland:
```bash
sudo raspi-config
# Advanced Options → Wayland → Select X11 → Reboot
```

Or use SSH with X11 forwarding:
```bash
ssh -Y pi@<RPI_IP>
```

#### Low FPS / Laggy Display

**Via SSH with X11:**
- X11 forwarding over network is slow
- Expected: 5-10 FPS display over SSH
- Actual inference: 10-15 FPS (check with `--metrics`)

**Solution:** Use VNC for better display performance, or run directly on RPi with monitor.

#### Thermal Throttling

**Check temperature:**
```bash
vcgencmd measure_temp
```

**If >70°C:**
- Ensure RPi has heatsink
- Improve ventilation
- Use `--fast` mode to reduce CPU load

---

## 📈 Expected Performance

### MacBook (Apple M1/M2 or Intel i5/i7)

| Mode | Resolution | FPS | MediaPipe Latency |
|------|------------|-----|-------------------|
| Standard | 640x480 | 10-15 | 25-30ms |
| Fast | 320x240 | 25-35 | 15-25ms |

### Raspberry Pi 4 (Quad-core ARM Cortex-A72 @ 1.5GHz)

| Mode | Resolution | FPS | MediaPipe Latency |
|------|------------|-----|-------------------|
| Standard | 640x480 | 5-8 | 100-150ms |
| Fast | 320x240 | 10-15 | 50-80ms |

---

## 🎯 Quick Reference Commands

### MacBook

```bash
# Setup
cd ~/Workspace/ai-hardware-project-proposal-visionmasters
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
cd src/emote_detector
python main.py --fast --metrics
```

### Raspberry Pi 4

```bash
# Setup (one-time)
pyenv install 3.11.9 && pyenv local 3.11.9
python -m venv venv
source venv/bin/activate
pip install numpy opencv-python-headless mediapipe scikit-learn pygame psutil

# Run (via SSH with X11)
ssh -Y pi@<RPI_IP>
cd ~/emote_detector
source venv/bin/activate
python main.py --fast --metrics
```

---

## 📚 Additional Resources

- [MediaPipe Documentation](https://google.github.io/mediapipe/solutions/holistic.html)
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [Project Proposal](docs/Project_Proposal.md)
- [Main README](README.md)

---

**For questions or issues, contact the VisionMasters team:**
- Allen Chen (wmm7wr@virginia.edu)
- Marvin Rivera (tkk9wg@virginia.edu)
- Sami Kang (ajp3cx@virginia.edu)

