# Deployment Guide: Raspberry Pi 4 Only (No Edge TPU)

Complete guide for deploying your facial expression recognition model on **Raspberry Pi 4 Model B** without external accelerators.

---

## 📋 What You Need

- ✅ Trained model from Colab (`model_int8.tflite` or `baseline_fp32_best.h5`)
- ✅ Raspberry Pi 4 Model B (4GB RAM recommended)
- ✅ MicroSD card (64GB, Class 10) with Raspberry Pi OS
- ✅ USB Webcam (720p recommended)
- ✅ Power supply (5V, 3A)
- ✅ HDMI monitor, keyboard, mouse
- ❌ ~~Google Coral USB Accelerator~~ (not needed!)

---

## Part 1: After Training

### Download Your Models from Colab

```python
# In Colab notebook
from google.colab import files

# Option 1: Use INT8 TFLite (smaller, faster)
files.download('/content/models/model_int8.tflite')

# Option 2: Use FP32 Keras (better accuracy)
files.download('/content/models/baseline_fp32_best.h5')

# Download results for presentation
files.download('/content/results/training_results.txt')

# Or zip everything
!zip -r project_results.zip models/ results/
files.download('project_results.zip')
```

**Which model to use?**
- **`model_int8.tflite`**: Smaller (3.5 MB), faster inference
- **`baseline_fp32_best.h5`**: Larger (14 MB), slightly better accuracy

**Recommendation**: Start with INT8 TFLite for better performance on Pi 4.

---

## Part 2: Raspberry Pi 4 Setup

### Step 1: Install Raspberry Pi OS

1. **Download Raspberry Pi Imager**: https://www.raspberrypi.com/software/
2. **Flash SD Card**:
   - Choose **Raspberry Pi OS (64-bit)** - recommended
   - Select your microSD card
   - Click "Write"
3. **First Boot**:
   - Insert SD card, connect peripherals
   - Power on and follow setup wizard
   - Connect to Wi-Fi, set password

### Step 2: System Update

```bash
# Update package lists
sudo apt update

# Upgrade installed packages
sudo apt upgrade -y

# Install basic tools
sudo apt install -y git vim curl wget

# Reboot
sudo reboot
```

### Step 3: Transfer Project Files

**Option A: Clone from GitHub** (recommended):
```bash
cd ~
git clone <your-repo-url> emotion-recognition
cd emotion-recognition
```

**Option B: Use USB Drive**:
```bash
# Insert USB drive (mounts at /media/pi/...)
cp -r /media/pi/USB_NAME/emotion-recognition ~/emotion-recognition
cd ~/emotion-recognition
```

**Option C: Use SCP from your computer**:
```bash
# From your computer
scp -r ai-hardware-project-proposal-visionmasters pi@<pi-ip>:~/emotion-recognition
```

### Step 4: Copy Trained Model

```bash
cd ~/emotion-recognition

# Make sure model is in models/ directory
ls -lh models/model_int8.tflite
# or
ls -lh models/baseline_fp32_best.h5

# If not there, copy from USB:
# cp /media/pi/USB_NAME/model_int8.tflite models/
```

---

## Part 3: Install Dependencies

### Step 1: Install System Dependencies

```bash
# OpenCV and system libraries
sudo apt install -y python3-opencv libopencv-dev

# Additional dependencies for ML
sudo apt install -y libatlas-base-dev libhdf5-dev libjpeg-dev libpng-dev

# For MediaPipe
sudo apt install -y libgl1-mesa-glx
```

### Step 2: Setup Python Virtual Environment

```bash
cd ~/emotion-recognition

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Python Packages

```bash
# Activate venv if not already
source venv/bin/activate

# Install core packages
pip install numpy opencv-python pillow

# Install MediaPipe (for face detection)
pip install mediapipe

# Install TensorFlow Lite Runtime (lighter than full TensorFlow)
pip install tflite-runtime

# If you want to use Keras model (.h5), install TensorFlow
# Note: This is large (~200MB) and slower
# pip install tensorflow
```

---

## Part 4: Test Camera

```bash
cd ~/emotion-recognition
source venv/bin/activate

# Check camera is detected
ls -l /dev/video*

# Test with Python
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK:', cap.isOpened()); cap.release()"

# Test face detection
python3 src/utils/face_detection.py
```

Press 'q' to quit. You should see yourself with a green box around your face.

---

## Part 5: Run the Demo!

### Basic Demo (TFLite Model)

```bash
cd ~/emotion-recognition
source venv/bin/activate

# Run with INT8 TFLite model
python3 src/hardware/inference_demo.py \
    --model models/model_int8.tflite \
    --no-edgetpu \
    --display-fps
```

### With Keras Model (if using .h5)

```bash
# If using Keras model, you need full TensorFlow
pip install tensorflow

# Run demo
python3 src/hardware/inference_demo.py \
    --model models/baseline_fp32_best.h5 \
    --no-edgetpu \
    --display-fps
```

### With Emotes (Optional)

```bash
# Setup emotes first
bash scripts/prepare_emotes.sh

# Run with emotes
python3 src/hardware/inference_demo.py \
    --model models/model_int8.tflite \
    --emotes-dir data/emotes \
    --no-edgetpu \
    --display-fps
```

**Controls:**
- Press **'q'** to quit
- Press **'s'** to save screenshot

---

## Part 6: Performance Optimization

### Tip 1: Increase GPU Memory

```bash
sudo nano /boot/config.txt

# Add or modify this line:
gpu_mem=256

# Save and reboot
sudo reboot
```

### Tip 2: Set CPU to Performance Mode

```bash
# Set all CPUs to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check current frequency
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

### Tip 3: Add Cooling

**Important**: The Pi 4 will run warm under continuous inference load.

Options:
- Add heatsinks to CPU/RAM chips
- Attach a small 5V fan
- Use official Pi 4 case with fan
- Monitor temperature: `watch -n 1 vcgencmd measure_temp`

Target: Keep temperature <75°C

### Tip 4: Reduce Input Resolution

Edit `src/hardware/inference_demo.py`:

```python
# Around line 200, modify camera settings:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Was 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Was 480

# You can reduce to:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
```

### Tip 5: Process Every Other Frame

For higher FPS, process every 2nd frame:

```python
# In the main loop, add frame skipping:
frame_count = 0
last_emotion = None
last_confidence = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    # Only process every 2nd frame
    if frame_count % 2 == 0:
        # Run face detection and inference
        ...
    else:
        # Use previous result
        emotion = last_emotion
        confidence = last_confidence
```

---

## Part 7: Benchmark Performance

```bash
cd ~/emotion-recognition
source venv/bin/activate

# Run benchmarks
python3 benchmarks/benchmark_model.py \
    --int8-tflite models/model_int8.tflite \
    --iterations 100
```

This measures:
- Face detection latency
- Model inference latency
- Total pipeline latency
- FPS capability

Results saved to `results/benchmark_results.csv`

---

## Part 8: Monitor System Resources

### While Demo is Running

Open a second terminal (SSH or separate session):

```bash
# Monitor CPU usage
htop

# Check temperature continuously
watch -n 1 vcgencmd measure_temp

# Check CPU frequency
watch -n 1 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# Check voltage (should be ~1.35V)
vcgencmd measure_volts

# Check memory usage
free -h
```

---

## Part 9: Expected Performance

### Realistic Expectations (RPi 4 without TPU)

| Metric | Expected | Notes |
|--------|----------|-------|
| **Inference Latency** | 40-80ms | Model inference only |
| **Face Detection** | 15-25ms | MediaPipe |
| **Total Pipeline** | 60-120ms | Detection + inference + display |
| **FPS** | 10-20 | Full pipeline |
| **Accuracy** | 85%+ | Same as training |
| **Power Consumption** | 3-5W | Depends on load |
| **Temperature** | 60-75°C | With cooling |
| **CPU Usage** | 70-90% | All cores active |

### Comparison: With vs Without Edge TPU

| Metric | RPi 4 Only | RPi 4 + Coral TPU |
|--------|------------|-------------------|
| **Inference Time** | 40-80ms | 8-15ms ⚡ |
| **FPS** | 10-20 | 30-45 ⚡ |
| **Setup Complexity** | Simple ✅ | More complex |
| **Cost** | Lower ✅ | +$60 for Coral |
| **Power** | 3-5W ✅ | 3-6W |

**Trade-off**: Without Edge TPU, you get simpler setup and lower cost, but slower inference and lower FPS.

---

## Part 10: Troubleshooting

### Low FPS (<10)

**Solutions**:
1. **Reduce camera resolution** (see Tip 4 above)
2. **Process every 2nd frame** (see Tip 5 above)
3. **Check CPU isn't throttling**:
   ```bash
   vcgencmd measure_temp  # Should be <75°C
   vcgencmd get_throttled  # Should be "0x0"
   ```
4. **Add cooling** if temperature >75°C
5. **Use INT8 TFLite model** instead of Keras (.h5)
6. **Close other applications**

### Camera Not Working

```bash
# Check camera is detected
ls -l /dev/video*

# Test different camera index
python3 src/hardware/inference_demo.py --camera 1 --no-edgetpu

# For Pi Camera Module (not USB webcam)
sudo raspi-config
# Interface Options > Camera > Enable
# Reboot
```

### Out of Memory

```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set: CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Close other applications
# Use lighter desktop environment
```

### CPU Thermal Throttling

```bash
# Check throttling status
vcgencmd get_throttled

# Output meanings:
# 0x0     = All good ✅
# 0x50000 = Throttling occurred
# 0x50005 = Currently throttling

# Solutions:
# 1. Add heatsinks
# 2. Add fan
# 3. Improve ventilation
# 4. Reduce workload (lower resolution, skip frames)
```

### ImportError for TensorFlow/TFLite

```bash
# For TFLite Runtime (recommended)
pip install --upgrade tflite-runtime

# Or if you need full TensorFlow
pip install tensorflow

# Check installation
python3 -c "import tflite_runtime.interpreter as tflite; print('TFLite OK')"
# or
python3 -c "import tensorflow as tf; print('TensorFlow OK')"
```

---

## Part 11: Create Demo Video

For your presentation:

```bash
# Option 1: Use built-in screen recording
sudo apt install -y simplescreenrecorder
simplescreenrecorder

# Option 2: Record with Python
python3 << EOF
import cv2
import subprocess

# Start demo in background
demo = subprocess.Popen(['python3', 'src/hardware/inference_demo.py', 
                         '--model', 'models/model_int8.tflite', '--no-edgetpu'])

# Record video
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('demo_recording.avi', fourcc, 15.0, (640, 480))

for i in range(450):  # 30 seconds at 15fps
    ret, frame = cap.read()
    if ret:
        out.write(frame)
    if i % 30 == 0:
        print(f"Recording... {i//15}s")

cap.release()
out.release()
demo.terminate()
print("Recording saved: demo_recording.avi")
EOF
```

---

## Part 12: Auto-Start on Boot (Optional)

```bash
# Create systemd service
sudo nano /etc/systemd/system/emotion-recognition.service
```

Add:
```ini
[Unit]
Description=Emotion Recognition Demo
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/emotion-recognition
Environment="DISPLAY=:0"
ExecStart=/home/pi/emotion-recognition/venv/bin/python3 src/hardware/inference_demo.py --model models/model_int8.tflite --no-edgetpu --display-fps
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable emotion-recognition
sudo systemctl start emotion-recognition

# Check status
sudo systemctl status emotion-recognition

# View logs
sudo journalctl -u emotion-recognition -f

# Stop
sudo systemctl stop emotion-recognition
```

---

## 🎯 Quick Command Reference

```bash
# Setup (one-time)
cd ~/emotion-recognition
python3 -m venv venv
source venv/bin/activate
pip install numpy opencv-python pillow mediapipe tflite-runtime

# Run demo
source venv/bin/activate
python3 src/hardware/inference_demo.py --model models/model_int8.tflite --no-edgetpu --display-fps

# Test components
python3 src/utils/face_detection.py
python3 benchmarks/benchmark_model.py

# Monitor system
vcgencmd measure_temp
htop
```

---

## ✅ Verification Checklist

Before final presentation:

- [ ] Model transferred to Pi (`model_int8.tflite`)
- [ ] All dependencies installed (OpenCV, MediaPipe, TFLite)
- [ ] Camera working (tested with face detection)
- [ ] Demo runs smoothly (10-20 FPS acceptable)
- [ ] Temperature stable (<75°C with cooling)
- [ ] Benchmark results collected
- [ ] Screenshots/video captured
- [ ] Performance metrics documented

---

## 📊 For Your Presentation

### What to Show

1. **Live Demo**:
   - Real-time emotion detection
   - Show different expressions
   - Display FPS and latency

2. **Performance Metrics**:
   - Accuracy: 85%+ (from training)
   - FPS: 10-20 (realistic for RPi 4)
   - Latency: 60-120ms total pipeline
   - Power: 3-5W

3. **Comparisons**:
   - Before/after model quantization
   - INT8 vs FP32 performance
   - Model size reduction (14MB → 3.5MB)

4. **Challenges & Solutions**:
   - Limited compute power → Used MobileNetV2, INT8 quantization
   - Real-time requirement → Optimized pipeline, frame skipping
   - Thermal management → Added cooling solution

### Adjusted Technical Objectives

Original (with Edge TPU) → Adjusted (RPi 4 only):

| Objective | Original Target | Adjusted Target |
|-----------|----------------|-----------------|
| **FPS** | >30 | >10 ✅ Achievable |
| **Inference Latency** | <20ms | <80ms ✅ Achievable |
| **Total Latency** | <50ms | <120ms ✅ Achievable |
| **Accuracy** | >85% | >85% ✅ Same |
| **Power** | <5W | <5W ✅ Same |
| **Model Size** | <10MB | <10MB ✅ 3.5MB |

**Key Point**: Without hardware acceleration, you still achieve real-time performance (10-20 FPS is sufficient for emotion detection), just not as fast as with Edge TPU.

---

## 🎓 What You Learned

Even without Edge TPU, this project demonstrates:

1. **Model Optimization**: Quantization reduces size/improves speed
2. **Edge Computing**: Running AI on constrained devices
3. **Hardware-Software Co-design**: Optimizing for limited compute
4. **Real-time Systems**: Meeting performance requirements
5. **Trade-offs**: Accuracy vs speed vs power vs cost

---

## 🚀 Next Steps

If you want to improve performance further:

1. **Use smaller input size**: 160x160 instead of 224x224
2. **Try MobileNetV1**: Slightly faster than V2
3. **Implement frame skipping**: Process every 2-3 frames
4. **Reduce face detection frequency**: Only every few frames
5. **Consider upgrading**: RPi 5 is significantly faster
6. **Add Coral later**: Can always add Edge TPU if needed

---

## 📞 Need Help?

- **Hardware issues**: Check `src/hardware/README.md`
- **Model issues**: Check `src/model/README.md`
- **General setup**: Check `GETTING_STARTED.md`

---

**You're all set for deployment on Raspberry Pi 4! 🎉**

The setup is simpler without Edge TPU, and 10-20 FPS is perfectly acceptable for emotion recognition. Focus on the AI/ML aspects and model optimization in your presentation!

