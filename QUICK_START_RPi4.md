# Quick Start: Raspberry Pi 4 Only

**TL;DR**: Deploy trained model on Raspberry Pi 4 (no external accelerators)

---

## ⚡ 5-Minute Setup

### 1. On Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv python3-opencv libatlas-base-dev

# Clone project
cd ~ && git clone <your-repo> emotion-recognition
cd emotion-recognition

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install numpy opencv-python pillow mediapipe tflite-runtime
```

### 2. Copy Your Trained Model

```bash
# Copy model_int8.tflite to models/ folder
# (via USB drive, SCP, or git)
ls models/model_int8.tflite  # Verify it's there
```

### 3. Run Demo

```bash
cd ~/emotion-recognition
source venv/bin/activate

python3 src/hardware/inference_demo.py \
    --model models/model_int8.tflite \
    --no-edgetpu \
    --display-fps
```

**Press 'q' to quit, 's' for screenshot**

---

## 📊 Expected Performance

✅ **FPS**: 10-20 (acceptable for emotion detection)  
✅ **Latency**: 60-120ms total pipeline  
✅ **Accuracy**: 85%+ (same as training)  
✅ **Power**: 3-5W  
✅ **Setup**: Simple, no external hardware needed  

---

## 🎯 Quick Commands

```bash
# Activate environment
cd ~/emotion-recognition && source venv/bin/activate

# Run demo
python3 src/hardware/inference_demo.py --model models/model_int8.tflite --no-edgetpu --display-fps

# Test camera
python3 src/utils/face_detection.py

# Check temperature
vcgencmd measure_temp

# Monitor CPU
htop
```

---

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| **Low FPS** | Add cooling, reduce resolution, skip frames |
| **Camera not working** | Try `--camera 1`, check `/dev/video*` |
| **Too hot (>75°C)** | Add heatsinks/fan, check throttling |
| **Out of memory** | Close other apps, increase swap |

---

## 🎓 Performance Tips

1. **Add cooling**: Heatsinks + fan keeps temp <70°C
2. **Reduce resolution**: 480x360 instead of 640x480
3. **Skip frames**: Process every 2nd frame for higher FPS
4. **Use INT8 model**: Faster than FP32
5. **Performance mode**: 
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

---

## 📁 What You Need

```
models/model_int8.tflite       # From training (3.5 MB)
src/hardware/inference_demo.py # Demo script
src/utils/face_detection.py    # Face detection
```

---

## 🎬 For Presentation

**Key Points**:
- Real-time emotion recognition on edge device ✅
- No cloud connectivity required ✅
- Low power consumption (<5W) ✅
- Model optimization (4x size reduction) ✅
- Achieves 10-20 FPS (sufficient for emotion detection) ✅

**Adjusted Objectives** (without hardware accelerator):
- FPS: 10-20 (vs 30+ with TPU) ✅ Still real-time
- Latency: 60-120ms (vs <20ms with TPU) ✅ Still responsive
- Demonstrates edge AI principles ✅
- Shows model optimization benefits ✅

---

## 📚 Full Docs

- **Complete Guide**: `DEPLOYMENT_RPi4_ONLY.md`
- **Training**: `notebooks/README.md`
- **Hardware Details**: `src/hardware/README.md`

---

**Simple setup, no extra hardware needed! 🚀**

