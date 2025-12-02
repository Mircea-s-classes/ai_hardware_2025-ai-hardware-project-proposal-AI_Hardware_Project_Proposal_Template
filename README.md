# Real-Time Facial Expression Recognition on Edge AI Hardware

**Team:** VisionMasters  
**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Platform:** Raspberry Pi 4 + Google Coral USB Accelerator (Edge TPU)

## 🎯 Project Overview

This project implements real-time facial expression recognition on edge AI hardware, demonstrating efficient deployment of deep learning models on resource-constrained devices. The system recognizes 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) and displays corresponding Clash Royale emotes.

### Platform
**Raspberry Pi 4 Model B** (CPU-only inference, no external accelerators)

### Key Objectives
- **Real-Time Performance**: 10-20 FPS with <120ms total latency
- **Power Efficiency**: <5W total system power consumption
- **Accuracy**: 85%+ accuracy on FER2013 dataset
- **Edge Optimization**: INT8 quantization for efficient inference

## 📁 Project Structure

```
.
├── src/
│   ├── model/           # Model training and conversion scripts
│   ├── hardware/        # Hardware integration and inference
│   └── utils/           # Utility functions
├── data/
│   ├── fer2013/         # FER2013 dataset
│   └── emotes/          # Clash Royale emote images and sounds
├── models/              # Trained models (FP32, INT8, TFLite, EdgeTPU)
├── benchmarks/          # Performance testing scripts
├── results/             # Performance data and charts
├── docs/                # Documentation and diagrams
├── presentations/       # Presentation slides
└── report/              # Final report
```

## 🚀 Getting Started

### Phase 1: Model Development (Current - No Hardware Required)

1. **Setup Environment**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Download FER2013 Dataset**
```bash
# Option 1: Using Kaggle API (requires Kaggle account and API key)
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/fer2013/

# Option 2: Manual download from Kaggle
# https://www.kaggle.com/datasets/msambare/fer2013
```

3. **Train Baseline Model**
```bash
python src/model/train_baseline.py
```

4. **Evaluate Model**
```bash
python src/model/evaluate.py --model models/baseline_fp32.h5
```

### Phase 2: Model Optimization (Week 3)

5. **Quantize Model to INT8**
```bash
python src/model/quantize_model.py
```

### Phase 3: Hardware Integration (Week 4-5)

6. **Deploy on Raspberry Pi 4**
```bash
# On Raspberry Pi
python src/hardware/inference_demo.py --model models/model_int8.tflite --no-edgetpu --display-fps
```

**See**: `DEPLOYMENT_RPi4_ONLY.md` for complete deployment guide

## 📊 Current Progress

- [x] Project proposal
- [ ] FER2013 dataset preparation
- [ ] Baseline MobileNetV2 model training
- [ ] Face detection pipeline (MediaPipe)
- [ ] Model quantization (INT8)
- [ ] TFLite conversion
- [ ] Edge TPU deployment
- [ ] Real-time demo application
- [ ] Benchmarking and optimization

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run benchmarks
python benchmarks/benchmark_model.py
```

## 📈 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Accuracy | >85% | TBD |
| FPS (RPi 4) | >10 | TBD |
| Inference Latency | <80ms | TBD |
| Total Latency | <120ms | TBD |
| Power | <5W | TBD |
| Model Size (INT8) | <5MB | ~3.5MB |

## 🎮 Emotion to Emote Mapping

| Emotion | Clash Royale Emote |
|---------|-------------------|
| Happy | 😂 Laughing King |
| Sad | 😢 Crying Face |
| Angry | 😠 Angry Face |
| Surprise | 😲 Shocked Face |
| Fear | 😱 Screaming Face |
| Disgust | 🤢 Sick Face |
| Neutral | 👍 Thumbs Up |

## 👥 Team Members

- **Allen Chen** (wmm7wr@virginia.edu) - Hardware Integration
- **Marvin Rivera** (tkk9wg@virginia.edu) - Team Lead, Documentation
- **Sami Kang** (ajp3cx@virginia.edu) - Model Training, Inference

## 📚 References

- [Google Coral Edge TPU](https://coral.ai/products/accelerator/)
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html)

## 📝 License

This project is for educational purposes as part of ECE 4332/6332 at the University of Virginia.
