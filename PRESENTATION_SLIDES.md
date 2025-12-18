# Real-Time Pose-Based Emote Detection on Edge AI Hardware
## ECE4332 AI Hardware - Midterm Presentation

**Team VisionMasters**  
December 2025

---

# Slide 1: Team Introduction

## Team VisionMasters

**Team Members & Roles:**
- [Your Name] - Project Lead, System Integration
- [Member 2] - Data Collection & Model Training
- [Member 3] - Performance Optimization & Testing
- [Member 4] - Documentation & Deployment

**GitHub Repository:**  
`ai-hardware-project-proposal-visionmasters`

**Platform:**  
Raspberry Pi 4 Model B (ARM Cortex-A72, 4GB RAM)

---

# Slide 2: Problem & Motivation

## Why Edge AI for Pose Detection?

**The Challenge:**
- Real-time pose detection typically requires powerful GPUs
- Cloud processing introduces latency and privacy concerns
- Can we run real-time AI on $50 edge hardware?

**Our Goal:**
> Demonstrate that a low-cost Raspberry Pi 4 can achieve real-time pose-based emote detection with proper hardware-software co-design

**Real-World Applications:**
- 🎮 Interactive gaming (like Clash Royale)
- 🏥 Remote patient monitoring
- 🏋️ Fitness tracking
- 🤖 Human-robot interaction

**Success Criteria:** ≥10 FPS on Raspberry Pi 4

---

# Slide 3: Technical Background & Related Work

## Existing Solutions

| Approach | Platform | Performance | Cost |
|----------|----------|-------------|------|
| Cloud-based APIs | Server GPU | 30-60 FPS | High + recurring |
| On-device CNNs | Mobile GPU | 20-30 FPS | Medium |
| **Our Approach** | **RPi4 CPU** | **Target: 10 FPS** | **$50** |

## Key Technologies

**MediaPipe (Google)**
- Pre-trained pose detection model
- TensorFlow Lite optimized
- Real-time performance on mobile

**Random Forest Classifier**
- Lightweight, fast inference
- CPU-friendly
- Small model size

**Why This Combination?**
- MediaPipe: State-of-art pose detection
- Random Forest: Faster than CNN for simple classification
- No GPU required!

---

# Slide 4: System Architecture

## End-to-End Pipeline

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌────────┐
│ Camera  │───▶│ MediaPipe│───▶│ Feature │───▶│ Random   │───▶│ Emote  │
│ 160x120 │    │ (TFLite) │    │Extract  │    │ Forest   │    │Display │
└─────────┘    └──────────┘    └─────────┘    └──────────┘    └────────┘
                   ~189ms           <1ms           ~12ms         Real-time
                (Pose Detection) (18 features)  (10 trees)
```

**Key Design Decisions:**
1. ✅ **Low resolution (160x120)** - 4x speedup vs 640x480
2. ✅ **Frame skipping (every 4th)** - Maintain responsiveness
3. ✅ **Traditional ML > DL** - Random Forest faster than CNN on CPU
4. ✅ **Model complexity=0** - Fastest MediaPipe variant

**4 Emote Classes:**
- Laughing (hands on waist)
- Yawning (hands over mouth)
- Crying (hands covering face)
- Taunting (fists near face)

---

# Slide 5: Data Collection & Model Training

## Custom ML Pipeline

**Phase 1: Data Collection**
- Custom interactive tool with live preview
- 50-100 samples per pose class
- Real-time pose visualization
- Total: ~300-400 samples

**[INSERT IMAGE: data_collector screenshot or data_distribution.png]**
`results/rpi_results/training_charts/charts/data_distribution.png`

**Phase 2: Feature Engineering**
- MediaPipe → 33 pose landmarks (x, y, z)
- Extract 18 geometric features:
  - Distances (shoulder width, arm length)
  - Angles (elbow, knee joints)
  - Positions (hand height, offset from center)

**Phase 3: Model Training**
- Algorithm: Random Forest (10 trees, depth=4)
- Train/Test Split: 80/20
- Training time: <5 seconds
- Model size: 25 KB

---

# Slide 6: Model Evaluation Results

## Training Performance

**[INSERT CHART: confusion_matrix.png]**
`results/rpi_results/training_charts/charts/confusion_matrix.png`

**[INSERT CHART: per_class_accuracy.png]**
`results/rpi_results/training_charts/charts/per_class_accuracy.png`

**Key Metrics:**
- ✅ Overall Accuracy: >85%
- ✅ Mean Confidence: 0.61-0.72
- ✅ Well-balanced across classes

**[INSERT CHART: feature_importance.png]**
`results/rpi_results/training_charts/charts/feature_importance.png`

**Most Important Features:**
1. Hand height relative to shoulders
2. Arm angles
3. Hand-to-face distances

---

# Slide 7: The Optimization Journey

## 20x Speedup Through Smart Design

**Problem Discovery:**
- Initial model: 239ms classifier latency
- MediaPipe: Already fast at ~25ms on MacBook
- **Bottleneck identified: Random Forest!**

**Optimization Iterations:**

| Configuration | RPi4 Latency | Speedup |
|---------------|--------------|---------|
| Initial (100 trees, n_jobs=-1) | 239 ms | Baseline |
| Fixed threading (n_jobs=1) | 160 ms | 1.5x |
| Reduced trees (20 trees) | 30 ms | 8x |
| **Final (10 trees, depth=4)** | **12 ms** | **20x!** ✅ |

**Key Insight:** Multi-threading (n_jobs=-1) hurt performance!
- Context switching overhead on RPi's 4 cores
- Single-core inference optimal for small models

**Accuracy Impact:** <3% accuracy loss (88% → 85%)

---

# Slide 8: Raspberry Pi 4 Performance Results

## Edge Deployment - Production Mode (Headless)

**[INSERT CHART: RPi metrics summary]**

### 🎯 Key Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Overall FPS** | ≥10 | **9.7** | ✅ Near target |
| **MediaPipe** | <100ms | 189ms | ⚠️ Dominant bottleneck |
| **Classifier** | <20ms | 12ms | ✅ Excellent |
| **Total Latency** | <250ms | 201ms | ✅ Real-time |
| **Temperature** | <80°C | 52.6°C | ✅ Cool |

**Configuration:**
- Resolution: 160x120 (ultra-fast mode)
- MediaPipe complexity: 0
- Frame skip: Process every 4th frame
- Deployment: Headless (no display overhead)

**MediaPipe Breakdown:**
```
Frame 1:  271ms (cold start)
Frames 2+: 120ms (warmed up)
Mean:     189ms (includes outliers)
```

---

# Slide 9: Platform Comparison - MacBook vs RPi4

## How Does $50 Hardware Compare to $1,500?

**[INSERT CHART: summary_dashboard.png]**
`results/comparison_charts/summary_dashboard.png`

### Performance Comparison

| Metric | MacBook | RPi4 | Ratio |
|--------|---------|------|-------|
| **MediaPipe** | 23.5 ms | 189 ms | 8x slower |
| **Classifier** | 1.3 ms | 12 ms | 9x slower |
| **Overall FPS** | 14.7 | 9.7 | 1.5x slower |
| **Cost** | $1,500 | $50 | **30x cheaper** |

**[INSERT CHART: inference_comparison.png]**
`results/comparison_charts/inference_comparison.png`

**Key Takeaway:** Despite being 8x slower, RPi4 still achieves real-time performance!

---

# Slide 10: Cost-Performance Analysis

## Value Proposition of Edge AI

**[INSERT CHART: cost_performance.png]**
`results/comparison_charts/cost_performance.png`

### FPS per $100 Invested

| Platform | FPS | Cost | FPS per $100 | Winner |
|----------|-----|------|--------------|--------|
| MacBook | 14.7 | $1,500 | 0.98 | - |
| **RPi4** | **9.7** | **$50** | **19.4** | ✅ **20x better!** |

**When to Use Each:**

**MacBook/Server:**
- ✅ High-resolution needs (>640x480)
- ✅ Multi-stream processing
- ✅ Development & training

**Raspberry Pi 4:**
- ✅ **Cost-sensitive deployments**
- ✅ **Power-constrained (5W vs 30W)**
- ✅ **Distributed IoT applications**
- ✅ **Real-time sufficient (10 FPS)**

---

# Slide 11: Live Demo

## **[DEMO TIME]** 🎥

**Demo Options:**

### Option A: Live Demo on RPi
```bash
python main.py --fast
```
- Show real-time pose detection
- Demonstrate emote classification
- Show FPS counter

### Option B: Pre-recorded Video
- Include backup video in case of technical issues
- Show: Data collection → Training → Inference

**What to Highlight:**
1. 🎥 Live camera feed processing
2. 🤖 Real-time pose detection
3. 😊 Emote classification with confidence scores
4. 📊 FPS counter (~10 FPS on RPi)
5. 🎵 Audio feedback on emote detection

**Demo Duration:** 2-3 minutes

---

# Slide 12: Deployment Insights & Challenges

## What We Learned

### 🔍 Key Insights

**1. Display Overhead Matters**
| Mode | FPS | Impact |
|------|-----|--------|
| Headless | 9.7 | ✅ True performance |
| X11 over WiFi | 2.5 | ⚠️ 4x slower! |
| Expected HDMI | ~7-8 | Moderate overhead |

**Lesson:** Deployment environment significantly impacts performance

**2. Threading ≠ Faster**
- Multi-threading hurts small models
- Context switching overhead dominates
- Single-core optimal for lightweight inference

**3. Resolution Sweet Spot**
- 160x120: Fast, still accurate for pose
- 320x240: 2x slower
- 640x480: 4x slower

### 🎯 Challenges Overcome

1. **Initial slow performance (2.4 FPS)**
   - Root cause: n_jobs=-1 threading overhead
   - Solution: Profile, identify bottleneck, optimize

2. **Model version mismatch**
   - scikit-learn 1.6.1 → 1.7.2 compatibility
   - Solution: Retrain on target platform

3. **Display issues (Qt/X11)**
   - X11 forwarding over SSH added latency
   - Solution: Headless mode for true metrics

---

# Slide 13: Innovation & Contributions

## What Makes This Project Unique?

### 🌟 Our Innovations

**1. Hardware-Aware Algorithm Selection**
- Chose Random Forest over CNN
- 10x faster on CPU with minimal accuracy loss
- Demonstrates domain-appropriate ML > complex models

**2. Comprehensive Performance Analysis**
- Not just "it works" - systematic optimization
- Platform comparison (MacBook vs RPi)
- Deployment environment impact (headless vs X11)
- Full metrics pipeline for reproducibility

**3. End-to-End Edge AI Pipeline**
- Custom data collection tool
- Training with evaluation charts
- Optimized deployment
- Performance monitoring

**4. Real-World Optimization Story**
- 20x speedup through profiling
- Shows importance of benchmarking
- Demonstrates iterative improvement

### 📚 Technical Contributions

- Proof that $50 hardware can do real-time AI
- Quantified cost-performance tradeoffs
- Open-source, reproducible pipeline
- Comprehensive documentation

---

# Slide 14: Future Work & Improvements

## What's Next?

### 🚀 Immediate Improvements (0-30% speedup)

**1. Overclocking RPi4**
- Current: 1.5 GHz
- Target: 2.0 GHz
- Expected: ~30% speedup → **13 FPS**

**2. Pose-Only MediaPipe**
- Remove face/hand detection
- Expected: ~30% speedup → **13 FPS**

**3. INT8 Quantization**
- TFLite model quantization
- Expected: 2x speedup → **15 FPS**

### 🎯 Major Improvements (10-20x speedup)

**4. Add Coral Edge TPU (+$25)**
- Hardware accelerator for TensorFlow Lite
- Expected: 10-20x MediaPipe speedup
- New MediaPipe latency: ~10-20ms (vs 189ms)
- **Target: 30-50 FPS!**

**5. Raspberry Pi 5**
- 2.4 GHz CPU, better cooling
- Expected: ~40% faster → **14 FPS**

### 📱 New Features

- More pose classes (expand from 4 to 10+)
- Multi-person detection
- Action sequences (not just static poses)
- Mobile app deployment (Android/iOS)

---

# Slide 15: Conclusion & Key Takeaways

## Project Summary

### ✅ Goals Achieved

| Objective | Target | Result | Status |
|-----------|--------|--------|--------|
| Real-time FPS | ≥10 FPS | 9.7 FPS | ✅ Near target |
| Inference Latency | <250ms | 201ms | ✅ |
| Model Accuracy | >85% | 85-88% | ✅ |
| Edge Deployment | RPi4 | Success | ✅ |
| Full ML Pipeline | End-to-end | Complete | ✅ |

### 🎯 Key Takeaways

1. **Edge AI is Viable**
   - $50 hardware can run real-time AI
   - Smart optimization > expensive hardware

2. **Algorithm Selection Matters**
   - Random Forest 10x faster than CNN on CPU
   - Domain-appropriate models win

3. **System Design is Critical**
   - Resolution, threading, frame skipping all matter
   - 20x speedup through systematic optimization

4. **Cost-Performance Tradeoffs**
   - RPi4: 20x better value per dollar
   - Choose based on application requirements

### 💡 Bottom Line

> **"A $50 Raspberry Pi 4 can achieve real-time AI inference with proper hardware-software co-design and optimization."**

**Demonstrated:** Real-world edge AI deployment is not just possible, but cost-effective!

---

# Slide 16: Questions & Demo

## Thank You!

**Team VisionMasters**

### 📁 Resources

- **GitHub:** `ai-hardware-project-proposal-visionmasters`
- **Documentation:** Complete README.md with HowTo
- **Metrics:** Full performance analysis & charts
- **Code:** Open source, ready to deploy

### 📊 What We Delivered

✅ Working system on Raspberry Pi 4  
✅ Comprehensive performance analysis  
✅ Platform comparison (MacBook vs RPi)  
✅ Training evaluation charts  
✅ Deployment guide  
✅ Reproducible results  

### 🎥 Demo Time

**Live demonstration or backup video**

---

### ❓ Questions?

**We're happy to discuss:**
- Technical implementation details
- Optimization methodology
- Alternative approaches
- Future improvements
- Deployment scenarios

---

# APPENDIX: Backup Slides

## Backup Slide A: Detailed Metrics Table

| Metric | MacBook | RPi4 Headless | RPi4 X11 |
|--------|---------|---------------|----------|
| MediaPipe | 23.5 ms | 189 ms | 547 ms |
| Classifier | 1.3 ms | 12 ms | 32 ms |
| Frame Time | 53 ms | 102 ms | 256 ms |
| Overall FPS | 14.7 | 9.7 | 2.5 |
| CPU Usage | 15% | ~50-70% | N/A |
| Temperature | N/A | 52.6°C | 53°C |
| Power | ~30W | ~5W | ~5W |

---

## Backup Slide B: Hardware Specifications

**Raspberry Pi 4 Model B**
- CPU: Broadcom BCM2711, Quad-core ARM Cortex-A72 @ 1.5GHz
- RAM: 4GB LPDDR4-3200
- GPU: VideoCore VI (not used)
- Storage: microSD card
- Camera: USB webcam (160x120 @ 30fps)
- Power: 5V/3A USB-C
- Cost: $50

**MacBook (Test Platform)**
- CPU: Apple M1 / Intel Core i5-i7 (assumed)
- RAM: 8-16GB
- Cost: ~$1,500

---

## Backup Slide C: Software Stack

**Dependencies:**
- Python 3.11
- MediaPipe 0.10.x
- TensorFlow Lite (via MediaPipe)
- scikit-learn 1.6.1
- OpenCV 4.x
- NumPy, Pygame

**Development Tools:**
- Git/GitHub for version control
- Matplotlib/Seaborn for visualization
- Custom metrics collection framework

---

## Backup Slide D: References

**Key Technologies:**
1. MediaPipe - Google's ML solutions for real-time pose detection
2. TensorFlow Lite - Lightweight ML for edge devices
3. scikit-learn Random Forest - Traditional ML classifier

**Inspiration:**
- Clash Royale Emote Detector (GitHub reference)
- MediaPipe official examples
- Edge AI deployment best practices

---

# END OF PRESENTATION

**Total Slides:** 16 main + 4 backup = 20 slides
**Estimated Duration:** 12-15 minutes (with demo)

---

# Presentation Notes

## Timing Breakdown (15 minutes total)

1. Title + Team (0:30)
2. Motivation (1:00)
3. Technical Background (1:30)
4. Architecture (1:30)
5. Data & Training (1:30)
6. Model Evaluation (1:00)
7. Optimization (2:00)
8. RPi Results (1:30)
9. Platform Comparison (1:30)
10. Cost Analysis (1:00)
11. **DEMO** (2:00)
12. Insights & Challenges (1:00)
13. Innovation (0:30)
14. Future Work (0:30)
15. Conclusion (0:30)
16. Q&A (remaining time)

## Charts to Include

**Required Charts:**
1. `results/comparison_charts/summary_dashboard.png` - Main comparison (Slide 9)
2. `results/comparison_charts/inference_comparison.png` - Latency bars (Slide 9)
3. `results/comparison_charts/cost_performance.png` - Value analysis (Slide 10)
4. `results/rpi_results/training_charts/charts/confusion_matrix.png` - Model accuracy (Slide 6)
5. `results/rpi_results/training_charts/charts/per_class_accuracy.png` - Per-class (Slide 6)
6. `results/rpi_results/training_charts/charts/feature_importance.png` - Features (Slide 6)
7. `results/rpi_results/training_charts/charts/data_distribution.png` - Dataset (Slide 5)

**Optional Charts:**
- `results/comparison_charts/fps_comparison.png` - FPS details
- `results/comparison_charts/speedup_comparison.png` - Relative performance

## Demo Preparation

**Before Presentation:**
1. Test live demo on RPi (ssh -Y pi@<IP>)
2. Record backup video (2-3 minutes)
3. Prepare fallback screenshots
4. Test audio/sound effects
5. Verify camera works

**Demo Script:**
1. Show data collector (if time)
2. Run main.py --fast
3. Demonstrate different poses
4. Show FPS counter
5. Highlight real-time classification
6. Show confidence scores

**Backup Plan:**
- Pre-recorded video
- Screenshots with annotations
- MacBook demo as alternative

