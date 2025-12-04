# University of Virginia

## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal

## 1. Project Title

**Real-Time Pose-Based Emote Detection on Raspberry Pi 4**

**Team Name:** VisionMasters

**Team Members:**

- Allen Chen - wmm7wr@virginia.edu
- Marvin Rivera - tkk9wg@virginia.edu
- Sami Kang - ajp3cx@virginia.edu

## 2. Platform Selection

**Selected Platform:** Edge-AI Platform

**Specific Hardware:** Raspberry Pi 4 Model B (4GB RAM)

**Reasoning:**

The Raspberry Pi 4 represents a widely-deployed edge computing platform that demonstrates the challenges and trade-offs of running AI workloads on resource-constrained devices:

1. **Representative Edge Hardware:** The Quad-core ARM Cortex-A72 @ 1.5GHz with 4GB LPDDR4 RAM mirrors the compute capabilities found in many embedded and IoT devices, making our findings applicable to real-world deployments.

2. **CPU-Only Inference:** By running inference without dedicated accelerators, we can analyze the full software stack and identify optimization opportunities that apply broadly to edge devices without specialized AI hardware.

3. **Real-World Constraints:** The platform exposes genuine limitations in power, thermal management, memory bandwidth, and compute that are central to AI hardware design decisions.

4. **Full ML Pipeline Demonstration:** The platform allows us to demonstrate the complete machine learning workflow from data collection through deployment, showcasing the end-to-end process required for edge AI applications.

## 3. Problem Definition

**Challenge:** Real-time human pose recognition for interactive applications requires processing video frames fast enough to feel responsive (>10 FPS) while running on power-constrained edge devices without dedicated AI accelerators.

**The AI Hardware Challenge:**

How can we achieve real-time pose classification on a CPU-only edge device by optimizing the ML pipeline, selecting appropriate model architectures, and engineering efficient feature representations?

**Why This Matters for AI Hardware:**

This project addresses fundamental AI hardware design considerations:

1. **Model Selection Trade-offs:** Comparing lightweight pose estimation (MediaPipe) vs. traditional computer vision approaches demonstrates how model architecture choices impact edge deployment feasibility.

2. **Feature Engineering vs. Deep Learning:** Using geometric features with classical ML (Random Forest) instead of end-to-end deep learning shows an alternative approach that can be more efficient on CPU-only devices.

3. **Pipeline Optimization:** Analyzing where time is spent (camera capture, pose detection, classification, rendering) reveals bottlenecks and guides hardware/software co-design decisions.

4. **Resource Profiling:** Measuring CPU utilization, memory usage, thermal behavior, and power consumption provides insights into what hardware resources limit performance.

**Real-World Use Case:**

Our demo maps detected body poses to Clash Royale game emotes, demonstrating applications in:

- Gaming accessories that react to player gestures
- Gesture-based communication interfaces
- Interactive entertainment systems
- Human-computer interaction research

## 4. Technical Objectives

1. **Real-Time Performance:** Achieve ≥10 FPS end-to-end processing (camera capture → pose detection → classification → display) on Raspberry Pi 4 CPU.

2. **Low Latency:** Maintain <100ms total latency from pose to emote display for responsive interaction.

3. **Classification Accuracy:** Achieve >85% accuracy on 5 pose classes using self-collected training data.

4. **Thermal Stability:** Ensure sustained operation without thermal throttling during extended use (>30 minutes).

5. **Comprehensive Metrics:** Collect detailed performance metrics (FPS, latency breakdown, CPU/memory usage, temperature) to analyze hardware utilization patterns.

## 5. Methodology

### 5.1 Hardware Setup

| Component | Specification |
|-----------|---------------|
| **Compute** | Raspberry Pi 4 Model B, 4GB RAM |
| **CPU** | Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5GHz |
| **Camera** | Logitech Brio 100 USB Webcam (1080p) |
| **Display** | HDMI monitor (via VNC for remote demo) |
| **Storage** | 64GB MicroSD Card |

### 5.2 Software Stack

| Layer | Technology |
|-------|------------|
| **OS** | Raspberry Pi OS (64-bit) |
| **Runtime** | Python 3.11 |
| **Pose Detection** | MediaPipe Holistic (TensorFlow Lite backend) |
| **Classification** | scikit-learn Random Forest |
| **Computer Vision** | OpenCV |
| **Audio** | pygame |
| **Metrics** | Custom performance profiler with psutil |

### 5.3 ML Pipeline Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────┐
│   Camera    │ ──► │  MediaPipe       │ ──► │  Feature        │ ──► │  Random    │
│   Capture   │     │  Holistic        │     │  Extraction     │     │  Forest    │
│  (OpenCV)   │     │  (TFLite)        │     │  (NumPy)        │     │  Classifier│
└─────────────┘     └──────────────────┘     └─────────────────┘     └────────────┘
      │                    │                        │                      │
      │              Pose, Face, Hand          Geometric              Pose Label +
      │               Landmarks               Features (45D)          Confidence
      ▼                    ▼                        ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Display & Emote Overlay                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Pose Detection (MediaPipe Holistic)

MediaPipe Holistic provides unified detection of:
- **33 Pose Landmarks** (body skeleton)
- **468 Face Landmarks** (facial mesh)
- **21 Hand Landmarks** (per hand)

**Model Complexity Options:**
| Level | Description | Use Case |
|-------|-------------|----------|
| 0 | Lite | Fastest, lower accuracy |
| 1 | Full | Balanced |
| 2 | Heavy | Most accurate, slowest |

We use **complexity=0** for optimal real-time performance on RPi4.

### 5.5 Feature Engineering

Instead of using raw landmarks (which would require deep learning), we extract **45 geometric features** from pose landmarks:

1. **Shoulder-Based Features:**
   - Shoulder width (normalized)
   - Shoulder center position
   - Shoulder tilt angle

2. **Arm Position Features:**
   - Elbow angles (left/right)
   - Wrist positions relative to shoulders
   - Arm extension ratios

3. **Head/Face Features:**
   - Head tilt angle
   - Nose-to-shoulder distances
   - Face visibility score

4. **Body Posture Features:**
   - Torso lean angle
   - Hip-shoulder alignment
   - Overall pose symmetry

### 5.6 Classification (Random Forest)

**Why Random Forest over Deep Learning:**

| Aspect | Random Forest | Deep Learning (CNN) |
|--------|---------------|---------------------|
| Training Time | Seconds | Hours |
| Inference Speed | <1ms | 10-50ms |
| Data Required | ~100 samples/class | 1000+ samples/class |
| CPU Efficiency | Excellent | Moderate |
| Interpretability | High (feature importance) | Low |

**Model Configuration:**
- 100 decision trees
- Max depth: 10
- Features: 45 geometric measurements
- Classes: 5 poses

### 5.7 Data Collection

We collect our own training data using the built-in data collector:

**Pose Classes:**
| ID | Pose Name | Description |
|----|-----------|-------------|
| 0 | Laughing | Hands raised, celebratory pose |
| 1 | Yawning | Hands near face, stretching |
| 2 | Crying | Hands covering face |
| 3 | Taunting | Arms crossed or dismissive gesture |
| 4 | Mean Laugh | Pointing/mocking gesture |

**Collection Process:**
1. Run `data_collector.py`
2. Press number keys (0-4) to record poses
3. Collect 50-100 samples per pose
4. Press 't' to train model
5. Model and data automatically saved

### 5.8 Performance Optimization

| Optimization | Impact |
|--------------|--------|
| Resolution reduction (320×240) | ~4x faster processing |
| Frame skipping (process every 2nd frame) | ~2x throughput |
| Model complexity 0 | ~2x faster inference |
| Display scaling (process small, display large) | Maintains usability |
| Single window mode | Reduces rendering overhead |
| Buffer size = 1 | Reduces latency |

### 5.9 Performance Metrics Collection

Our custom metrics system captures:

| Metric Category | Measurements |
|-----------------|--------------|
| **Timing** | Frame time, MediaPipe inference, Classifier inference |
| **Throughput** | FPS (mean, min, max, std) |
| **Latency** | P50, P95, P99 latencies |
| **System** | CPU %, Memory %, Temperature |
| **Classification** | Confidence scores, prediction distribution |

**Output Formats:**
- JSON (raw data for analysis)
- CSV (for plotting)
- Markdown report (for presentation)

## 6. Expected Deliverables

### 6.1 Working Demo System

- [x] Real-time pose detection running on Raspberry Pi 4
- [x] Interactive emote display with 5 pose classes
- [x] Audio feedback for detected poses
- [x] Live FPS and confidence display
- [x] Screenshot capability

### 6.2 GitHub Repository Structure

```
/ai-hardware-project-proposal-visionmasters/
├── src/emote_detector/
│   ├── main.py                 # Main application
│   ├── holistic_detector.py    # MediaPipe wrapper
│   ├── pose_classifier.py      # Random Forest classifier
│   ├── data_collector.py       # Training data collection
│   ├── train_model.py          # Model training & evaluation
│   ├── performance_metrics.py  # Metrics collection system
│   ├── pose_classifier_model.pkl  # Trained model
│   ├── emotes/                 # Emote images and sounds
│   ├── pose_data/              # Collected training data
│   └── results/                # Charts and metrics
├── docs/
│   └── Project_Proposal.md     # This document
├── presentations/              # Presentation slides
├── report/                     # Final report
├── requirements.txt            # Python dependencies
└── README.md                   # Setup instructions
```

### 6.3 Trained Models & Data

- [x] Self-collected pose dataset (~500 samples)
- [x] Trained Random Forest classifier
- [x] Feature extraction pipeline
- [x] Model evaluation metrics

### 6.4 Benchmark Results

- [ ] Latency breakdown by component
- [ ] FPS measurements under different configurations
- [ ] CPU/Memory utilization profiles
- [ ] Thermal behavior over time
- [ ] Comparison charts (resolution, complexity, skip frames)

### 6.5 Technical Documentation

- [x] System architecture diagram
- [x] Data collection instructions
- [x] Model training pipeline
- [ ] Deployment guide for Raspberry Pi
- [ ] Performance optimization guide

### 6.6 Presentations

- **Midterm:** Problem definition, approach, initial implementation
- **Final:** Live demo, performance analysis, lessons learned

### 6.7 Final Report

- Introduction and motivation
- Background on edge AI and pose estimation
- System design and implementation
- Performance analysis and optimization
- Hardware utilization insights
- Challenges and solutions
- Conclusions and future work

## 7. Team Responsibilities

| Name | Role | Responsibilities |
|------|------|------------------|
| Marvin Rivera | Team Lead | Coordination, documentation, presentation |
| Allen Chen | Hardware | RPi setup, camera integration, deployment |
| Sami Kang | Software | Model training, optimization, metrics |
| ALL | Evaluation | Testing, benchmarking, data collection |

## 8. Timeline and Milestones

| Week | Date | Task | Deliverable |
|------|------|------|-------------|
| 1 | Nov 5 | **Project Proposal** | Submit proposal to Canvas and GitHub |
| 2 | Nov 12 | Environment Setup | RPi configured, MediaPipe running |
| 3 | Nov 19 | Data Collection & Training | Pose dataset, trained classifier |
| 4 | Nov 19 | **Midterm Presentation** | Demo basic pose detection |
| 5 | Dec 3 | Integration & Optimization | Full pipeline working, performance tuning |
| 6 | Dec 10 | Benchmarking & Analysis | Complete metrics, performance charts |
| 7 | Dec 17 | **Final Presentation** | Live demo, final report, all deliverables |

## 9. Resources Required

### Hardware (Provided by Course)

- Raspberry Pi 4 Model B (4GB RAM) — 1 unit
- MicroSD card (64GB) — 1 unit

### Hardware (Team Provided)

- Logitech Brio 100 USB Webcam — ~$30
- HDMI monitor (personal)
- USB-C power supply

### Software (Free/Open Source)

| Software | Purpose |
|----------|---------|
| MediaPipe | Pose detection (TFLite) |
| scikit-learn | Random Forest classifier |
| OpenCV | Camera/image processing |
| pygame | Audio playback |
| NumPy | Numerical computation |
| matplotlib/seaborn | Visualization |
| psutil | System monitoring |

### Compute

- Personal laptops for development
- Raspberry Pi 4 for deployment/testing

## 10. Key Differences from Original Proposal

| Aspect | Original Plan | Actual Implementation |
|--------|---------------|----------------------|
| **Hardware** | RPi4 + Coral TPU | RPi4 only (CPU) |
| **Task** | Facial expression recognition | Pose-based gesture recognition |
| **Dataset** | FER2013 (35K images) | Self-collected (~500 samples) |
| **Model** | MobileNetV2 (CNN) | Random Forest (classical ML) |
| **Features** | Raw pixels | Engineered geometric features |
| **Quantization** | INT8 for Edge TPU | Not needed (RF is efficient) |
| **Target FPS** | 30+ FPS | 10-15 FPS |
| **Complexity** | Deep learning pipeline | Lightweight ML pipeline |

**Rationale for Changes:**
1. Coral TPU was removed to simplify deployment and focus on CPU optimization
2. Pose-based approach is more robust and requires less training data
3. Self-collected data demonstrates the full ML pipeline for educational purposes
4. Random Forest provides interpretable results and fast inference

## 11. References

1. MediaPipe Holistic Documentation - https://google.github.io/mediapipe/solutions/holistic
2. scikit-learn Random Forest - https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
3. Raspberry Pi 4 Specifications - https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/
4. TensorFlow Lite for Microcontrollers - https://www.tensorflow.org/lite/microcontrollers
5. Real-time Pose Estimation on Edge Devices - https://arxiv.org/abs/2006.10204
6. Original Clash Royale Emote Detector (Reference) - https://github.com/example/clash-royale-emote-detector
