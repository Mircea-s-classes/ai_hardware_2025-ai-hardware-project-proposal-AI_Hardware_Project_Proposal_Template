"""
Performance Metrics Collector for AI Hardware Project
Measures and records performance metrics for edge AI deployment on Raspberry Pi 4
"""

import time
import json
import csv
import os
from datetime import datetime
from pathlib import Path
from collections import deque
import numpy as np

# Optional imports for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - CPU/memory metrics disabled")


class PerformanceMetrics:
    """Collects and analyzes performance metrics for edge AI inference"""
    
    def __init__(self, output_dir="results/metrics", buffer_size=1000):
        """
        Initialize metrics collector
        
        Args:
            output_dir: Directory to save metrics
            buffer_size: Number of samples to keep in memory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timing buffers (in milliseconds)
        self.frame_times = deque(maxlen=buffer_size)
        self.mediapipe_times = deque(maxlen=buffer_size)
        self.classifier_times = deque(maxlen=buffer_size)
        self.total_inference_times = deque(maxlen=buffer_size)
        self.fps_history = deque(maxlen=buffer_size)
        
        # System metrics
        self.cpu_usage = deque(maxlen=buffer_size)
        self.memory_usage = deque(maxlen=buffer_size)
        self.temperature = deque(maxlen=buffer_size)
        
        # Classification metrics
        self.predictions = []
        self.confidences = deque(maxlen=buffer_size)
        
        # Session info
        self.start_time = None
        self.end_time = None
        self.total_frames = 0
        self.config = {}
        
        # Timing helpers
        self._frame_start = None
        self._mediapipe_start = None
        self._classifier_start = None
        
    def start_session(self, config=None):
        """Start a new metrics collection session"""
        self.start_time = datetime.now()
        self.config = config or {}
        self.total_frames = 0
        print(f"📊 Metrics collection started at {self.start_time.strftime('%H:%M:%S')}")
        
    def end_session(self):
        """End the current session and save metrics"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"📊 Metrics collection ended. Duration: {duration:.1f}s, Frames: {self.total_frames}")
        
    # === Timing Methods ===
    
    def start_frame(self):
        """Mark the start of frame processing"""
        self._frame_start = time.perf_counter()
        self.total_frames += 1
        
    def end_frame(self):
        """Mark the end of frame processing"""
        if self._frame_start:
            elapsed_ms = (time.perf_counter() - self._frame_start) * 1000
            self.frame_times.append(elapsed_ms)
            if elapsed_ms > 0:
                self.fps_history.append(1000 / elapsed_ms)
            
    def start_mediapipe(self):
        """Mark the start of MediaPipe inference"""
        self._mediapipe_start = time.perf_counter()
        
    def end_mediapipe(self):
        """Mark the end of MediaPipe inference"""
        if self._mediapipe_start:
            elapsed_ms = (time.perf_counter() - self._mediapipe_start) * 1000
            self.mediapipe_times.append(elapsed_ms)
            self._mediapipe_start = None  # Reset for next frame
            
    def start_classifier(self):
        """Mark the start of classifier inference"""
        self._classifier_start = time.perf_counter()
        
    def end_classifier(self):
        """Mark the end of classifier inference"""
        if self._classifier_start:
            elapsed_ms = (time.perf_counter() - self._classifier_start) * 1000
            self.classifier_times.append(elapsed_ms)
            self._classifier_start = None  # Reset for next frame
            
    def record_inference(self, mediapipe_ms, classifier_ms):
        """Record inference times directly"""
        self.mediapipe_times.append(mediapipe_ms)
        self.classifier_times.append(classifier_ms)
        self.total_inference_times.append(mediapipe_ms + classifier_ms)
        
    def record_prediction(self, pose_name, confidence):
        """Record a classification prediction"""
        self.predictions.append({
            'timestamp': time.time(),
            'pose': pose_name,
            'confidence': confidence
        })
        self.confidences.append(confidence)
        
    # === System Metrics ===
    
    def collect_system_metrics(self):
        """Collect current system metrics (CPU, memory, temperature)"""
        if PSUTIL_AVAILABLE:
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().percent)
        
        # RPi temperature (Linux-specific)
        temp = self._get_rpi_temperature()
        if temp:
            self.temperature.append(temp)
            
    def _get_rpi_temperature(self):
        """Get Raspberry Pi CPU temperature"""
        try:
            # Try RPi-specific path
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000
                return temp
        except:
            return None
            
    # === Statistics ===
    
    def get_stats(self):
        """Get current performance statistics"""
        stats = {
            'total_frames': self.total_frames,
            'session_duration_s': None,
        }
        
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            stats['session_duration_s'] = duration
            stats['average_fps_overall'] = self.total_frames / duration if duration > 0 else 0
        
        # Timing stats (convert to numpy for calculations)
        if self.frame_times:
            frame_arr = np.array(self.frame_times)
            stats['frame_time_ms'] = {
                'mean': float(np.mean(frame_arr)),
                'std': float(np.std(frame_arr)),
                'min': float(np.min(frame_arr)),
                'max': float(np.max(frame_arr)),
                'p50': float(np.percentile(frame_arr, 50)),
                'p95': float(np.percentile(frame_arr, 95)),
                'p99': float(np.percentile(frame_arr, 99)),
            }
            
        if self.mediapipe_times:
            mp_arr = np.array(list(self.mediapipe_times))  # Convert deque to list first
            stats['mediapipe_inference_ms'] = {
                'mean': float(np.mean(mp_arr)),
                'std': float(np.std(mp_arr)),
                'min': float(np.min(mp_arr)),
                'max': float(np.max(mp_arr)),
                'p50': float(np.percentile(mp_arr, 50)),
                'p95': float(np.percentile(mp_arr, 95)),
            }
            
        if self.classifier_times:
            cls_arr = np.array(self.classifier_times)
            stats['classifier_inference_ms'] = {
                'mean': float(np.mean(cls_arr)),
                'std': float(np.std(cls_arr)),
                'min': float(np.min(cls_arr)),
                'max': float(np.max(cls_arr)),
            }
            
        if self.fps_history:
            fps_arr = np.array(self.fps_history)
            stats['fps'] = {
                'mean': float(np.mean(fps_arr)),
                'std': float(np.std(fps_arr)),
                'min': float(np.min(fps_arr)),
                'max': float(np.max(fps_arr)),
            }
            
        # System stats
        if self.cpu_usage:
            stats['cpu_percent'] = {
                'mean': float(np.mean(self.cpu_usage)),
                'max': float(np.max(self.cpu_usage)),
            }
            
        if self.memory_usage:
            stats['memory_percent'] = {
                'mean': float(np.mean(self.memory_usage)),
                'max': float(np.max(self.memory_usage)),
            }
            
        if self.temperature:
            stats['temperature_celsius'] = {
                'mean': float(np.mean(self.temperature)),
                'max': float(np.max(self.temperature)),
            }
            
        # Classification stats
        if self.confidences:
            stats['confidence'] = {
                'mean': float(np.mean(self.confidences)),
                'std': float(np.std(self.confidences)),
            }
            
        return stats
    
    def print_summary(self):
        """Print a formatted summary of metrics"""
        stats = self.get_stats()
        
        # Debug: Show collected sample counts
        print(f"\n[DEBUG] Collected samples: MediaPipe={len(self.mediapipe_times)}, Classifier={len(self.classifier_times)}, Frames={len(self.frame_times)}")
        if self.mediapipe_times:
            import numpy as np
            mp_vals = list(self.mediapipe_times)
            print(f"[DEBUG] MediaPipe sample values (first 5): {mp_vals[:5]}")
            print(f"[DEBUG] MediaPipe mean: {np.mean(mp_vals):.2f}ms")
        print(f"[DEBUG] Stats keys: {list(stats.keys())}")
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE METRICS SUMMARY - AI Hardware Project")
        print("="*60)
        
        print(f"\n🎯 Session Info:")
        print(f"   Total Frames: {stats['total_frames']}")
        if stats.get('session_duration_s'):
            print(f"   Duration: {stats['session_duration_s']:.1f} seconds")
            print(f"   Overall FPS: {stats.get('average_fps_overall', 0):.1f}")
        
        if 'fps' in stats:
            print(f"\n⚡ Frame Rate:")
            print(f"   Mean FPS: {stats['fps']['mean']:.1f}")
            print(f"   Min/Max: {stats['fps']['min']:.1f} / {stats['fps']['max']:.1f}")
            
        if 'frame_time_ms' in stats:
            print(f"\n⏱️ Frame Processing Time:")
            print(f"   Mean: {stats['frame_time_ms']['mean']:.2f} ms")
            print(f"   P50/P95/P99: {stats['frame_time_ms']['p50']:.2f} / {stats['frame_time_ms']['p95']:.2f} / {stats['frame_time_ms']['p99']:.2f} ms")
            
        print(f"[DEBUG] Checking MediaPipe: 'mediapipe_inference_ms' in stats = {'mediapipe_inference_ms' in stats}")
        if 'mediapipe_inference_ms' in stats:
            print(f"\n🧠 MediaPipe Inference (Pose Detection):")
            print(f"   Mean: {stats['mediapipe_inference_ms']['mean']:.2f} ms")
            print(f"   P50/P95: {stats['mediapipe_inference_ms']['p50']:.2f} / {stats['mediapipe_inference_ms']['p95']:.2f} ms")
        else:
            print(f"[DEBUG] MediaPipe stats NOT found, but data exists!")
            
        if 'classifier_inference_ms' in stats:
            print(f"\n🌲 Random Forest Classifier:")
            print(f"   Mean: {stats['classifier_inference_ms']['mean']:.3f} ms")
            
        if 'cpu_percent' in stats:
            print(f"\n💻 System Resources:")
            print(f"   CPU Usage: {stats['cpu_percent']['mean']:.1f}% (max: {stats['cpu_percent']['max']:.1f}%)")
            
        if 'memory_percent' in stats:
            print(f"   Memory Usage: {stats['memory_percent']['mean']:.1f}%")
            
        if 'temperature_celsius' in stats:
            print(f"   🌡️ Temperature: {stats['temperature_celsius']['mean']:.1f}°C (max: {stats['temperature_celsius']['max']:.1f}°C)")
            
        if 'confidence' in stats:
            print(f"\n🎯 Classification Confidence:")
            print(f"   Mean: {stats['confidence']['mean']:.2f}")
            
        print("\n" + "="*60)
        
    # === Save/Export ===
    
    def save_metrics(self, filename=None):
        """Save all metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
            
        filepath = self.output_dir / filename
        
        data = {
            'session': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'total_frames': self.total_frames,
                'config': self.config,
            },
            'statistics': self.get_stats(),
            'raw_data': {
                'frame_times_ms': list(self.frame_times),
                'mediapipe_times_ms': list(self.mediapipe_times),
                'classifier_times_ms': list(self.classifier_times),
                'fps_history': list(self.fps_history),
                'cpu_usage': list(self.cpu_usage),
                'memory_usage': list(self.memory_usage),
                'temperature': list(self.temperature),
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"📁 Metrics saved to: {filepath}")
        return filepath
        
    def save_csv(self, filename=None):
        """Save timing data to CSV for easy plotting"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.csv"
            
        filepath = self.output_dir / filename
        
        # Align all arrays to same length
        max_len = max(
            len(self.frame_times),
            len(self.mediapipe_times),
            len(self.classifier_times),
            len(self.fps_history),
            len(self.cpu_usage),
            len(self.temperature)
        )
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'frame_time_ms', 'mediapipe_ms', 'classifier_ms', 
                           'fps', 'cpu_percent', 'temperature_c'])
            
            for i in range(max_len):
                row = [
                    i,
                    self.frame_times[i] if i < len(self.frame_times) else '',
                    self.mediapipe_times[i] if i < len(self.mediapipe_times) else '',
                    self.classifier_times[i] if i < len(self.classifier_times) else '',
                    self.fps_history[i] if i < len(self.fps_history) else '',
                    self.cpu_usage[i] if i < len(self.cpu_usage) else '',
                    self.temperature[i] if i < len(self.temperature) else '',
                ]
                writer.writerow(row)
                
        print(f"📁 CSV saved to: {filepath}")
        return filepath


def generate_report(metrics_file, output_file=None):
    """Generate a markdown report from saved metrics"""
    with open(metrics_file, 'r') as f:
        data = json.load(f)
        
    stats = data['statistics']
    config = data['session'].get('config', {})
    
    if output_file is None:
        output_file = Path(metrics_file).with_suffix('.md')
        
    report = f"""# AI Hardware Performance Report
## Clash Royale Emote Detector on Raspberry Pi 4

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Model Complexity | {config.get('complexity', 'N/A')} |
| Resolution | {config.get('resolution', 'N/A')} |
| Skip Frames | {config.get('skip_frames', 'N/A')} |
| Fast Mode | {config.get('fast_mode', 'N/A')} |

---

## Performance Summary

### Throughput
- **Total Frames Processed:** {stats['total_frames']}
- **Session Duration:** {stats.get('session_duration_s', 0):.1f} seconds
- **Average FPS:** {stats.get('average_fps_overall', 0):.1f}

### Latency (Frame Processing)
| Metric | Value |
|--------|-------|
| Mean | {stats.get('frame_time_ms', {}).get('mean', 0):.2f} ms |
| P50 (Median) | {stats.get('frame_time_ms', {}).get('p50', 0):.2f} ms |
| P95 | {stats.get('frame_time_ms', {}).get('p95', 0):.2f} ms |
| P99 | {stats.get('frame_time_ms', {}).get('p99', 0):.2f} ms |
| Min | {stats.get('frame_time_ms', {}).get('min', 0):.2f} ms |
| Max | {stats.get('frame_time_ms', {}).get('max', 0):.2f} ms |

### Inference Breakdown
| Component | Mean Latency |
|-----------|-------------|
| MediaPipe (TFLite) | {stats.get('mediapipe_inference_ms', {}).get('mean', 0):.2f} ms |
| Random Forest Classifier | {stats.get('classifier_inference_ms', {}).get('mean', 0):.3f} ms |

### System Resources
| Metric | Mean | Max |
|--------|------|-----|
| CPU Usage | {stats.get('cpu_percent', {}).get('mean', 0):.1f}% | {stats.get('cpu_percent', {}).get('max', 0):.1f}% |
| Memory Usage | {stats.get('memory_percent', {}).get('mean', 0):.1f}% | - |
| Temperature | {stats.get('temperature_celsius', {}).get('mean', 0):.1f}°C | {stats.get('temperature_celsius', {}).get('max', 0):.1f}°C |

---

## Hardware Platform

- **Device:** Raspberry Pi 4 Model B
- **CPU:** Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5GHz
- **RAM:** 4GB/8GB LPDDR4
- **Accelerator:** None (CPU-only inference)
- **ML Framework:** MediaPipe (TensorFlow Lite backend)
- **Classifier:** scikit-learn Random Forest

---

## Key Findings

1. **Real-time Performance:** {'Achieved' if stats.get('average_fps_overall', 0) >= 10 else 'Below target'} ({stats.get('average_fps_overall', 0):.1f} FPS vs 10 FPS target)
2. **Inference Bottleneck:** MediaPipe pose detection dominates inference time
3. **Thermal:** {'Stable' if stats.get('temperature_celsius', {}).get('max', 0) < 70 else 'Throttling risk'} (max {stats.get('temperature_celsius', {}).get('max', 0):.1f}°C)

---

*Report generated by AI Hardware Project - VisionMasters Team*
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
        
    print(f"📄 Report saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Test the metrics collector
    metrics = PerformanceMetrics()
    metrics.start_session({'test': True})
    
    # Simulate some frames
    for i in range(100):
        metrics.start_frame()
        metrics.start_mediapipe()
        time.sleep(0.02)  # Simulate 20ms inference
        metrics.end_mediapipe()
        metrics.start_classifier()
        time.sleep(0.001)  # Simulate 1ms classifier
        metrics.end_classifier()
        metrics.collect_system_metrics()
        metrics.record_prediction("Laughing", 0.85)
        metrics.end_frame()
        
    metrics.end_session()
    metrics.print_summary()
    metrics.save_metrics()
    metrics.save_csv()

