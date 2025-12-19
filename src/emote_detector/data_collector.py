"""
Data Collection Tool for Training Custom Poses
Collect samples of yourself doing different poses/gestures
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

from holistic_detector import HolisticDetector
from pose_classifier import PoseClassifier


class PoseDataCollector:
    """Tool for collecting training data for pose classification"""
    
    def __init__(self, data_dir="pose_data"):
        """
        Initialize data collector
        
        Args:
            data_dir: Directory to save collected data
        """
        self.detector = HolisticDetector(model_complexity=1)
        self.classifier = PoseClassifier()
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Collection settings
        self.current_pose = 0
        self.collected_samples = 0
        self.samples_per_pose = 100  # Collect 100 samples per pose
        self.auto_collect = False
        self.collection_delay = 5  # Collect every N frames
        self.frame_counter = 0
        
        # Pose labels - CUSTOMIZE THESE FOR YOUR PROJECT!
        self.pose_labels = {
            0: "Laughing",    # Hands on waist or hips
            1: "Yawning",     # Hand(s) covering mouth
            2: "Crying",      # Hands covering face/eyes
            3: "Taunting",    # Fists near face, taunting pose
        }
        
        # Collected data
        self.collected_data = []
        self.collected_labels = []
        
        self._print_instructions()
    
    def _print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("📸 Pose Data Collector")
        print("="*60)
        print("\nCollect training data for your custom poses!")
        print("\nControls:")
        for idx, name in self.pose_labels.items():
            print(f"  '{idx}' - Select pose: {name}")
        print(f"  'a' - Toggle auto-collection (ON/OFF)")
        print(f"  's' - Save collected data")
        print(f"  't' - Train model with collected data")
        print(f"  'l' - Load previously saved data")
        print(f"  'c' - Clear current collection")
        print(f"  'q' - Quit")
        print("\n💡 Tips:")
        print(f"   - Collect at least {self.samples_per_pose} samples per pose")
        print(f"   - Vary your position slightly while collecting")
        print(f"   - Make sure your full upper body is visible")
        print("="*60 + "\n")
    
    def collect_data(self):
        """Main data collection loop"""
        print("📷 Initializing camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Trying camera 1...")
            cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("❌ Could not open camera!")
            return
        
        print("✅ Camera ready!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            self.frame_counter += 1
            
            # Detect landmarks
            results = self.detector.detect(frame)
            
            # Draw landmarks
            frame = self.detector.draw_landmarks(frame, results)
            
            # Get pose data
            landmark_data = self.detector.get_landmark_data(results)
            pose_landmarks = landmark_data.get('pose')
            
            # Auto-collect if enabled
            if (self.auto_collect and 
                pose_landmarks is not None and 
                self.frame_counter % self.collection_delay == 0):
                
                # Extract features and store
                features = self.classifier.extract_features(pose_landmarks)
                self.collected_data.append(features)
                self.collected_labels.append(self.current_pose)
                self.collected_samples += 1
                
                print(f"  ✓ Sample {self.collected_samples} for {self.pose_labels[self.current_pose]}")
                
                # Auto-advance to next pose if enough samples
                if self.collected_samples >= self.samples_per_pose:
                    print(f"\n✅ Completed {self.samples_per_pose} samples for {self.pose_labels[self.current_pose]}")
                    self.collected_samples = 0
                    self.current_pose = (self.current_pose + 1) % len(self.pose_labels)
                    
                    if self.current_pose == 0:
                        print("\n🎉 All poses collected! Press 't' to train or keep collecting.")
                        self.auto_collect = False
                    else:
                        print(f"📍 Now collecting: {self.pose_labels[self.current_pose]}")
            
            # Draw UI
            self._draw_ui(frame, pose_landmarks)
            
            cv2.imshow('Pose Data Collector', frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key >= ord('0') and key <= ord('9'):
                pose_idx = key - ord('0')
                if pose_idx in self.pose_labels:
                    self.current_pose = pose_idx
                    self.collected_samples = 0
                    print(f"📍 Switched to: {self.pose_labels[self.current_pose]}")
            elif key == ord('a'):
                self.auto_collect = not self.auto_collect
                status = "ON 🟢" if self.auto_collect else "OFF 🔴"
                print(f"Auto-collection: {status}")
            elif key == ord('s'):
                self._save_data()
            elif key == ord('l'):
                self._load_data()
            elif key == ord('t'):
                self._train_model()
            elif key == ord('c'):
                self.collected_data = []
                self.collected_labels = []
                self.collected_samples = 0
                print("🗑️ Cleared all collected data")
        
        cap.release()
        cv2.destroyAllWindows()
        self.detector.release()
    
    def _draw_ui(self, frame, pose_landmarks):
        """Draw modern UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # Modern header bar (dark blue-gray)
        header_height = 100
        cv2.rectangle(frame, (0, 0), (w, header_height), (35, 35, 45), -1)
        
        # Accent line
        cv2.rectangle(frame, (0, header_height-3), (w, header_height), (0, 140, 255), -1)
        
        # Title with modern font
        cv2.putText(frame, "POSE DATA COLLECTOR", (20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Current pose indicator box
        pose_name = self.pose_labels[self.current_pose]
        cv2.rectangle(frame, (20, 50), (300, 88), (50, 50, 60), -1)
        cv2.rectangle(frame, (20, 50), (300, 88), (0, 200, 100), 2)
        cv2.putText(frame, f"Recording: {pose_name}", (30, 73),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2, cv2.LINE_AA)
        
        # Sample counter box
        total = len(self.collected_data)
        cv2.rectangle(frame, (320, 50), (520, 88), (50, 50, 60), -1)
        cv2.rectangle(frame, (320, 50), (520, 88), (0, 140, 255), 2)
        cv2.putText(frame, f"{self.collected_samples}/{self.samples_per_pose}", (335, 73),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        
        # Total samples
        cv2.putText(frame, f"Total: {total}", (445, 73),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Status indicators in top right
        status_x = w - 180
        
        # Auto-collect badge
        if self.auto_collect:
            cv2.rectangle(frame, (status_x, 25), (status_x + 80, 55), (0, 60, 60), -1)
            cv2.rectangle(frame, (status_x, 25), (status_x + 80, 55), (0, 255, 255), 2)
            cv2.putText(frame, "AUTO", (status_x + 15, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (status_x, 25), (status_x + 80, 55), (40, 40, 50), -1)
            cv2.rectangle(frame, (status_x, 25), (status_x + 80, 55), (100, 100, 120), 2)
            cv2.putText(frame, "MANUAL", (status_x + 8, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Detection indicator
        status_x += 95
        if pose_landmarks is not None:
            cv2.rectangle(frame, (status_x, 25), (status_x + 70, 55), (0, 60, 0), -1)
            cv2.rectangle(frame, (status_x, 25), (status_x + 70, 55), (0, 255, 0), 2)
            cv2.putText(frame, "READY", (status_x + 8, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (status_x, 25), (status_x + 70, 55), (60, 0, 0), -1)
            cv2.rectangle(frame, (status_x, 25), (status_x + 70, 55), (200, 0, 0), 2)
            cv2.putText(frame, "NO POSE", (status_x + 2, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1, cv2.LINE_AA)
        
        # Modern footer with controls
        footer_height = 50
        cv2.rectangle(frame, (0, h - footer_height), (w, h), (35, 35, 45), -1)
        cv2.rectangle(frame, (0, h - footer_height), (w, h - footer_height + 2), (0, 140, 255), -1)
        
        # Control instructions
        controls = [
            ("0-3", "Pose"),
            ("SPACE", "Capture"),
            ("A", "Auto"),
            ("S", "Save"),
            ("T", "Train"),
            ("Q", "Quit")
        ]
        
        x_offset = 15
        for key, action in controls:
            # Key button
            key_width = len(key) * 11 + 16
            cv2.rectangle(frame, (x_offset, h - 38), (x_offset + key_width, h - 15),
                         (60, 60, 70), -1)
            cv2.rectangle(frame, (x_offset, h - 38), (x_offset + key_width, h - 15),
                         (100, 100, 120), 1)
            
            cv2.putText(frame, key, (x_offset + 8, h - 23),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Action label
            cv2.putText(frame, action, (x_offset + key_width + 6, h - 23),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)
            
            x_offset += key_width + len(action) * 6 + 20
        
        # Recording indicator when auto-collecting
        if self.auto_collect and pose_landmarks is not None:
            cv2.circle(frame, (w - 25, h - 25), 10, (0, 255, 0), -1)
            cv2.circle(frame, (w - 25, h - 25), 12, (0, 200, 0), 2)
    
    def _save_data(self):
        """Save collected data to files"""
        if len(self.collected_data) == 0:
            print("⚠️ No data to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save arrays
        features_file = self.data_dir / f"pose_features_{timestamp}.npy"
        labels_file = self.data_dir / f"pose_labels_{timestamp}.npy"
        
        np.save(features_file, np.array(self.collected_data))
        np.save(labels_file, np.array(self.collected_labels))
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "num_samples": len(self.collected_data),
            "pose_labels": self.pose_labels,
            "samples_per_pose": self.samples_per_pose,
            "collection_date": datetime.now().isoformat()
        }
        
        metadata_file = self.data_dir / f"pose_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save as "latest" for easy loading
        np.save(self.data_dir / "pose_features_latest.npy", np.array(self.collected_data))
        np.save(self.data_dir / "pose_labels_latest.npy", np.array(self.collected_labels))
        
        print(f"\n✅ Data saved!")
        print(f"   Samples: {len(self.collected_data)}")
        print(f"   Location: {self.data_dir}")
    
    def _load_data(self):
        """Load previously saved data"""
        features_file = self.data_dir / "pose_features_latest.npy"
        labels_file = self.data_dir / "pose_labels_latest.npy"
        
        if not features_file.exists() or not labels_file.exists():
            print("⚠️ No saved data found!")
            return
        
        try:
            self.collected_data = np.load(features_file).tolist()
            self.collected_labels = np.load(labels_file).tolist()
            print(f"✅ Loaded {len(self.collected_data)} samples")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def _train_model(self):
        """Train the classifier with collected data"""
        if len(self.collected_data) < 20:
            print(f"⚠️ Need at least 20 samples to train (have {len(self.collected_data)})")
            return
        
        # Auto-save data before training
        print("\n💾 Auto-saving data before training...")
        self._save_data()
        
        X = np.array(self.collected_data)
        y = np.array(self.collected_labels)
        
        # Update classifier labels
        self.classifier.set_pose_labels(self.pose_labels)
        
        # Train
        print(f"\n🚀 Training with {len(X)} samples...")
        accuracy = self.classifier.train_model(X, y)
        
        print(f"\n✅ Model trained and saved!")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"\n📊 To generate charts, run: python train_model.py")


if __name__ == "__main__":
    collector = PoseDataCollector()
    collector.collect_data()

