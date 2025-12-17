"""
Clash Royale Emote Detector - Main Application
Real-time pose detection with emote display

Based on clash-royale-emote-detector, adapted for RPi4 deployment
"""

import cv2
import os
import time
import threading
import argparse
from pathlib import Path
import numpy as np

from holistic_detector import HolisticDetector
from pose_classifier import PoseClassifier
from performance_metrics import PerformanceMetrics

# Try to import pygame for audio
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(8)
    AUDIO_AVAILABLE = True
    print("✅ Audio support enabled")
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ Audio disabled (install pygame for sound)")


class EmoteDetector:
    """Main application for pose-based emote detection"""
    
    def __init__(self, model_path=None, emotes_dir=None, model_complexity=1):
        """
        Initialize emote detector
        
        Args:
            model_path: Path to trained pose classifier model
            emotes_dir: Directory containing emote images and sounds
            model_complexity: MediaPipe complexity (0=fast, 1=balanced, 2=accurate)
        """
        # Initialize detector and classifier
        self.detector = HolisticDetector(model_complexity=model_complexity)
        self.classifier = PoseClassifier(model_path=model_path)
        
        # Load emote assets
        self.emotes_dir = Path(emotes_dir) if emotes_dir else Path("emotes")
        self.reference_images = self._load_emote_images()
        self.sounds = self._load_sounds() if AUDIO_AVAILABLE else {}
        
        # Sound cooldown
        self.last_sound_time = 0
        self.sound_cooldown = 0.5  # seconds
        
        # FPS tracking
        self.fps_history = []
        self.last_frame_time = time.time()
        
        print(f"\n✅ EmoteDetector initialized")
        print(f"   Model complexity: {model_complexity}")
        print(f"   Emotes loaded: {len(self.reference_images)}")
    
    def _load_emote_images(self):
        """Load emote reference images"""
        images = {}
        images_dir = self.emotes_dir / "images"
        
        if not images_dir.exists():
            print(f"⚠️ Emotes directory not found: {images_dir}")
            return images
        
        # Map pose names to image files
        pose_images = {
            "Laughing": "laughing.png",
            "Yawning": "yawning.png",
            "Crying": "crying.png",
            "Taunting": "taunting.png",
            "Neutral": "neutral.png"
        }
        
        for pose_name, filename in pose_images.items():
            image_path = images_dir / filename
            if image_path.exists():
                img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    images[pose_name] = img
                    print(f"   Loaded emote: {pose_name}")
        
        return images
    
    def _load_sounds(self):
        """Load emote sound files"""
        sounds = {}
        sounds_dir = self.emotes_dir / "sounds"
        
        if not sounds_dir.exists():
            return sounds
        
        pose_sounds = {
            "Laughing": "laughing.mp3",
            "Yawning": "yawning.mp3",
            "Crying": "crying.mp3",
            "Taunting": "taunting.mp3"
        }
        
        for pose_name, filename in pose_sounds.items():
            sound_path = sounds_dir / filename
            if sound_path.exists():
                try:
                    sounds[pose_name] = pygame.mixer.Sound(str(sound_path))
                    print(f"   Loaded sound: {pose_name}")
                except Exception as e:
                    print(f"   ⚠️ Could not load sound {filename}: {e}")
        
        return sounds
    
    def _play_sound(self, pose_name):
        """Play sound for detected pose (with cooldown)"""
        if not AUDIO_AVAILABLE:
            return
        
        current_time = time.time()
        if current_time - self.last_sound_time < self.sound_cooldown:
            return
        
        if pose_name in self.sounds:
            try:
                self.sounds[pose_name].play()
                self.last_sound_time = current_time
            except Exception as e:
                print(f"⚠️ Error playing sound: {e}")
    
    def _overlay_emote(self, frame, pose_name, position=(10, 10), size=(120, 120)):
        """Overlay emote image on frame"""
        if pose_name not in self.reference_images:
            return frame
        
        emote = self.reference_images[pose_name]
        emote = cv2.resize(emote, size)
        
        x, y = position
        h, w = emote.shape[:2]
        
        # Handle transparency if PNG with alpha
        if emote.shape[2] == 4:
            alpha = emote[:, :, 3] / 255.0
            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    alpha * emote[:, :, c] +
                    (1 - alpha) * frame[y:y+h, x:x+w, c]
                )
        else:
            frame[y:y+h, x:x+w] = emote[:, :, :3]
        
        return frame
    
    def _calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if delta > 0:
            fps = 1.0 / delta
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
        
        return np.mean(self.fps_history) if self.fps_history else 0
    
    def run(self, camera_index=0, show_fps=True, show_confidence=True, 
            fast_mode=False, resolution=(640, 480), skip_frames=1, display_scale=1.0,
            collect_metrics=False):
        """
        Run the emote detector
        
        Args:
            camera_index: Camera device index
            show_fps: Display FPS counter
            show_confidence: Show confidence for all poses
            fast_mode: Enable performance optimizations
            resolution: Frame resolution (width, height)
            skip_frames: Process every Nth frame (1=all, 2=every other, etc)
            display_scale: Scale factor for display window (2.0 = 2x larger)
            collect_metrics: Enable performance metrics collection
        """
        print(f"\n🎮 Starting Emote Detector...")
        print(f"Press 'q' to quit, 's' to save screenshot")
        
        # Initialize metrics collector if enabled
        self.metrics = None
        if collect_metrics:
            self.metrics = PerformanceMetrics()
            self.metrics.start_session({
                'complexity': self.detector.model_complexity if hasattr(self.detector, 'model_complexity') else 'unknown',
                'resolution': f"{resolution[0]}x{resolution[1]}",
                'skip_frames': skip_frames,
                'fast_mode': fast_mode,
                'display_scale': display_scale,
            })
            print("📊 Performance metrics collection ENABLED")
        
        # Try multiple camera indices
        cap = None
        for idx in [camera_index, 0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"✅ Using camera index {idx}")
                break
            cap.release()
        
        if cap is None or not cap.isOpened():
            print(f"❌ Could not open any camera")
            return
        
        # Set camera resolution
        width, height = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Calculate display size
        display_width = int(width * display_scale)
        display_height = int(height * display_scale)
        
        print(f"✅ Camera ready! Process: {width}x{height}, Display: {display_width}x{display_height}")
        if fast_mode:
            print(f"⚡ Fast mode: skip_frames={skip_frames}")
        
        # Only create emote window if not in fast mode
        if not fast_mode:
            cv2.namedWindow('Emote', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Emote', 200, 200)
        
        frame_count = 0
        last_pose_name = "No Pose"
        last_confidence = 0.0
        last_all_confidences = {}
        
        while True:
            # Start frame timing
            if self.metrics:
                self.metrics.start_frame()
                
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % skip_frames == 0:
                # Detect landmarks (MediaPipe inference)
                if self.metrics:
                    self.metrics.start_mediapipe()
                results = self.detector.detect(frame)
                if self.metrics:
                    self.metrics.end_mediapipe()
                
                # Draw landmarks
                frame = self.detector.draw_landmarks(frame, results)
                
                # Get pose data and classify
                landmark_data = self.detector.get_landmark_data(results)
                pose_landmarks = landmark_data.get('pose')
                
                pose_name = "No Pose"
                confidence = 0.0
                all_confidences = {}
                
                if pose_landmarks is not None:
                    # Classifier inference
                    if self.metrics:
                        self.metrics.start_classifier()
                    pose_name, confidence = self.classifier.predict(pose_landmarks)
                    all_confidences = self.classifier.get_all_confidences(pose_landmarks)
                    if self.metrics:
                        self.metrics.end_classifier()
                        self.metrics.record_prediction(pose_name, confidence)
                    
                    # Play sound
                    if pose_name != "No Pose" and confidence > 0.5:
                        self._play_sound(pose_name)
                
                last_pose_name = pose_name
                last_confidence = confidence
                last_all_confidences = all_confidences
                
                # Collect system metrics periodically
                if self.metrics and frame_count % 10 == 0:
                    self.metrics.collect_system_metrics()
            else:
                # Use cached results, still draw landmarks if available
                pose_name = last_pose_name
                confidence = last_confidence
                all_confidences = last_all_confidences
            
            # Calculate FPS
            fps = self._calculate_fps()
            
            # Scale up frame for display if needed
            if display_scale != 1.0:
                frame = cv2.resize(frame, (display_width, display_height), 
                                  interpolation=cv2.INTER_LINEAR)
            
            # Draw UI (after scaling so text is readable)
            self._draw_ui(frame, pose_name, confidence, all_confidences, 
                         fps, show_fps, show_confidence)
            
            # Show emote in separate window (skip in fast mode)
            if not fast_mode:
                self._show_emote_window(pose_name)
            
            # Display main frame
            cv2.imshow('Clash Royale Emote Detector', frame)
            
            # End frame timing
            if self.metrics:
                self.metrics.end_frame()
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('m') and self.metrics:
                # Print live metrics summary
                self.metrics.print_summary()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.detector.release()
        
        # Save metrics if enabled
        if self.metrics:
            self.metrics.end_session()
            self.metrics.print_summary()
            json_file = self.metrics.save_metrics()
            self.metrics.save_csv()
            # Generate report
            from performance_metrics import generate_report
            generate_report(json_file)
        
        print("\n✅ Detector stopped")
    
    def _draw_ui(self, frame, pose_name, confidence, all_confidences, 
                 fps, show_fps, show_confidence):
        """Draw modern UI overlay"""
        h, w = frame.shape[:2]
        
        # Modern header bar
        header_height = 90
        cv2.rectangle(frame, (0, 0), (w, header_height), (35, 35, 45), -1)
        cv2.rectangle(frame, (0, header_height-3), (w, header_height), (0, 140, 255), -1)
        
        # Title with modern styling
        cv2.putText(frame, "EMOTE DETECTOR", (20, 38),
                   cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "CLASH ROYALE", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 120), 1, cv2.LINE_AA)
        
        # Pose prediction box with confidence-based styling
        pose_x, pose_y = 20, 50
        pose_width = 350
        pose_height = 35
        
        # Color coding based on confidence
        if confidence > 0.7:
            accent_color = (0, 255, 100)  # Green
            bg_color = (0, 60, 30)
            status = "HIGH"
        elif confidence > 0.4:
            accent_color = (0, 255, 255)  # Yellow
            bg_color = (60, 60, 0)
            status = "MEDIUM"
        else:
            accent_color = (0, 100, 255)  # Blue
            bg_color = (40, 40, 60)
            status = "LOW"
        
        # Pose box
        cv2.rectangle(frame, (pose_x, pose_y), (pose_x + pose_width, pose_y + pose_height),
                     bg_color, -1)
        cv2.rectangle(frame, (pose_x, pose_y), (pose_x + pose_width, pose_y + pose_height),
                     accent_color, 2)
        
        # Pose name
        cv2.putText(frame, pose_name, (pose_x + 10, pose_y + 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, accent_color, 2, cv2.LINE_AA)
        
        # Confidence badge
        conf_text = f"{confidence:.0%}"
        cv2.putText(frame, conf_text, (pose_x + pose_width - 80, pose_y + 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, accent_color, 2, cv2.LINE_AA)
        
        # FPS counter in top right
        if show_fps:
            fps_x = w - 150
            fps_y = 25
            fps_width = 130
            fps_height = 45
            
            # FPS box
            cv2.rectangle(frame, (fps_x, fps_y), (fps_x + fps_width, fps_y + fps_height),
                         (50, 50, 60), -1)
            cv2.rectangle(frame, (fps_x, fps_y), (fps_x + fps_width, fps_y + fps_height),
                         (0, 200, 255), 2)
            
            # FPS label
            cv2.putText(frame, "FPS", (fps_x + 10, fps_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
            
            # FPS value
            fps_color = (0, 255, 100) if fps > 15 else (0, 200, 255) if fps > 8 else (0, 100, 255)
            cv2.putText(frame, f"{fps:.1f}", (fps_x + 50, fps_y + 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)
        
        # Confidence breakdown sidebar (if enabled)
        if show_confidence and all_confidences:
            sidebar_x = w - 240
            sidebar_y = 105
            sidebar_width = 220
            
            # Sidebar background
            sidebar_height = min(len(all_confidences) * 35 + 40, 200)
            cv2.rectangle(frame, (sidebar_x, sidebar_y),
                         (sidebar_x + sidebar_width, sidebar_y + sidebar_height),
                         (30, 30, 40), -1)
            
            # Sidebar title
            cv2.putText(frame, "CONFIDENCE", (sidebar_x + 15, sidebar_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
            
            # Confidence bars
            y_offset = sidebar_y + 45
            for pose, conf in sorted(all_confidences.items(),
                                    key=lambda x: x[1], reverse=True):
                if conf > 0.05:
                    # Pose name
                    text_color = (0, 255, 150) if pose == pose_name else (180, 180, 180)
                    cv2.putText(frame, pose, (sidebar_x + 15, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
                    
                    # Progress bar
                    bar_x = sidebar_x + 15
                    bar_y = y_offset + 5
                    bar_width = 190
                    bar_height = 8
                    bar_fill = int(bar_width * conf)
                    
                    # Bar background
                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + bar_width, bar_y + bar_height),
                                 (50, 50, 60), -1)
                    
                    # Bar fill
                    if bar_fill > 0:
                        bar_color = accent_color if pose == pose_name else (0, 140, 255)
                        cv2.rectangle(frame, (bar_x, bar_y),
                                     (bar_x + bar_fill, bar_y + bar_height),
                                     bar_color, -1)
                    
                    # Percentage
                    cv2.putText(frame, f"{conf:.0%}", (bar_x + bar_width - 30, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (130, 130, 130), 1, cv2.LINE_AA)
                    
                    y_offset += 30
        
        # Overlay emote on main frame (larger, with shadow)
        if pose_name in self.reference_images:
            emote_size = 160
            emote_x = w - emote_size - 20
            emote_y = h - emote_size - 70
            
            # Shadow effect
            cv2.rectangle(frame, (emote_x - 3, emote_y - 3),
                         (emote_x + emote_size + 3, emote_y + emote_size + 3),
                         (0, 0, 0), -1)
            
            # Emote overlay
            frame = self._overlay_emote(frame, pose_name,
                                       position=(emote_x, emote_y),
                                       size=(emote_size, emote_size))
        
        # Modern footer
        footer_height = 45
        cv2.rectangle(frame, (0, h - footer_height), (w, h), (35, 35, 45), -1)
        cv2.rectangle(frame, (0, h - footer_height), (w, h - footer_height + 2),
                     (0, 140, 255), -1)
        
        # Control buttons
        controls = [
            ("Q", "Quit"),
            ("S", "Save"),
            ("M", "Metrics")
        ]
        
        x_offset = 15
        for key, action in controls:
            key_width = 30
            # Key button
            cv2.rectangle(frame, (x_offset, h - 35), (x_offset + key_width, h - 12),
                         (60, 60, 70), -1)
            cv2.rectangle(frame, (x_offset, h - 35), (x_offset + key_width, h - 12),
                         (100, 100, 120), 1)
            
            cv2.putText(frame, key, (x_offset + 9, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Action label
            cv2.putText(frame, action, (x_offset + key_width + 8, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
            
            x_offset += key_width + len(action) * 7 + 25
    
    def _show_emote_window(self, pose_name):
        """Show emote in separate window"""
        if pose_name in self.reference_images:
            emote = self.reference_images[pose_name]
            emote_display = cv2.resize(emote, (200, 200))
            if emote_display.shape[2] == 4:
                emote_display = emote_display[:, :, :3]
            cv2.imshow('Emote', emote_display)
        else:
            blank = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.putText(blank, "No Pose", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            cv2.imshow('Emote', blank)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Clash Royale Emote Detector')
    parser.add_argument('--model', type=str, default='pose_classifier_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--emotes', type=str, default='emotes',
                       help='Path to emotes directory')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index')
    parser.add_argument('--complexity', type=int, default=1, choices=[0, 1, 2],
                       help='MediaPipe model complexity (0=fast, 1=balanced, 2=accurate)')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS counter')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Hide confidence scores')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: lower resolution, skip frames')
    parser.add_argument('--resolution', type=str, default='medium',
                       choices=['low', 'medium', 'high'],
                       help='Resolution: low=320x240, medium=640x480, high=1280x720')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth frame (1=all, 2=every other)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Display scale factor (2.0 = 2x larger window)')
    parser.add_argument('--metrics', action='store_true',
                       help='Enable performance metrics collection (saved to results/metrics/)')
    
    args = parser.parse_args()
    
    # Resolution presets
    resolutions = {
        'low': (320, 240),
        'medium': (640, 480),
        'high': (1280, 720)
    }
    resolution = resolutions[args.resolution]
    
    # Fast mode defaults
    if args.fast:
        resolution = (320, 240)
        skip_frames = 2
        complexity = 0
        display_scale = 2.0  # Scale up to 640x480 for display
        print("⚡ Fast mode enabled: process 320x240, display 640x480, skip=2, complexity=0")
    else:
        skip_frames = args.skip
        complexity = args.complexity
        display_scale = args.scale
    
    # Create and run detector
    detector = EmoteDetector(
        model_path=args.model,
        emotes_dir=args.emotes,
        model_complexity=complexity if args.fast else args.complexity
    )
    
    detector.run(
        camera_index=args.camera,
        show_fps=not args.no_fps,
        show_confidence=not args.no_confidence,
        fast_mode=args.fast,
        resolution=resolution,
        skip_frames=skip_frames,
        display_scale=display_scale,
        collect_metrics=args.metrics
    )


if __name__ == "__main__":
    main()

