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
            "Mean Laugh": "mean_laugh.png",
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
            "Taunting": "taunting.mp3",
            "Mean Laugh": "mean_laugh.mp3"
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
        """Draw UI overlay"""
        h, w = frame.shape[:2]
        
        # Title
        cv2.putText(frame, "Clash Royale Emote Detector", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Pose prediction with color coding
        if confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif confidence > 0.4:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.putText(frame, f"Pose: {pose_name} ({confidence:.2f})", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # All confidences
        if show_confidence and all_confidences:
            y_offset = 100
            for pose, conf in sorted(all_confidences.items(), 
                                    key=lambda x: x[1], reverse=True):
                if conf > 0.1:
                    cv2.putText(frame, f"  {pose}: {conf:.2f}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20
        
        # FPS
        if show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Overlay emote on main frame
        if pose_name in self.reference_images:
            frame = self._overlay_emote(frame, pose_name, 
                                       position=(w - 140, h - 140),
                                       size=(120, 120))
        
        # Instructions
        cv2.putText(frame, "q=quit, s=save, m=metrics", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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

