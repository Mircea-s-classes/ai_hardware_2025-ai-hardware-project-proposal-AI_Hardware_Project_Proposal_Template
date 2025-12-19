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
            collect_metrics=False, headless=False):
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
        
        # Only create emote window if not in fast mode or headless
        if not fast_mode and not headless:
            cv2.namedWindow('Emote', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Emote', 200, 200)
        
        if headless:
            print("🚀 Headless mode: Display disabled, pure AI performance test")
            print("   Will run for 60 seconds. Press Ctrl+C to stop early.")
        
        frame_count = 0
        last_pose_name = "No Pose"
        last_confidence = 0.0
        last_all_confidences = {}
        
        # Headless mode timer
        headless_start_time = time.time() if headless else None
        headless_duration = 60  # seconds
        
        try:
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
                    # Resize frame for MediaPipe inference if needed
                    process_frame = frame
                    if fast_mode:
                        # Ensure frame is actually 320x240 for MediaPipe
                        if frame.shape[0] != 240 or frame.shape[1] != 320:
                            process_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
                    
                    # Debug: Print frame shape on first inference
                    if frame_count == skip_frames:
                        print(f"🔍 [DEBUG] Camera frame shape: {frame.shape}, MediaPipe process frame: {process_frame.shape}")
                    
                    # Detect landmarks (MediaPipe inference)
                    if self.metrics:
                        self.metrics.start_mediapipe()
                    results = self.detector.detect(process_frame)
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
                
                # Show emote in separate window (skip in fast mode or headless)
                if not fast_mode and not headless:
                    self._show_emote_window(pose_name)
                
                # Display main frame (skip if headless)
                if not headless:
                    cv2.imshow('Clash Royale Emote Detector', frame)
                
                # End frame timing
                if self.metrics:
                    self.metrics.end_frame()
                
                # Handle keyboard or auto-quit in headless mode
                if headless:
                    # Check time limit
                    if time.time() - headless_start_time >= headless_duration:
                        print(f"\n⏱️  {headless_duration} seconds elapsed, stopping...")
                        break
                    # Print progress every 10 seconds
                    elapsed = time.time() - headless_start_time
                    if frame_count % 100 == 0:
                        print(f"⏱️  Running... {elapsed:.0f}s / {headless_duration}s (Frames: {frame_count})", end='\r', flush=True)
                    key = 0xFF  # No keyboard input
                else:
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
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        finally:
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
    
    def _draw_rounded_rect(self, frame, pt1, pt2, color, thickness=-1, radius=10, alpha=1.0):
        """Draw a rounded rectangle with optional transparency"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        if alpha < 1.0:
            overlay = frame.copy()
        else:
            overlay = frame
        
        # Clamp radius
        radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        
        if thickness == -1:
            # Filled rounded rectangle
            cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            # Border only
            cv2.line(overlay, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(overlay, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(overlay, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(overlay, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
            cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def _draw_gradient_bar(self, frame, x, y, width, height, fill_ratio, base_color, glow=False):
        """Draw a modern gradient progress bar"""
        # Background track
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 50), -1)
        
        fill_width = int(width * fill_ratio)
        if fill_width > 0:
            # Create gradient effect by drawing multiple rectangles
            for i in range(fill_width):
                ratio = i / max(fill_width, 1)
                # Lighten color towards the end
                color = tuple(min(255, int(c * (0.7 + 0.3 * ratio))) for c in base_color)
                cv2.line(frame, (x + i, y), (x + i, y + height), color, 1)
            
            # Add highlight on top
            highlight_y = y + 1
            cv2.line(frame, (x, highlight_y), (x + fill_width, highlight_y), 
                    tuple(min(255, c + 40) for c in base_color), 1)
    
    def _draw_ui(self, frame, pose_name, confidence, all_confidences, 
                 fps, show_fps, show_confidence):
        """Draw modern, professional UI overlay - scales with frame size"""
        h, w = frame.shape[:2]
        
        # Scale factor based on frame width (1.0 at 1280px width)
        scale = w / 1280.0
        
        # Color palette - modern dark theme with gold/amber accents
        DARK_BG = (18, 18, 24)
        PANEL_BG = (28, 28, 38)
        ACCENT_GOLD = (0, 180, 255)  # Gold/amber in BGR
        ACCENT_CYAN = (220, 180, 0)  # Cyan accent
        TEXT_PRIMARY = (255, 255, 255)
        TEXT_SECONDARY = (160, 160, 170)
        TEXT_MUTED = (100, 100, 110)
        
        # Confidence-based accent colors
        if confidence > 0.7:
            accent_color = (50, 205, 50)  # Lime green
            status_text = "HIGH CONFIDENCE"
        elif confidence > 0.4:
            accent_color = (0, 200, 255)  # Amber/gold
            status_text = "MEDIUM"
        else:
            accent_color = (180, 130, 70)  # Steel blue
            status_text = "DETECTING..."
        
        # ═══════════════════════════════════════════════════════════
        # HEADER SECTION - Clean minimal header
        # ═══════════════════════════════════════════════════════════
        header_height = int(110 * scale)
        
        # Dark header background with subtle gradient effect
        for i in range(header_height):
            alpha = 1.0 - (i / header_height) * 0.15
            color = tuple(int(c * alpha) for c in DARK_BG)
            cv2.line(frame, (0, i), (w, i), color, 1)
        
        # Thin accent line at bottom of header
        cv2.line(frame, (0, header_height - 1), (w, header_height - 1), ACCENT_GOLD, 3)
        
        # App title - clean typography
        title_y = int(55 * scale)
        cv2.putText(frame, "EMOTE", (int(30 * scale), title_y),
                   cv2.FONT_HERSHEY_DUPLEX, 1.6 * scale, TEXT_PRIMARY, int(2 * max(1, scale)), cv2.LINE_AA)
        cv2.putText(frame, "DETECTOR", (int(175 * scale), title_y),
                   cv2.FONT_HERSHEY_DUPLEX, 1.6 * scale, ACCENT_GOLD, int(2 * max(1, scale)), cv2.LINE_AA)
        
        # Subtitle
        cv2.putText(frame, "CLASH ROYALE  •  REAL-TIME AI", (int(32 * scale), int(85 * scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, TEXT_MUTED, 1, cv2.LINE_AA)
        
        # ═══════════════════════════════════════════════════════════
        # MAIN DETECTION CARD - Shows current pose
        # ═══════════════════════════════════════════════════════════
        card_x = int(25 * scale)
        card_y = int(130 * scale)
        card_width = int(480 * scale)
        card_height = int(70 * scale)
        
        # Card background with transparency effect
        self._draw_rounded_rect(frame, (card_x, card_y), 
                               (card_x + card_width, card_y + card_height),
                               PANEL_BG, -1, radius=int(12 * scale), alpha=0.85)
        
        # Left accent bar
        cv2.rectangle(frame, (card_x, card_y + int(8 * scale)), 
                     (card_x + int(6 * scale), card_y + card_height - int(8 * scale)),
                     accent_color, -1)
        
        # Pose name - larger, bolder
        display_name = pose_name if pose_name != "No Pose" else "Analyzing..."
        cv2.putText(frame, display_name, (card_x + int(22 * scale), card_y + int(47 * scale)),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2 * scale, accent_color, int(2 * max(1, scale)), cv2.LINE_AA)
        
        # Confidence percentage - right aligned
        conf_text = f"{confidence:.0%}"
        (text_w, _), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_DUPLEX, 1.3 * scale, 2)
        cv2.putText(frame, conf_text, (card_x + card_width - text_w - int(20 * scale), card_y + int(48 * scale)),
                   cv2.FONT_HERSHEY_DUPLEX, 1.3 * scale, TEXT_PRIMARY, int(2 * max(1, scale)), cv2.LINE_AA)
        
        # ═══════════════════════════════════════════════════════════
        # FPS INDICATOR - Minimal, top-right
        # ═══════════════════════════════════════════════════════════
        if show_fps:
            fps_width = int(120 * scale)
            fps_height = int(70 * scale)
            fps_x = w - fps_width - int(25 * scale)
            fps_y = int(25 * scale)
            
            # FPS card
            self._draw_rounded_rect(frame, (fps_x, fps_y), 
                                   (fps_x + fps_width, fps_y + fps_height),
                                   PANEL_BG, -1, radius=int(10 * scale), alpha=0.9)
            
            # Border accent
            self._draw_rounded_rect(frame, (fps_x, fps_y), 
                                   (fps_x + fps_width, fps_y + fps_height),
                                   ACCENT_GOLD, 2, radius=int(10 * scale))
            
            # FPS label
            cv2.putText(frame, "FPS", (fps_x + int(18 * scale), fps_y + int(25 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, TEXT_MUTED, 1, cv2.LINE_AA)
            
            # FPS value with color coding
            if fps > 20:
                fps_color = (100, 220, 100)
            elif fps > 12:
                fps_color = (0, 200, 255)
            else:
                fps_color = (100, 100, 220)
            
            cv2.putText(frame, f"{fps:.1f}", (fps_x + int(18 * scale), fps_y + int(55 * scale)),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0 * scale, fps_color, int(2 * max(1, scale)), cv2.LINE_AA)
        
        # ═══════════════════════════════════════════════════════════
        # CONFIDENCE SIDEBAR - Right side panel
        # ═══════════════════════════════════════════════════════════
        if show_confidence and all_confidences:
            sidebar_width = int(250 * scale)
            sidebar_padding = int(18 * scale)
            sidebar_x = w - sidebar_width - int(25 * scale)
            sidebar_y = int(220 * scale)
            
            # Count visible items
            visible_items = [(p, c) for p, c in all_confidences.items() if c > 0.03]
            visible_items = sorted(visible_items, key=lambda x: x[1], reverse=True)[:5]
            
            item_height = int(50 * scale)
            sidebar_height = len(visible_items) * item_height + int(55 * scale)
            
            # Sidebar background
            self._draw_rounded_rect(frame, (sidebar_x, sidebar_y),
                                   (sidebar_x + sidebar_width, sidebar_y + sidebar_height),
                                   PANEL_BG, -1, radius=int(12 * scale), alpha=0.88)
            
            # Sidebar header
            cv2.putText(frame, "CONFIDENCE", (sidebar_x + sidebar_padding, sidebar_y + int(30 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, TEXT_MUTED, 1, cv2.LINE_AA)
            
            # Separator line
            cv2.line(frame, (sidebar_x + sidebar_padding, sidebar_y + int(42 * scale)),
                    (sidebar_x + sidebar_width - sidebar_padding, sidebar_y + int(42 * scale)),
                    (50, 50, 60), 1)
            
            # Confidence bars
            y_offset = sidebar_y + int(65 * scale)
            for pose, conf in visible_items:
                is_active = pose == pose_name
                
                # Pose label
                label_color = accent_color if is_active else TEXT_SECONDARY
                cv2.putText(frame, pose, (sidebar_x + sidebar_padding, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, label_color, 1, cv2.LINE_AA)
                
                # Percentage
                pct_text = f"{conf:.0%}"
                (pct_w, _), _ = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, 1)
                cv2.putText(frame, pct_text, 
                           (sidebar_x + sidebar_width - sidebar_padding - pct_w, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, TEXT_MUTED, 1, cv2.LINE_AA)
                
                # Progress bar
                bar_x = sidebar_x + sidebar_padding
                bar_y = y_offset + int(10 * scale)
                bar_width = sidebar_width - sidebar_padding * 2
                bar_height = int(10 * scale)
                
                bar_color = accent_color if is_active else (80, 80, 90)
                self._draw_gradient_bar(frame, bar_x, bar_y, bar_width, bar_height, conf, bar_color)
                
                y_offset += item_height
        
        # ═══════════════════════════════════════════════════════════
        # EMOTE DISPLAY - Bottom right with glow effect
        # ═══════════════════════════════════════════════════════════
        if pose_name in self.reference_images:
            emote_size = int(180 * scale)
            emote_margin = int(30 * scale)
            emote_x = w - emote_size - emote_margin
            emote_y = h - emote_size - int(90 * scale)
            
            # Glow/shadow effect
            glow_padding = int(10 * scale)
            self._draw_rounded_rect(frame, 
                                   (emote_x - glow_padding, emote_y - glow_padding),
                                   (emote_x + emote_size + glow_padding, emote_y + emote_size + glow_padding),
                                   (15, 15, 20), -1, radius=int(15 * scale))
            
            # Border
            self._draw_rounded_rect(frame,
                                   (emote_x - 4, emote_y - 4),
                                   (emote_x + emote_size + 4, emote_y + emote_size + 4),
                                   accent_color, 3, radius=int(12 * scale))
            
            # Emote image
            frame = self._overlay_emote(frame, pose_name,
                                       position=(emote_x, emote_y),
                                       size=(emote_size, emote_size))
        
        # ═══════════════════════════════════════════════════════════
        # FOOTER - Clean control bar
        # ═══════════════════════════════════════════════════════════
        footer_height = int(65 * scale)
        footer_y = h - footer_height
        
        # Footer background
        for i in range(footer_height):
            alpha = 0.85 + (i / footer_height) * 0.15
            color = tuple(int(c * alpha) for c in DARK_BG)
            cv2.line(frame, (0, footer_y + i), (w, footer_y + i), color, 1)
        
        # Top accent line
        cv2.line(frame, (0, footer_y), (w, footer_y), ACCENT_GOLD, 2)
        
        # Control buttons - modern pill style
        controls = [
            ("Q", "Quit", (100, 100, 220)),
            ("S", "Save", (100, 180, 100)),
            ("M", "Metrics", (180, 140, 80))
        ]
        
        x_offset = int(25 * scale)
        button_height = int(38 * scale)
        button_y = footer_y + (footer_height - button_height) // 2
        
        for key, action, color in controls:
            # Calculate button width based on text
            (action_w, _), _ = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, 1)
            button_width = int(50 * scale) + action_w + int(25 * scale)
            
            # Button background
            self._draw_rounded_rect(frame, (x_offset, button_y),
                                   (x_offset + button_width, button_y + button_height),
                                   (45, 45, 55), -1, radius=int(8 * scale))
            
            # Key badge
            key_size = int(28 * scale)
            key_x = x_offset + int(8 * scale)
            key_y_center = button_y + button_height // 2
            self._draw_rounded_rect(frame, (key_x, key_y_center - key_size//2),
                                   (key_x + key_size, key_y_center + key_size//2),
                                   color, -1, radius=int(5 * scale))
            
            # Key letter
            cv2.putText(frame, key, (key_x + int(7 * scale), key_y_center + int(6 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Action text
            cv2.putText(frame, action, (key_x + key_size + int(10 * scale), key_y_center + int(6 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, TEXT_SECONDARY, 1, cv2.LINE_AA)
            
            x_offset += button_width + int(15 * scale)
    
    def _show_emote_window(self, pose_name):
        """Show emote in separate window with modern styling"""
        window_size = 220
        display = np.zeros((window_size, window_size, 3), dtype=np.uint8)
        
        # Dark gradient background
        for i in range(window_size):
            intensity = 20 + int(10 * (i / window_size))
            cv2.line(display, (0, i), (window_size, i), (intensity, intensity, intensity + 5), 1)
        
        if pose_name in self.reference_images:
            emote = self.reference_images[pose_name]
            emote_size = 180
            margin = (window_size - emote_size) // 2
            
            # Emote glow effect
            glow_color = (30, 35, 45)
            cv2.rectangle(display, (margin - 5, margin - 5), 
                         (margin + emote_size + 5, margin + emote_size + 5), 
                         glow_color, -1)
            
            # Border
            cv2.rectangle(display, (margin - 2, margin - 2),
                         (margin + emote_size + 2, margin + emote_size + 2),
                         (0, 180, 255), 2)
            
            # Resize and place emote
            emote_resized = cv2.resize(emote, (emote_size, emote_size))
            if emote_resized.shape[2] == 4:
                # Handle alpha channel
                alpha = emote_resized[:, :, 3] / 255.0
                for c in range(3):
                    display[margin:margin+emote_size, margin:margin+emote_size, c] = (
                        alpha * emote_resized[:, :, c] +
                        (1 - alpha) * display[margin:margin+emote_size, margin:margin+emote_size, c]
                    )
            else:
                display[margin:margin+emote_size, margin:margin+emote_size] = emote_resized[:, :, :3]
        else:
            # No pose detected - show placeholder
            cv2.putText(display, "Waiting...", (55, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 110), 1, cv2.LINE_AA)
            # Animated dots effect (simple version)
            cv2.circle(display, (85, 130), 4, (60, 60, 70), -1)
            cv2.circle(display, (110, 130), 4, (80, 80, 90), -1)
            cv2.circle(display, (135, 130), 4, (100, 100, 110), -1)
        
        cv2.imshow('Emote', display)


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
                       help='Fast mode: 320x240, skip=3, complexity=0')
    parser.add_argument('--ultra-fast', action='store_true',
                       help='Ultra-fast mode for RPi: 160x120, skip=4, complexity=0')
    parser.add_argument('--resolution', type=str, default='high',
                       choices=['low', 'medium', 'high'],
                       help='Resolution: low=320x240, medium=640x480, high=1280x720')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth frame (1=all, 2=every other)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Display scale factor (2.0 = 2x larger window)')
    parser.add_argument('--metrics', action='store_true',
                       help='Enable performance metrics collection (saved to results/metrics/)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without display (for pure performance testing)')
    
    args = parser.parse_args()
    
    # Resolution presets
    resolutions = {
        'low': (320, 240),
        'medium': (640, 480),
        'high': (1280, 720)
    }
    resolution = resolutions[args.resolution]
    
    # Fast mode defaults
    if args.ultra_fast:
        resolution = (160, 120)
        skip_frames = 4
        complexity = 0
        display_scale = 4.0  # Scale up to 640x480 for display
        print("⚡⚡ ULTRA-FAST mode enabled: process 160x120, display 640x480, skip=4, complexity=0")
    elif args.fast:
        resolution = (320, 240)
        skip_frames = 3  # Increased from 2 to 3 for RPi
        complexity = 0
        display_scale = 2.0  # Scale up to 640x480 for display
        print("⚡ Fast mode enabled: process 320x240, display 640x480, skip=3, complexity=0")
    else:
        skip_frames = args.skip
        complexity = args.complexity
        display_scale = args.scale
    
    # Create and run detector
    fast_mode_enabled = args.fast or args.ultra_fast
    detector = EmoteDetector(
        model_path=args.model,
        emotes_dir=args.emotes,
        model_complexity=complexity
    )
    
    detector.run(
        camera_index=args.camera,
        show_fps=not args.no_fps,
        show_confidence=not args.no_confidence,
        fast_mode=fast_mode_enabled,
        resolution=resolution,
        skip_frames=skip_frames,
        display_scale=display_scale,
        collect_metrics=args.metrics,
        headless=args.headless
    )


if __name__ == "__main__":
    main()

