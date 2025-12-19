"""
MediaPipe Holistic Detector for Pose/Gesture Recognition
Based on clash-royale-emote-detector, adapted for RPi4
"""

import cv2
import mediapipe as mp
import numpy as np


class HolisticDetector:
    """MediaPipe Holistic detector for body pose, face, and hand landmarks"""
    
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Holistic detector
        
        Args:
            model_complexity: 0, 1, or 2. Higher = more accurate but slower.
                             Use 0 for RPi4 to maximize FPS.
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize holistic model
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        print(f"✅ HolisticDetector initialized (complexity={model_complexity})")
    
    def detect(self, frame):
        """
        Detect holistic landmarks in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            results: MediaPipe holistic results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.holistic.process(rgb_frame)
        
        return results
    
    def draw_landmarks(self, frame, results, draw_pose=True, draw_face=True, 
                       draw_hands=True):
        """
        Draw landmarks on the frame
        
        Args:
            frame: Input frame
            results: MediaPipe holistic results
            draw_pose: Draw body pose landmarks
            draw_face: Draw face landmarks
            draw_hands: Draw hand landmarks
            
        Returns:
            frame: Frame with landmarks drawn
        """
        if draw_pose and results.pose_landmarks:
            # Pose landmarks (green)
            pose_connections = self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=3)
            pose_landmarks = self.mp_drawing.DrawingSpec(
                color=(0, 0, 255), thickness=5, circle_radius=3)
            
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_landmarks,
                connection_drawing_spec=pose_connections
            )
        
        if draw_face and results.face_landmarks:
            # Just draw key face points (not all 468)
            face_landmarks = results.face_landmarks.landmark
            key_points = [1, 13, 14, 17, 18]  # Nose tip, mouth corners
            
            h, w = frame.shape[:2]
            for point_idx in key_points:
                if point_idx < len(face_landmarks):
                    point = face_landmarks[point_idx]
                    x, y = int(point.x * w), int(point.y * h)
                    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        
        if draw_hands:
            # Hand landmarks (blue/yellow)
            hand_connections = self.mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2)
            hand_landmarks = self.mp_drawing.DrawingSpec(
                color=(255, 255, 0), thickness=3, circle_radius=2)
            
            # Left hand
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    hand_landmarks,
                    hand_connections
                )
            
            # Right hand
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    hand_landmarks,
                    hand_connections
                )
        
        return frame
    
    def get_landmark_data(self, results):
        """
        Extract landmark coordinates from results
        
        Args:
            results: MediaPipe holistic results
            
        Returns:
            dict: Dictionary containing landmark arrays
        """
        landmark_data = {
            'pose': None,       # 33 landmarks (x, y, z)
            'face': None,       # 468 landmarks (x, y, z)
            'left_hand': None,  # 21 landmarks (x, y, z)
            'right_hand': None  # 21 landmarks (x, y, z)
        }
        
        if results.pose_landmarks:
            landmark_data['pose'] = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.pose_landmarks.landmark
            ])
        
        if results.face_landmarks:
            landmark_data['face'] = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.face_landmarks.landmark
            ])
        
        if results.left_hand_landmarks:
            landmark_data['left_hand'] = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.left_hand_landmarks.landmark
            ])
        
        if results.right_hand_landmarks:
            landmark_data['right_hand'] = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.right_hand_landmarks.landmark
            ])
        
        return landmark_data
    
    def release(self):
        """Release resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()


def test_detector():
    """Test the holistic detector with webcam"""
    print("Testing Holistic Detector...")
    print("Press 'q' to quit")
    
    detector = HolisticDetector(model_complexity=1)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect landmarks
        results = detector.detect(frame)
        
        # Draw landmarks
        frame = detector.draw_landmarks(frame, results)
        
        # Get landmark data
        data = detector.get_landmark_data(results)
        
        # Show status
        status = []
        if data['pose'] is not None:
            status.append("Pose: ✓")
        if data['left_hand'] is not None:
            status.append("L-Hand: ✓")
        if data['right_hand'] is not None:
            status.append("R-Hand: ✓")
        
        status_text = " | ".join(status) if status else "No detection"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Holistic Detector Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    print("✅ Test complete!")


if __name__ == '__main__':
    test_detector()

