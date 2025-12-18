"""
Pose Classifier using Random Forest
Extracts geometric features from MediaPipe landmarks
Based on clash-royale-emote-detector
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


class PoseClassifier:
    """Classifier for pose/gesture recognition using extracted features"""
    
    def __init__(self, model_path=None):
        """
        Initialize pose classifier
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        
        # Default pose labels (customize these!)
        self.pose_labels = {
            0: "Laughing",   # Hands on waist, mouth open
            1: "Yawning",    # Hands over mouth
            2: "Crying",     # Hands covering face
            3: "Taunting",   # Fists close to face
            4: "Neutral"     # Default pose
        }
        
        self.model_path = model_path or "pose_classifier_model.pkl"
        
        # Load existing model if available
        if os.path.exists(self.model_path):
            self.load_model()
    
    def extract_features(self, pose_landmarks):
        """
        Extract geometric features from pose landmarks
        
        These features capture body pose characteristics:
        - Distances between body parts
        - Angles at joints
        - Relative positions
        
        Args:
            pose_landmarks: Array of pose landmarks (33, 3)
            
        Returns:
            features: Feature vector (18 dimensions - matches pre-trained model)
        """
        if pose_landmarks is None:
            return np.zeros(18)  # Match pre-trained model's feature count
        
        landmarks = np.array(pose_landmarks)
        features = []
        
        # Key landmark indices in MediaPipe Pose:
        # 0: Nose
        # 11, 12: Left/Right shoulder
        # 13, 14: Left/Right elbow
        # 15, 16: Left/Right wrist
        # 23, 24: Left/Right hip
        # 25, 26: Left/Right knee
        # 27, 28: Left/Right ankle
        
        # Get key landmarks
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Feature 1: Shoulder width
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        features.append(shoulder_width)
        
        # Feature 2: Hip width
        hip_width = np.linalg.norm(left_hip - right_hip)
        features.append(hip_width)
        
        # Feature 3: Body height (shoulder to hip)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        body_height = np.linalg.norm(shoulder_center - hip_center)
        features.append(body_height)
        
        # Feature 4-5: Arm angles
        left_arm_angle = self._calculate_angle(left_elbow, left_shoulder, left_wrist)
        right_arm_angle = self._calculate_angle(right_elbow, right_shoulder, right_wrist)
        features.extend([left_arm_angle, right_arm_angle])
        
        # Feature 6-7: Hand heights relative to shoulders (important for gestures!)
        left_hand_height = left_shoulder[1] - left_wrist[1]
        right_hand_height = right_shoulder[1] - right_wrist[1]
        features.extend([left_hand_height, right_hand_height])
        
        # Feature 8-9: Knee angles
        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        features.extend([left_knee_angle, right_knee_angle])
        
        # Feature 10: Torso orientation
        torso_angle = self._calculate_angle(left_shoulder, hip_center, right_hip)
        features.append(torso_angle)
        
        # Feature 11-12: Arm extension (shoulder to wrist distance)
        left_arm_extension = np.linalg.norm(left_shoulder - left_wrist)
        right_arm_extension = np.linalg.norm(right_shoulder - right_wrist)
        features.extend([left_arm_extension, right_arm_extension])
        
        # Feature 13-14: Leg angles
        left_leg_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        features.extend([left_leg_angle, right_leg_angle])
        
        # Feature 15: Body symmetry
        left_side = np.mean([left_shoulder, left_elbow, left_wrist, left_hip, left_knee], axis=0)
        right_side = np.mean([right_shoulder, right_elbow, right_wrist, right_hip, right_knee], axis=0)
        body_symmetry = np.linalg.norm(left_side - right_side)
        features.append(body_symmetry)
        
        # Feature 16-17: Hand offset from center (important for detecting hands near face)
        center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        left_hand_offset = abs(left_wrist[0] - center_x)
        right_hand_offset = abs(right_wrist[0] - center_x)
        features.extend([left_hand_offset, right_hand_offset])
        
        # Feature 18: Vertical alignment (head position)
        vertical_alignment = abs(nose[0] - center_x)
        features.append(vertical_alignment)
        
        # Note: The pre-trained model uses 18 features, so we stop here
        # If you retrain, you can add more features like hand-to-face distance
        
        return np.array(features)
    
    def _calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points
        
        Args:
            point1, point2, point3: Three 3D points
            
        Returns:
            angle: Angle in degrees
        """
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def train_model(self, X, y, test_size=0.2):
        """
        Train the pose classification model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            test_size: Fraction of data for testing
        """
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n📊 Training Random Forest Classifier...")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Classes: {len(np.unique(y))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train Random Forest (ultra-light for RPi inference)
        self.model = RandomForestClassifier(
            n_estimators=10,  # Ultra-light for RPi inference
            max_depth=4,      # Reduced for faster inference
            random_state=42,
            n_jobs=1  # Use single core for better inference performance
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model trained!")
        print(f"   Accuracy: {accuracy:.2%}")
        
        # Classification report
        unique_labels = np.unique(y)
        target_names = [self.pose_labels.get(label, f"Class_{label}") 
                       for label in unique_labels]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   labels=unique_labels,
                                   target_names=target_names))
        
        # Save model
        self.save_model()
        
        return accuracy
    
    def predict(self, pose_landmarks):
        """
        Predict pose from landmarks
        
        Args:
            pose_landmarks: Pose landmarks array (33, 3)
            
        Returns:
            prediction: Predicted pose label (string)
            confidence: Prediction confidence (0-1)
        """
        if self.model is None:
            return "No Model", 0.0
        
        features = self.extract_features(pose_landmarks)
        features = features.reshape(1, -1)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        pose_name = self.pose_labels.get(prediction, "Unknown")
        
        return pose_name, confidence
    
    def get_all_confidences(self, pose_landmarks):
        """
        Get confidence scores for all pose classes
        
        Args:
            pose_landmarks: Pose landmarks array
            
        Returns:
            confidences: Dictionary of {pose_name: confidence}
        """
        if self.model is None:
            return {}
        
        features = self.extract_features(pose_landmarks)
        features = features.reshape(1, -1)
        
        probabilities = self.model.predict_proba(features)[0]
        
        confidences = {}
        for i, prob in enumerate(probabilities):
            pose_name = self.pose_labels.get(self.model.classes_[i], f"Class_{i}")
            confidences[pose_name] = prob
        
        return confidences
    
    def save_model(self, path=None):
        """Save the trained model"""
        path = path or self.model_path
        if self.model is not None:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'pose_labels': self.pose_labels
                }, f)
            print(f"✅ Model saved to {path}")
    
    def load_model(self, path=None):
        """Load a saved model"""
        path = path or self.model_path
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.model = data['model']
                    self.pose_labels = data.get('pose_labels', self.pose_labels)
                else:
                    # Backward compatibility
                    self.model = data
            print(f"✅ Model loaded from {path}")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            self.model = None
    
    def set_pose_labels(self, labels):
        """
        Set custom pose labels
        
        Args:
            labels: Dictionary mapping class index to label name
        """
        self.pose_labels = labels
        print(f"✅ Pose labels updated: {labels}")

