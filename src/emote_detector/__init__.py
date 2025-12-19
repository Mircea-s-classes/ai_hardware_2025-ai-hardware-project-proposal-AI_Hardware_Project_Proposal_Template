"""
Clash Royale Emote Detector
Pose-based gesture recognition using MediaPipe and Random Forest
"""

from .holistic_detector import HolisticDetector
from .pose_classifier import PoseClassifier

__all__ = ['HolisticDetector', 'PoseClassifier']

