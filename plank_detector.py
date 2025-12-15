"""
Plank Detection System using MediaPipe Pose Estimation
======================================================
A production-ready system for detecting and validating plank exercise posture
using computer vision and pose landmark analysis.

Architecture:
- MediaPipe Pose for 33-point body landmark detection
- Geometric angle calculations for posture validation
- Real-time feedback with visual overlays
- Session management with timing and scoring

Key Metrics for Proper Plank:
1. Body alignment (shoulder-hip-ankle should be ~180°)
2. Hip position (not too high or sagging)
3. Arm position (shoulders over wrists)
4. Head alignment (neutral spine)
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlankState(Enum):
    """States of plank detection"""
    NOT_DETECTED = "not_detected"
    POOR_FORM = "poor_form"
    NEEDS_ADJUSTMENT = "needs_adjustment"
    GOOD_FORM = "good_form"
    PERFECT_FORM = "perfect_form"


@dataclass
class PlankConfig:
    """Configuration parameters for plank detection thresholds"""
    # Angle thresholds (in degrees)
    # Body alignment: shoulder-hip-ankle angle (ideal: 180°)
    body_alignment_min: float = 160.0
    body_alignment_max: float = 195.0
    body_alignment_perfect_min: float = 170.0
    body_alignment_perfect_max: float = 190.0

    # Hip angle: prevents sagging or piking (ideal: 180°)
    hip_angle_min: float = 155.0
    hip_angle_max: float = 200.0
    hip_angle_perfect_min: float = 165.0
    hip_angle_perfect_max: float = 195.0

    # Arm angle: elbow should be relatively straight for high plank
    # For forearm plank, this is less relevant
    arm_angle_min: float = 150.0
    arm_angle_max: float = 195.0

    # Knee angle: legs should be straight (ideal: 180°)
    knee_angle_min: float = 160.0
    knee_angle_max: float = 200.0

    # Shoulder-wrist alignment tolerance (normalized units)
    shoulder_wrist_tolerance: float = 0.15

    # Visibility threshold for landmarks
    visibility_threshold: float = 0.5

    # Temporal smoothing (number of frames to average)
    smoothing_window: int = 5

    # Minimum confidence for pose detection
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Model complexity (0=lite, 1=full, 2=heavy)
    model_complexity: int = 1

    # Minimum time in good form to count as valid (seconds)
    min_hold_time: float = 0.5

    # Grace period for brief form breaks (seconds)
    grace_period: float = 0.3


@dataclass
class PlankMetrics:
    """Metrics for current plank form"""
    body_alignment_angle: float = 0.0
    hip_angle: float = 0.0
    left_arm_angle: float = 0.0
    right_arm_angle: float = 0.0
    left_knee_angle: float = 0.0
    right_knee_angle: float = 0.0
    shoulder_wrist_alignment: float = 0.0
    head_alignment: float = 0.0

    # Derived metrics
    overall_score: float = 0.0
    state: PlankState = PlankState.NOT_DETECTED
    feedback: List[str] = field(default_factory=list)

    # Visibility info
    landmarks_visible: bool = False
    visibility_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class PlankSession:
    """Tracks a plank session"""
    start_time: Optional[float] = None
    total_time: float = 0.0
    good_form_time: float = 0.0
    perfect_form_time: float = 0.0
    is_active: bool = False
    last_good_form_time: Optional[float] = None
    tokens_earned: float = 0.0

    # Historical data for this session
    form_history: List[PlankState] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)


class PlankDetector:
    """
    Main class for detecting and analyzing plank exercise form.

    Uses MediaPipe Pose to detect body landmarks and calculates
    geometric angles to determine if the user is in proper plank position.
    """

    # MediaPipe landmark indices for key body parts
    LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_heel': 29,
        'right_heel': 30,
    }

    def __init__(self, config: Optional[PlankConfig] = None):
        """Initialize the plank detector with optional configuration."""
        self.config = config or PlankConfig()

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.config.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

        # Temporal smoothing buffers
        self._angle_buffers: Dict[str, deque] = {
            'body_alignment': deque(maxlen=self.config.smoothing_window),
            'hip': deque(maxlen=self.config.smoothing_window),
            'left_arm': deque(maxlen=self.config.smoothing_window),
            'right_arm': deque(maxlen=self.config.smoothing_window),
            'left_knee': deque(maxlen=self.config.smoothing_window),
            'right_knee': deque(maxlen=self.config.smoothing_window),
        }

        # Current session tracking
        self.session = PlankSession()
        self._last_frame_time: Optional[float] = None

        logger.info("PlankDetector initialized with config: %s", self.config)

    @staticmethod
    def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate the angle ABC where B is the vertex.

        Args:
            a: First point (numpy array [x, y] or [x, y, z])
            b: Middle point / vertex (numpy array)
            c: End point (numpy array)

        Returns:
            Angle in degrees (0-180)
        """
        # Use only x, y for 2D angle calculation
        a = np.array(a[:2])
        b = np.array(b[:2])
        c = np.array(c[:2])

        # Calculate vectors
        ba = a - b
        bc = c - b

        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    def _get_landmark_coords(
        self,
        landmarks,
        idx: int
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract landmark coordinates and visibility.

        Returns:
            Tuple of (coordinates as numpy array, visibility score)
        """
        landmark = landmarks.landmark[idx]
        visibility = landmark.visibility

        if visibility < self.config.visibility_threshold:
            return None, visibility

        return np.array([landmark.x, landmark.y, landmark.z]), visibility

    def _smooth_angle(self, angle: float, buffer_name: str) -> float:
        """Apply temporal smoothing to reduce jitter."""
        self._angle_buffers[buffer_name].append(angle)
        return np.mean(self._angle_buffers[buffer_name])

    def _calculate_body_alignment(
        self,
        shoulder: np.ndarray,
        hip: np.ndarray,
        ankle: np.ndarray
    ) -> float:
        """
        Calculate the body alignment angle.

        For a proper plank, shoulder-hip-ankle should form approximately
        a straight line (180 degrees).
        """
        return self.calculate_angle(shoulder, hip, ankle)

    def _calculate_hip_angle(
        self,
        shoulder: np.ndarray,
        hip: np.ndarray,
        knee: np.ndarray
    ) -> float:
        """
        Calculate the hip angle to detect sagging or piking.

        Should be close to 180 degrees for proper form.
        """
        return self.calculate_angle(shoulder, hip, knee)

    def _check_shoulder_wrist_alignment(
        self,
        shoulder: np.ndarray,
        wrist: np.ndarray
    ) -> float:
        """
        Check if shoulders are stacked over wrists.

        Returns the horizontal distance (smaller is better).
        """
        return abs(shoulder[0] - wrist[0])

    def _evaluate_form(self, metrics: PlankMetrics) -> Tuple[PlankState, float, List[str]]:
        """
        Evaluate the plank form based on calculated metrics.

        Returns:
            Tuple of (state, score 0-100, list of feedback messages)
        """
        feedback = []
        score_components = []

        # Check body alignment (most important - weighted heavily)
        body_score = 0.0
        if self.config.body_alignment_min <= metrics.body_alignment_angle <= self.config.body_alignment_max:
            if self.config.body_alignment_perfect_min <= metrics.body_alignment_angle <= self.config.body_alignment_perfect_max:
                body_score = 100.0
            else:
                # Linear interpolation for good but not perfect
                if metrics.body_alignment_angle < self.config.body_alignment_perfect_min:
                    body_score = 70 + 30 * (metrics.body_alignment_angle - self.config.body_alignment_min) / \
                                 (self.config.body_alignment_perfect_min - self.config.body_alignment_min)
                else:
                    body_score = 70 + 30 * (self.config.body_alignment_max - metrics.body_alignment_angle) / \
                                 (self.config.body_alignment_max - self.config.body_alignment_perfect_max)
        else:
            if metrics.body_alignment_angle < self.config.body_alignment_min:
                feedback.append("Raise your hips - body is sagging")
            else:
                feedback.append("Lower your hips - they're too high")
            body_score = max(0, 40 - abs(180 - metrics.body_alignment_angle))

        score_components.append(('body', body_score, 0.4))

        # Check hip angle
        hip_score = 0.0
        if self.config.hip_angle_min <= metrics.hip_angle <= self.config.hip_angle_max:
            if self.config.hip_angle_perfect_min <= metrics.hip_angle <= self.config.hip_angle_perfect_max:
                hip_score = 100.0
            else:
                hip_score = 75.0
        else:
            if metrics.hip_angle < self.config.hip_angle_min:
                feedback.append("Straighten your core - hips dropping")
            else:
                feedback.append("Lower your hips slightly")
            hip_score = max(0, 40 - abs(180 - metrics.hip_angle))

        score_components.append(('hip', hip_score, 0.3))

        # Check knee angle (legs should be straight)
        avg_knee_angle = (metrics.left_knee_angle + metrics.right_knee_angle) / 2
        knee_score = 0.0
        if avg_knee_angle >= self.config.knee_angle_min:
            knee_score = min(100, 60 + 40 * (avg_knee_angle - self.config.knee_angle_min) / 20)
        else:
            feedback.append("Straighten your legs")
            knee_score = max(0, 40 - abs(180 - avg_knee_angle))

        score_components.append(('knee', knee_score, 0.15))

        # Check arm position (for high plank)
        avg_arm_angle = (metrics.left_arm_angle + metrics.right_arm_angle) / 2
        arm_score = 0.0
        if avg_arm_angle >= self.config.arm_angle_min:
            arm_score = min(100, 60 + 40 * (avg_arm_angle - self.config.arm_angle_min) / 30)
        else:
            # Could be forearm plank - don't penalize too much
            arm_score = 60.0

        score_components.append(('arm', arm_score, 0.15))

        # Calculate weighted overall score
        overall_score = sum(score * weight for _, score, weight in score_components)

        # Determine state based on score
        if overall_score >= 90:
            state = PlankState.PERFECT_FORM
            if not feedback:
                feedback.append("Perfect form! Hold it!")
        elif overall_score >= 75:
            state = PlankState.GOOD_FORM
            if not feedback:
                feedback.append("Good form! Keep going!")
        elif overall_score >= 50:
            state = PlankState.NEEDS_ADJUSTMENT
        else:
            state = PlankState.POOR_FORM

        return state, overall_score, feedback

    def process_frame(
        self,
        frame: np.ndarray,
        draw_overlay: bool = True
    ) -> Tuple[np.ndarray, PlankMetrics]:
        """
        Process a single video frame for plank detection.

        Args:
            frame: BGR image from camera (numpy array)
            draw_overlay: Whether to draw visualization on the frame

        Returns:
            Tuple of (processed frame with overlay, PlankMetrics)
        """
        current_time = time.time()
        metrics = PlankMetrics()

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Run pose detection
        results = self.pose.process(rgb_frame)

        rgb_frame.flags.writeable = True
        output_frame = frame.copy()

        if not results.pose_landmarks:
            metrics.state = PlankState.NOT_DETECTED
            metrics.feedback = ["No person detected - please position yourself in frame"]
            self._update_session(current_time, metrics)

            if draw_overlay:
                self._draw_status(output_frame, metrics)

            return output_frame, metrics

        landmarks = results.pose_landmarks

        # Extract key landmarks
        landmark_coords = {}
        visibility_scores = {}

        for name, idx in self.LANDMARKS.items():
            coords, vis = self._get_landmark_coords(landmarks, idx)
            landmark_coords[name] = coords
            visibility_scores[name] = vis

        metrics.visibility_scores = visibility_scores

        # Check if we have minimum required landmarks visible
        required_landmarks = [
            'left_shoulder', 'right_shoulder',
            'left_hip', 'right_hip',
            'left_ankle', 'right_ankle'
        ]

        missing_landmarks = [
            name for name in required_landmarks
            if landmark_coords.get(name) is None
        ]

        if missing_landmarks:
            metrics.state = PlankState.NOT_DETECTED
            metrics.feedback = [f"Can't see: {', '.join(missing_landmarks)}. Adjust camera angle."]
            metrics.landmarks_visible = False
            self._update_session(current_time, metrics)

            if draw_overlay:
                self._draw_pose(output_frame, landmarks)
                self._draw_status(output_frame, metrics)

            return output_frame, metrics

        metrics.landmarks_visible = True

        # Calculate midpoints for body analysis
        mid_shoulder = (landmark_coords['left_shoulder'] + landmark_coords['right_shoulder']) / 2
        mid_hip = (landmark_coords['left_hip'] + landmark_coords['right_hip']) / 2
        mid_ankle = (landmark_coords['left_ankle'] + landmark_coords['right_ankle']) / 2

        # Calculate body alignment angle (primary metric)
        body_alignment = self._calculate_body_alignment(mid_shoulder, mid_hip, mid_ankle)
        metrics.body_alignment_angle = self._smooth_angle(body_alignment, 'body_alignment')

        # Calculate hip angle
        left_knee = landmark_coords.get('left_knee')
        right_knee = landmark_coords.get('right_knee')

        if left_knee is not None and right_knee is not None:
            mid_knee = (left_knee + right_knee) / 2
            hip_angle = self._calculate_hip_angle(mid_shoulder, mid_hip, mid_knee)
            metrics.hip_angle = self._smooth_angle(hip_angle, 'hip')
        else:
            metrics.hip_angle = 180.0  # Assume good if not visible

        # Calculate arm angles
        left_elbow = landmark_coords.get('left_elbow')
        left_wrist = landmark_coords.get('left_wrist')
        if left_elbow is not None and left_wrist is not None:
            left_arm_angle = self.calculate_angle(
                landmark_coords['left_shoulder'],
                left_elbow,
                left_wrist
            )
            metrics.left_arm_angle = self._smooth_angle(left_arm_angle, 'left_arm')
        else:
            metrics.left_arm_angle = 180.0

        right_elbow = landmark_coords.get('right_elbow')
        right_wrist = landmark_coords.get('right_wrist')
        if right_elbow is not None and right_wrist is not None:
            right_arm_angle = self.calculate_angle(
                landmark_coords['right_shoulder'],
                right_elbow,
                right_wrist
            )
            metrics.right_arm_angle = self._smooth_angle(right_arm_angle, 'right_arm')
        else:
            metrics.right_arm_angle = 180.0

        # Calculate knee angles
        if left_knee is not None:
            left_ankle = landmark_coords.get('left_ankle')
            if left_ankle is not None:
                left_knee_angle = self.calculate_angle(
                    landmark_coords['left_hip'],
                    left_knee,
                    left_ankle
                )
                metrics.left_knee_angle = self._smooth_angle(left_knee_angle, 'left_knee')
        else:
            metrics.left_knee_angle = 180.0

        if right_knee is not None:
            right_ankle = landmark_coords.get('right_ankle')
            if right_ankle is not None:
                right_knee_angle = self.calculate_angle(
                    landmark_coords['right_hip'],
                    right_knee,
                    right_ankle
                )
                metrics.right_knee_angle = self._smooth_angle(right_knee_angle, 'right_knee')
        else:
            metrics.right_knee_angle = 180.0

        # Check shoulder-wrist alignment
        left_wrist = landmark_coords.get('left_wrist')
        right_wrist = landmark_coords.get('right_wrist')
        if left_wrist is not None and right_wrist is not None:
            mid_wrist = (left_wrist + right_wrist) / 2
            metrics.shoulder_wrist_alignment = self._check_shoulder_wrist_alignment(
                mid_shoulder, mid_wrist
            )

        # Evaluate overall form
        state, score, feedback = self._evaluate_form(metrics)
        metrics.state = state
        metrics.overall_score = score
        metrics.feedback = feedback

        # Update session tracking
        self._update_session(current_time, metrics)

        # Draw visualization
        if draw_overlay:
            self._draw_pose(output_frame, landmarks)
            self._draw_angles(output_frame, landmark_coords, metrics)
            self._draw_status(output_frame, metrics)
            self._draw_session_info(output_frame)

        return output_frame, metrics

    def _update_session(self, current_time: float, metrics: PlankMetrics) -> None:
        """Update session tracking based on current frame."""
        if self._last_frame_time is None:
            self._last_frame_time = current_time
            return

        frame_duration = current_time - self._last_frame_time
        self._last_frame_time = current_time

        # Detect if user is in plank position
        is_in_plank = metrics.state in [
            PlankState.GOOD_FORM,
            PlankState.PERFECT_FORM,
            PlankState.NEEDS_ADJUSTMENT
        ]

        if is_in_plank:
            if not self.session.is_active:
                # Start new session
                self.session = PlankSession()
                self.session.start_time = current_time
                self.session.is_active = True
                logger.info("Plank session started")

            self.session.total_time += frame_duration

            if metrics.state in [PlankState.GOOD_FORM, PlankState.PERFECT_FORM]:
                self.session.good_form_time += frame_duration
                self.session.last_good_form_time = current_time

                if metrics.state == PlankState.PERFECT_FORM:
                    self.session.perfect_form_time += frame_duration
        else:
            # Check grace period
            if self.session.is_active:
                if self.session.last_good_form_time is not None:
                    time_since_good_form = current_time - self.session.last_good_form_time
                    if time_since_good_form > self.config.grace_period:
                        self._end_session()

        # Track history
        if self.session.is_active:
            self.session.form_history.append(metrics.state)
            self.session.score_history.append(metrics.overall_score)

        # Calculate tokens (example: 1 token per second of good form)
        self.session.tokens_earned = self.session.good_form_time

    def _end_session(self) -> None:
        """End the current plank session."""
        if self.session.is_active:
            logger.info(
                "Plank session ended - Total: %.1fs, Good form: %.1fs, Tokens: %.2f",
                self.session.total_time,
                self.session.good_form_time,
                self.session.tokens_earned
            )
            self.session.is_active = False

    def _draw_pose(self, frame: np.ndarray, landmarks) -> None:
        """Draw pose landmarks on the frame."""
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

    def _draw_angles(
        self,
        frame: np.ndarray,
        landmark_coords: Dict[str, Optional[np.ndarray]],
        metrics: PlankMetrics
    ) -> None:
        """Draw angle measurements on the frame."""
        h, w = frame.shape[:2]

        # Draw body alignment angle at hip
        mid_hip = landmark_coords.get('left_hip')
        if mid_hip is not None:
            hip_px = (int(mid_hip[0] * w), int(mid_hip[1] * h))
            cv2.putText(
                frame,
                f"Body: {metrics.body_alignment_angle:.0f}°",
                (hip_px[0] + 10, hip_px[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    def _draw_status(self, frame: np.ndarray, metrics: PlankMetrics) -> None:
        """Draw status overlay on the frame."""
        h, w = frame.shape[:2]

        # Status background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # State color coding
        state_colors = {
            PlankState.NOT_DETECTED: (128, 128, 128),
            PlankState.POOR_FORM: (0, 0, 255),
            PlankState.NEEDS_ADJUSTMENT: (0, 165, 255),
            PlankState.GOOD_FORM: (0, 255, 0),
            PlankState.PERFECT_FORM: (255, 215, 0),
        }
        color = state_colors.get(metrics.state, (255, 255, 255))

        # Draw state
        cv2.putText(
            frame,
            f"Status: {metrics.state.value.replace('_', ' ').title()}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Draw score
        cv2.putText(
            frame,
            f"Score: {metrics.overall_score:.0f}/100",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Draw feedback
        if metrics.feedback:
            cv2.putText(
                frame,
                metrics.feedback[0][:40],
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        # Draw angle info
        cv2.putText(
            frame,
            f"Body Angle: {metrics.body_alignment_angle:.0f}° | Hip: {metrics.hip_angle:.0f}°",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1
        )

    def _draw_session_info(self, frame: np.ndarray) -> None:
        """Draw session information on the frame."""
        h, w = frame.shape[:2]

        if self.session.is_active:
            # Session background
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - 200, 10), (w - 10, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Draw session info
            cv2.putText(
                frame,
                f"Time: {self.session.total_time:.1f}s",
                (w - 190, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                f"Good Form: {self.session.good_form_time:.1f}s",
                (w - 190, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            cv2.putText(
                frame,
                f"Tokens: {self.session.tokens_earned:.2f}",
                (w - 190, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 215, 0),
                2
            )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current/last session."""
        return {
            'total_time': self.session.total_time,
            'good_form_time': self.session.good_form_time,
            'perfect_form_time': self.session.perfect_form_time,
            'tokens_earned': self.session.tokens_earned,
            'average_score': np.mean(self.session.score_history) if self.session.score_history else 0,
            'is_active': self.session.is_active,
        }

    def reset_session(self) -> None:
        """Reset the session tracking."""
        self.session = PlankSession()
        self._last_frame_time = None
        for buffer in self._angle_buffers.values():
            buffer.clear()
        logger.info("Session reset")

    def release(self) -> None:
        """Release resources."""
        self.pose.close()
        logger.info("PlankDetector resources released")


def run_camera_detection(
    camera_id: int = 0,
    config: Optional[PlankConfig] = None,
    window_name: str = "Plank Detection"
) -> None:
    """
    Run real-time plank detection from camera feed.

    Args:
        camera_id: Camera device ID (default 0 for primary camera)
        config: Optional PlankConfig for customization
        window_name: Name of the display window
    """
    detector = PlankDetector(config)
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        logger.error("Failed to open camera %d", camera_id)
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logger.info("Starting plank detection. Press 'q' to quit, 'r' to reset session.")

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process frame
            processed_frame, metrics = detector.process_frame(frame)

            # Calculate and display FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            cv2.putText(
                processed_frame,
                f"FPS: {current_fps}",
                (processed_frame.shape[1] - 100, processed_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # Display
            cv2.imshow(window_name, processed_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset_session()
                logger.info("Session reset by user")

    except KeyboardInterrupt:
        logger.info("Detection stopped by user")

    finally:
        # Print session summary
        summary = detector.get_session_summary()
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        print(f"Total Time:       {summary['total_time']:.1f} seconds")
        print(f"Good Form Time:   {summary['good_form_time']:.1f} seconds")
        print(f"Perfect Form:     {summary['perfect_form_time']:.1f} seconds")
        print(f"Average Score:    {summary['average_score']:.1f}/100")
        print(f"Tokens Earned:    {summary['tokens_earned']:.2f}")
        print("=" * 50)

        # Cleanup
        detector.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage with custom configuration
    config = PlankConfig(
        model_complexity=1,  # Balanced accuracy/speed
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smoothing_window=5,
    )

    run_camera_detection(camera_id=0, config=config)
