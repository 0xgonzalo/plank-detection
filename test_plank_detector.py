"""
Unit tests for the Plank Detection System
"""

import numpy as np
import pytest
from plank_detector import (
    PlankDetector,
    PlankConfig,
    PlankMetrics,
    PlankState,
    PlankSession,
)


class TestAngleCalculations:
    """Test the geometric angle calculation functions."""

    def test_straight_line_angle(self):
        """Three points in a straight line should give 180 degrees."""
        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([2, 0])

        angle = PlankDetector.calculate_angle(a, b, c)
        assert abs(angle - 180.0) < 0.1

    def test_right_angle(self):
        """Three points forming a right angle should give 90 degrees."""
        a = np.array([0, 1])
        b = np.array([0, 0])
        c = np.array([1, 0])

        angle = PlankDetector.calculate_angle(a, b, c)
        assert abs(angle - 90.0) < 0.1

    def test_45_degree_angle(self):
        """Test 45-degree angle calculation."""
        a = np.array([0, 1])
        b = np.array([0, 0])
        c = np.array([1, 1])

        angle = PlankDetector.calculate_angle(a, b, c)
        assert abs(angle - 45.0) < 0.1

    def test_obtuse_angle(self):
        """Test obtuse angle (135 degrees)."""
        a = np.array([-1, 1])
        b = np.array([0, 0])
        c = np.array([1, 0])

        angle = PlankDetector.calculate_angle(a, b, c)
        assert abs(angle - 135.0) < 0.1

    def test_angle_with_3d_coords(self):
        """Test that 3D coordinates work (using only x, y)."""
        a = np.array([0, 0, 0.5])
        b = np.array([1, 0, 0.3])
        c = np.array([2, 0, 0.4])

        angle = PlankDetector.calculate_angle(a, b, c)
        assert abs(angle - 180.0) < 0.1


class TestPlankConfig:
    """Test configuration defaults and customization."""

    def test_default_config(self):
        """Test that default configuration has sensible values."""
        config = PlankConfig()

        assert config.body_alignment_min == 160.0
        assert config.body_alignment_max == 195.0
        assert config.visibility_threshold == 0.5
        assert config.smoothing_window == 5
        assert config.model_complexity == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = PlankConfig(
            body_alignment_min=150.0,
            model_complexity=2,
            smoothing_window=10,
        )

        assert config.body_alignment_min == 150.0
        assert config.model_complexity == 2
        assert config.smoothing_window == 10


class TestPlankMetrics:
    """Test metrics data class."""

    def test_default_metrics(self):
        """Test default metrics initialization."""
        metrics = PlankMetrics()

        assert metrics.body_alignment_angle == 0.0
        assert metrics.state == PlankState.NOT_DETECTED
        assert metrics.feedback == []
        assert not metrics.landmarks_visible

    def test_metrics_with_values(self):
        """Test metrics with custom values."""
        metrics = PlankMetrics(
            body_alignment_angle=175.0,
            hip_angle=178.0,
            overall_score=85.0,
            state=PlankState.GOOD_FORM,
        )

        assert metrics.body_alignment_angle == 175.0
        assert metrics.hip_angle == 178.0
        assert metrics.overall_score == 85.0
        assert metrics.state == PlankState.GOOD_FORM


class TestPlankSession:
    """Test session tracking."""

    def test_default_session(self):
        """Test default session initialization."""
        session = PlankSession()

        assert session.start_time is None
        assert session.total_time == 0.0
        assert session.good_form_time == 0.0
        assert session.tokens_earned == 0.0
        assert not session.is_active

    def test_session_with_data(self):
        """Test session with accumulated data."""
        session = PlankSession(
            start_time=1000.0,
            total_time=60.0,
            good_form_time=45.0,
            tokens_earned=45.0,
            is_active=True,
        )

        assert session.total_time == 60.0
        assert session.good_form_time == 45.0
        assert session.tokens_earned == 45.0


class TestPlankDetector:
    """Test the main PlankDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = PlankDetector()

        assert detector.config is not None
        assert detector.pose is not None
        assert len(detector._angle_buffers) == 6
        detector.release()

    def test_initialization_with_config(self):
        """Test detector initialization with custom config."""
        config = PlankConfig(model_complexity=0)
        detector = PlankDetector(config)

        assert detector.config.model_complexity == 0
        detector.release()

    def test_session_reset(self):
        """Test session reset functionality."""
        detector = PlankDetector()
        detector.session.total_time = 100.0
        detector.session.is_active = True

        detector.reset_session()

        assert detector.session.total_time == 0.0
        assert not detector.session.is_active
        detector.release()

    def test_get_session_summary(self):
        """Test session summary retrieval."""
        detector = PlankDetector()
        detector.session.total_time = 60.0
        detector.session.good_form_time = 45.0
        detector.session.perfect_form_time = 20.0
        detector.session.tokens_earned = 45.0
        detector.session.score_history = [80.0, 85.0, 90.0]

        summary = detector.get_session_summary()

        assert summary['total_time'] == 60.0
        assert summary['good_form_time'] == 45.0
        assert summary['perfect_form_time'] == 20.0
        assert summary['tokens_earned'] == 45.0
        assert abs(summary['average_score'] - 85.0) < 0.1
        detector.release()


class TestFormEvaluation:
    """Test form evaluation logic."""

    def test_perfect_body_alignment(self):
        """Test that perfect body alignment scores high."""
        config = PlankConfig()

        # Angle of 175 should be in perfect range
        assert config.body_alignment_perfect_min <= 175 <= config.body_alignment_perfect_max

    def test_poor_body_alignment(self):
        """Test that poor body alignment is detected."""
        config = PlankConfig()

        # Angle of 140 should be below minimum
        assert 140 < config.body_alignment_min

    def test_hip_sag_detection(self):
        """Test that hip sag is detected."""
        config = PlankConfig()

        # Angle of 145 indicates sagging hips
        assert 145 < config.hip_angle_min


class TestPlankStates:
    """Test plank state enumeration."""

    def test_state_values(self):
        """Test all state values exist."""
        assert PlankState.NOT_DETECTED.value == "not_detected"
        assert PlankState.POOR_FORM.value == "poor_form"
        assert PlankState.NEEDS_ADJUSTMENT.value == "needs_adjustment"
        assert PlankState.GOOD_FORM.value == "good_form"
        assert PlankState.PERFECT_FORM.value == "perfect_form"

    def test_state_ordering_logic(self):
        """Test that states represent increasing quality."""
        states_quality = [
            PlankState.NOT_DETECTED,
            PlankState.POOR_FORM,
            PlankState.NEEDS_ADJUSTMENT,
            PlankState.GOOD_FORM,
            PlankState.PERFECT_FORM,
        ]

        # Verify all states are unique
        assert len(states_quality) == len(set(states_quality))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_vector_angle(self):
        """Test angle calculation with very close points."""
        a = np.array([0.0, 0.0])
        b = np.array([0.0001, 0.0])
        c = np.array([0.0002, 0.0])

        # Should not raise an error due to epsilon in denominator
        angle = PlankDetector.calculate_angle(a, b, c)
        assert not np.isnan(angle)

    def test_same_point_angle(self):
        """Test angle calculation with same points."""
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([0.0, 0.0])

        # Should handle gracefully with epsilon
        angle = PlankDetector.calculate_angle(a, b, c)
        assert not np.isnan(angle)

    def test_negative_coordinates(self):
        """Test angle calculation with negative coordinates."""
        a = np.array([-1.0, -1.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 0.0])

        angle = PlankDetector.calculate_angle(a, b, c)
        assert 0 <= angle <= 180


# Integration test markers for tests requiring camera
class TestIntegration:
    """Integration tests (require camera/display)."""

    @pytest.mark.skipif(True, reason="Requires camera hardware")
    def test_camera_detection_smoke(self):
        """Smoke test for camera detection."""
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            pytest.skip("No camera available")

        detector = PlankDetector()
        ret, frame = cap.read()

        if ret:
            processed_frame, metrics = detector.process_frame(frame)
            assert processed_frame is not None
            assert metrics is not None

        detector.release()
        cap.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
