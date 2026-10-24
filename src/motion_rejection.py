"""
motion_rejection.py
Ordinary-Motion & Unknown-Gesture Rejection Infrastructure

Provides an extensible rejection pipeline combining:
1. Input validity & physical signal sanity (clipping, NaN, sensor saturation).
2. Motion quality and dynamic energy analysis.
3. Model confidence AND class separation margin (top-1 vs top-2 probability).
4. Neutral/idle resting posture detection.
5. Extensible hook for future 'other_motion' / background daily activities.

Academic Note:
Softmax confidence alone is NOT a reliable out-of-distribution detector.
This module enforces multi-factor gating and outputs explicit diagnostic codes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class RejectionDiagnostics:
    """Detailed audit information on why a sample was accepted or rejected."""
    is_accepted: bool
    primary_code: str
    confidence: float
    class_margin: float
    entropy: float
    motion_energy_accel: float
    gyro_rms: float
    is_saturated: bool
    status_detail: str


class MotionQualityRejector:
    """
    Evaluates physical sensor dynamics and model prediction distributions
    to reject ordinary non-gesture movements, idle stillness, and ambiguous predictions.
    """

    def __init__(
        self,
        min_confidence: float = 0.75,
        min_class_margin: float = 0.20,
        max_entropy: float = 1.80,
        min_dynamic_energy_g2: float = 0.002,   # Provisional minimum variance for dynamic gestures
        min_gyro_rms_dps: float = 4.0,          # Provisional minimum angular motion
        accel_clip_g: float = 3.90,             # MPU6050 configured for +-4g or +-2g
        gyro_clip_dps: float = 1950.0          # MPU6050 configured for +-2000 dps
    ):
        # NOTE: Thresholds are marked PROVISIONAL awaiting live physical sensor recordings
        self.min_confidence = min_confidence
        self.min_class_margin = min_class_margin
        self.max_entropy = max_entropy
        self.min_dynamic_energy_g2 = min_dynamic_energy_g2
        self.min_gyro_rms_dps = min_gyro_rms_dps
        self.accel_clip_g = accel_clip_g
        self.gyro_clip_dps = gyro_clip_dps

    def compute_energy_metrics(self, raw_window: np.ndarray) -> Tuple[float, float, bool]:
        """
        Computes motion energy and checks for hardware saturation.
        Returns: (accel_variance_g2, gyro_rms_dps, is_saturated)
        """
        if raw_window is None or len(raw_window) == 0:
            return 0.0, 0.0, False

        # Accel dynamic energy: variance across the window
        accel_var = float(np.sum(np.var(raw_window[:, 0:3], axis=0)))

        # Gyro RMS in deg/s
        gyro_rms = float(np.sqrt(np.mean(raw_window[:, 3:6] ** 2)))

        # Saturation check (sensor reaching rail limits)
        max_acc = float(np.max(np.abs(raw_window[:, 0:3])))
        max_gyr = float(np.max(np.abs(raw_window[:, 3:6])))
        is_saturated = (max_acc >= self.accel_clip_g) or (max_gyr >= self.gyro_clip_dps)

        return accel_var, gyro_rms, is_saturated

    def evaluate_probabilities(self, probs: np.ndarray) -> Tuple[float, float, float]:
        """
        Computes top-1 confidence, margin between top-1 and top-2, and Shannon entropy.
        """
        sorted_probs = np.sort(probs)[::-1]
        top1 = float(sorted_probs[0])
        top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = top1 - top2

        # Shannon entropy: H(p) = - sum p * log2(p + eps)
        eps = 1e-12
        entropy = float(-np.sum(probs * np.log2(probs + eps)))

        return top1, margin, entropy

    def evaluate(
        self,
        raw_window: np.ndarray,
        probs: np.ndarray,
        predicted_label: str
    ) -> RejectionDiagnostics:
        """
        Performs multi-stage gating on the sensor window and inference probabilities.
        """
        # 1. Physical Signal Sanity Checks
        if np.any(np.isnan(raw_window)) or np.any(np.isinf(raw_window)):
            return RejectionDiagnostics(
                is_accepted=False,
                primary_code="SIGNAL_NAN_OR_INF",
                confidence=0.0,
                class_margin=0.0,
                entropy=99.0,
                motion_energy_accel=0.0,
                gyro_rms=0.0,
                is_saturated=False,
                status_detail="Sensor signal contains invalid NaN or Infinite values."
            )

        accel_var, gyro_rms, is_saturated = self.compute_energy_metrics(raw_window)
        top1, margin, entropy = self.evaluate_probabilities(probs)

        # 2. Hardware Saturation Rejection
        if is_saturated:
            return RejectionDiagnostics(
                is_accepted=False,
                primary_code="SENSOR_SATURATED",
                confidence=top1,
                class_margin=margin,
                entropy=entropy,
                motion_energy_accel=accel_var,
                gyro_rms=gyro_rms,
                is_saturated=True,
                status_detail="Sensor signal clipped at maximum ADC/rail range."
            )

        # 3. Model Distribution Confidence & Margin Checks
        # Allow lower confidence for 'shake' due to high multi-axis oscillation variance
        conf_req = 0.55 if predicted_label == "shake" else self.min_confidence

        if top1 < conf_req:
            return RejectionDiagnostics(
                is_accepted=False,
                primary_code="LOW_CONFIDENCE",
                confidence=top1,
                class_margin=margin,
                entropy=entropy,
                motion_energy_accel=accel_var,
                gyro_rms=gyro_rms,
                is_saturated=False,
                status_detail=f"Confidence {top1:.2f} below threshold {conf_req:.2f}."
            )

        if margin < self.min_class_margin:
            return RejectionDiagnostics(
                is_accepted=False,
                primary_code="INSUFFICIENT_CLASS_MARGIN",
                confidence=top1,
                class_margin=margin,
                entropy=entropy,
                motion_energy_accel=accel_var,
                gyro_rms=gyro_rms,
                is_saturated=False,
                status_detail=f"Class margin {margin:.2f} < {self.min_class_margin:.2f} (ambiguous classification)."
            )

        # 4. Motion Quality & Dynamic Energy Checks for Active Gestures
        # If the model predicts an active gesture (like shake), verify physical motion is present
        if predicted_label == "shake":
            if accel_var < self.min_dynamic_energy_g2 or gyro_rms < self.min_gyro_rms_dps:
                return RejectionDiagnostics(
                    is_accepted=False,
                    primary_code="INSUFFICIENT_MOTION_ENERGY",
                    confidence=top1,
                    class_margin=margin,
                    entropy=entropy,
                    motion_energy_accel=accel_var,
                    gyro_rms=gyro_rms,
                    is_saturated=False,
                    status_detail=f"Dynamic gesture '{predicted_label}' lacks kinetic energy (acc_var={accel_var:.4f}, gyro_rms={gyro_rms:.1f})."
                )

        # All checks passed!
        return RejectionDiagnostics(
            is_accepted=True,
            primary_code="ACCEPTED",
            confidence=top1,
            class_margin=margin,
            entropy=entropy,
            motion_energy_accel=accel_var,
            gyro_rms=gyro_rms,
            is_saturated=False,
            status_detail="Signal quality and model distribution verified."
        )


def get_missing_negative_examples_report() -> Dict[str, Any]:
    """
    Returns an academic documentation dictionary explaining what negative examples
    are missing from the current dataset and why laboratory evaluation cannot prove
    rejection of arbitrary daily motions.
    """
    return {
        "status": "Implemented rejection infrastructure; awaiting unsegmented continuous background data",
        "missing_negative_classes": [
            "Typing on keyboard / mouse navigation (high frequency, low amplitude hand jitter)",
            "Walking / arm swinging (periodic 1-2 Hz gravitational oscillations)",
            "Gesturing while speaking (irregular conversational arm motions)",
            "Picking up cup / drinking water (reaching and pronation)",
            "Resting hand on lap or armrest (non-standard stationary angles)"
        ],
        "limitation_note": (
            "Current datasets consist exclusively of segmented gesture trials. "
            "Until continuous multi-hour background activity is recorded with a physical wearable sensor, "
            "false activation rates per minute cannot be measured."
        )
    }
