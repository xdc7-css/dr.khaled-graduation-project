"""
profile_manager.py
User Setup, Personalization, and Guided Acquisition Workflow

Capabilities:
1. Participant profile management (ID, session, sensor placement, hand, sampling rate).
2. Neutral reference posture acquisition and sensor bias calibration.
3. Guided repetition workflow with quality/completeness checks.
4. Non-destructive saving under 'data/profiles/<participant_id>/' (never overwrites raw datasets).
5. Exercisable with mock/replay data for offline testing.
6. Clear academic distinction between neutral configuration, bias calibration, and model fine-tuning.
"""

import os
import sys
import json
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.gesture_schema import CANONICAL_CLASSES
from src.data_pipeline import WINDOW_SIZE, NUM_CHANNELS, FEATURE_COLS


@dataclass
class NeutralReference:
    """Resting baseline parameters recorded for an individual user."""
    mean_ax_g: float = 0.0
    mean_ay_g: float = 0.0
    mean_az_g: float = 1.0
    bias_gx_dps: float = 0.0
    bias_gy_dps: float = 0.0
    bias_gz_dps: float = 0.0
    recorded_samples: int = 0
    timestamp_utc: str = ""


@dataclass
class ParticipantProfile:
    """Metadata detailing user physiology and hardware mounting conditions."""
    participant_id: str = "P_DEMO"
    session_id: str = "S01"
    sensor_placement: str = "dorsal_wrist"  # 'dorsal_wrist' or 'palmar_wrist'
    hand: str = "dominant_right"           # 'dominant_right', 'dominant_left', etc.
    sampling_rate_hz: float = 50.0
    window_size: int = WINDOW_SIZE
    neutral_reference: Optional[NeutralReference] = None
    created_at: str = ""
    notes: str = ""


class GuidedSetupWorkflow:
    """
    Step-by-step interactive workflow guiding a participant through:
    1. Mounting & Profile initialization
    2. Neutral resting posture recording
    3. Guided gesture collection trials
    4. Quality and completeness audit
    """

    GESTURE_SCHEDULE = ["still", "left", "right", "up", "down", "shake"]

    def __init__(self, profile: Optional[ParticipantProfile] = None, repetitions_per_gesture: int = 5):
        self.profile = profile or ParticipantProfile(created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.target_reps = repetitions_per_gesture
        
        # Workflow state
        self.current_gesture_idx: int = 0
        self.current_rep: int = 1
        self.collected_trials: Dict[str, List[np.ndarray]] = {g: [] for g in self.GESTURE_SCHEDULE}
        self.neutral_samples: List[np.ndarray] = []
        self.is_completed: bool = False

    @property
    def current_gesture(self) -> Optional[str]:
        if self.current_gesture_idx < len(self.GESTURE_SCHEDULE):
            return self.GESTURE_SCHEDULE[self.current_gesture_idx]
        return None

    def record_neutral_window(self, window: np.ndarray) -> NeutralReference:
        """Calculates neutral posture baseline and zero-rate gyroscope bias."""
        if window is None or len(window) < 10:
            raise ValueError("Insufficient samples to compute neutral reference.")

        ax_mean = float(np.mean(window[:, 0]))
        ay_mean = float(np.mean(window[:, 1]))
        az_mean = float(np.mean(window[:, 2]))
        gx_bias = float(np.mean(window[:, 3]))
        gy_bias = float(np.mean(window[:, 4]))
        gz_bias = float(np.mean(window[:, 5]))

        ref = NeutralReference(
            mean_ax_g=round(ax_mean, 4),
            mean_ay_g=round(ay_mean, 4),
            mean_az_g=round(az_mean, 4),
            bias_gx_dps=round(gx_bias, 4),
            bias_gy_dps=round(gy_bias, 4),
            bias_gz_dps=round(gz_bias, 4),
            recorded_samples=len(window),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        self.profile.neutral_reference = ref
        return ref

    def add_trial(self, window: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Accepts a recorded trial window for the current guided gesture.
        Performs data quality checks before appending.
        """
        if self.is_completed:
            return False, "Guided workflow is already completed.", {}

        curr_g = self.current_gesture
        if curr_g is None:
            self.is_completed = True
            return False, "All scheduled gestures completed.", {}

        # 1. Quality & Shape check
        if window.shape != (WINDOW_SIZE, NUM_CHANNELS):
            return False, f"Invalid shape {window.shape}, expected ({WINDOW_SIZE}, {NUM_CHANNELS})", {}

        if np.any(np.isnan(window)) or np.any(np.isinf(window)):
            return False, "Trial contains NaN or Infinite values.", {}

        # 2. Dynamic check
        acc_var = float(np.sum(np.var(window[:, 0:3], axis=0)))
        if curr_g == "still" and acc_var > 0.05:
            return False, f"Excessive motion during 'still' trial (variance {acc_var:.3f} > 0.05). Please repeat.", {}

        # 3. Store valid trial
        self.collected_trials[curr_g].append(window.copy())
        rep_recorded = self.current_rep

        # Step counters
        if self.current_rep < self.target_reps:
            self.current_rep += 1
        else:
            self.current_rep = 1
            self.current_gesture_idx += 1
            if self.current_gesture_idx >= len(self.GESTURE_SCHEDULE):
                self.is_completed = True

        status_info = {
            "gesture": curr_g,
            "rep": rep_recorded,
            "target_reps": self.target_reps,
            "next_gesture": self.current_gesture,
            "next_rep": self.current_rep,
            "is_workflow_completed": self.is_completed
        }
        return True, f"Accepted trial {rep_recorded}/{self.target_reps} for '{curr_g}'.", status_info

    def generate_mock_trial(self, gesture: str) -> np.ndarray:
        """Generates a realistic synthetic window to test the guided workflow offline."""
        t = np.linspace(0, 1.0, WINDOW_SIZE)
        noise = np.random.normal(0, 0.01, size=(WINDOW_SIZE, NUM_CHANNELS))
        window = np.zeros((WINDOW_SIZE, NUM_CHANNELS), dtype=np.float32)

        if gesture == "still":
            window[:, 2] = 1.0  # +1.0g on Z
        elif gesture == "left":
            window[:, 0] = -0.5 * np.sin(np.pi * t)
            window[:, 2] = 0.8
            window[:, 4] = 60.0 * np.sin(np.pi * t)
        elif gesture == "right":
            window[:, 0] = 0.5 * np.sin(np.pi * t)
            window[:, 2] = 0.8
            window[:, 4] = -60.0 * np.sin(np.pi * t)
        elif gesture == "up":
            window[:, 1] = 0.5 * np.sin(np.pi * t)
            window[:, 2] = 0.8
            window[:, 3] = 60.0 * np.sin(np.pi * t)
        elif gesture == "down":
            window[:, 1] = -0.5 * np.sin(np.pi * t)
            window[:, 2] = 0.8
            window[:, 3] = -60.0 * np.sin(np.pi * t)
        elif gesture == "shake":
            window[:, 2] = 0.7
            window[:, 3] = 180.0 * np.sin(4 * np.pi * t)
            window[:, 4] = 180.0 * np.cos(4 * np.pi * t)

        return (window + noise).astype(np.float32)

    def save_session(self, output_dir: Optional[str] = None) -> Tuple[str, str]:
        """
        Saves the profile metadata and recorded trials safely without overwriting originals.
        Returns: (profile_json_path, trials_csv_path)
        """
        base_dir = output_dir or os.path.join(REPO_ROOT, "data", "profiles", self.profile.participant_id)
        os.makedirs(base_dir, exist_ok=True)

        # 1. Save Profile JSON
        profile_fname = f"profile_{self.profile.session_id}_{int(time.time())}.json"
        profile_path = os.path.join(base_dir, profile_fname)

        p_dict = asdict(self.profile)
        p_dict["total_collected_trials"] = {g: len(trials) for g, trials in self.collected_trials.items()}
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(p_dict, f, indent=2)

        # 2. Save Collected Trials CSV
        import pandas as pd
        csv_rows = []
        for g, trial_list in self.collected_trials.items():
            for t_idx, w in enumerate(trial_list, start=1):
                for s_idx in range(len(w)):
                    row = {
                        "participant_id": self.profile.participant_id,
                        "session_id": self.profile.session_id,
                        "gesture": g,
                        "trial_id": t_idx,
                        "sample_idx": s_idx,
                        "timestamp_ms": s_idx * int(1000.0 / self.profile.sampling_rate_hz),
                        "ax_g": float(w[s_idx, 0]),
                        "ay_g": float(w[s_idx, 1]),
                        "az_g": float(w[s_idx, 2]),
                        "gx_dps": float(w[s_idx, 3]),
                        "gy_dps": float(w[s_idx, 4]),
                        "gz_dps": float(w[s_idx, 5]),
                    }
                    csv_rows.append(row)

        trials_fname = f"trials_{self.profile.session_id}_{int(time.time())}.csv"
        trials_path = os.path.join(base_dir, trials_fname)
        if csv_rows:
            pd.DataFrame(csv_rows).to_csv(trials_path, index=False)
        else:
            with open(trials_path, "w", encoding="utf-8") as f:
                f.write(",".join(["participant_id", "session_id", "gesture", "trial_id", "sample_idx", "timestamp_ms"] + FEATURE_COLS) + "\n")

        return profile_path, trials_path


def get_personalization_concepts() -> Dict[str, str]:
    """
    Returns precise definitions separating neutral calibration from model retraining.
    """
    return {
        "Neutral Reference Configuration": (
            "Measures the participant's natural anatomical resting arm posture "
            "to establish gravitational vector orientation [ax, ay, az] and re-arming tolerances."
        ),
        "Sensor Bias Calibration": (
            "Measures zero-rate gyroscope offsets [gx, gy, gz] during static resting periods "
            "to cancel MEMS sensor thermal drift without changing model weights."
        ),
        "Model Personalization / Fine-Tuning": (
            "Retraining or fine-tuning network layers on user-specific trial data. "
            "Must be kept in isolated directories and evaluated on independent unseen test splits."
        )
    }
