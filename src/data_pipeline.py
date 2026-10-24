"""
data_pipeline.py
Core Data Validation, Feature Specification, and Normalization Pipeline

Responsibilities:
1. Canonical feature order and window length specification.
2. Z-score feature normalization (fitted strictly on training data).
3. Window shape and numerical validity verification (NaN/Inf detection).
4. Robust sensor string parsing.
"""

from typing import List, Tuple, Optional, Union
import os
import numpy as np

# Canonical feature column ordering expected by all models
FEATURE_COLS: List[str] = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]

# Expected temporal dimension (51 samples ~ 1.0 - 1.1s at ~45-50 Hz)
WINDOW_SIZE: int = 51

# Number of IMU axes
NUM_CHANNELS: int = len(FEATURE_COLS)  # 6


class ZScoreNormalizer:
    """
    Standardizes IMU features by subtracting the mean and dividing by std.
    Computes parameters strictly from training data to avoid data leakage.
    """

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None, epsilon: float = 1e-8):
        self.mean: Optional[np.ndarray] = mean.astype(np.float32) if mean is not None else None
        self.std: Optional[np.ndarray] = std.astype(np.float32) if std is not None else None
        self.epsilon: float = epsilon

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, X_train: np.ndarray) -> "ZScoreNormalizer":
        """
        Fits mean and std from training windows.
        X_train shape: (N, 51, 6) or (N*51, 6)
        """
        flat = X_train.reshape(-1, NUM_CHANNELS)
        self.mean = np.mean(flat, axis=0).astype(np.float32)
        self.std = (np.std(flat, axis=0) + self.epsilon).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normalizes input array using fitted mean and std.
        X shape: (51, 6) or (N, 51, 6)
        """
        if not self.is_fitted:
            raise ValueError("Normalizer has not been fitted or loaded with parameters.")
        return (X - self.mean) / self.std

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def save(self, mean_path: str, std_path: str) -> None:
        """Saves mean and std to .npy files."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted normalizer parameters.")
        os.makedirs(os.path.dirname(os.path.abspath(mean_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(std_path)), exist_ok=True)
        np.save(mean_path, self.mean)
        np.save(std_path, self.std)

    @classmethod
    def load(cls, mean_path: str, std_path: str) -> "ZScoreNormalizer":
        """Loads mean and std from .npy files."""
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            raise FileNotFoundError(f"Missing normalization file(s): {mean_path} or {std_path}")
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        return cls(mean=mean, std=std)


def validate_window(window: np.ndarray, expected_shape: Tuple[int, int] = (WINDOW_SIZE, NUM_CHANNELS)) -> Tuple[bool, Optional[str]]:
    """
    Validates that a gesture window has the correct shape and finite numeric values.
    """
    if not isinstance(window, np.ndarray):
        return False, f"Expected numpy ndarray, got {type(window)}"
    
    if window.shape != expected_shape:
        return False, f"Window shape {window.shape} does not match expected {expected_shape}"
    
    if not np.all(np.isfinite(window)):
        return False, "Window contains non-finite values (NaN or Inf)"
    
    return True, None


def parse_sensor_line(line: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """
    Parses a single line from serial stream or CSV into 6 sensor floats.

    Expected formats:
    1. Raw serial: "timestamp_ms,ax,ay,az,gx,gy,gz" (7 tokens)
    2. Logged CSV: "session_id,gesture,trial_id,timestamp_ms,ax,ay,az,gx,gy,gz" (10 tokens)

    Returns:
        (sensor_array, timestamp_ms, error_message)
    """
    clean_line = line.strip()
    if not clean_line or clean_line.startswith("#") or clean_line.startswith("timestamp_ms") or clean_line.startswith("session_id"):
        return None, None, None

    parts = [p.strip() for p in clean_line.split(",")]

    try:
        if len(parts) == 7:
            # Serial format
            ts = int(float(parts[0]))
            values = [float(v) for v in parts[1:7]]
            return np.array(values, dtype=np.float32), ts, None
        elif len(parts) == 10:
            # Full logged dataset format
            ts = int(float(parts[3]))
            values = [float(v) for v in parts[4:10]]
            return np.array(values, dtype=np.float32), ts, None
        else:
            return None, None, f"Unexpected token count: {len(parts)} (expected 7 or 10)"
    except ValueError as e:
        return None, None, f"Non-numeric sensor data: {e}"
