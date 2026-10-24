"""
replay_source.py
Hardware-Free CSV Replay Source for IMU Gesture Testing

Key Features:
1. Loads single CSV files or entire dataset directories.
2. Preserves strict trial boundaries and chronological sample ordering.
3. NEVER combines samples across different trials.
4. Identifies malformed rows, missing columns, incomplete windows (<51 samples),
   and unrecognized labels into a visible audit record.
5. Provides Play, Pause, Stop, Step, and adjustable playback speed.
6. Replay restart cleanly resets state and counters without off-by-one errors.
7. Explicit clock abstraction ensures replay speed does not alter model inputs.
"""

import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import glob
import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Any

import pandas as pd
import numpy as np

from src.data_pipeline import FEATURE_COLS, WINDOW_SIZE, NUM_CHANNELS, validate_window
from src.gesture_schema import normalize_label


@dataclass
class ReplayTrial:
    """Represents a single independent gesture trial extracted from recorded data."""
    trial_key: str
    source_file: str
    session_id: str
    raw_gesture: str
    canonical_gesture: Optional[str]
    trial_id: int
    timestamps: np.ndarray
    data: np.ndarray  # Shape: (N, 6)
    is_valid: bool
    status_message: str


class CsvReplaySource:
    """
    Parses and replays recorded IMU trial data without requiring hardware.
    """

    def __init__(self, target_path: Optional[str] = None):
        self.target_path: Optional[str] = None
        self.trials: List[ReplayTrial] = []
        self.quarantine_log: List[Dict[str, Any]] = []
        
        # Playback control state
        self._current_index: int = 0
        self._is_playing: bool = False
        self._is_paused: bool = False
        self._stop_event = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
        self._speed_multiplier: float = 1.0  # 1.0 = real-time, 0.0 = instantaneous
        self._lock = threading.Lock()

        if target_path:
            self.load(target_path)

    @property
    def total_trials(self) -> int:
        return len(self.trials)

    @property
    def valid_trials(self) -> List[ReplayTrial]:
        return [t for t in self.trials if t.is_valid]

    @property
    def quarantined_trials(self) -> List[ReplayTrial]:
        return [t for t in self.trials if not t.is_valid]

    @property
    def current_index(self) -> int:
        with self._lock:
            return self._current_index

    @property
    def speed_multiplier(self) -> float:
        with self._lock:
            return self._speed_multiplier

    def set_speed(self, speed: float) -> None:
        """Sets replay speed multiplier. >1 is faster, <1 is slower, <=0 is instantaneous."""
        with self._lock:
            self._speed_multiplier = max(0.0, float(speed))

    def reset(self) -> None:
        """Resets playback position to the start."""
        self.stop()
        with self._lock:
            self._current_index = 0

    def load(self, path: str) -> Tuple[int, int]:
        """
        Loads CSV files from a file or folder path.
        Returns: (valid_trial_count, quarantined_trial_count)
        """
        self.stop()
        with self._lock:
            self.target_path = os.path.abspath(path)
            self.trials.clear()
            self.quarantine_log.clear()
            self._current_index = 0

            csv_files = []
            if os.path.isfile(self.target_path):
                if self.target_path.lower().endswith(".csv"):
                    csv_files.append(self.target_path)
            elif os.path.isdir(self.target_path):
                csv_files = sorted(glob.glob(os.path.join(self.target_path, "**", "*.csv"), recursive=True))

            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in: {self.target_path}")

            for fpath in csv_files:
                self._parse_file(fpath)

        valid_count = len(self.valid_trials)
        quarantine_count = len(self.quarantine_log)
        return valid_count, quarantine_count

    def _parse_file(self, file_path: str) -> None:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            self.quarantine_log.append({
                "file": file_path,
                "reason": f"Failed to read CSV: {e}",
                "trial_key": "unreadable"
            })
            return

        # Check required sensor columns
        missing_sensors = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_sensors:
            self.quarantine_log.append({
                "file": file_path,
                "reason": f"Missing required sensor columns: {missing_sensors}",
                "trial_key": "missing_columns"
            })
            return

        has_session = "session_id" in df.columns
        has_gesture = "gesture" in df.columns
        has_trial = "trial_id" in df.columns
        has_timestamp = "timestamp_ms" in df.columns

        fname_base = os.path.splitext(os.path.basename(file_path))[0]

        # Group rows into independent trials
        if has_trial:
            group_cols = []
            if has_session:
                group_cols.append("session_id")
            if has_gesture:
                group_cols.append("gesture")
            group_cols.append("trial_id")
            groups = df.groupby(group_cols, sort=False)
        else:
            # Fallback if trial_id column missing: chunk into 51-row blocks
            groups = [(f"chunk_{i}", df.iloc[i:i+WINDOW_SIZE]) for i in range(0, len(df), WINDOW_SIZE)]

        for group_key, grp in groups:
            sess_id = str(grp["session_id"].iloc[0]) if has_session else "1"
            raw_gest = str(grp["gesture"].iloc[0]) if has_gesture else fname_base
            tid = int(grp["trial_id"].iloc[0]) if has_trial else 0

            trial_key = f"sess_{sess_id}_{raw_gest}_trial_{tid}"
            num_rows = len(grp)

            # Sort by timestamp if present
            if has_timestamp:
                grp_sorted = grp.sort_values("timestamp_ms")
                ts_arr = grp_sorted["timestamp_ms"].values
            else:
                grp_sorted = grp
                ts_arr = np.arange(num_rows) * 20  # Synthesize nominal 50Hz timestamps

            sensor_data = grp_sorted[FEATURE_COLS].values.astype(np.float32)

            # Validation checks
            canon_label, err_label = normalize_label(raw_gest)
            is_valid = True
            reasons = []

            if num_rows != WINDOW_SIZE:
                is_valid = False
                reasons.append(f"Incomplete/Invalid window: {num_rows} rows (expected {WINDOW_SIZE})")

            valid_shape, err_shape = validate_window(sensor_data, expected_shape=(WINDOW_SIZE, NUM_CHANNELS))
            if not valid_shape:
                is_valid = False
                reasons.append(f"Data validation failed: {err_shape}")

            if canon_label is None:
                is_valid = False
                reasons.append(f"Label error: {err_label}")

            status_msg = "OK" if is_valid else "; ".join(reasons)

            trial_obj = ReplayTrial(
                trial_key=trial_key,
                source_file=file_path,
                session_id=sess_id,
                raw_gesture=raw_gest,
                canonical_gesture=canon_label,
                trial_id=tid,
                timestamps=ts_arr,
                data=sensor_data,
                is_valid=is_valid,
                status_message=status_msg
            )

            self.trials.append(trial_obj)
            if not is_valid:
                self.quarantine_log.append({
                    "trial_key": trial_key,
                    "file": file_path,
                    "raw_gesture": raw_gest,
                    "trial_id": tid,
                    "rows": num_rows,
                    "reason": status_msg
                })

    def get_trial(self, index: int) -> Optional[ReplayTrial]:
        if 0 <= index < len(self.trials):
            return self.trials[index]
        return None

    def seek(self, index: int) -> Optional[ReplayTrial]:
        with self._lock:
            if 0 <= index < len(self.trials):
                self._current_index = index
                return self.trials[index]
        return None

    def step_next(self, only_valid: bool = False) -> Optional[ReplayTrial]:
        """
        Steps forward by one trial and returns it.
        Returns None when end of sequence is reached.
        """
        with self._lock:
            trial_list = self.valid_trials if only_valid else self.trials
            total = len(trial_list)
            if total == 0:
                return None

            if self._current_index < total:
                trial = trial_list[self._current_index]
                self._current_index += 1
                return trial
            return None

    def start_replay(
        self,
        on_trial_emit: Callable[[ReplayTrial, int, int], None],
        on_complete: Optional[Callable[[], None]] = None,
        only_valid: bool = True
    ) -> None:
        """
        Starts background playback of loaded trials.
        on_trial_emit: callback invoked per trial window with (trial, current_1based_idx, total_trials).
        """
        self.stop()
        self._stop_event.clear()
        self._is_playing = True
        self._is_paused = False

        def _worker():
            trial_list = self.valid_trials if only_valid else self.trials
            total = len(trial_list)
            if total == 0:
                with self._lock:
                    self._is_playing = False
                return

            with self._lock:
                # If already at the end, restart cleanly from index 0
                if self._current_index >= total:
                    self._current_index = 0
                idx = self._current_index

            while idx < total and not self._stop_event.is_set():
                while self._is_paused and not self._stop_event.is_set():
                    time.sleep(0.05)

                if self._stop_event.is_set():
                    break

                trial = trial_list[idx]
                with self._lock:
                    self._current_index = idx + 1
                    current_1based = self._current_index

                on_trial_emit(trial, current_1based, total)
                idx += 1

                # Inter-trial playback pacing
                spd = self.speed_multiplier
                if spd > 0.0:
                    delay_sec = max(0.01, 1.0 / spd)
                    deadline = time.time() + delay_sec
                    while time.time() < deadline and not self._stop_event.is_set():
                        time.sleep(0.02)

            with self._lock:
                self._is_playing = False
                self._is_paused = False

            if on_complete and not self._stop_event.is_set():
                on_complete()

        self._playback_thread = threading.Thread(target=_worker, daemon=True)
        self._playback_thread.start()

    def pause(self) -> None:
        with self._lock:
            self._is_paused = True

    def resume(self) -> None:
        with self._lock:
            self._is_paused = False

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._is_playing = False
            self._is_paused = False
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=0.5)
        self._playback_thread = None
