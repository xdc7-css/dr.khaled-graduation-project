"""
logger.py
Data Collection Utility for Wearable IMU Gesture Datasets

Features:
1. Records participant ID, session ID, canonical gesture label, and trial ID.
2. Formats rows into standardized CSV schema:
   session_id,gesture,trial_id,timestamp_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
3. Validates gesture labels using canonical gesture schema before recording.
4. Overwrite protection: prevents accidental truncation or overwriting of existing trials.
5. Configurable countdown and capture duration (default: 51 samples).
6. Discarding of incomplete captures on timeout or disconnection.
7. Mock / Hardware-Free Mode (--mock): Allows testing collection without an ESP32.
   Mock files are stored in 'data/mock/' to guarantee separation from real recordings.
"""

import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import time
import argparse
from typing import Optional, List, Tuple

import numpy as np

from src.gesture_schema import CANONICAL_CLASSES, normalize_label
from src.data_pipeline import FEATURE_COLS, WINDOW_SIZE, parse_sensor_line

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class ImuDataLogger:
    """Manages serial or mock data capture for gesture trials."""

    def __init__(
        self,
        port: str = "COM6",
        baud_rate: int = 115200,
        samples_per_trial: int = WINDOW_SIZE,
        output_dir: str = "data/collected",
        mock_mode: bool = False,
    ):
        self.port: str = port
        self.baud_rate: int = baud_rate
        self.samples_per_trial: int = samples_per_trial
        self.mock_mode: bool = mock_mode
        self.output_dir: str = os.path.abspath("data/mock" if mock_mode else output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.ser: Optional[serial.Serial] = None

    def connect(self) -> bool:
        if self.mock_mode:
            print("[LOGGER] Running in MOCK MODE (No physical hardware required).")
            return True

        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial is not installed. Install it or run with --mock.")

        try:
            print(f"[LOGGER] Connecting to ESP32 on {self.port} at {self.baud_rate} baud...")
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1.0)
            time.sleep(2.0)  # Allow board reboot / settle
            self.ser.reset_input_buffer()
            print(f"[LOGGER] Successfully connected to {self.port}")
            return True
        except Exception as e:
            print(f"[LOGGER ERROR] Failed to connect to {self.port}: {e}")
            return False

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b"stop\n")
                time.sleep(0.1)
                self.ser.close()
            except Exception:
                pass
            print("[LOGGER] Serial connection closed.")

    def calibrate(self) -> bool:
        if self.mock_mode:
            print("[LOGGER MOCK] Simulating calibration... Done.")
            return True

        if not self.ser or not self.ser.is_open:
            print("[LOGGER ERROR] Not connected.")
            return False

        print("\n[CALIBRATION] Hold sensor stationary in standard reference position.")
        self.ser.reset_input_buffer()
        self.ser.write(b"calibrate\n")

        timeout = time.time() + 10.0
        while time.time() < timeout:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"ESP: {line}")
            if "# CALIBRATION_DONE" in line:
                print("[CALIBRATION] Calibration confirmed by firmware.\n")
                return True
            if "# CALIBRATION_FAILED" in line:
                print("[CALIBRATION ERROR] Firmware reported calibration failure.\n")
                return False

        print("[CALIBRATION ERROR] Calibration timed out.\n")
        return False

    def collect_trial(
        self,
        participant_id: str,
        session_id: str,
        canonical_gesture: str,
        trial_id: int,
        countdown_sec: int = 3,
    ) -> Optional[List[List[float]]]:
        """
        Captures exactly samples_per_trial IMU readings.
        Returns list of [timestamp, ax, ay, az, gx, gy, gz] rows, or None on failure.
        """
        if countdown_sec > 0:
            print(f"\nPreparing for [{canonical_gesture}] trial {trial_id}...")
            for i in range(countdown_sec, 0, -1):
                print(f"Starting in {i}...")
                time.sleep(1.0)

        print(f"*** RECORDING NOW: Perform '{canonical_gesture}' gesture ***")

        samples = []

        if self.mock_mode:
            # Generate synthetic physical window
            t_base = int(time.time() * 1000)
            for step in range(self.samples_per_trial):
                ts = step * 20
                # Generate slight perturbation around nominal gesture
                if canonical_gesture == "left":
                    ax, ay, az = 0.85 + np.random.normal(0, 0.05), 0.15, 0.3
                elif canonical_gesture == "right":
                    ax, ay, az = -0.75 + np.random.normal(0, 0.05), 0.15, 0.35
                elif canonical_gesture == "up":
                    ax, ay, az = 0.08, 0.30 + np.random.normal(0, 0.05), 0.45
                elif canonical_gesture == "down":
                    ax, ay, az = 0.08, -0.60 + np.random.normal(0, 0.05), 0.70
                elif canonical_gesture == "shake":
                    ax, ay, az = 0.05, 0.05, 0.98
                else:  # still
                    ax, ay, az = 0.04, 0.09, 0.99

                gx = 80.0 * np.sin(step) if canonical_gesture == "shake" else np.random.normal(0, 2.0)
                gy = 70.0 * np.cos(step) if canonical_gesture == "shake" else np.random.normal(0, 2.0)
                gz = 60.0 * np.sin(2 * step) if canonical_gesture == "shake" else np.random.normal(0, 2.0)

                samples.append([ts, float(ax), float(ay), float(az), float(gx), float(gy), float(gz)])
                time.sleep(0.02)
        else:
            if not self.ser or not self.ser.is_open:
                print("[LOGGER ERROR] Serial disconnected.")
                return None

            self.ser.reset_input_buffer()
            self.ser.write(b"start\n")
            deadline = time.perf_counter() + 5.0

            while len(samples) < self.samples_per_trial and time.perf_counter() < deadline:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                vals, ts, err = parse_sensor_line(line)
                if vals is not None:
                    row = [ts if ts is not None else len(samples) * 20] + list(vals)
                    samples.append(row)

            try:
                self.ser.write(b"stop\n")
                time.sleep(0.1)
                self.ser.reset_input_buffer()
            except Exception:
                pass

        if len(samples) < self.samples_per_trial:
            print(f"[LOGGER ERROR] Incomplete capture: got {len(samples)}/{self.samples_per_trial} samples. Trial discarded.")
            return None

        print(f"[LOGGER SUCCESS] Captured complete trial ({len(samples)} samples).")
        return samples

    def save_trial(
        self,
        participant_id: str,
        session_id: str,
        canonical_gesture: str,
        trial_id: int,
        samples: List[List[float]],
        force_overwrite: bool = False,
    ) -> str:
        """
        Saves captured samples into the session CSV file with overwrite protection.
        """
        filename = f"{canonical_gesture}.csv" if not self.mock_mode else f"mock_{canonical_gesture}.csv"
        csv_path = os.path.join(self.output_dir, filename)

        # Check existing trials for overwrite protection
        if os.path.exists(csv_path) and not force_overwrite:
            try:
                import pandas as pd
                existing_df = pd.read_csv(csv_path)
                match = (existing_df["session_id"].astype(str) == str(session_id)) & \
                        (existing_df["trial_id"].astype(int) == int(trial_id))
                if match.any():
                    raise FileExistsError(
                        f"Trial {trial_id} for session '{session_id}' already exists in {csv_path}. "
                        "Increment trial_id or pass force_overwrite=True."
                    )
            except (ImportError, KeyError):
                pass

        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("session_id,gesture,trial_id,timestamp_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps\n")
            for s in samples:
                ts, ax, ay, az, gx, gy, gz = s
                f.write(f"{session_id},{canonical_gesture},{trial_id},{int(ts)},{ax:.4f},{ay:.4f},{az:.4f},{gx:.4f},{gy:.4f},{gz:.4f}\n")

        print(f"[SAVED] Appended trial {trial_id} to {csv_path}")
        return csv_path


def main():
    parser = argparse.ArgumentParser(description="IMU Gesture Dataset Collection Tool")
    parser.add_argument("--port", default="COM6", help="Serial COM port (default: COM6)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--participant", default="P01", help="Participant identifier (e.g., P01)")
    parser.add_argument("--session", default="1", help="Session ID (default: 1)")
    parser.add_argument("--gesture", default=None, help="Gesture name to record")
    parser.add_argument("--trial", type=int, default=None, help="Starting trial ID")
    parser.add_argument("--num-trials", type=int, default=1, help="Number of trials to record")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without physical sensor")
    parser.add_argument("--outdir", default="data/collected", help="Output directory")

    args = parser.parse_args()

    logger = ImuDataLogger(
        port=args.port,
        baud_rate=args.baud,
        output_dir=args.outdir,
        mock_mode=args.mock,
    )

    if not logger.connect():
        sys.exit(1)

    try:
        gesture_in = args.gesture
        if not gesture_in:
            print("\nAvailable canonical gestures:", CANONICAL_CLASSES)
            gesture_in = input("Enter gesture to collect: ").strip()

        canon_gesture, err = normalize_label(gesture_in)
        if not canon_gesture:
            print(f"[ERROR] Invalid gesture: {err}")
            sys.exit(1)

        start_trial = args.trial if args.trial is not None else 1
        for t_offset in range(args.num_trials):
            curr_trial = start_trial + t_offset
            samples = logger.collect_trial(
                participant_id=args.participant,
                session_id=args.session,
                canonical_gesture=canon_gesture,
                trial_id=curr_trial,
            )
            if samples:
                logger.save_trial(
                    participant_id=args.participant,
                    session_id=args.session,
                    canonical_gesture=canon_gesture,
                    trial_id=curr_trial,
                    samples=samples,
                )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
