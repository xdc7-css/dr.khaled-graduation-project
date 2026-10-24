"""
replay_demo.py
Hardware-Free CLI Replay Demonstration for IMU Gesture Recognition

Demonstrates:
1. Loading recorded trials from CSV.
2. Window validation and Z-score normalization.
3. Model prediction (1D-CNN or MLP).
4. PowerPoint command mapping (Dry-Run by default).
5. Comprehensive summary table of expected vs. predicted gestures and commands.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
from typing import List
from src.data_pipeline import ZScoreNormalizer
from src.replay_source import CsvReplaySource, ReplayTrial
from src.recognition_engine import GestureRecognitionEngine


def main():
    parser = argparse.ArgumentParser(description="Hardware-Free CLI IMU Gesture Replay")
    parser.add_argument("--dataset", default="data/raw_new", help="Path to dataset directory or single CSV")
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn", help="Model architecture")
    parser.add_argument("--limit", type=int, default=12, help="Number of trials to replay (default: 12)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold")
    parser.add_argument("--cooldown", type=float, default=0.0, help="Action cooldown in seconds (default: 0.0 for batch replay)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Simulate without real keystrokes")

    args = parser.parse_args()


    cnn_path = os.path.join(REPO_ROOT, "models", "imu_gesture_cnn.keras")
    mlp_path = os.path.join(REPO_ROOT, "models", "imu_gesture_mlp.pkl")
    labels_path = os.path.join(REPO_ROOT, "models", "label_classes.npy")
    mean_path = os.path.join(REPO_ROOT, "models", "feature_mean.npy")
    std_path = os.path.join(REPO_ROOT, "models", "feature_std.npy")

    model_path = cnn_path if args.model == "cnn" else mlp_path

    print("=" * 75)
    print("IMU GESTURE REPLAY - HARDWARE-FREE VALIDATION DEMO")
    print("=" * 75)
    print(f"Dataset Path : {args.dataset}")
    print(f"Model Chosen : {args.model.upper()} ({os.path.basename(model_path)})")
    print(f"Mode         : {'DRY RUN (Simulated)' if args.dry_run else 'LIVE KEYSTROKES'}")
    print(f"Confidence   : Threshold >= {args.threshold}")
    print("-" * 75)

    normalizer = ZScoreNormalizer.load(mean_path, std_path)
    engine = GestureRecognitionEngine(
        model_path=model_path,
        labels_path=labels_path,
        normalizer=normalizer,
        confidence_threshold=args.threshold,
        cooldown_seconds=args.cooldown,
        dry_run=args.dry_run,
    )


    replay_source = CsvReplaySource(args.dataset)
    valid_trials = replay_source.valid_trials
    print(f"Loaded {len(valid_trials)} valid trials ({len(replay_source.quarantine_log)} quarantined).")

    trials_to_run = valid_trials[:args.limit] if args.limit > 0 else valid_trials

    print(f"\n{'Trial Key':<26} | {'Expected':<10} | {'Predicted':<10} | {'Conf':<6} | {'PowerPoint Action':<22} | {'Status'}")
    print("-" * 95)

    matches = 0
    for trial in trials_to_run:
        res = engine.process_window(trial.data)
        is_match = (res.filtered_label == trial.canonical_gesture)
        if is_match:
            matches += 1
            status_str = "MATCH"
        else:
            status_str = "MISMATCH/REJECTED"

        print(
            f"{trial.trial_key:<26} | "
            f"{str(trial.canonical_gesture):<10} | "
            f"{res.filtered_label:<10} | "
            f"{res.confidence*100:>5.1f}% | "
            f"{res.command.get('action', 'none'):<22} | "
            f"{status_str}"
        )

    print("-" * 95)
    print(f"Replay Summary: {matches}/{len(trials_to_run)} trials recognized correctly ({matches/len(trials_to_run)*100:.1f}%)")
    print("Notice: Replayed dataset consists of pre-recorded trials and does not prove real-world continuous spotting accuracy.")
    print("=" * 75)


if __name__ == "__main__":
    main()
