"""
test_software_suite.py
Comprehensive Offline Validation Test Suite for IMU Gesture Recognition

Automated verification covering:
1. Canonical label normalization and PowerPoint command mapping.
2. Preprocessing consistency (feature shapes, Z-score normalizer fit/transform).
3. Trained model artifact compatibility (1D-CNN and MLP loading and predicting).
4. Trial-boundary preservation and isolation (no mixing of consecutive trials).
5. Quarantining of invalid, incomplete, or malformed data rows.
6. Confidence threshold rejection and repetition cooldown logic.
7. Dry-Run safety verification (zero keystrokes dispatched in dry-run mode).
8. End-to-end hardware-free CSV replay execution.
"""

import os
import sys
import unittest
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.gesture_schema import (
    CANONICAL_CLASSES,
    LABEL_SYNONYMS,
    DEFAULT_POWERPOINT_COMMANDS,
    normalize_label,
    get_command_for_gesture,
)
from src.data_pipeline import (
    FEATURE_COLS,
    WINDOW_SIZE,
    NUM_CHANNELS,
    ZScoreNormalizer,
    validate_window,
    parse_sensor_line,
)
from src.replay_source import CsvReplaySource, ReplayTrial
from src.recognition_engine import GestureRecognitionEngine, CommandDispatcher, RecognitionResult


class TestCanonicalLabelsAndCommands(unittest.TestCase):
    """Verifies label mapping, synonym translation, and PowerPoint command dictionary."""

    def test_canonical_classes_count(self):
        self.assertEqual(len(CANONICAL_CLASSES), 6)
        self.assertIn("left", CANONICAL_CLASSES)
        self.assertIn("right", CANONICAL_CLASSES)
        self.assertIn("up", CANONICAL_CLASSES)
        self.assertIn("down", CANONICAL_CLASSES)
        self.assertIn("shake", CANONICAL_CLASSES)
        self.assertIn("still", CANONICAL_CLASSES)

    def test_synonym_normalization(self):
        cases = {
            "tilt_left": "left",
            "swipe_left": "left",
            "LEFT": "left",
            "tilt_right": "right",
            "swipe_right": "right",
            "tilt_up": "up",
            "tilt_down": "down",
            "shake": "shake",
            "still": "still",
        }
        for raw, expected in cases.items():
            norm, err = normalize_label(raw)
            self.assertEqual(norm, expected, f"Failed for {raw}")
            self.assertIsNone(err)

    def test_unknown_label_rejection(self):
        # 'n' from tilt_up.csv trial 15 should be reported as invalid
        norm, err = normalize_label("n")
        self.assertIsNone(norm)
        self.assertIn("Unknown or unmapped label", err)

        norm_rand, err_rand = normalize_label("wave_hand")
        self.assertIsNone(norm_rand)
        self.assertIsNotNone(err_rand)

    def test_powerpoint_commands_mapping(self):
        left_cmd = get_command_for_gesture("left")
        self.assertEqual(left_cmd["key"], "left")
        self.assertEqual(left_cmd["action"], "previous_slide")

        right_cmd = get_command_for_gesture("right")
        self.assertEqual(right_cmd["key"], "right")
        self.assertEqual(right_cmd["action"], "next_slide")

        shake_cmd = get_command_for_gesture("shake")
        self.assertEqual(shake_cmd["key"], "esc")
        self.assertEqual(shake_cmd["action"], "end_presentation")

        still_cmd = get_command_for_gesture("still")
        self.assertIsNone(still_cmd["key"])
        self.assertEqual(still_cmd["action"], "none")


class TestDataPipelineAndNormalization(unittest.TestCase):
    """Verifies window validation, parsing, and non-leaking Z-score normalization."""

    def test_window_validation(self):
        valid_win = np.zeros((51, 6), dtype=np.float32)
        is_val, err = validate_window(valid_win)
        self.assertTrue(is_val)
        self.assertIsNone(err)

        # Invalid shape
        bad_shape = np.zeros((47, 6), dtype=np.float32)
        is_val, err = validate_window(bad_shape)
        self.assertFalse(is_val)
        self.assertIn("does not match expected", err)

        # Non-finite values
        nan_win = np.zeros((51, 6), dtype=np.float32)
        nan_win[10, 2] = np.nan
        is_val, err = validate_window(nan_win)
        self.assertFalse(is_val)
        self.assertIn("non-finite", err)

    def test_normalizer_train_only_fit(self):
        # Synthetic train windows (10 windows of 51x6)
        np.random.seed(42)
        X_train = np.random.normal(loc=5.0, scale=2.0, size=(10, 51, 6)).astype(np.float32)
        norm = ZScoreNormalizer()
        self.assertFalse(norm.is_fitted)

        norm.fit(X_train)
        self.assertTrue(norm.is_fitted)
        self.assertEqual(norm.mean.shape, (6,))
        self.assertEqual(norm.std.shape, (6,))

        # Verify normalized train has mean ~ 0, std ~ 1
        X_norm = norm.transform(X_train)
        np.testing.assert_allclose(X_norm.reshape(-1, 6).mean(axis=0), np.zeros(6), atol=1e-5)
        np.testing.assert_allclose(X_norm.reshape(-1, 6).std(axis=0), np.ones(6), atol=1e-3)

    def test_sensor_line_parser(self):
        serial_line = "1050,-0.444,-0.061,0.827,-18.71,-24.16,95.43"
        vals, ts, err = parse_sensor_line(serial_line)
        self.assertIsNotNone(vals)
        self.assertEqual(len(vals), 6)
        self.assertEqual(ts, 1050)
        self.assertIsNone(err)

        csv_line = "1,shake,3,1050,-0.444,-0.061,0.827,-18.71,-24.16,95.43"
        vals, ts, err = parse_sensor_line(csv_line)
        self.assertIsNotNone(vals)
        self.assertEqual(len(vals), 6)
        self.assertEqual(ts, 1050)
        self.assertIsNone(err)

        malformed = "1050,corrupt,value"
        vals, ts, err = parse_sensor_line(malformed)
        self.assertIsNone(vals)
        self.assertIsNotNone(err)


class TestModelArtifactsAndInference(unittest.TestCase):
    """Verifies both original and newly trained models load and predict properly."""

    @classmethod
    def setUpClass(cls):
        cls.cnn_path = os.path.join(REPO_ROOT, "models", "imu_gesture_cnn.keras")
        cls.mlp_path = os.path.join(REPO_ROOT, "models", "imu_gesture_mlp.pkl")
        cls.labels_path = os.path.join(REPO_ROOT, "models", "label_classes.npy")
        cls.mean_path = os.path.join(REPO_ROOT, "models", "feature_mean.npy")
        cls.std_path = os.path.join(REPO_ROOT, "models", "feature_std.npy")

    def test_artifacts_exist(self):
        self.assertTrue(os.path.exists(self.cnn_path))
        self.assertTrue(os.path.exists(self.mlp_path))
        self.assertTrue(os.path.exists(self.labels_path))
        self.assertTrue(os.path.exists(self.mean_path))
        self.assertTrue(os.path.exists(self.std_path))

    def test_cnn_engine_prediction(self):
        norm = ZScoreNormalizer.load(self.mean_path, self.std_path)
        engine = GestureRecognitionEngine(
            model_path=self.cnn_path,
            labels_path=self.labels_path,
            normalizer=norm,
            dry_run=True
        )
        self.assertEqual(engine.model_type, "cnn")

        # Create still test window
        still_win = np.zeros((51, 6), dtype=np.float32)
        still_win[:, 2] = 1.0  # +1g gravity on z-axis
        res = engine.process_window(still_win)

        self.assertIsInstance(res, RecognitionResult)
        self.assertIn(res.predicted_label, CANONICAL_CLASSES)
        self.assertTrue(res.dry_run)
        self.assertFalse(res.action_dispatched)

    def test_mlp_engine_prediction(self):
        norm = ZScoreNormalizer.load(self.mean_path, self.std_path)
        engine = GestureRecognitionEngine(
            model_path=self.mlp_path,
            labels_path=self.labels_path,
            normalizer=norm,
            dry_run=True
        )
        self.assertEqual(engine.model_type, "mlp")

        left_win = np.zeros((51, 6), dtype=np.float32)
        left_win[:, 0] = 0.8  # +0.8g roll on x-axis
        left_win[:, 2] = 0.3
        res = engine.process_window(left_win)

        self.assertIsInstance(res, RecognitionResult)
        self.assertIn(res.predicted_label, CANONICAL_CLASSES)
        self.assertTrue(res.dry_run)


class TestReplaySourceAndTrialBoundaries(unittest.TestCase):
    """Verifies CSV replay preserves trial boundaries and quarantines bad rows."""

    def test_replay_load_dataset_and_quarantine(self):
        dataset_path = os.path.join(REPO_ROOT, "data", "raw_new")
        source = CsvReplaySource()
        valid_c, quaran_c = source.load(dataset_path)

        # In raw_new, there are 120 trials recorded, but trial 15 of tilt_up has label 'n'.
        # That single trial must be quarantined!
        self.assertEqual(valid_c, 119)
        self.assertEqual(quaran_c, 1)

        # Check quarantine log
        q_entry = source.quarantine_log[0]
        self.assertEqual(q_entry["raw_gesture"], "n")
        self.assertIn("Label error", q_entry["reason"])

    def test_replay_stepping_and_trial_isolation(self):
        dataset_path = os.path.join(REPO_ROOT, "data", "raw_new")
        source = CsvReplaySource()
        source.load(dataset_path)

        t1 = source.step_next()
        t2 = source.step_next()

        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)
        # Ensure separate trials
        self.assertNotEqual(t1.trial_key, t2.trial_key)
        self.assertEqual(t1.data.shape, (51, 6))
        self.assertEqual(t2.data.shape, (51, 6))


class TestSafetyAndCommandDispatcher(unittest.TestCase):
    """Verifies that Dry Run mode strictly prevents OS keystroke execution."""

    def test_dry_run_suppression(self):
        dispatcher = CommandDispatcher(dry_run=True)
        cmd, dispatched, msg = dispatcher.dispatch("right")

        self.assertFalse(dispatched)
        self.assertIn("[DRY RUN]", msg)
        self.assertEqual(cmd["key"], "right")

    def test_cooldown_suppression(self):
        norm = ZScoreNormalizer.load(
            os.path.join(REPO_ROOT, "models", "feature_mean.npy"),
            os.path.join(REPO_ROOT, "models", "feature_std.npy")
        )
        engine = GestureRecognitionEngine(
            model_path=os.path.join(REPO_ROOT, "models", "imu_gesture_cnn.keras"),
            labels_path=os.path.join(REPO_ROOT, "models", "label_classes.npy"),
            normalizer=norm,
            cooldown_seconds=2.0,
            dry_run=True
        )

        sample = np.zeros((51, 6), dtype=np.float32)
        sample[:, 0] = 0.85  # left tilt
        t0 = 1000.0

        # First trigger
        r1 = engine.process_window(sample, timestamp=t0)
        self.assertFalse(r1.is_suppressed_cooldown)

        # Second trigger 0.5s later (within 2.0s cooldown)
        r2 = engine.process_window(sample, timestamp=t0 + 0.5)
        self.assertTrue(r2.is_suppressed_cooldown)
        self.assertEqual(r2.filtered_label, "unknown")

        # Verify status separation: correct prediction remains marked MATCHED during cooldown
        status_text, status_color = r2.get_recognition_status(expected_gesture="left")
        self.assertEqual(status_text, "Prediction Matched")

        self.assertEqual(status_color, "#10B981")

        dispatch_text, dispatch_color = r2.get_dispatch_status()
        self.assertEqual(dispatch_text, "Command Suppressed by Cooldown")
        self.assertEqual(dispatch_color, "#F59E0B")


class TestInternalPreviewAndReplayRestart(unittest.TestCase):
    """Verifies SlidePreviewWidget transitions and ReplaySource completion/restart."""

    def test_internal_slide_preview_transitions(self):
        import tkinter as tk
        from src.ui_widgets import SlidePreviewWidget

        root = tk.Tk()
        root.withdraw()  # Headless test without opening display window

        try:
            preview = SlidePreviewWidget(root)
            self.assertEqual(preview.current_slide_idx, 0)
            self.assertFalse(preview.is_black_screen)

            # Advance
            preview.handle_gesture_action("next_slide")
            self.assertEqual(preview.current_slide_idx, 1)

            # Previous
            preview.handle_gesture_action("previous_slide")
            self.assertEqual(preview.current_slide_idx, 0)

            # Black screen toggle
            preview.handle_gesture_action("black_screen")
            self.assertTrue(preview.is_black_screen)

            # Action removes black screen
            preview.handle_gesture_action("next_slide")
            self.assertFalse(preview.is_black_screen)
            self.assertEqual(preview.current_slide_idx, 1)

            # End presentation
            preview.handle_gesture_action("end_presentation")
            self.assertEqual(preview.current_slide_idx, len(preview.SAMPLE_DECK) - 1)

            # Start presentation
            preview.handle_gesture_action("start_presentation")
            self.assertEqual(preview.current_slide_idx, 0)
        finally:
            root.destroy()

    def test_replay_completion_and_restart(self):
        dataset_path = os.path.join(REPO_ROOT, "data", "raw_new")
        source = CsvReplaySource(dataset_path)
        valid_trials = source.valid_trials
        self.assertEqual(len(valid_trials), 119)

        # Step to end
        count = 0
        while source.step_next() is not None:
            count += 1
        self.assertEqual(count, len(source.trials))

        # Further steps return None
        self.assertIsNone(source.step_next())

        # Restart via seek(0)
        source.seek(0)
        first_trial = source.step_next()
        self.assertIsNotNone(first_trial)
        self.assertEqual(source.current_index, 1)


if __name__ == "__main__":
    unittest.main()

