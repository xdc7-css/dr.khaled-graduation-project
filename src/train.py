"""
train.py
Reproducible Preprocessing, Training, and Evaluation Pipeline

Key Engineering Principles:
1. Validates and audits dataset before training; quarantines malformed or ambiguous trials (e.g. 'n' typo) with an audit trail.
2. Fits Z-score normalization strictly on the training set (zero data leakage).
3. Evaluates models on a strictly held-out test split (15%) distinct from validation data (15%).
4. Evaluates a simple rule-based baseline on the EXACT SAME held-out test split.
5. Saves new artifacts to 'models/reproducible/' with full provenance metadata.json, preserving legacy artifacts intact.
"""

import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import joblib

from src.data_pipeline import FEATURE_COLS, WINDOW_SIZE, NUM_CHANNELS, ZScoreNormalizer
from src.gesture_schema import CANONICAL_CLASSES, normalize_label


class TrainingPipeline:
    """Orchestrates dataset loading, splitting, normalization, training, and evaluation."""

    def __init__(
        self,
        data_dir: str = "data/raw_new",
        output_models_dir: str = "models/reproducible",
        random_state: int = 42,
    ):
        self.data_dir: str = os.path.abspath(data_dir)
        self.output_models_dir: str = os.path.abspath(output_models_dir)
        self.random_state: int = random_state
        os.makedirs(self.output_models_dir, exist_ok=True)

        self.normalizer = ZScoreNormalizer()
        self.label_encoder = LabelEncoder()
        self.quarantine_log: List[Dict[str, Any]] = []

    def load_and_validate_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Loads CSV files from data_dir, audits every trial, and extracts valid 51x6 windows.
        Quarantines ambiguous or malformed trials.
        """
        import glob
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV dataset files found in {self.data_dir}")

        windows = []
        labels = []
        trial_keys = []

        for fpath in csv_files:
            df = pd.read_csv(fpath)
            has_sess = "session_id" in df.columns
            has_gest = "gesture" in df.columns
            has_trial = "trial_id" in df.columns

            group_cols = []
            if has_sess: group_cols.append("session_id")
            if has_gest: group_cols.append("gesture")
            if has_trial: group_cols.append("trial_id")

            groups = df.groupby(group_cols, sort=False) if group_cols else [(os.path.basename(fpath), df)]

            for key, grp in groups:
                raw_gest = str(grp["gesture"].iloc[0]) if has_gest else os.path.splitext(os.path.basename(fpath))[0]
                sess = str(grp["session_id"].iloc[0]) if has_sess else "1"
                tid = int(grp["trial_id"].iloc[0]) if has_trial else 0
                trial_key = f"{sess}_{raw_gest}_{tid}"

                # Validate gesture label
                canon_label, err_label = normalize_label(raw_gest)
                if canon_label is None:
                    self.quarantine_log.append({
                        "file": fpath,
                        "trial_key": trial_key,
                        "raw_gesture": raw_gest,
                        "trial_id": tid,
                        "rows": len(grp),
                        "reason": f"Ambiguous/Unrecognized label: {err_label}. Quarantined."
                    })
                    continue

                if len(grp) != WINDOW_SIZE:
                    self.quarantine_log.append({
                        "file": fpath,
                        "trial_key": trial_key,
                        "raw_gesture": raw_gest,
                        "trial_id": tid,
                        "rows": len(grp),
                        "reason": f"Invalid row count: {len(grp)} (expected {WINDOW_SIZE})"
                    })
                    continue

                grp_sorted = grp.sort_values("timestamp_ms") if "timestamp_ms" in grp.columns else grp
                sample = grp_sorted[FEATURE_COLS].values.astype(np.float32)

                if not np.all(np.isfinite(sample)):
                    self.quarantine_log.append({
                        "file": fpath,
                        "trial_key": trial_key,
                        "raw_gesture": raw_gest,
                        "trial_id": tid,
                        "reason": "Non-finite values (NaN/Inf) detected."
                    })
                    continue

                windows.append(sample)
                labels.append(canon_label)
                trial_keys.append(trial_key)

        X = np.stack(windows)
        y = np.array(labels)
        return X, y, trial_keys

    def evaluate_rule_baseline(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates a simple static tilt/gyro heuristic on the EXACT SAME held-out test split.
        """
        preds = []
        for w in X_test:
            # w: shape (51, 6) -> [ax, ay, az, gx, gy, gz]
            ax_m = np.mean(w[:, 0])
            ay_m = np.mean(w[:, 1])
            az_m = np.mean(w[:, 2])
            gyro_std = np.std(w[:, 3:])
            gyro_mag = np.mean(np.sqrt(np.sum(w[:, 3:]**2, axis=-1)))

            if gyro_std > 20.0 and az_m > 0.85 and abs(ax_m) < 0.3 and abs(ay_m) < 0.3:
                preds.append("shake")
            elif gyro_mag < 15.0 and az_m > 0.85 and abs(ax_m) < 0.3 and abs(ay_m) < 0.3:
                preds.append("still")
            elif ax_m > 0.4:
                preds.append("left")
            elif ax_m < -0.3:
                preds.append("right")
            elif ay_m < -0.2:
                preds.append("down")
            elif ay_m > 0.0 and az_m < 0.7:
                preds.append("up")
            elif gyro_std > 20.0:
                preds.append("shake")
            else:
                preds.append("still")

        preds_arr = np.array(preds)
        acc = float(accuracy_score(y_test, preds_arr))
        f1 = float(f1_score(y_test, preds_arr, average="macro", zero_division=0))
        return {
            "accuracy": acc,
            "macro_f1": f1,
            "predictions": preds,
            "true_labels": list(y_test)
        }

    def run(self, train_cnn: bool = True) -> Dict[str, Any]:
        print("=" * 60)
        print("STARTING REPRODUCIBLE IMU GESTURE TRAINING PIPELINE")
        print("=" * 60)

        # 1. Ingestion & Validation
        X, y, trial_keys = self.load_and_validate_dataset()
        print(f"[DATASET] Validated windows: {len(X)}, Shape: {X.shape}")
        print(f"[DATASET] Unique classes: {np.unique(y)}, Counts: {pd.Series(y).value_counts().to_dict()}")

        if self.quarantine_log:
            q_path = os.path.join(self.output_models_dir, "dataset_quarantine_log.json")
            with open(q_path, "w", encoding="utf-8") as f:
                json.dump(self.quarantine_log, f, indent=2)
            print(f"[AUDIT] Quarantined {len(self.quarantine_log)} anomalous trials. Saved audit record to {q_path}")

        # 2. Label Encoding
        self.label_encoder.fit(CANONICAL_CLASSES)
        y_encoded = self.label_encoder.transform(y)

        # 3. Deterministic Splitting: 70% Train, 15% Val, 15% Test
        indices = np.arange(len(X))
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            indices, y_encoded, test_size=0.30, random_state=self.random_state, stratify=y_encoded
        )
        idx_val, idx_test, y_val, y_test = train_test_split(
            idx_temp, y_temp, test_size=0.50, random_state=self.random_state, stratify=y_temp
        )

        X_train, y_train_raw = X[idx_train], y[idx_train]
        X_val, y_val_raw = X[idx_val], y[idx_val]
        X_test, y_test_raw = X[idx_test], y[idx_test]

        print(f"[SPLITS] Train: {len(X_train)} (70%), Val: {len(X_val)} (15%), Test: {len(X_test)} (15%)")

        # 4. Train-Only Normalization Fitting (No Leakage)
        self.normalizer.fit(X_train)
        X_train_norm = self.normalizer.transform(X_train)
        X_val_norm = self.normalizer.transform(X_val)
        X_test_norm = self.normalizer.transform(X_test)

        mean_save = os.path.join(self.output_models_dir, "feature_mean.npy")
        std_save = os.path.join(self.output_models_dir, "feature_std.npy")
        labels_save = os.path.join(self.output_models_dir, "label_classes.npy")
        self.normalizer.save(mean_save, std_save)
        np.save(labels_save, np.array(CANONICAL_CLASSES))
        print(f"[NORMALIZATION] Saved parameters to {mean_save} and {std_save}")

        # 5. Evaluate Rule-Based Baseline on Held-Out Test Set
        rule_results = self.evaluate_rule_baseline(X_test, y_test_raw)
        print(f"[BASELINE] Simple Tilt Rule Accuracy on Held-Out Test Set: {rule_results['accuracy']:.4f} (Macro F1: {rule_results['macro_f1']:.4f})")

        # 6. Train MLP Model
        print("\n--- Training MLP Baseline ---")
        X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
        X_val_flat = X_val_norm.reshape(len(X_val_norm), -1)
        X_test_flat = X_test_norm.reshape(len(X_test_norm), -1)

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=300, random_state=self.random_state)
        mlp.fit(X_train_flat, y_train)

        mlp_val_pred = mlp.predict(X_val_flat)
        mlp_test_pred = mlp.predict(X_test_flat)

        mlp_val_acc = accuracy_score(y_val, mlp_val_pred)
        mlp_test_acc = accuracy_score(y_test, mlp_test_pred)
        mlp_test_f1 = f1_score(y_test, mlp_test_pred, average="macro")

        mlp_path = os.path.join(self.output_models_dir, "imu_gesture_mlp.pkl")
        joblib.dump(mlp, mlp_path)
        print(f"[MLP] Val Accuracy: {mlp_val_acc:.4f}, Test Accuracy: {mlp_test_acc:.4f}, Test Macro F1: {mlp_test_f1:.4f}")
        print(f"[MLP] Saved model artifact to {mlp_path}")

        # 7. Train 1D-CNN (if enabled)
        cnn_test_acc, cnn_test_f1 = None, None
        if train_cnn:
            print("\n--- Training 1D-CNN Main Architecture ---")
            import keras
            from keras.models import Sequential
            from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
            from keras.utils import to_categorical

            y_train_cat = to_categorical(y_train, num_classes=len(CANONICAL_CLASSES))
            y_val_cat = to_categorical(y_val, num_classes=len(CANONICAL_CLASSES))
            y_test_cat = to_categorical(y_test, num_classes=len(CANONICAL_CLASSES))

            cnn = Sequential([
                Input(shape=(WINDOW_SIZE, NUM_CHANNELS)),
                Conv1D(filters=32, kernel_size=3, activation="relu"),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=64, kernel_size=3, activation="relu"),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(64, activation="relu"),
                Dropout(0.3),
                Dense(len(CANONICAL_CLASSES), activation="softmax")
            ])
            cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            # Clean validation isolation: train monitors validation data, NEVER test data!
            cnn.fit(
                X_train_norm, y_train_cat,
                epochs=15,
                batch_size=8,
                validation_data=(X_val_norm, y_val_cat),
                verbose=0
            )

            test_loss, cnn_test_acc = cnn.evaluate(X_test_norm, y_test_cat, verbose=0)
            cnn_preds = np.argmax(cnn.predict(X_test_norm, verbose=0), axis=1)
            cnn_test_f1 = f1_score(y_test, cnn_preds, average="macro")

            cnn_path = os.path.join(self.output_models_dir, "imu_gesture_cnn.keras")
            cnn.save(cnn_path)
            print(f"[CNN] Test Accuracy: {cnn_test_acc:.4f}, Test Macro F1: {cnn_test_f1:.4f}")
            print(f"[CNN] Saved model artifact to {cnn_path}")

        # 8. Export Metadata Record
        metadata = {
            "pipeline_version": "2.0-reproducible",
            "feature_columns": FEATURE_COLS,
            "window_size": WINDOW_SIZE,
            "canonical_classes": CANONICAL_CLASSES,
            "split_counts": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
            "held_out_test_metrics": {
                "rule_baseline_accuracy": rule_results["accuracy"],
                "rule_baseline_macro_f1": rule_results["macro_f1"],
                "mlp_test_accuracy": float(mlp_test_acc),
                "mlp_test_macro_f1": float(mlp_test_f1),
                "cnn_test_accuracy": float(cnn_test_acc) if cnn_test_acc is not None else None,
                "cnn_test_macro_f1": float(cnn_test_f1) if cnn_test_f1 is not None else None,
            },
            "quarantined_trials_count": len(self.quarantine_log),
            "random_state": self.random_state,
            "data_provenance_note": (
                "Evaluated on single-user, single-session recordings (session_1). "
                "Subject-independent generalization cannot be verified from available metadata."
            )
        }

        meta_path = os.path.join(self.output_models_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n[METADATA] Exported full experiment metadata to {meta_path}")

        return metadata


def main():
    parser = argparse.ArgumentParser(description="Reproducible IMU Gesture Training Pipeline")
    parser.add_argument("--data-dir", default="data/raw_new", help="Path to raw CSV dataset")
    parser.add_argument("--out-dir", default="models/reproducible", help="Output directory for new model artifacts")
    parser.add_argument("--skip-cnn", action="store_true", help="Skip CNN training (MLP only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")

    args = parser.parse_args()

    pipeline = TrainingPipeline(
        data_dir=args.data_dir,
        output_models_dir=args.out_dir,
        random_state=args.seed,
    )
    pipeline.run(train_cnn=not args.skip_cnn)


if __name__ == "__main__":
    main()
