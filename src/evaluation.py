"""
evaluation.py
Comprehensive Reproducible Evaluation Engine

Computes:
1. Classification Accuracy and Macro F1.
2. Per-Class Precision, Recall, F1-Score, and Confusion Matrix.
3. Accepted vs Rejected vs Suppressed command tallies.
4. Latency breakdown (Model inference time vs Window buffering vs End-to-end latency).
5. Duplicate-command rate (enforcing state-machine guarantees).
6. False activations per minute (explicitly labeled 'Not measurable with current data' when only segmented trials exist).
7. Benchmark comparison (1D-CNN vs Baseline MLP vs Rule-based Baseline) on identical held-out splits.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.gesture_schema import CANONICAL_CLASSES, normalize_label
from src.data_pipeline import WINDOW_SIZE


@dataclass
class EvaluationReport:
    """Standardized academic evaluation metrics report."""
    model_name: str
    total_evaluated: int
    accuracy_pct: float
    macro_f1: float
    per_class_results: Dict[str, Dict[str, float]]
    confusion_matrix: Dict[str, Dict[str, int]]
    
    # Intentional command tallies
    accepted_commands: int
    suppressed_commands: int
    rejected_ambiguous: int
    duplicate_command_rate_pct: float
    false_activations_per_min: str
    
    # Latency breakdown (in milliseconds)
    inference_ms_mean: float
    inference_ms_p95: float
    window_buffer_ms: float
    end_to_end_est_ms: float
    
    timestamp_utc: str = ""


class RuleBasedBaseline:
    """
    Interpretable baseline using simple physical tilt and gyroscope thresholds.
    Used for comparative benchmark against neural network models.
    """

    def predict_window(self, window: np.ndarray) -> str:
        # window shape: (51, 6) -> [ax, ay, az, gx, gy, gz]
        # Calculate mean gravity tilt
        mean_ax = float(np.mean(window[:, 0]))
        mean_ay = float(np.mean(window[:, 1]))
        mean_az = float(np.mean(window[:, 2]))
        
        # Calculate dynamic gyro activity
        gyro_rms = float(np.sqrt(np.mean(window[:, 3:6] ** 2)))

        # Rule 1: High oscillation -> shake
        if gyro_rms > 40.0:
            return "shake"

        # Rule 2: Gravity alignment -> tilt directions
        if mean_ax < -0.35:
            return "left"
        elif mean_ax > 0.35:
            return "right"
        elif mean_ay > 0.35:
            return "up"
        elif mean_ay < -0.35:
            return "down"

        # Rule 3: Resting near vertical -> still
        return "still"


class EvaluationEngine:
    """
    Evaluates gesture recognition pipelines on held-out test splits
    without data leakage or test-set tuning.
    """

    def __init__(self, canonical_classes: Optional[List[str]] = None):
        self.classes = canonical_classes or list(CANONICAL_CLASSES)

    def evaluate_model(
        self,
        model_fn,
        test_windows: List[np.ndarray],
        true_labels: List[str],
        model_name: str = "1D-CNN",
        nominal_hz: float = 50.0
    ) -> EvaluationReport:
        """
        Runs comprehensive evaluation on a test split.
        model_fn: callable taking a (51, 6) window and returning (pred_label, confidence, is_suppressed, is_rejected)
        """
        assert len(test_windows) == len(true_labels), "Mismatch between windows and labels"
        total = len(test_windows)
        if total == 0:
            raise ValueError("Test set is empty.")

        y_true = []
        y_pred = []
        inference_times_ms = []

        accepted_count = 0
        suppressed_count = 0
        rejected_count = 0
        duplicates = 0
        last_dispatched = None

        for w, true_l in zip(test_windows, true_labels):
            t0 = time.perf_counter()
            pred_l, conf, suppressed, rejected = model_fn(w)
            t_ms = (time.perf_counter() - t0) * 1000.0
            inference_times_ms.append(t_ms)

            # Map true label to canonical form
            canon_true, _ = normalize_label(true_l)
            canon_true = canon_true or "unknown"

            y_true.append(canon_true)
            y_pred.append(pred_l)

            if rejected:
                rejected_count += 1
            elif suppressed:
                suppressed_count += 1
            else:
                accepted_count += 1
                if pred_l == last_dispatched and pred_l not in ["still", "unknown"]:
                    duplicates += 1
                last_dispatched = pred_l

        # Compute Metrics
        matches = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        acc_pct = (matches / total) * 100.0

        # Confusion Matrix
        cm = {c1: {c2: 0 for c2 in self.classes} for c1 in self.classes}
        for yt, yp in zip(y_true, y_pred):
            if yt in cm and yp in cm[yt]:
                cm[yt][yp] += 1

        # Per-class precision, recall, F1
        per_class = {}
        f1_scores = []
        for c in self.classes:
            tp = cm[c][c]
            fp = sum(cm[other][c] for other in self.classes if other != c)
            fn = sum(cm[c][other] for other in self.classes if other != c)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[c] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "support": tp + fn
            }
            if (tp + fn) > 0:
                f1_scores.append(f1)

        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        dup_rate = (duplicates / accepted_count * 100.0) if accepted_count > 0 else 0.0

        # Latency breakdown
        inf_mean = float(np.mean(inference_times_ms))
        inf_p95 = float(np.percentile(inference_times_ms, 95))
        buffer_ms = (WINDOW_SIZE / nominal_hz) * 1000.0  # 51 / 50Hz = 1020 ms
        e2e_ms = buffer_ms + inf_mean

        return EvaluationReport(
            model_name=model_name,
            total_evaluated=total,
            accuracy_pct=round(acc_pct, 2),
            macro_f1=round(macro_f1, 4),
            per_class_results=per_class,
            confusion_matrix=cm,
            accepted_commands=accepted_count,
            suppressed_commands=suppressed_count,
            rejected_ambiguous=rejected_count,
            duplicate_command_rate_pct=round(dup_rate, 2),
            false_activations_per_min="Not measurable with current data (requires unsegmented continuous recordings)",
            inference_ms_mean=round(inf_mean, 2),
            inference_ms_p95=round(inf_p95, 2),
            window_buffer_ms=round(buffer_ms, 1),
            end_to_end_est_ms=round(e2e_ms, 1),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    def format_summary_table(self, report: EvaluationReport) -> str:
        """Returns a clean ASCII summary table suitable for GUI and logs."""
        lines = [
            f"=== REPRODUCIBLE EVALUATION REPORT: {report.model_name.upper()} ===",
            f"Total Held-Out Trials: {report.total_evaluated}",
            f"Overall Accuracy:      {report.accuracy_pct:.2f}%",
            f"Macro F1-Score:        {report.macro_f1:.4f}",
            "",
            "--- Command Dispatch Statistics ---",
            f"Accepted Commands:     {report.accepted_commands}",
            f"Suppressed (Cooldown): {report.suppressed_commands}",
            f"Rejected (Ambiguous):  {report.rejected_ambiguous}",
            f"Duplicate Rate:        {report.duplicate_command_rate_pct:.1f}%",
            f"False Activations/Min: {report.false_activations_per_min}",
            "",
            "--- Latency Analysis ---",
            f"Model Inference:       {report.inference_ms_mean:.2f} ms (p95: {report.inference_ms_p95:.2f} ms)",
            f"Temporal Buffer:       {report.window_buffer_ms:.1f} ms (51 samples @ 50 Hz)",
            f"Est. End-to-End:       {report.end_to_end_est_ms:.1f} ms",
            "",
            "--- Per-Class Performance ---",
            f"{'Gesture':<10} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Support':<7}",
            "-" * 54
        ]
        for g, m in report.per_class_results.items():
            lines.append(f"{g:<10} | {m['precision']:<9.3f} | {m['recall']:<8.3f} | {m['f1_score']:<8.3f} | {m['support']:<7}")

        return "\n".join(lines)
