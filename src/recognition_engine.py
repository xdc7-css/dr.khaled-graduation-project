"""
recognition_engine.py
Core IMU Gesture Recognition & PowerPoint Command Dispatch Engine

Pipeline:
Window Validation -> Z-Score Normalization -> Model Inference ->
Motion Quality & Margin Rejection -> State Machine Evaluation ->
PowerPoint Command Mapping -> (Dry-Run / Live Keystroke Dispatch)

Guarantees:
1. 1 Intentional Gesture -> 1 Command (Enforced via GestureStateMachine).
2. Holding a tilted posture CANNOT repeatedly trigger commands.
3. Decoupled Concepts:
   Prediction -> Mapped Command -> Eligibility -> Dispatch Result.
4. Cooldown and threshold rejections have distinct, unambiguous reason codes.
"""

import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Optional keyboard automation for actual live PowerPoint execution
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    PYAUTOGUI_AVAILABLE = True
except Exception:
    PYAUTOGUI_AVAILABLE = False

from src.data_pipeline import ZScoreNormalizer, validate_window, WINDOW_SIZE, NUM_CHANNELS
from src.gesture_schema import (
    CANONICAL_CLASSES,
    DEFAULT_POWERPOINT_COMMANDS,
    get_command_for_gesture,
)
from src.state_machine import GestureStateMachine, GestureState, DecisionReason, NeutralConfig
from src.motion_rejection import MotionQualityRejector, RejectionDiagnostics


@dataclass
class RecognitionResult:
    """Comprehensive outcome of a single window inference cycle."""
    timestamp: float
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    filtered_label: str
    command: Dict[str, Any]
    is_suppressed_threshold: bool
    is_suppressed_cooldown: bool
    is_holding_blocked: bool
    dry_run: bool
    action_dispatched: bool
    status_detail: str
    # Diagnostic fields
    state_machine_state: str = "READY"
    suppression_reason: str = "NONE"
    class_margin: float = 0.0
    motion_energy: float = 0.0

    def get_recognition_status(self, expected_gesture: Optional[str] = None) -> Tuple[str, str]:
        """
        Returns (status_text, color_hex).
        Explicitly separates model prediction accuracy from command dispatch suppression.
        A correct prediction is ALWAYS 'Prediction Matched' even if suppressed by cooldown.
        """
        if self.is_suppressed_threshold:
            return "Low Confidence", "#F59E0B"

        if expected_gesture:
            if self.predicted_label.lower() == expected_gesture.lower():
                return "Prediction Matched", "#10B981"
            else:
                return "Prediction Differed", "#EF4444"

        return f"Predicted: {self.predicted_label.upper()}", "#38BDF8"

    def get_dispatch_status(self) -> Tuple[str, str]:
        """
        Returns (dispatch_status_text, color_hex).
        Accurately reports command eligibility and dispatch outcome.
        """
        if self.is_suppressed_threshold:
            return "Suppressed (Low Confidence)", "#64748B"
        if self.is_holding_blocked:
            return "Suppressed: Awaiting Neutral", "#F59E0B"
        if self.is_suppressed_cooldown:
            return "Command Suppressed by Cooldown", "#F59E0B"
        if self.suppression_reason == "INSUFFICIENT_CLASS_MARGIN":
            return "Rejected (Ambiguous Margin)", "#F59E0B"
        if self.suppression_reason == "INSUFFICIENT_MOTION_ENERGY":
            return "Rejected (Low Motion Energy)", "#F59E0B"
        if self.filtered_label in ["still", "unknown"] or self.command.get("action") == "none":
            return "No Action Mapped", "#64748B"
        if self.dry_run:
            return "Command Simulated (Dry Run)", "#38BDF8"
        if self.action_dispatched:
            return "Command Sent to Active Window", "#10B981"
        return "Command Suppressed", "#F59E0B"


class CommandDispatcher:
    """
    Safely dispatches keyboard commands to the operating system.
    CRITICAL: Defaults to DRY RUN so no accidental keystrokes are sent.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run: bool = dry_run
        self.command_map: Dict[str, Dict[str, Any]] = dict(DEFAULT_POWERPOINT_COMMANDS)

    def set_mapping(self, gesture: str, action: str, key: Optional[str], description: str) -> None:
        self.command_map[gesture] = {
            "action": action,
            "key": key,
            "description": description
        }

    def dispatch(self, gesture: str) -> Tuple[Dict[str, Any], bool, str]:
        """
        Dispatches keystroke if valid and not in dry_run mode.

        Returns:
            (command_dict, was_dispatched, status_message)
        """
        cmd = get_command_for_gesture(gesture, self.command_map)
        key = cmd.get("key")

        if key is None or cmd.get("action") == "none":
            return cmd, False, "No keystroke mapped for this state."

        if self.dry_run:
            msg = f"[DRY RUN] Would send key '{key}' ({cmd.get('description')}). Actual keystrokes disabled."
            return cmd, False, msg

        if not PYAUTOGUI_AVAILABLE:
            msg = f"[ERROR] PyAutoGUI not available. Cannot send key '{key}'."
            return cmd, False, msg

        try:
            pyautogui.press(key)
            msg = f"[LIVE DISPATCH] Sent keystroke '{key}' -> {cmd.get('description')} (PowerPoint must have focus)"
            return cmd, True, msg
        except Exception as e:
            msg = f"[DISPATCH ERROR] Failed to send key '{key}': {e}"
            return cmd, False, msg


class GestureRecognitionEngine:
    """
    Loads trained models, applies normalization, predicts gestures,
    filters by confidence/quality/state-machine, and maps to presentation commands.
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        normalizer: Optional[ZScoreNormalizer] = None,
        confidence_threshold: float = 0.75,
        shake_threshold: float = 0.55,
        cooldown_seconds: float = 1.2,
        dry_run: bool = True,
        neutral_config: Optional[NeutralConfig] = None
    ):
        self.model_path: str = os.path.abspath(model_path)
        self.labels_path: str = os.path.abspath(labels_path)
        self.confidence_threshold: float = confidence_threshold
        self.shake_threshold: float = shake_threshold
        self.cooldown_seconds: float = cooldown_seconds
        
        self.dispatcher = CommandDispatcher(dry_run=dry_run)
        self.normalizer: Optional[ZScoreNormalizer] = normalizer
        
        # Motion Quality Rejector & State Machine
        self.rejector = MotionQualityRejector(min_confidence=confidence_threshold)
        self.state_machine = GestureStateMachine(
            neutral_config=neutral_config,
            cooldown_seconds=cooldown_seconds
        )

        # Model state
        self.model = None
        self.model_type: str = "unknown"
        self.label_classes: List[str] = []
        
        self._load_labels()
        self._load_model()

    @property
    def dry_run(self) -> bool:
        return self.dispatcher.dry_run

    @dry_run.setter
    def dry_run(self, val: bool) -> None:
        self.dispatcher.dry_run = bool(val)

    def reset_state_machine(self, current_time: Optional[float] = None) -> None:
        """Resets the state machine and neutral tracking (e.g. on replay restart)."""
        self.state_machine.reset(new_state=GestureState.READY, current_time=current_time)

    def _load_labels(self) -> None:
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Label classes file not found: {self.labels_path}")
        labels_arr = np.load(self.labels_path, allow_pickle=True)
        self.label_classes = [str(l) for l in labels_arr]

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        ext = os.path.splitext(self.model_path)[1].lower()
        if ext == ".keras":
            import keras
            self.model = keras.models.load_model(self.model_path)
            self.model_type = "cnn"
        elif ext in [".pkl", ".joblib"]:
            import joblib
            self.model = joblib.load(self.model_path)
            self.model_type = "mlp"
        else:
            raise ValueError(f"Unsupported model extension: {ext} (expected .keras or .pkl)")

    def process_window(self, raw_window: np.ndarray, timestamp: Optional[float] = None) -> RecognitionResult:
        """
        Executes complete recognition pipeline on a single 51x6 window:
        1. Window validation
        2. Normalization
        3. Model inference
        4. Motion quality & margin rejection
        5. State machine intent evaluation (1 intentional gesture -> 1 command)
        6. Command mapping & dispatch
        """
        now = time.time() if timestamp is None else timestamp

        # 1. Validation
        is_valid, err = validate_window(raw_window, expected_shape=(WINDOW_SIZE, NUM_CHANNELS))
        if not is_valid:
            cmd = get_command_for_gesture("unknown", self.dispatcher.command_map)
            return RecognitionResult(
                timestamp=now,
                predicted_label="unknown",
                confidence=0.0,
                probabilities={},
                filtered_label="unknown",
                command=cmd,
                is_suppressed_threshold=True,
                is_suppressed_cooldown=False,
                is_holding_blocked=False,
                dry_run=self.dry_run,
                action_dispatched=False,
                status_detail=f"Window validation failed: {err}",
                state_machine_state=self.state_machine.state.value,
                suppression_reason="INVALID_WINDOW"
            )

        # 2. Normalization
        if self.normalizer is not None and self.normalizer.is_fitted:
            norm_window = self.normalizer.transform(raw_window)
        else:
            norm_window = raw_window

        # 3. Model Inference
        if self.model_type == "cnn":
            x_input = np.expand_dims(norm_window, axis=0)  # Shape (1, 51, 6)
            probs = self.model.predict(x_input, verbose=0)[0]
        else:  # MLP
            x_input = norm_window.reshape(1, -1)  # Shape (1, 306)
            probs = self.model.predict_proba(x_input)[0]

        pred_idx = int(np.argmax(probs))
        pred_label = self.label_classes[pred_idx]
        confidence = float(probs[pred_idx])
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(self.label_classes)}

        # 4. Motion Quality & Multi-Factor Rejection
        diag: RejectionDiagnostics = self.rejector.evaluate(raw_window, probs, pred_label)

        # 5. Intent & State Machine Evaluation (One Intentional Gesture -> One Command)
        # Check if gesture can produce a command
        is_eligible, decision_reason, state_detail = self.state_machine.evaluate_intent(
            predicted_gesture=pred_label,
            is_motion_valid=diag.is_accepted,
            raw_window=raw_window,
            current_time=now
        )

        # Identify mapped command for the PREDICTED gesture (concept separation)
        mapped_cmd = get_command_for_gesture(pred_label, self.dispatcher.command_map)

        is_suppressed_threshold = (diag.primary_code == "LOW_CONFIDENCE")
        is_suppressed_cooldown = (decision_reason == DecisionReason.COOLDOWN_ACTIVE)
        is_holding_blocked = (decision_reason == DecisionReason.HOLDING_POSTURE_BLOCKED)

        # 6. Command Dispatch
        filtered_label = pred_label if is_eligible else "unknown"
        dispatched = False
        dispatch_msg = state_detail

        if is_eligible and filtered_label not in ["still", "unknown"]:
            cmd_out, dispatched, dispatch_msg = self.dispatcher.dispatch(filtered_label)
        else:
            cmd_out = mapped_cmd

        return RecognitionResult(
            timestamp=now,
            predicted_label=pred_label,
            confidence=confidence,
            probabilities=prob_dict,
            filtered_label=filtered_label,
            command=cmd_out,
            is_suppressed_threshold=is_suppressed_threshold,
            is_suppressed_cooldown=is_suppressed_cooldown,
            is_holding_blocked=is_holding_blocked,
            dry_run=self.dry_run,
            action_dispatched=dispatched,
            status_detail=f"{diag.status_detail}; {dispatch_msg}",
            state_machine_state=self.state_machine.state.value,
            suppression_reason=decision_reason.value,
            class_margin=diag.class_margin,
            motion_energy=diag.motion_energy_accel
        )
