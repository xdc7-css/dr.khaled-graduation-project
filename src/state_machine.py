"""
state_machine.py
Gesture Capture & Command Dispatch State Machine

Replaces reliance on cooldown alone with an intentional state machine:
States:
  IDLE -> CAPTURING -> EVALUATING -> WAITING_FOR_NEUTRAL -> READY

Core Principles:
1. One Intentional Gesture -> Exactly One Command.
2. Holding a tilted posture CANNOT repeatedly trigger commands.
3. Returning to a configurable neutral state re-arms recognition.
4. Includes hysteresis (minimum consecutive neutral samples/windows),
   minimum neutral dwell duration, and timeout safety.
5. Cooldown remains as an additional secondary protection layer.
6. Safe reset on stop, disconnect, or invalid data.
7. Explicit decision and suppression reason tracking.
"""

import enum
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


class GestureState(enum.Enum):
    IDLE = "IDLE"
    CAPTURING = "CAPTURING"
    EVALUATING = "EVALUATING"
    WAITING_FOR_NEUTRAL = "WAITING_FOR_NEUTRAL"
    READY = "READY"


class DecisionReason(enum.Enum):
    COMMAND_ISSUED = "COMMAND_ISSUED"
    COMMAND_SIMULATED = "COMMAND_SIMULATED"
    HOLDING_POSTURE_BLOCKED = "HOLDING_POSTURE_BLOCKED"  # Held gesture blocked until neutral return
    AWAITING_NEUTRAL = "AWAITING_NEUTRAL"                # State machine waiting for neutral
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"                  # Secondary cooldown filter active
    LOW_CONFIDENCE = "LOW_CONFIDENCE"                    # Model confidence below threshold
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"          # Class separation too small
    LOW_MOTION_ENERGY = "LOW_MOTION_ENERGY"              # Static/insufficient motion for dynamic gesture
    NEUTRAL_IDLE = "NEUTRAL_IDLE"                        # Neutral posture, no action intended
    INVALID_WINDOW = "INVALID_WINDOW"                    # Data validation or shape failure
    NO_ACTION_MAPPED = "NO_ACTION_MAPPED"                # Gesture valid but mapped to no action
    DISCONNECTED = "DISCONNECTED"                        # Hardware disconnected / reset
    TIMEOUT_RESET = "TIMEOUT_RESET"                      # Stuck state reset by timeout


@dataclass
class NeutralConfig:
    """
    Configuration for neutral posture detection and hysteresis.
    Marked as provisional: thresholds will be calibrated when physical sensor is attached.
    """
    # Nominal flat reference: gravity approximately +1.0g on Z, 0 on X and Y
    ref_ax: float = 0.0
    ref_ay: float = 0.0
    ref_az: float = 1.0
    
    # Tolerances for accelerometer gravity vector alignment (in g)
    accel_tolerance_g: float = 0.35
    
    # Maximum angular velocity variance/magnitude for neutral stillness (in deg/s)
    gyro_max_dps: float = 25.0
    
    # Hysteresis: minimum consecutive neutral windows or time before re-arming
    min_neutral_windows: int = 2
    min_neutral_duration_s: float = 0.30
    
    # Safety timeout: if stuck in WAITING_FOR_NEUTRAL for too long without seeing neutral,
    # force re-arm after this duration to prevent permanent lockup (in seconds)
    neutral_timeout_s: float = 6.0


class GestureStateMachine:
    """
    Robust state machine governing gesture capture, evaluation, command dispatch,
    and neutral-return re-arming.
    """

    def __init__(
        self,
        neutral_config: Optional[NeutralConfig] = None,
        cooldown_seconds: float = 1.2,
        triggered_mode_default: bool = True
    ):
        self.config = neutral_config or NeutralConfig()
        self.cooldown_seconds = cooldown_seconds
        self.triggered_mode = triggered_mode_default  # True = triggered (e.g. button/trial), False = continuous stream

        # State machine variables
        self.state: GestureState = GestureState.READY
        self.last_command_time: float = -100.0
        self.last_state_change_time: float = 0.0
        self.last_dispatched_gesture: Optional[str] = None
        
        # Neutral hysteresis counters
        self.consecutive_neutral_count: int = 0
        self.neutral_dwell_start_time: Optional[float] = None
        
        # Last decision audit
        self.last_reason: DecisionReason = DecisionReason.NEUTRAL_IDLE
        self.last_reason_detail: str = "Initialized in READY state."

    def reset(self, new_state: GestureState = GestureState.READY, current_time: Optional[float] = None) -> None:
        """Safely resets the state machine to an initial state."""
        now = time.time() if current_time is None else current_time
        self.state = new_state
        self.last_state_change_time = now
        self.consecutive_neutral_count = 0
        self.neutral_dwell_start_time = None
        self.last_reason = DecisionReason.DISCONNECTED
        self.last_reason_detail = f"State machine reset to {new_state.value}."

    def is_neutral_window(self, raw_window: np.ndarray) -> Tuple[bool, str]:
        """
        Tests whether a 51x6 window represents a neutral resting posture.
        Checks gravity alignment and low gyroscope motion.
        """
        if raw_window is None or raw_window.shape[0] == 0:
            return False, "Empty window"

        mean_ax = float(np.mean(raw_window[:, 0]))
        mean_ay = float(np.mean(raw_window[:, 1]))
        mean_az = float(np.mean(raw_window[:, 2]))

        # Calculate deviation from neutral gravity reference
        err_x = abs(mean_ax - self.config.ref_ax)
        err_y = abs(mean_ay - self.config.ref_ay)
        err_z = abs(mean_az - self.config.ref_az)
        accel_err = max(err_x, err_y, err_z)

        # Gyro motion magnitude (RMS across gx, gy, gz in deg/s)
        gyro_rms = float(np.sqrt(np.mean(raw_window[:, 3:6] ** 2)))

        is_neutral_accel = accel_err <= self.config.accel_tolerance_g
        is_neutral_gyro = gyro_rms <= self.config.gyro_max_dps

        if is_neutral_accel and is_neutral_gyro:
            return True, f"Neutral confirmed (accel_err={accel_err:.2f}g, gyro_rms={gyro_rms:.1f} dps)"
        
        reasons = []
        if not is_neutral_accel:
            reasons.append(f"accel deviation {accel_err:.2f}g > {self.config.accel_tolerance_g:.2f}g")
        if not is_neutral_gyro:
            reasons.append(f"gyro motion {gyro_rms:.1f} dps > {self.config.gyro_max_dps:.1f} dps")
        return False, "; ".join(reasons)

    def update_neutral_tracking(self, is_neutral: bool, current_time: float) -> None:
        """Updates hysteresis counters and dwell times for neutral return."""
        if is_neutral:
            self.consecutive_neutral_count += 1
            if self.neutral_dwell_start_time is None:
                self.neutral_dwell_start_time = current_time
        else:
            # Broken neutral resets hysteresis counter
            self.consecutive_neutral_count = 0
            self.neutral_dwell_start_time = None

    def is_neutral_confirmed(self, current_time: float) -> bool:
        """Returns True only if both window count and minimum dwell duration are satisfied."""
        if self.consecutive_neutral_count < self.config.min_neutral_windows:
            return False
        if self.neutral_dwell_start_time is None:
            return False
        dwell_time = current_time - self.neutral_dwell_start_time
        return dwell_time >= self.config.min_neutral_duration_s

    def evaluate_intent(
        self,
        predicted_gesture: str,
        is_motion_valid: bool,
        raw_window: np.ndarray,
        current_time: Optional[float] = None
    ) -> Tuple[bool, DecisionReason, str]:
        """
        Evaluates whether a recognized gesture is eligible to produce a command.
        
        Enforces:
        - 1 intentional gesture -> 1 command.
        - Holding a tilted posture produces at most 1 command until neutral return.
        - Timeout recovery if user holds posture indefinitely.
        
        Returns:
            (is_eligible, decision_reason, explanation_detail)
        """
        now = time.time() if current_time is None else current_time
        is_neutral, neutral_detail = self.is_neutral_window(raw_window)
        self.update_neutral_tracking(is_neutral, now)

        # 1. Check for neutral/still gesture
        if predicted_gesture in ["still", "neutral"] or is_neutral:
            if self.state == GestureState.WAITING_FOR_NEUTRAL:
                if self.is_neutral_confirmed(now):
                    self.state = GestureState.READY
                    self.last_state_change_time = now
                    self.last_reason = DecisionReason.NEUTRAL_IDLE
                    self.last_reason_detail = "Neutral confirmed via hysteresis. System re-armed."
                    return False, DecisionReason.NEUTRAL_IDLE, self.last_reason_detail
                else:
                    detail = f"Neutral detected, satisfying hysteresis ({self.consecutive_neutral_count}/{self.config.min_neutral_windows} windows)..."
                    return False, DecisionReason.AWAITING_NEUTRAL, detail
            else:
                self.state = GestureState.READY
                return False, DecisionReason.NEUTRAL_IDLE, f"At rest in neutral posture ({neutral_detail})."

        # 2. Check secondary cooldown
        time_since_last_cmd = now - self.last_command_time
        if time_since_last_cmd < self.cooldown_seconds:
            self.last_reason = DecisionReason.COOLDOWN_ACTIVE
            detail = f"Secondary cooldown active ({time_since_last_cmd:.2f}s < {self.cooldown_seconds:.2f}s)."
            self.last_reason_detail = detail
            return False, DecisionReason.COOLDOWN_ACTIVE, detail

        # 3. Check if waiting for neutral (Holding posture protection)
        if self.state == GestureState.WAITING_FOR_NEUTRAL:
            time_waiting = now - self.last_state_change_time
            if time_waiting >= self.config.neutral_timeout_s:
                # Force re-arm via safety timeout
                self.state = GestureState.READY
                self.last_state_change_time = now
                self.consecutive_neutral_count = 0
                self.neutral_dwell_start_time = None
                self.last_reason = DecisionReason.TIMEOUT_RESET
                self.last_reason_detail = f"Neutral wait timeout expired ({time_waiting:.1f}s). Re-armed."
            else:
                # Holding or repeating gesture without returning to neutral
                self.last_reason = DecisionReason.HOLDING_POSTURE_BLOCKED
                detail = (
                    f"Holding posture '{predicted_gesture}' blocked. "
                    f"Must return to neutral posture to re-arm ({time_waiting:.1f}s / {self.config.neutral_timeout_s:.1f}s timeout)."
                )
                self.last_reason_detail = detail
                return False, DecisionReason.HOLDING_POSTURE_BLOCKED, detail

        # 4. If we are READY and gesture is dynamic/actionable:
        if self.state in [GestureState.READY, GestureState.CAPTURING, GestureState.EVALUATING]:
            if not is_motion_valid:
                self.last_reason = DecisionReason.INVALID_WINDOW
                self.last_reason_detail = "Motion quality or window validity check failed."
                return False, DecisionReason.INVALID_WINDOW, self.last_reason_detail

            # ELIGIBLE! Transition to WAITING_FOR_NEUTRAL
            self.state = GestureState.WAITING_FOR_NEUTRAL
            self.last_state_change_time = now
            self.last_command_time = now
            self.last_dispatched_gesture = predicted_gesture
            self.consecutive_neutral_count = 0
            self.neutral_dwell_start_time = None

            self.last_reason = DecisionReason.COMMAND_ISSUED
            detail = f"Gesture '{predicted_gesture}' accepted! Dispatched single command. Awaiting return to neutral."
            self.last_reason_detail = detail
            return True, DecisionReason.COMMAND_ISSUED, detail

        return False, DecisionReason.AWAITING_NEUTRAL, "State machine not ready for gesture dispatch."
