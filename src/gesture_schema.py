"""
gesture_schema.py
Canonical Gesture-Label Mapping & PowerPoint Command Definitions

Provides a single source of truth for:
1. Canonical gesture class names
2. Normalization of synonym labels (e.g., 'tilt_left' -> 'left')
3. Detection and reporting of invalid/unknown labels
4. Default and customizable PowerPoint command mappings
"""

from typing import Dict, List, Optional, Tuple, Any

# Ordered canonical class names matching model output dimensions (6 classes)
CANONICAL_CLASSES: List[str] = ["down", "left", "right", "shake", "still", "up"]

# Mapping from raw/legacy synonyms to canonical class names
LABEL_SYNONYMS: Dict[str, str] = {
    # Tilt variants
    "down": "down",
    "tilt_down": "down",
    "left": "left",
    "tilt_left": "left",
    "right": "right",
    "tilt_right": "right",
    "up": "up",
    "tilt_up": "up",
    "shake": "shake",
    "still": "still",
    # Legacy swipe variants (from raw_old)
    "swipe_down": "down",
    "swipe_left": "left",
    "swipe_right": "right",
    "swipe_up": "up",
}

# Standard PowerPoint Command Mappings
# Focus Notice: Keyboard shortcuts require PowerPoint (or presentation window) to be active.
DEFAULT_POWERPOINT_COMMANDS: Dict[str, Dict[str, Any]] = {
    "left": {
        "action": "previous_slide",
        "key": "left",
        "description": "Previous Slide (Left Arrow)",
    },
    "right": {
        "action": "next_slide",
        "key": "right",
        "description": "Next Slide (Right Arrow)",
    },
    "up": {
        "action": "start_presentation",
        "key": "f5",
        "description": "Start Presentation from Beginning (F5)",
    },
    "down": {
        "action": "black_screen",
        "key": "b",
        "description": "Toggle Black / Un-black Screen (B)",
    },
    "shake": {
        "action": "end_presentation",
        "key": "esc",
        "description": "End Presentation / Cancel (Esc)",
    },
    "still": {
        "action": "none",
        "key": None,
        "description": "Resting / Neutral - No Command",
    },
    "unknown": {
        "action": "none",
        "key": None,
        "description": "No Action Mapped (Unrecognized / Suppressed)",
    },
}


def normalize_label(raw_label: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalizes any raw label string to its canonical class.

    Returns:
        (canonical_label, error_reason)
        If valid: (canonical_name, None)
        If invalid/unknown: (None, error_message)
    """
    if raw_label is None:
        return None, "Label is None"
    
    clean_str = str(raw_label).strip().lower()
    if clean_str in LABEL_SYNONYMS:
        return LABEL_SYNONYMS[clean_str], None
    
    return None, f"Unknown or unmapped label: '{raw_label}'"


def is_canonical(label: str) -> bool:
    """Checks if label is in canonical class list."""
    return label in CANONICAL_CLASSES


def get_command_for_gesture(gesture: str, command_map: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Retrieves command metadata for a canonical gesture label.
    Falls back to 'unknown' entry if gesture is unrecognized.
    """
    mapping = command_map if command_map is not None else DEFAULT_POWERPOINT_COMMANDS
    return mapping.get(gesture, mapping["unknown"])
