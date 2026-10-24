# live_demo_button_desktop_app.py
# Polished IMU desktop controller app
#
# What this version does:
#   1. Opens as a desktop app with no model-choice terminal prompt
#   2. Automatically loads the CNN model by default
#   3. Connects to the ESP32 on COM6
#   4. Listens for the ESP physical button over Serial
#   5. When the ESP prints "# BUTTON_PRESSED", the app records one 51-sample window
#   6. Runs the trained gesture model
#   7. Maps the prediction to a desktop command:
#        tilt_left  -> previous slide
#        tilt_right -> next slide
#        tilt_up    -> volume +10%
#        tilt_down  -> volume -10%
#        shake      -> Escape / cancel
#        still      -> no action
#
# Requirements:
#   pip install pyserial numpy joblib tensorflow keyboard pycaw comtypes
#
# Important ESP firmware requirement:
#   When the physical button is pressed, the ESP should print this exact line:
#       # BUTTON_PRESSED
#
# Run from the folder containing:
#   imu_gesture_cnn.keras
#   imu_gesture_mlp.pkl       optional
#   label_classes.npy
#   feature_mean.npy          optional but recommended
#   feature_std.npy           optional but recommended

import os
import time
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

import serial
import numpy as np
import joblib
import keyboard
from tensorflow.keras.models import load_model

# -----------------------------
# Windows volume control support
# -----------------------------
# PyCAW lets us change the Windows master volume by an exact percentage amount.
# If PyCAW is not installed, the app falls back to keyboard media keys, but that
# fallback is NOT guaranteed to be exactly 10%.
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYCAW_AVAILABLE = True
except Exception:
    PYCAW_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
PORT = "COM6"
BAUD_RATE = 115200

WINDOW_SIZE = 51
CAPTURE_TIMEOUT_SECONDS = 5.0

DEFAULT_MODEL_TYPE = "cnn"  # app auto-loads CNN so you do not have to type anything
CNN_MODEL_PATH = "imu_gesture_cnn.keras"
MLP_MODEL_PATH = "imu_gesture_mlp.pkl"
LABELS_PATH = "label_classes.npy"

# Optional backup hotkeys
ARM_HOTKEY = "`"
CALIBRATE_HOTKEY = "ctrl+shift+c"
QUIT_HOTKEY = "ctrl+shift+q"

# Model confidence rules
CONFIDENCE_THRESHOLD = 0.75
SHAKE_THRESHOLD = 0.55

# Action cooldown prevents duplicate commands from accidental double triggers
ACTION_COOLDOWN_SECONDS = 1.2
last_action_time = 0.0

# Gesture-to-action mapping.
# Keep both naming styles in case your model labels are "up/down" or "tilt_up/tilt_down".
ACTION_MAP = {
    "left": "previous_slide",
    "tilt_left": "previous_slide",

    "right": "next_slide",
    "tilt_right": "next_slide",

    "up": "volume_up_10",
    "tilt_up": "volume_up_10",

    "down": "volume_down_10",
    "tilt_down": "volume_down_10",

    # Default shake command: Escape / cancel.
    # If you prefer mute/unmute instead, change this to: "toggle_mute"
    "shake": "escape",

    "still": None,
}

# Lines that count as an ESP button press.
# Your ESP should ideally print "# BUTTON_PRESSED".
BUTTON_TRIGGER_PHRASES = (
    "# BUTTON_PRESSED",
    "BUTTON_PRESSED",
    "#BUTTON_PRESSED",
    "BUTTON",
)


# -----------------------------
# Shared state
# -----------------------------
ui_queue = queue.Queue()
serial_lock = threading.Lock()
state_lock = threading.Lock()
shutdown_event = threading.Event()

ser = None
model = None
label_classes = None
feature_mean = None
feature_std = None
model_display_name = "Not loaded"
volume_endpoint = None

is_busy = False
listener_thread = None


# -----------------------------
# UI setup
# -----------------------------
root = tk.Tk()
root.title("IMU Gesture Desktop Controller")
root.geometry("620x520+30+30")
root.attributes("-topmost", True)
root.resizable(True, True)

# Simple cleaner look
BG = "#f4f6f8"
CARD = "#ffffff"
TEXT = "#1f2937"
MUTED = "#6b7280"
ACCENT = "#2563eb"

root.configure(bg=BG)

style = ttk.Style()
try:
    style.theme_use("clam")
except Exception:
    pass

style.configure("TFrame", background=BG)
style.configure("Card.TFrame", background=CARD, relief="flat")
style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
style.configure("Header.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 16, "bold"))
style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
style.configure("Card.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
style.configure("CardBold.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10, "bold"))
style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

main = ttk.Frame(root, padding=14)
main.pack(fill="both", expand=True)

title_label = ttk.Label(main, text="IMU Gesture Desktop Controller", style="Header.TLabel")
title_label.pack(anchor="w")

subtitle_label = ttk.Label(
    main,
    text="Press the ESP button. The app will count down, then record.",
    style="Sub.TLabel",
)
subtitle_label.pack(anchor="w", pady=(2, 10))

status_card = tk.Frame(main, bg=CARD, bd=0, highlightthickness=1, highlightbackground="#e5e7eb")
status_card.pack(fill="x", pady=(0, 10))

model_var = tk.StringVar(value="Model: loading...")
port_var = tk.StringVar(value=f"Port: {PORT}")
status_var = tk.StringVar(value="Starting app...")
last_output_var = tk.StringVar(value="Last output: none")
volume_var = tk.StringVar(value="Volume: not checked yet")
recording_phase_var = tk.StringVar(value="READY")

tk.Label(status_card, textvariable=model_var, bg=CARD, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
tk.Label(status_card, textvariable=port_var, bg=CARD, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=12)
tk.Label(status_card, textvariable=status_var, bg=CARD, fg=TEXT, font=("Segoe UI", 10), wraplength=580, justify="left").pack(anchor="w", padx=12, pady=(8, 2))
tk.Label(status_card, textvariable=last_output_var, bg=CARD, fg=ACCENT, font=("Segoe UI", 10, "bold"), wraplength=580, justify="left").pack(anchor="w", padx=12, pady=(4, 2))
tk.Label(status_card, textvariable=volume_var, bg=CARD, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(2, 8))

tk.Label(
    status_card,
    textvariable=recording_phase_var,
    bg=CARD,
    fg=ACCENT,
    font=("Segoe UI", 22, "bold"),
    wraplength=580,
    justify="left",
).pack(anchor="w", padx=12, pady=(0, 12))

button_frame = ttk.Frame(main)
button_frame.pack(fill="x", pady=(0, 10))

record_button = ttk.Button(button_frame, text="Record One Gesture", command=lambda: start_prediction_thread("manual_button"))
record_button.pack(side="left", padx=(0, 8))

calibrate_button = ttk.Button(button_frame, text="Calibrate IMU", command=lambda: start_calibration_thread())
calibrate_button.pack(side="left", padx=(0, 8))

test_up_button = ttk.Button(button_frame, text="Test +10% Volume", command=lambda: execute_action_label("volume_up_10"))
test_up_button.pack(side="left", padx=(0, 8))

test_down_button = ttk.Button(button_frame, text="Test -10% Volume", command=lambda: execute_action_label("volume_down_10"))
test_down_button.pack(side="left", padx=(0, 8))

prob_card = tk.Frame(main, bg=CARD, bd=0, highlightthickness=1, highlightbackground="#e5e7eb")
prob_card.pack(fill="both", expand=True)

tk.Label(prob_card, text="Prediction Probabilities", bg=CARD, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))

prob_box = scrolledtext.ScrolledText(
    prob_card,
    height=10,
    width=70,
    font=("Consolas", 10),
    bg="#f9fafb",
    fg=TEXT,
    relief="flat",
    borderwidth=0,
)
prob_box.pack(fill="both", expand=True, padx=12, pady=(0, 10))
prob_box.insert("1.0", "No predictions yet.\n")
prob_box.config(state="disabled")

help_text = (
    "Controls:\n"
    "  ESP physical button = 3, 2, 1, then RECORDING NOW\n"
    f"  {ARM_HOTKEY} = backup keyboard trigger\n"
    f"  {CALIBRATE_HOTKEY} = calibrate IMU\n"
    f"  {QUIT_HOTKEY} = quit\n\n"
    "Mapping:\n"
    "  tilt_left = previous slide | tilt_right = next slide\n"
    "  tilt_up = volume +10% | tilt_down = volume -10%\n"
    "  shake = Escape / cancel | still = no action"
)

help_label = ttk.Label(main, text=help_text, style="Sub.TLabel", justify="left")
help_label.pack(anchor="w", pady=(10, 0))


# -----------------------------
# UI helpers
# -----------------------------
def enqueue_ui(kind, payload):
    ui_queue.put((kind, payload))


def log_ui(message):
    print(message)
    enqueue_ui("status", message)


def set_last_output(message):
    print(message)
    enqueue_ui("last_output", message)


def set_probabilities(text):
    enqueue_ui("probs", text)


def set_volume_display(text):
    enqueue_ui("volume", text)


def set_recording_phase(text):
    enqueue_ui("phase", text)


def process_ui_queue():
    try:
        while True:
            kind, payload = ui_queue.get_nowait()

            if kind == "status":
                status_var.set(payload)
            elif kind == "last_output":
                last_output_var.set(payload)
            elif kind == "probs":
                prob_box.config(state="normal")
                prob_box.delete("1.0", tk.END)
                prob_box.insert("1.0", payload)
                prob_box.config(state="disabled")
            elif kind == "volume":
                volume_var.set(payload)
            elif kind == "model":
                model_var.set(payload)
            elif kind == "port":
                port_var.set(payload)
            elif kind == "phase":
                recording_phase_var.set(payload)

    except queue.Empty:
        pass

    if not shutdown_event.is_set():
        root.after(50, process_ui_queue)


# -----------------------------
# Volume helpers
# -----------------------------
def init_volume_controller():
    global volume_endpoint

    if not PYCAW_AVAILABLE:
        volume_endpoint = None
        set_volume_display("Volume: PyCAW not installed; fallback media keys will be used.")
        return

    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_endpoint = cast(interface, POINTER(IAudioEndpointVolume))

        current = get_volume_percent()
        set_volume_display(f"Volume: {current}%")
    except Exception as e:
        volume_endpoint = None
        set_volume_display(f"Volume: PyCAW unavailable ({e}); fallback media keys will be used.")


def get_volume_percent():
    if volume_endpoint is None:
        return None
    current_scalar = volume_endpoint.GetMasterVolumeLevelScalar()
    return int(round(current_scalar * 100))


def set_volume_percent(new_percent):
    if volume_endpoint is None:
        return None

    new_percent = max(0, min(100, int(round(new_percent))))
    volume_endpoint.SetMasterVolumeLevelScalar(new_percent / 100.0, None)
    set_volume_display(f"Volume: {new_percent}%")
    return new_percent


def change_volume_by_10(direction):
    """
    direction = +1 for volume up by 10 percentage points
    direction = -1 for volume down by 10 percentage points
    """
    if volume_endpoint is not None:
        current = get_volume_percent()
        new_percent = current + (10 * direction)
        return set_volume_percent(new_percent)

    # Fallback if PyCAW is missing.
    # This is not guaranteed to be exactly 10%, but keeps the app usable.
    media_key = "volume up" if direction > 0 else "volume down"
    for _ in range(5):
        keyboard.send(media_key)
        time.sleep(0.02)
    set_volume_display("Volume: changed using fallback media keys, not exact 10%.")
    return None


def toggle_mute():
    if volume_endpoint is not None:
        current_mute = volume_endpoint.GetMute()
        volume_endpoint.SetMute(0 if current_mute else 1, None)
        set_volume_display("Volume: muted" if not current_mute else f"Volume: {get_volume_percent()}%")
    else:
        keyboard.send("volume mute")
        set_volume_display("Volume: mute toggled using fallback media key.")


# -----------------------------
# Model and serial initialization
# -----------------------------
def load_models_and_files():
    global model, label_classes, feature_mean, feature_std, model_display_name

    log_ui("Loading model files...")

    if DEFAULT_MODEL_TYPE == "cnn":
        if not os.path.exists(CNN_MODEL_PATH):
            raise FileNotFoundError(f"Missing model file: {CNN_MODEL_PATH}")
        model = load_model(CNN_MODEL_PATH)
        model_display_name = "CNN (Keras)"
    else:
        if not os.path.exists(MLP_MODEL_PATH):
            raise FileNotFoundError(f"Missing model file: {MLP_MODEL_PATH}")
        model = joblib.load(MLP_MODEL_PATH)
        model_display_name = "MLP (scikit-learn)"

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")

    label_classes = np.load(LABELS_PATH, allow_pickle=True)

    if os.path.exists("feature_mean.npy") and os.path.exists("feature_std.npy"):
        feature_mean = np.load("feature_mean.npy")
        feature_std = np.load("feature_std.npy")
        normalization_text = "normalization loaded"
    else:
        feature_mean = None
        feature_std = None
        normalization_text = "no normalization files found"

    enqueue_ui("model", f"Model: {model_display_name} | Classes: {list(label_classes)} | {normalization_text}")
    log_ui("Model loaded successfully.")


def connect_serial():
    global ser

    log_ui(f"Connecting to ESP32 on {PORT}...")
    ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    ser.reset_input_buffer()
    enqueue_ui("port", f"Port: {PORT} connected")
    log_ui("Connected. Waiting for ESP button press...")


# -----------------------------
# Serial / sensor functions
# -----------------------------
def is_button_trigger_line(line):
    upper_line = line.strip().upper()
    return any(phrase.upper() in upper_line for phrase in BUTTON_TRIGGER_PHRASES)


def parse_sensor_line(line):
    parts = line.split(",")
    if len(parts) != 7:
        return None

    try:
        _, ax, ay, az, gx, gy, gz = parts
        return np.array(
            [float(ax), float(ay), float(az), float(gx), float(gy), float(gz)],
            dtype=np.float32,
        )
    except ValueError:
        return None


def is_system_busy():
    with state_lock:
        return is_busy


def serial_button_listener():
    """
    Background thread:
    - constantly listens for ESP serial messages
    - when it sees "# BUTTON_PRESSED", it starts one prediction cycle
    - it does not read while the app is already recording
    """
    log_ui("ESP button listener started.")

    while not shutdown_event.is_set():
        if ser is None or not ser.is_open:
            time.sleep(0.1)
            continue

        if is_system_busy():
            time.sleep(0.05)
            continue

        try:
            line = ""
            with serial_lock:
                if ser.in_waiting > 0:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()

            if line:
                print("ESP:", line)

                if is_button_trigger_line(line):
                    log_ui("ESP button pressed. Starting gesture recording...")
                    start_prediction_thread("esp_button")

            time.sleep(0.03)

        except Exception as e:
            log_ui(f"Serial listener error: {e}")
            time.sleep(0.5)


def countdown_before_recording():
    """Simple demo countdown: 3, 2, 1, then RECORDING NOW."""
    for number in [3, 2, 1]:
        set_recording_phase(str(number))
        time.sleep(1.0)

    set_recording_phase("RECORDING NOW")
    time.sleep(0.2)


def collect_window():
    """
    Sends 'start' to ESP32, collects exactly WINDOW_SIZE valid IMU rows,
    then sends 'stop'. Includes a timeout so the app cannot freeze forever.

    User-facing flow:
      1. Show countdown
      2. Tell user to do the gesture
      3. Start ESP streaming
      4. Collect 51 valid IMU rows
      5. Stop ESP streaming
    """
    if ser is None or not ser.is_open:
        raise RuntimeError("Serial port is not connected.")

    countdown_before_recording()

    samples = []
    deadline = time.perf_counter() + CAPTURE_TIMEOUT_SECONDS

    with serial_lock:
        ser.reset_input_buffer()
        ser.write(b"start\n")

        while len(samples) < WINDOW_SIZE and time.perf_counter() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                continue

            if line.startswith("# STREAMING_STARTED"):
                continue

            if line.startswith("#") or line.startswith("timestamp_ms"):
                continue

            sensor_values = parse_sensor_line(line)
            if sensor_values is None:
                continue

            samples.append(sensor_values)

            # No extra recording text during demo. Keep display on RECORDING NOW.

        ser.write(b"stop\n")
        time.sleep(0.05)
        ser.reset_input_buffer()

    if len(samples) < WINDOW_SIZE:
        set_recording_phase("CAPTURE FAILED")
        raise RuntimeError(
            f"Only captured {len(samples)}/{WINDOW_SIZE} rows. "
            "Check ESP firmware, COM port, and that the MPU6050 is streaming after 'start'."
        )

    x_live = np.array(samples, dtype=np.float32)
    set_recording_phase("CAPTURE COMPLETE")
    log_ui(f"Captured window shape: {x_live.shape}")
    return x_live


def calibrate_imu():
    global is_busy

    with state_lock:
        if is_busy:
            log_ui("Busy. Calibration ignored.")
            return
        is_busy = True

    try:
        if ser is None or not ser.is_open:
            raise RuntimeError("Serial port is not connected.")

        log_ui("Starting calibration. Place the IMU flat and keep it still.")

        with serial_lock:
            ser.reset_input_buffer()
            ser.write(b"calibrate\n")

            timeout = time.time() + 12
            while time.time() < timeout:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    log_ui(line)
                if "# CALIBRATION_DONE" in line or "# CALIBRATION_FAILED" in line:
                    break

            time.sleep(0.2)
            ser.reset_input_buffer()

        log_ui("Calibration finished. Waiting for ESP button press...")

    except Exception as e:
        log_ui(f"Calibration error: {e}")

    finally:
        with state_lock:
            is_busy = False


# -----------------------------
# Prediction and actions
# -----------------------------
def predict_window(x_live):
    x_live = np.expand_dims(x_live, axis=0)  # (1, 51, 6)

    if feature_mean is not None and feature_std is not None:
        x_live = (x_live - feature_mean) / feature_std

    if DEFAULT_MODEL_TYPE == "mlp":
        x_input = x_live.reshape(1, -1)
        probs = model.predict_proba(x_input)[0]
    else:
        probs = model.predict(x_live, verbose=0)[0]

    pred_index = int(np.argmax(probs))
    pred_label = str(label_classes[pred_index])
    confidence = float(probs[pred_index])

    threshold = SHAKE_THRESHOLD if pred_label == "shake" else CONFIDENCE_THRESHOLD
    final_output = pred_label if confidence >= threshold else "unknown"

    return final_output, pred_label, confidence, probs


def format_probabilities(probs):
    pairs = list(zip(label_classes, probs))
    pairs.sort(key=lambda x: float(x[1]), reverse=True)

    lines = []
    for label, prob in pairs:
        lines.append(f"{str(label):<14} {float(prob):>7.3f}   ({float(prob) * 100:>6.2f}%)")
    return "\n".join(lines)


def execute_action_label(action):
    """
    Executes an app-level action name directly.
    Used by both gesture predictions and test buttons.
    """
    global last_action_time

    if action is None:
        log_ui("No action mapped.")
        return

    now = time.time()
    if now - last_action_time < ACTION_COOLDOWN_SECONDS:
        log_ui("Action blocked by cooldown.")
        return

    try:
        if action == "previous_slide":
            keyboard.send("page up")
            log_ui("← Sent PREVIOUS SLIDE")

        elif action == "next_slide":
            keyboard.send("page down")
            log_ui("→ Sent NEXT SLIDE")

        elif action == "volume_up_10":
            new_vol = change_volume_by_10(+1)
            if new_vol is not None:
                log_ui(f"🔊 Volume increased to {new_vol}%")
            else:
                log_ui("🔊 Volume up sent using fallback")

        elif action == "volume_down_10":
            new_vol = change_volume_by_10(-1)
            if new_vol is not None:
                log_ui(f"🔉 Volume decreased to {new_vol}%")
            else:
                log_ui("🔉 Volume down sent using fallback")

        elif action == "toggle_mute":
            toggle_mute()
            log_ui("🔇 Sent MUTE / UNMUTE")

        elif action == "escape":
            keyboard.send("esc")
            log_ui("⎋ Sent ESCAPE / CANCEL")

        else:
            keyboard.send(action)
            log_ui(f"Sent key/action: {action}")

        last_action_time = now

    except Exception as e:
        log_ui(f"Action error: {e}")


def execute_action(final_output):
    action = ACTION_MAP.get(final_output)

    if action is None:
        log_ui("No desktop action mapped for this gesture.")
        return

    execute_action_label(action)


def run_one_prediction(source="manual"):
    global is_busy

    with state_lock:
        if is_busy:
            log_ui("Already recording. Trigger ignored.")
            return
        is_busy = True

    try:
        x_live = collect_window()
        final_output, pred_label, confidence, probs = predict_window(x_live)

        result_line = (
            f"Final output: {final_output} | "
            f"Raw prediction: {pred_label} | "
            f"Confidence: {confidence:.3f}"
        )

        set_last_output(result_line)
        set_probabilities(format_probabilities(probs))

        print("\n=== PREDICTION ===")
        print(f"Model used: {model_display_name}")
        print(result_line)
        print("Probabilities:")
        print(format_probabilities(probs))
        print()

        if final_output != "unknown":
            execute_action(final_output)
        else:
            log_ui("Prediction rejected as unknown. No action sent.")

    except Exception as e:
        log_ui(f"Prediction error: {e}")

    finally:
        with state_lock:
            is_busy = False

        time.sleep(0.8)
        if not shutdown_event.is_set():
            set_recording_phase("READY")


def start_prediction_thread(source):
    threading.Thread(target=run_one_prediction, args=(source,), daemon=True).start()


def start_calibration_thread():
    threading.Thread(target=calibrate_imu, daemon=True).start()


# -----------------------------
# App startup / shutdown
# -----------------------------
def request_quit():
    shutdown_event.set()
    log_ui("Quit requested. Shutting down...")
    root.after(200, root.destroy)


def on_close():
    request_quit()


def initialize_app():
    global listener_thread

    try:
        load_models_and_files()
        init_volume_controller()
        connect_serial()

        listener_thread = threading.Thread(target=serial_button_listener, daemon=True)
        listener_thread.start()

        # Backup hotkeys
        try:
            keyboard.add_hotkey(ARM_HOTKEY, lambda: start_prediction_thread("keyboard_hotkey"))
            keyboard.add_hotkey(CALIBRATE_HOTKEY, start_calibration_thread)
            keyboard.add_hotkey(QUIT_HOTKEY, request_quit)
        except Exception as e:
            log_ui(f"Hotkey warning: {e}. ESP button and GUI buttons still work.")

        log_ui("Ready. Press the ESP button to record a gesture.")

    except Exception as e:
        log_ui(f"Startup error: {e}")
        enqueue_ui("model", "Model: failed to load")
        enqueue_ui("port", f"Port: {PORT} not connected")


def cleanup():
    shutdown_event.set()

    try:
        keyboard.unhook_all_hotkeys()
    except Exception:
        pass

    try:
        if ser is not None and ser.is_open:
            with serial_lock:
                ser.write(b"stop\n")
                time.sleep(0.1)
                ser.close()
    except Exception:
        pass

    print("App closed.")


root.protocol("WM_DELETE_WINDOW", on_close)
root.after(100, process_ui_queue)
root.after(300, initialize_app)

try:
    root.mainloop()
finally:
    cleanup()
