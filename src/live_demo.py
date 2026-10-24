# live_demo.py — Basic live gesture recognition demo
#
# Step 3 of the pipeline (after training models in the notebook).
# Connects to the ESP32, collects one 51-sample window on demand,
# runs it through the CNN, and optionally sends left/right arrow keys
# to control a slideshow.
#
# Controls:
#   Enter = record and classify one gesture
#   c     = recalibrate IMU
#   a     = toggle slideshow arrow-key output on/off
#   q     = quit
#
# Run from inside imu_live_demo/ so it finds the model files.

import serial
import time
import os
import numpy as np
import pyautogui
import joblib
from tensorflow.keras.models import load_model

# --- Config ---
PORT = "COM6"
BAUD_RATE = 115200

WINDOW_SIZE = 51
COUNTDOWN_SECONDS = 3
CONFIDENCE_THRESHOLD = 0.75
SHAKE_THRESHOLD = 0.55

MODEL_TYPE = "cnn"   # "cnn" or "mlp"
CNN_MODEL_PATH = "imu_gesture_cnn.keras"
MLP_MODEL_PATH = "imu_gesture_mlp.pkl"
LABELS_PATH = "label_classes.npy"

# --- Slide control state ---
ACTIONS_ENABLED = False
ACTION_COOLDOWN_SECONDS = 1.2
last_action_time = 0.0

# PyAutoGUI safety
pyautogui.FAILSAFE = True

# --- Load model and normalization ---
print("Loading model...")

if MODEL_TYPE == "cnn":
    model = load_model(CNN_MODEL_PATH)
else:
    model = joblib.load(MLP_MODEL_PATH)

label_classes = np.load(LABELS_PATH, allow_pickle=True)

USE_NORMALIZATION = os.path.exists("feature_mean.npy") and os.path.exists("feature_std.npy")
if USE_NORMALIZATION:
    feature_mean = np.load("feature_mean.npy")
    feature_std = np.load("feature_std.npy")
    print("Loaded normalization files.")
else:
    feature_mean = None
    feature_std = None
    print("No normalization files found. Using raw inputs.")

print("Loaded classes:", label_classes)

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print(f"Connected to ESP32 on {PORT}\n")


# --- Helper functions ---

def parse_sensor_line(line):
    parts = line.split(",")
    if len(parts) != 7:
        return None
    try:
        _, ax, ay, az, gx, gy, gz = parts
        return np.array(
            [float(ax), float(ay), float(az), float(gx), float(gy), float(gz)],
            dtype=np.float32
        )
    except ValueError:
        return None


def calibrate_imu():
    print("\nStarting calibration...")
    print("Place the IMU flat and keep it still.")
    ser.reset_input_buffer()
    ser.write(b"calibrate\n")

    timeout = time.time() + 12
    while time.time() < timeout:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(line)
        if "# CALIBRATION_DONE" in line or "# CALIBRATION_FAILED" in line:
            break

    time.sleep(0.2)
    ser.reset_input_buffer()
    print("Calibration finished.\n")


def collect_window():
    print("\nGet ready...")
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    ser.reset_input_buffer()
    ser.write(b"start\n")

    start_confirmed = False
    wait_timeout = time.perf_counter() + 3
    samples = []

    while time.perf_counter() < wait_timeout:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith("# STREAMING_STARTED"):
            start_confirmed = True
            print("RECORD NOW")
            break

    if not start_confirmed:
        print("RECORD NOW")

    while len(samples) < WINDOW_SIZE:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line or line.startswith("#") or line.startswith("timestamp_ms"):
            continue

        sensor_values = parse_sensor_line(line)
        if sensor_values is None:
            continue

        samples.append(sensor_values)

    ser.write(b"stop\n")
    time.sleep(0.2)
    ser.reset_input_buffer()

    x_live = np.array(samples, dtype=np.float32)
    print(f"Captured window shape: {x_live.shape}")
    return x_live


def predict_window(x_live):
    x_live = np.expand_dims(x_live, axis=0)  # (1, 51, 6)

    if feature_mean is not None and feature_std is not None:
        x_live = (x_live - feature_mean) / feature_std

    if MODEL_TYPE == "mlp":
        x_input = x_live.reshape(1, -1)
        probs = model.predict_proba(x_input)[0]
    else:
        probs = model.predict(x_live, verbose=0)[0]

    pred_index = int(np.argmax(probs))
    pred_label = label_classes[pred_index]
    confidence = float(probs[pred_index])

    threshold = SHAKE_THRESHOLD if pred_label == "shake" else CONFIDENCE_THRESHOLD
    final_output = pred_label if confidence >= threshold else "unknown"

    return final_output, pred_label, confidence, probs


def execute_action(final_output):
    global last_action_time

    now = time.time()
    if now - last_action_time < ACTION_COOLDOWN_SECONDS:
        print("Action blocked by cooldown.")
        return

    if final_output in ["tilt_right", "right"]:
        pyautogui.press("right")
        last_action_time = now
        print("→ Sent NEXT SLIDE (Right Arrow)")
    elif final_output in ["tilt_left", "left"]:
        pyautogui.press("left")
        last_action_time = now
        print("← Sent PREVIOUS SLIDE (Left Arrow)")
    elif final_output in ["tilt_up", "up"]:
        pyautogui.press("f5")
        last_action_time = now
        print("↑ Sent START PRESENTATION (F5)")
    elif final_output in ["tilt_down", "down"]:
        pyautogui.press("b")
        last_action_time = now
        print("↓ Sent BLACK SCREEN (B)")
    elif final_output == "shake":
        pyautogui.press("esc")
        last_action_time = now
        print("⚡ Sent END PRESENTATION (Escape)")
    else:
        print(f"No slideshow action mapped for gesture '{final_output}'.")



# --- Main loop ---
print("Controls:")
print("  Enter = record one gesture")
print("  c = calibrate IMU")
print("  a = toggle slideshow actions on/off")
print("  q = quit\n")

try:
    while True:
        cmd = input("Enter = record | c = calibrate | a = toggle actions | q = quit: ").strip().lower()

        if cmd == "q":
            break

        if cmd == "c":
            calibrate_imu()
            continue

        if cmd == "a":
            ACTIONS_ENABLED = not ACTIONS_ENABLED
            print(f"ACTIONS_ENABLED = {ACTIONS_ENABLED}\n")
            continue

        x_live = collect_window()
        final_output, pred_label, confidence, probs = predict_window(x_live)

        print("\n=== PREDICTION ===")
        print("Final output:", final_output)
        print("Raw prediction:", pred_label)
        print("Confidence:", round(confidence, 3))
        print("Probabilities:", dict(zip(label_classes, probs)))

        if ACTIONS_ENABLED and final_output != "unknown":
            execute_action(final_output)
        elif not ACTIONS_ENABLED:
            print("Actions are OFF. No keystroke sent.")

        print()

except KeyboardInterrupt:
    print("\nStopping demo...")

finally:
    try:
        ser.write(b"stop\n")
        time.sleep(0.2)
    except Exception:
        pass
    ser.close()
    print("Serial connection closed.")