# V2_live_demo.py — Clean modular demo for testing (no keyboard output)
#
# Same pipeline step as the other demos, but does NOT send any keystrokes.
# Good for testing model accuracy without accidentally pressing keys.
#
# Differences from live_demo.py:
#   - Prompts to choose CNN or MLP at startup
#   - Confidence threshold is 0.80 (stricter than live_demo's 0.75)
#   - No slideshow control — prediction only
#
# Run from inside imu_live_demo/ so it finds the model files.

import os
import time
import serial
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# --- Load normalization if available ---
USE_NORMALIZATION = os.path.exists("feature_mean.npy") and os.path.exists("feature_std.npy")

if USE_NORMALIZATION:
    feature_mean = np.load("feature_mean.npy")
    feature_std = np.load("feature_std.npy")
    print("Loaded normalization files.")
else:
    feature_mean = None
    feature_std = None
    print("No normalization files found. Using raw inputs.")

# --- Config ---
PORT = "COM6"
BAUD_RATE = 115200
WINDOW_SIZE = 51  # Must match training data window size exactly
COUNTDOWN_SECONDS = 3
CONFIDENCE_THRESHOLD = 0.80

# Change these if your filenames are different
CNN_MODEL_PATH = "imu_gesture_cnn.keras"
MLP_MODEL_PATH = "imu_gesture_mlp.pkl"
LABELS_PATH = "label_classes.npy"
MEAN_PATH = "feature_mean.npy"
STD_PATH = "feature_std.npy"


# --- Helper functions ---

def load_label_classes(labels_path):
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Could not find {labels_path}. Put label_classes.npy in the same folder as this script."
        )
    return np.load(labels_path, allow_pickle=True)


label_classes = load_label_classes(LABELS_PATH)

print("Choose model:")
print("1 = CNN (.keras)")
print("2 = MLP (.pkl)")
model_choice = input("Enter 1 or 2 [default 1]: ").strip()
if model_choice not in {"", "1", "2"}:
    model_choice = "1"

if model_choice in {"", "1"}:
    model_type = "cnn"
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {CNN_MODEL_PATH}. Put the trained CNN file in the same folder as this script."
        )
    print("Loading CNN model...")
    model = load_model(CNN_MODEL_PATH)
else:
    model_type = "mlp"
    if not os.path.exists(MLP_MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find {MLP_MODEL_PATH}. Put the trained MLP file in the same folder as this script."
        )
    print("Loading MLP model...")
    model = joblib.load(MLP_MODEL_PATH)

print("Loaded classes:", label_classes)
print(f"Using {model_type.upper()} model.\n")

# Optional normalization, only if you saved these during training
feature_mean = None
feature_std = None
if os.path.exists(MEAN_PATH) and os.path.exists(STD_PATH):
    feature_mean = np.load(MEAN_PATH)
    feature_std = np.load(STD_PATH)
    print("Loaded feature normalization files.")
else:
    print("feature_mean.npy / feature_std.npy not found. Running without normalization.\n")


def parse_sensor_line(line):
    """Expected format: timestamp_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"""
    parts = line.split(",")
    if len(parts) != 7:
        return None

    try:
        _, ax, ay, az, gx, gy, gz = parts
        return np.array([
            float(ax),
            float(ay),
            float(az),
            float(gx),
            float(gy),
            float(gz),
        ], dtype=np.float32)
    except ValueError:
        return None



def maybe_normalize(sample_window):
    """sample_window shape: (WINDOW_SIZE, 6)"""
    if feature_mean is None or feature_std is None:
        return sample_window
    return (sample_window - feature_mean) / feature_std



def wait_for_stream_start(ser, timeout_seconds=3.0):
    deadline = time.perf_counter() + timeout_seconds

    while time.perf_counter() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith("# STREAMING_STARTED"):
            return True
    return False



def send_calibrate_command(ser):
    print("\nPlace the IMU flat and completely still.")
    input("Press Enter when ready to calibrate...")

    ser.reset_input_buffer()
    ser.write(b"calibrate\n")

    calibration_timeout = time.time() + 12
    while time.time() < calibration_timeout:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(line)
        if "# CALIBRATION_DONE" in line or "# CALIBRATION_FAILED" in line:
            break
    else:
        print("Calibration response timed out.")

    time.sleep(0.2)
    ser.reset_input_buffer()
    print()



def collect_one_window(ser):
    print("\nGet ready...")
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    ser.reset_input_buffer()
    ser.write(b"start\n")

    started = wait_for_stream_start(ser, timeout_seconds=3.0)
    if not started:
        print("Warning: did not receive # STREAMING_STARTED, but continuing anyway.")

    print("RECORD NOW")
    sample_rows = []

    while len(sample_rows) < WINDOW_SIZE:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line or line.startswith("#") or line.startswith("timestamp_ms"):
            continue

        sensor_values = parse_sensor_line(line)
        if sensor_values is None:
            continue

        sample_rows.append(sensor_values)

    ser.write(b"stop\n")
    time.sleep(0.2)
    ser.reset_input_buffer()

    sample_window = np.array(sample_rows, dtype=np.float32)
    return sample_window



def predict_window(sample_window):
    sample_window = maybe_normalize(sample_window)

    if model_type == "cnn":
        x_input = np.expand_dims(sample_window, axis=0)  # (1, 51, 6)
        probs = model.predict(x_input, verbose=0)[0]
    else:
        x_input = sample_window.reshape(1, -1)  # (1, 51*6)
        probs = model.predict_proba(x_input)[0]

    pred_index = int(np.argmax(probs))
    pred_label = label_classes[pred_index]
    confidence = float(probs[pred_index])

    if confidence < CONFIDENCE_THRESHOLD:
        final_label = "unknown"
    else:
        final_label = str(pred_label)

    return final_label, pred_label, confidence, probs


# --- Connect and run ---
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print(f"Connected to ESP32 on {PORT}\n")

calibrate_choice = input("Calibrate IMU before demo? (y/n): ").strip().lower()
if calibrate_choice == "y":
    send_calibrate_command(ser)

print("Manual one-shot demo ready.")
print(f"Window size: {WINDOW_SIZE} samples")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print("Press Enter to record one gesture, 'c' to recalibrate, or 'q' to quit.\n")

try:
    while True:
        user_cmd = input("Enter = record | c = calibrate | q = quit: ").strip().lower()

        if user_cmd == "q":
            print("Exiting demo.")
            break

        if user_cmd == "c":
            send_calibrate_command(ser)
            continue

        sample_window = collect_one_window(ser)
        print(f"Captured window shape: {sample_window.shape}")

        final_label, raw_label, confidence, probs = predict_window(sample_window)
        probs_dict = {str(label_classes[i]): float(probs[i]) for i in range(len(label_classes))}

        print("\n=== PREDICTION ===")
        print(f"Final output: {final_label}")
        print(f"Raw prediction: {raw_label}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Probabilities: {probs_dict}\n")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    try:
        ser.write(b"stop\n")
        time.sleep(0.2)
    except Exception:
        pass
    ser.close()
    print("Serial connection closed.")
