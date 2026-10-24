# Graduation Project Engineering Manual & Architecture Guide

**Academic Project Title:** Machine Learning-Based Hand Gesture Recognition Using a Wearable IMU Sensor for PowerPoint Control  
**Short Application Heading:** IMU Gesture Controller  
**Academic Credits:**  
- Supervisor: Dr. Khalid Ali Abbas  
- Institution: University of Basrah · College of Engineering · Computer Engineering Department  
**Software Version:** 2.1 (Intent State Machine, Quality Rejection, Guided Profiles & Compatibility Layer)  
**Target Environment:** Windows 10/11, Python 3.12 (64-bit), ESP32 + MPU6050 (when hardware is acquired)

---

## 1. Quick Start Commands (Hardware-Free Verification)

All core features can be run and evaluated on your computer today without an ESP32 or MPU6050 sensor.

### 1.1 Environment Setup (Project-Local Virtual Environment)
Initialize the project environment and install dependencies pinned to compatible versions (including `scikit-learn==1.8.0` for exact model compatibility):
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 1.2 Run Automated Software Verification Test Suite
Executes the comprehensive 27-test suite verifying canonical labeling, window validation, Z-score normalization, model inference, cooldown filtering, holding posture protection, neutral return hysteresis, motion quality rejection, model bundle loading, and dry-run safety:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

### 1.3 Launch Interactive Desktop GUI (Redesigned Engineering Application)
Launches the 5-tab dark-mode engineering instrument desktop application with vector oscilloscope waveforms, built-in demonstration slide preview, decoupled recognition/dispatch status cards, guided setup protocol, and state machine visualization. **Defaults to DRY RUN mode** to guarantee safety:
```powershell
# Recommended: Double-click or run the Windows launcher
.\run_app.bat

# Or run via project-local Python interpreter:
.\.venv\Scripts\python.exe main.py
```

### 1.4 Run Command-Line Replay Demonstration
Replays recorded gesture trials through the 1D-CNN or MLP recognition pipeline and prints a structured evaluation table:
```powershell
# Replay 12 trials using 1D-CNN
.\.venv\Scripts\python.exe src/replay_demo.py --dataset data/raw_new/ --model cnn --limit 12

# Replay 12 trials using baseline MLP
.\.venv\Scripts\python.exe src/replay_demo.py --dataset data/raw_new/ --model mlp --limit 12
```

### 1.4 Test Data Collection Logger (Mock Mode)
Tests the complete data recording and trial-appending pipeline without hardware. Files are placed safely into `data/mock/`:
```powershell
py -3.12 src/logger.py --mock --gesture tilt_left --trial 1 --num-trials 2
```

### 1.5 Run Reproducible Training & Evaluation Pipeline
Audits the raw dataset, quarantines anomalous trials with a logged audit trail, fits Z-score normalization strictly on the training split, trains MLP/CNN models, and evaluates both models and a rule-based baseline on the held-out test split:
```powershell
py -3.12 src/train.py --out-dir models/reproducible
```

---

## 2. Component Implementation Status & Honest Verification Labels

| Component / Subsystem | Status Label | Technical Description & Offline Evidence |
|---|---|---|
| **Intent Capture State Machine** | **Implemented & Verified Offline** | `src/state_machine.py`: Replaces cooldown-alone logic. Enforces 1 intentional gesture $\rightarrow$ 1 command. Holding a tilted posture is blocked (`HOLDING_POSTURE_BLOCKED`) until user returns to neutral. Tested with dwell times & hysteresis. |
| **Motion Quality & Rejection** | **Implemented & Verified Offline** | `src/motion_rejection.py`: Evaluates physical signal sanity (NaN/Inf, clipping/saturation), softmax margin ($\ge 0.20$), and dynamic kinetic motion energy. |
| **Unconstrained Background Rejection** | **Implemented, Awaiting Representative Data** | Rejection infrastructure is complete, but unsegmented continuous daily-activity recordings (typing, walking, reaching) do not yet exist in the dataset. |
| **User Setup & Personalization** | **Implemented & Verified Offline** | `src/profile_manager.py`: Guided workflow managing participant profiles, neutral resting baseline calibration, and guided repetition collection. Exercised with mock/replay data and non-destructive profile export. |
| **Reproducible Evaluation Engine** | **Implemented & Verified Offline** | `src/evaluation.py`: Computes Accuracy, Macro F1, confusion matrices, latency breakdown (inference vs buffer), and duplicate command rates without test-set leakage. |
| **Model Bundle Compatibility** | **Implemented & Verified Offline** | `src/compatibility.py`: Loads models with matching normalization, labels, units, and metadata. Flags legacy ~45.44 Hz vs 50 Hz temporal differences and handles scikit-learn version differences. |
| **Desktop GUI Application** | **Implemented & Verified Offline** | `src/gui_app.py`: 5-tab engineering UI with vector waveforms, decoupled recognition & dispatch cards, state machine visualization badge, guided setup tab, model bundle browser, and strictly bounded counters (119/119). |
| **CSV Replay Engine** | **Implemented & Verified Offline** | `src/replay_source.py`: Preserves trial boundaries, isolates corrupt rows into audit logs, and supports clean reset and restart. |
| **Upgraded ESP32 Firmware** | **Prepared, Awaiting Hardware** | `firmware/imu_remote.ino`: Button debouncing on GPIO 4, `# BUTTON_PRESSED` event, and 50 Hz interval timing. Prepared but uncompiled on physical chip. |
| **True Wearable Wrist Dataset** | **Not Implemented (Future Work)** | Existing recordings represent desktop/handheld orientations. True wrist-worn multi-participant dataset collection requires physical hardware. |
| **On-Chip BLE Wireless Transport** | **Not Implemented (Future Work)** | Wireless transport deferred to maintain simplicity; firmware structures serial stream abstraction for future BLE swap. |

---

## 3. Engineering Upgrades and Problem Resolutions

### 3.1 Canonical Label Normalization & PowerPoint Command Bug Fix
- **Previous Issue:** The trained model exported class names `['down', 'left', 'right', 'shake', 'still', 'up']`, but `src/live_demo.py` checked for `tilt_right` and `tilt_left`, causing slide navigation commands to silently fail.
- **Resolution:** Implemented `src/gesture_schema.py`. All components share canonical classes:
  - `left` $\rightarrow$ Previous Slide (`left` arrow)
  - `right` $\rightarrow$ Next Slide (`right` arrow)
  - `up` $\rightarrow$ Start Presentation from Beginning (`F5`)
  - `down` $\rightarrow$ Toggle Black Screen (`B`)
  - `shake` $\rightarrow$ End Presentation / Cancel (`Esc`)
  - `still` $\rightarrow$ Neutral / No Action
  - `unknown` $\rightarrow$ Action Suppressed

### 3.2 Detection and Quarantining of Anomalous Data (Trial 15 Bug)
- **Previous Issue:** In `data/raw_new/tilt_up.csv`, trial 15 had its label recorded as `'n'`. Training notebooks mapped this to `NaN` and silently dropped it, leaving only 19 trials for `up`.
- **Resolution:** `src/replay_source.py` and `src/train.py` actively audit dataset rows. Unmapped labels are not guessed or discarded silently; they are formally flagged, quarantined, and exported to `dataset_quarantine_log.json` for reviewer auditing.

### 3.3 Test Set Contamination Elimination
- **Previous Issue:** Notebook Cell 41 passed `validation_data=(X_test, y_test_cat)` to monitor CNN training during the reduced-data experiment, leaking test set information.
- **Resolution:** `src/train.py` isolates a 70% train split, a 15% validation split, and a 15% final test split. Model selection and training loss monitor only `X_val`; `X_test` is touched solely for final evaluation.

### 3.4 Preservation of Legacy Artifacts
- **Policy Followed:** Original datasets in `data/` and original models in `models/` were kept strictly intact as benchmark baselines. All new training runs export to `models/reproducible/` with complete `metadata.json` records.

---

## 4. Hardware Assumptions Awaiting Physical Sensor Validation

The following assumptions cannot be verified without physical hardware:

1. **MPU6050 I2C Bus Timing & Clock Speed:** Firmware configures 400 kHz Fast-Mode I2C. Bus pull-up resistors and cable capacitance on breadboards must be validated for signal integrity without bus lockups.
2. **Physical Button Bounce Characteristics:** Firmware implements a 200 ms non-blocking debounce timer on GPIO 4 with internal pull-up. The mechanical bounce of the specific push-button must be checked.
3. **USB Serial Latency & Buffer Overrun:** Live serial communication at 115200 baud produces $\approx 1.8\text{ KB/s}$. Serial buffer overrun must be verified during rapid movements.
4. **Thermal Drift and Zero-Rate Bias:** Low-cost MEMS IMUs (MPU6050) experience thermal gyroscope drift over continuous 20–30 minute presentations.
5. **Wearable Kinematics (Wrist vs. Desk):** Desktop flat calibration assumes gravity vector $+1.0g$ on $+z$. A sensor worn on the dorsal wrist in a natural standing presentation posture has gravity distributed between axes.

---

## 5. Next Steps After Acquiring Hardware

1. **Step 1: Flash Firmware**
   - Open `firmware/imu_remote.ino` in Arduino IDE or VS Code with ESP32 board support.
   - Select your ESP32 board and upload at 115200 baud.
   - Verify startup banner in Serial Monitor: `# MPU6050 initialized successfully`.
2. **Step 2: Verify Button Event in Serial Monitor**
   - Connect a push button between GPIO 4 and GND.
   - Press the button and confirm `# BUTTON_PRESSED` appears.
3. **Step 3: Collect First Wrist-Worn Dataset**
   - Strap the sensor securely to the dorsal wrist of your dominant hand.
   - Run: `py -3.12 src/logger.py --port COMx --participant P01 --session 1`
   - Collect 20 trials per gesture with realistic presentation arm motions.
4. **Step 4: Retrain & Test with GUI**
   - Retrain: `py -3.12 src/train.py --data-dir data/collected`
   - Launch GUI: `py -3.12 src/gui_app.py`, switch to Serial Mode, uncheck Dry Run, and control an active PowerPoint slideshow.
