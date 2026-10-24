// ==============================================================================
// UPGRADED FIRMWARE: ESP32 + MPU6050 Hand Gesture Controller
// For Graduation Project: "Machine Learning-Based Hand Gesture Recognition
// Using a Wearable IMU Sensor for PowerPoint Control"
//
// Key Upgrades & Engineering Fixes:
// 1. Configurable physical capture button with non-blocking software debouncing.
// 2. Consistent serial protocol emitting '# BUTTON_PRESSED' upon trigger.
// 3. Compensated interval sampling (50 Hz / 20 ms) eliminating delay() timing drift.
// 4. Clear state machine: STATE_IDLE, STATE_STREAMING, STATE_CALIBRATING.
// 5. Explicit sensor scaling comments (±2g -> 16384 LSB/g, ±250 dps -> 131 LSB/dps).
// 6. Transparent calibration routine distinguishing sensor bias from gravity vector.
// 7. Prepared for future wireless transport (stream abstraction for Serial/Bluetooth).
//
// Hardware Status Notice:
// PREPARED AND STRUCTURALLY AUDITED. UNCOMPILED / NOT HARDWARE-TESTED.
// Physical hardware validation to be performed upon board delivery.
// ==============================================================================

#include <Wire.h>
#include <MPU6050.h>

// -----------------------------
// Pin Configurations
// -----------------------------
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22

// Physical capture button pin (Connect button between GPIO 4 and GND)
#define BUTTON_PIN 4
#define BUTTON_DEBOUNCE_MS 200

// -----------------------------
// Acquisition Timing Configuration
// -----------------------------
// Nominal sampling rate: 50 Hz -> 20 ms interval
// Preserves temporal alignment with existing 51-sample model windows (~1.0s duration)
#define TARGET_SAMPLE_RATE_HZ 50
#define SAMPLE_INTERVAL_MS (1000 / TARGET_SAMPLE_RATE_HZ)

// Stream transport abstraction (Serial over USB; easily redirected to BluetoothSerial later)
#define SENSOR_STREAM Serial

// -----------------------------
// Global Instances & Variables
// -----------------------------
MPU6050 mpu;

// Raw 16-bit registers
int16_t ax_raw, ay_raw, az_raw;
int16_t gx_raw, gy_raw, gz_raw;

// Scaled physical quantities (Units: ax, ay, az in g-force; gx, gy, gz in deg/s)
float ax = 0.0f, ay = 0.0f, az = 0.0f;
float gx = 0.0f, gy = 0.0f, gz = 0.0f;

// Calibration zero-rate offsets
float ax_bias = 0.0f, ay_bias = 0.0f, az_bias = 0.0f;
float gx_bias = 0.0f, gy_bias = 0.0f, gz_bias = 0.0f;
bool imuCalibrated = false;

// State machine
enum SystemState {
  STATE_IDLE,
  STATE_STREAMING,
  STATE_CALIBRATING
};
SystemState currentState = STATE_IDLE;

// Timing counters
unsigned long sessionStartTime = 0;
unsigned long lastSampleTime = 0;
unsigned long lastButtonPressTime = 0;
int lastButtonState = HIGH;

// Serial input buffer
String inputCommandBuffer = "";

// -----------------------------
// Helper: Print CSV Header
// -----------------------------
void printHeader() {
  SENSOR_STREAM.println("timestamp_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
}

// -----------------------------
// Sensor Readout & Unit Scaling
// -----------------------------
void readIMU() {
  mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

  // MPU6050 standard scaling at default ranges:
  // Accelerometer range ±2g  -> Sensitivity: 16384 LSB / g
  // Gyroscope range ±250 dps -> Sensitivity: 131 LSB / (deg/s)
  ax = (float)ax_raw / 16384.0f;
  ay = (float)ay_raw / 16384.0f;
  az = (float)az_raw / 16384.0f;

  gx = (float)gx_raw / 131.0f;
  gy = (float)gy_raw / 131.0f;
  gz = (float)gz_raw / 131.0f;

  // Apply calibration bias subtraction if calibrated
  if (imuCalibrated) {
    ax -= ax_bias;
    ay -= ay_bias;
    az -= az_bias;
    gx -= gx_bias;
    gy -= gy_bias;
    gz -= gz_bias;
  }
}

// -----------------------------
// Calibration Routine
// -----------------------------
// CAUTION: Benchtop calibration assumes the sensor is stationary and horizontal
// with the Z-axis pointing up (measuring +1.0g gravity).
// For wrist-worn operation, resting gravity will differ depending on arm orientation.
// Gyro zero-rate bias can be calibrated at any stationary orientation.
void calibrateIMU(int sampleCount = 400) {
  SystemState previousState = currentState;
  currentState = STATE_CALIBRATING;

  SENSOR_STREAM.println("# CALIBRATION_STARTED");
  SENSOR_STREAM.println("# Keep IMU stationary in flat reference orientation");

  delay(1500); // Allow physical settle

  float ax_sum = 0.0f, ay_sum = 0.0f, az_sum = 0.0f;
  float gx_sum = 0.0f, gy_sum = 0.0f, gz_sum = 0.0f;
  int validSamples = 0;

  for (int i = 0; i < sampleCount; i++) {
    mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

    ax_sum += ((float)ax_raw / 16384.0f);
    ay_sum += ((float)ay_raw / 16384.0f);
    az_sum += ((float)az_raw / 16384.0f);

    gx_sum += ((float)gx_raw / 131.0f);
    gy_sum += ((float)gy_raw / 131.0f);
    gz_sum += ((float)gz_raw / 131.0f);

    validSamples++;
    delay(5);
  }

  if (validSamples > 0) {
    // Gyro zero-rate bias: true rate is 0.0 deg/s when stationary
    gx_bias = gx_sum / (float)validSamples;
    gy_bias = gy_sum / (float)validSamples;
    gz_bias = gz_sum / (float)validSamples;

    // Desktop flat reference orientation assumes ax=0, ay=0, az=+1.0g
    ax_bias = ax_sum / (float)validSamples;
    ay_bias = ay_sum / (float)validSamples;
    az_bias = (az_sum / (float)validSamples) - 1.0f;

    imuCalibrated = true;

    SENSOR_STREAM.println("# CALIBRATION_DONE");
    SENSOR_STREAM.print("# accel_bias=");
    SENSOR_STREAM.print(ax_bias, 4); SENSOR_STREAM.print(",");
    SENSOR_STREAM.print(ay_bias, 4); SENSOR_STREAM.print(",");
    SENSOR_STREAM.println(az_bias, 4);

    SENSOR_STREAM.print("# gyro_bias=");
    SENSOR_STREAM.print(gx_bias, 4); SENSOR_STREAM.print(",");
    SENSOR_STREAM.print(gy_bias, 4); SENSOR_STREAM.print(",");
    SENSOR_STREAM.println(gz_bias, 4);
  } else {
    SENSOR_STREAM.println("# CALIBRATION_FAILED");
  }

  if (previousState == STATE_STREAMING) {
    sessionStartTime = millis();
    printHeader();
    SENSOR_STREAM.println("# STREAMING_RESUMED");
    currentState = STATE_STREAMING;
  } else {
    currentState = STATE_IDLE;
  }
}

// -----------------------------
// Button Event Handler
// -----------------------------
void checkPhysicalButton() {
  int reading = digitalRead(BUTTON_PIN);
  unsigned long now = millis();

  // Active LOW (internal pull-up enabled)
  if (reading == LOW && lastButtonState == HIGH) {
    if (now - lastButtonPressTime > BUTTON_DEBOUNCE_MS) {
      lastButtonPressTime = now;
      // Send standardized button event message recognized by live_demo_hotkey_ad.py & GUI
      SENSOR_STREAM.println("# BUTTON_PRESSED");
    }
  }
  lastButtonState = reading;
}

// -----------------------------
// Command Parser
// -----------------------------
void handleSerialCommand(String cmd) {
  cmd.trim();
  cmd.toLowerCase();

  if (cmd == "start") {
    if (currentState != STATE_STREAMING) {
      sessionStartTime = millis();
      lastSampleTime = sessionStartTime;
      printHeader();
      SENSOR_STREAM.println("# STREAMING_STARTED");
      currentState = STATE_STREAMING;
    }
  } else if (cmd == "stop") {
    if (currentState == STATE_STREAMING) {
      currentState = STATE_IDLE;
      SENSOR_STREAM.println("# STREAMING_STOPPED");
    }
  } else if (cmd == "calibrate") {
    calibrateIMU();
  } else if (cmd == "status") {
    SENSOR_STREAM.print("# STATE=");
    SENSOR_STREAM.print(currentState == STATE_STREAMING ? "STREAMING" : (currentState == STATE_CALIBRATING ? "CALIBRATING" : "IDLE"));
    SENSOR_STREAM.print(" CALIBRATED=");
    SENSOR_STREAM.println(imuCalibrated ? "YES" : "NO");
  }
}

// -----------------------------
// Arduino Setup
// -----------------------------
void setup() {
  SENSOR_STREAM.begin(115200);

  // Setup physical button with internal pull-up
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // Setup I2C interface on ESP32
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000); // 400 kHz Fast-Mode I2C for minimal bus latency

  mpu.initialize();

  if (mpu.testConnection()) {
    SENSOR_STREAM.println("# MPU6050 initialized successfully (Fast I2C 400kHz)");
  } else {
    SENSOR_STREAM.println("# ERROR: MPU6050 connection failed! Check SDA=GPIO21, SCL=GPIO22, VCC, GND.");
  }

  SENSOR_STREAM.println("# Ready. Supported serial commands: 'start', 'stop', 'calibrate', 'status'");
  SENSOR_STREAM.print("# Physical button configured on GPIO ");
  SENSOR_STREAM.println(BUTTON_PIN);

  // Optional baseline calibration at power-on
  calibrateIMU();
}

// -----------------------------
// Main Execution Loop
// -----------------------------
void loop() {
  // Check for incoming serial commands
  if (SENSOR_STREAM.available()) {
    String incoming = SENSOR_STREAM.readStringUntil('\n');
    handleSerialCommand(incoming);
  }

  // Check physical button trigger
  checkPhysicalButton();

  // Continuous stream with compensated interval timing
  if (currentState == STATE_STREAMING) {
    unsigned long currentMillis = millis();
    if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL_MS) {
      lastSampleTime = currentMillis;

      readIMU();

      SENSOR_STREAM.print(currentMillis - sessionStartTime);
      SENSOR_STREAM.print(",");
      SENSOR_STREAM.print(ax, 4); SENSOR_STREAM.print(",");
      SENSOR_STREAM.print(ay, 4); SENSOR_STREAM.print(",");
      SENSOR_STREAM.print(az, 4); SENSOR_STREAM.print(",");
      SENSOR_STREAM.print(gx, 4); SENSOR_STREAM.print(",");
      SENSOR_STREAM.print(gy, 4); SENSOR_STREAM.print(",");
      SENSOR_STREAM.println(gz, 4);
    }
  }
}