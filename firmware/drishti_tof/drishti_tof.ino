/*
 * Drishti ESP32 sensor MCU firmware
 * ---------------------------------
 * Board:  ESP32 DevKit V1 (30-pin, CP2102/CH340 USB-UART)
 * Sensor: VL53L0X time-of-flight range finder (I2C, address 0x29)
 * Inputs: two momentary push buttons to GND (internal pull-ups)
 *
 * Responsibilities
 *   - Own the deterministic VL53L0X ranging loop (20 ms timing budget).
 *   - Reject invalid measurements (range status != 0).
 *   - Median-of-5 + EMA filtering.
 *   - Latch both button edges and stream them over USB serial.
 *   - Emit a 1 Hz health line and honour a host heartbeat.
 *
 * Protocol (ASCII CSV, newline terminated, 921600 8N1)
 *   Host -> MCU : "P\n" heartbeat | "V\n" version/banner | "R\n" reset
 *   MCU -> host : "I,<fw>,<model>"
 *                 "D,<seq>,<t_ms>,<mm_f>,<mm_raw>,<st>,<btn>"
 *                 "B,<id>,<t_ms>,<edge>"      edge: 1=press, 0=release
 *                 "S,<hz>,<invalid_pct>,<i2c_ok>,<link_ok>"
 *                 "E,<code>,<msg>"
 *
 * WiFi and Bluetooth are never started, keeping the ranging loop jitter-free.
 */

#include <Arduino.h>
#include <Wire.h>
#include <VL53L0X.h>

namespace {

constexpr uint8_t PIN_SDA = 21;
constexpr uint8_t PIN_SCL = 22;
constexpr uint8_t PIN_BUTTON_1 = 32;  // Narration
constexpr uint8_t PIN_BUTTON_2 = 33;  // Currency / OCR

constexpr uint32_t SERIAL_BAUD = 921600;
constexpr uint16_t TIMING_BUDGET_MS = 33;
constexpr uint8_t FILTER_WINDOW = 5;
constexpr uint16_t MIN_VALID_MM = 30;
constexpr uint16_t MAX_VALID_MM = 2000;
constexpr uint32_t STATUS_INTERVAL_MS = 1000;
constexpr uint32_t HOST_TIMEOUT_MS = 2000;
constexpr uint32_t DEBOUNCE_MS = 30;
constexpr const char *FW_VERSION = "1.0.0";

}  // namespace

VL53L0X sensor;
bool sensorReady = false;
bool hostAnnounced = false;

uint32_t sequenceNumber = 0;
uint32_t lastStatusMs = 0;
uint32_t lastHostMs = 0;
uint32_t statusSamples = 0;
uint32_t statusInvalid = 0;

uint16_t filterBuffer[FILTER_WINDOW];
uint8_t filterCount = 0;
uint8_t filterIndex = 0;
float emaMm = -1.0f;

struct ButtonState {
  uint8_t pin;
  uint8_t id;
  bool stablePressed;
  bool lastReading;
  uint32_t lastChangeMs;
};

ButtonState buttons[2] = {
    {PIN_BUTTON_1, 1, false, false, 0},
    {PIN_BUTTON_2, 2, false, false, 0},
};

uint16_t medianFilter(uint16_t sample) {
  if (sample >= MIN_VALID_MM && sample <= MAX_VALID_MM) {
    filterBuffer[filterIndex] = sample;
    filterIndex = (filterIndex + 1) % FILTER_WINDOW;
    if (filterCount < FILTER_WINDOW) {
      filterCount++;
    }
  }

  if (filterCount == 0) {
    return 0;
  }

  uint16_t sorted[FILTER_WINDOW];
  for (uint8_t i = 0; i < filterCount; i++) {
    sorted[i] = filterBuffer[i];
  }
  for (uint8_t i = 1; i < filterCount; i++) {
    uint16_t key = sorted[i];
    int8_t j = static_cast<int8_t>(i) - 1;
    while (j >= 0 && sorted[j] > key) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = key;
  }

  uint16_t median = sorted[filterCount / 2];
  if (emaMm < 0.0f) {
    emaMm = static_cast<float>(median);
  } else {
    emaMm = 0.3f * static_cast<float>(median) + 0.7f * emaMm;
  }
  return static_cast<uint16_t>(emaMm + 0.5f);
}

uint8_t buttonMask() {
  uint8_t mask = 0;
  if (buttons[0].stablePressed) {
    mask |= 0x01;
  }
  if (buttons[1].stablePressed) {
    mask |= 0x02;
  }
  return mask;
}

void pollButtons() {
  uint32_t now = millis();
  for (uint8_t i = 0; i < 2; i++) {
    ButtonState &button = buttons[i];
    bool reading = (digitalRead(button.pin) == LOW);  // active low

    if (reading != button.lastReading) {
      button.lastReading = reading;
      button.lastChangeMs = now;
    }

    if ((now - button.lastChangeMs) >= DEBOUNCE_MS && reading != button.stablePressed) {
      button.stablePressed = reading;
      Serial.printf("B,%u,%lu,%u\n", button.id, static_cast<unsigned long>(now),
                    reading ? 1u : 0u);
    }
  }
}

void pollHost() {
  while (Serial.available() > 0) {
    char command = static_cast<char>(Serial.read());
    if (command == 'P' || command == 'p') {
      lastHostMs = millis();
      // A host that connects after boot still needs the firmware banner.
      if (!hostAnnounced) {
        Serial.printf("I,%s,VL53L0X\n", FW_VERSION);
        hostAnnounced = true;
      }
    } else if (command == 'V' || command == 'v') {
      // Version query: sent by the host on every (re)connect.
      Serial.printf("I,%s,VL53L0X\n", FW_VERSION);
      hostAnnounced = true;
    } else if (command == 'R' || command == 'r') {
      delay(20);
      ESP.restart();
    }
    // "T,<ms>" is reserved for future timing-budget changes and ignored here.
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(50);

  pinMode(PIN_BUTTON_1, INPUT_PULLUP);
  pinMode(PIN_BUTTON_2, INPUT_PULLUP);

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(100000);  // 100 kHz standard mode for robust breadboard wiring

  sensor.setTimeout(200);
  sensorReady = sensor.init();

  if (!sensorReady) {
    Serial.println("E,1,sensor_init_failed");
  } else {
    // Long-range profile (up to ~2 m on a cooperative target).
    sensor.setSignalRateLimit(0.1);
    sensor.setVcselPulsePeriod(VL53L0X::VcselPeriodPreRange, 18);
    sensor.setVcselPulsePeriod(VL53L0X::VcselPeriodFinalRange, 14);
    sensor.setMeasurementTimingBudget(TIMING_BUDGET_MS * 1000);
    sensor.startContinuous();
    Serial.printf("I,%s,VL53L0X\n", FW_VERSION);
  }

  lastHostMs = millis();
  lastStatusMs = millis();
}

void loop() {
  pollHost();
  pollButtons();

  if (!sensorReady) {
    static uint32_t lastRetryMs = 0;
    if (millis() - lastRetryMs >= 1500) {
      lastRetryMs = millis();
      Wire.begin(PIN_SDA, PIN_SCL);
      Wire.setClock(100000);
      sensor.setTimeout(200);
      sensorReady = sensor.init();
      if (sensorReady) {
        sensor.setSignalRateLimit(0.1);
        sensor.setVcselPulsePeriod(VL53L0X::VcselPeriodPreRange, 18);
        sensor.setVcselPulsePeriod(VL53L0X::VcselPeriodFinalRange, 14);
        sensor.setMeasurementTimingBudget(TIMING_BUDGET_MS * 1000);
        sensor.startContinuous();
        Serial.printf("I,%s,VL53L0X\n", FW_VERSION);
      } else {
        Wire.beginTransmission(0x29);
        uint8_t err = Wire.endTransmission();
        if (err == 0) {
          Serial.println("E,1,I2C device at 0x29 responded but init failed. Ensure XSHUT is tied to 3V3.");
        } else {
          Serial.println("E,1,No I2C ack at 0x29. Move SDA to D21, SCL to D22, VIN to 3V3, XSHUT to 3V3.");
        }
      }
    }
  }

  if (sensorReady) {
    uint16_t rawMm = sensor.readRangeContinuousMillimeters();
    uint8_t rangeStatus = 0;
    if (sensor.timeoutOccurred()) {
      rangeStatus = 254;
      // Recover from timeout by explicitly clearing the ST hardware interrupt
      Wire.beginTransmission(0x29);
      Wire.write(0x0B);  // SYSTEM_INTERRUPT_CLEAR
      Wire.write(0x01);
      Wire.endTransmission();
    } else if (rawMm >= 8000 || rawMm > MAX_VALID_MM) {
      rangeStatus = 1;  // Target out of range / no reflection
    } else if (rawMm < MIN_VALID_MM) {
      rangeStatus = 2;  // Below minimum valid distance
    } else {
      rangeStatus = 0;  // Clean, valid range measurement
    }

    uint16_t filteredMm = medianFilter(rangeStatus == 0 ? rawMm : 0);

    sequenceNumber++;
    statusSamples++;
    if (rangeStatus != 0) {
      statusInvalid++;
    }

    Serial.printf("D,%lu,%lu,%u,%u,%u,%u\n", static_cast<unsigned long>(sequenceNumber),
                  static_cast<unsigned long>(millis()), filteredMm, rawMm, rangeStatus,
                  buttonMask());
  }

  uint32_t now = millis();
  if (now - lastStatusMs >= STATUS_INTERVAL_MS) {
    float elapsedSeconds = static_cast<float>(now - lastStatusMs) / 1000.0f;
    float measuredHz = (elapsedSeconds > 0.0f)
                           ? static_cast<float>(statusSamples) / elapsedSeconds
                           : 0.0f;
    float invalidPct = (statusSamples > 0)
                           ? 100.0f * static_cast<float>(statusInvalid) /
                                 static_cast<float>(statusSamples)
                           : 0.0f;
    bool linkOk = (now - lastHostMs) < HOST_TIMEOUT_MS;

    Serial.printf("S,%.1f,%.1f,%u,%u\n", measuredHz, invalidPct, sensorReady ? 1u : 0u,
                  linkOk ? 1u : 0u);

    lastStatusMs = now;
    statusSamples = 0;
    statusInvalid = 0;
  }
}
