# Drishti: Troubleshooting Guide & Diagnostic Manual

This manual documents known failure modes, root causes, diagnostic procedures, and permanent solutions across development workstations and target Raspberry Pi 5 hardware.

Last updated: 2026-11-09.

---

## 1. Quick Diagnostic Decision Matrix

```text
[System Issue Encountered]
         │
         ├──► Console prints "NAVIGATION NOT READY"?
         │       └──► See Section 2.1: Fail-Closed Safety Readiness Diagnostics
         │
         ├──► Book / Text Reading returns "No clear text detected"?
         │       └──► See Section 3.1: Anonymizer False-Positive Blur on Documents & Currency
         │
         ├──► Hearing "Unclassified obstacle" after pressing a button?
         │       └──► See Section 3.2: Distinguishing 30 FPS Local Sonar from Cloud Vision
         │
         ├──► Cloud Vision queries take >15 seconds or time out?
         │       └──► See Section 3.3: Gemini Vision API Latency & Thinking Budget
         │
         ├──► Buttons appear permanently pressed or stuck?
         │       └──► See Section 4.1: Tactile Switch 90-Degree Breadboard Rotation
         │
         ├──► ESP32 Serial Link missing or timeouts?
         │       └──► See Section 4.2: ESP32 USB-Serial & Permission Diagnostics
         │
         ├──► Audio stuttering, pops, or underruns?
         │       └──► See Section 5.1: ALSA Audio Buffer Underruns (xrun)
         │
         └──► Camera unavailable or busy?
                 └──► See Section 6.1: Video Device Locks & Permissions
```

---

## 2. Safety Readiness & Fail-Closed Navigation

### 2.1 Navigational State Stuck in `NAVIGATION NOT READY`
#### Symptoms:
- The terminal displays:
  ```text
  Cycle 1     | NAVIGATION NOT READY | CALIBRATION_UNVERIFIED, RANGE_NO_VALID_SAMPLE
  ```
- No binaural obstacle pings are emitted, even when objects are in front of the camera.
- Debug viewer displays status: `NAV NOT READY: UNKNOWN`.

#### Root Cause:
Drishti enforces an uncompromising **fail-closed** safety standard. Navigational spatial pings are synthesized *only* when physical sensors are authoritative. Inferred-distance audio pings (from 2D YOLO bounding box priors) are strictly suppressed if physical ToF range is missing, stale, or unverified.

#### Diagnostic Checklist:
1. **Check Physical ToF Connection**:
   ```bash
   python3 scripts/probe_esp32.py --port auto --duration 5
   ```
   If no samples arrive, the serial link is disconnected or the sensor is unpowered.
2. **Inspect Readiness Diagnostic Codes**:
   - `RANGE_NO_VALID_SAMPLE`: The ESP32 is disconnected, unpowered, or has not sent a valid range packet within $100\text{ ms}$.
   - `CALIBRATION_UNVERIFIED`: Camera FOV and optical-to-ToF baseline measurements have not been confirmed in `config/pipeline_config.yaml`.
   - `CAMERA_UNAVAILABLE`: Physical camera is not returning frames.
   - `MODEL_CONTRACT_INVALID`: `models/yolov8n.onnx` is missing or corrupted.

#### Resolution:
- For development testing where physical ToF is absent, run explicit simulation:
  ```bash
  python3 scripts/run_live_wearable.py --hardware synthetic
  ```
- For production operation with hardware, verify the ESP32 USB cable is plugged in and recognized under `/dev/serial/by-id/`.

---

## 3. Visual Cognition & Edge Privacy Subsystem

### 3.1 Document & Banknote Blurring (False-Positive Anonymization)
#### Symptoms:
- Holding a book or document in front of the camera and pressing **Button 3 (Text Reading)** results in:
  ```text
  >>> [TEXT READING] Spoken: "No clear text detected."
  ```
- Holding a banknote and pressing **Button 2 (Currency Identification)** results in:
  ```text
  >>> [CURRENCY] Spoken: "No banknote clearly detected."
  ```
- Debug viewer snapshot reveals the book or banknote is Gaussian-blurred across the entire page.

#### Root Cause:
1. **Synthetic Face Detection Heuristic Collision**: In earlier revisions, `safety/anonymizer.py` contained a color heuristic (`mask = (B > 150) & (G > 180) & (R > 220)`) intended for synthetic simulation markers. Because white book pages, paper, and bright currency notes contain high BGR values, the heuristic flagged the entire document as a "face" `(0, 0, 640, 480)` and blurred the entire frame prior to upload.
2. **License Plate Cascade False Positives**: OpenCV's `haarcascade_russian_plate_number.xml` was running over every frame, detecting printed lines of text on books and documents as license plates and blurring each line.

#### Verification & Fix (Implemented in `aa55387`):
1. **Synthetic Marker Scoping**: The synthetic color marker heuristic is now strictly guarded by the background signature of `SyntheticCameraSource` (`frame[0, 0, 1] == 40 and frame[0, 0, 2] == 40`). Real webcam images never trigger this heuristic.
2. **Plate Cascade Scoping**: License plate detection executes **only** when YOLO detects an actual vehicle (`car`, `bus`, `truck`, `motorcycle`) and only searches within the vehicle bounding box.
3. **Verify with Regression Test**:
   ```bash
   python3 -m pytest tests/unit/test_anonymizer.py -k "test_anonymizer_does_not_blur_text"
   ```

---

### 3.2 Mistaking 30 FPS Obstacle Sonar for Button Response
#### Symptoms:
- The user holds a book or document close to the camera, presses Button 3 for text reading, and immediately hears:
  ```text
  "Caution: unclassified obstacle 0.4 meters ahead"
  ```
- The terminal displays:
  ```text
  Cycle 42    | unclassified obstacle | 0.38 m   | +0.0°   | 3D: [+0.00, +0.00, 0.38]
  ```
- The user believes the text reading model misclassified their book as an "unclassified object."

#### Explanation:
Drishti runs two distinct perception streams simultaneously:
1. **Fast Perception Stream (30 FPS, Local Edge)**: Continuously evaluates the ToF laser sensor. When any object is held $<0.8\text{ m}$ from the sensor, it immediately alerts the user of a physical obstruction in their walking path. Because a close-up book or hand is not a standard YOLO hazard (person/car/chair), it is labeled `unclassified obstacle`.
2. **Cognitive Vision Stream (~1.5 s, Cloud Multimodal)**: Asynchronously triggered by the physical button.

#### Resolution & Operation:
- When you press a button, notice the instant audio and console confirmation:
  ```text
  >>> [BUTTON 3 PRESSED] Multilingual Text Reading queued for Gemini Vision...
  Audio Prompt: "Reading text..."
  ```
- Approximately 1.5 seconds later, the cloud vision response will arrive and be spoken:
  ```text
  >>> [TEXT READING] Spoken: "Chapter 1: Artificial Intelligence and Neural Networks"
  ```
- Maintain the reading material at a natural distance ($25\text{--}40\text{ cm}$) with adequate lighting.

---

### 3.3 Gemini API Latency & Thinking Budget Overheads
#### Symptoms:
- Scene narration, currency, or OCR takes 15 to 25 seconds to speak after pressing a button.
- Cognitive requests time out with network exceptions.

#### Root Cause:
Google's Gemini 3.1 Flash-Lite and 2.5 Flash reasoning models allocate an internal "thinking budget" by default, generating multi-step internal reasoning tokens before outputting the final JSON response.

#### Permanent Fix:
In `cognitive/online_vision.py`, `generationConfig` sets `thinkingBudget` to `0`:
```python
"generationConfig": {
    "temperature": 0.0,
    "maxOutputTokens": 300,
    "thinkingConfig": {
        "thinkingBudget": 0
    }
}
```
This forces immediate single-turn inference, dropping end-to-end response times from $>20\text{ s}$ to **$1.5\text{--}1.8\text{ s}$**.

---

### 3.4 Ineffective Custom Transport / IPv4 Adapter on Requests 2.32+
#### Symptoms:
- Runtime warnings or `TypeError: create_connection() missing required argument 'timeout'` in modern virtualenvs.

#### Root Cause:
Requests 2.32+ routes HTTPS requests through `get_connection_with_tls_context`, bypassing legacy `get_connection()` overrides in custom `HTTPAdapter` subclasses.

#### Permanent Fix:
`OnlineVisionService` utilizes standard `requests.Session()` with clean connection pooling, native DNS resolution, and explicit timeouts ($5.0\text{ s}$ connect, $15.0\text{ s}$ read).

---

## 4. Hardware Sensors & ESP32 Serial Link

### 4.1 Tactile Switches Stuck Active (`buttonMask = 7` or Continuous Press)
#### Symptoms:
- Probing the ESP32 reports constant button presses without physical contact:
  ```text
  >>> [BUTTON EVENT] Button 1 Pressed: Scene Narration queued...
  ```
- Probe output shows `buttonMask: 7` (binary `111`).

#### Root Cause:
Standard 4-pin 6×6 mm tactile switches have internally connected leg pairs on opposite sides. Inserting the switch rotated by $90^\circ$ across the breadboard center trench connects GPIO directly to the GND rail permanently, pulling the active-LOW input to ground ($0\text{ V}$).

#### Diagnostic:
```bash
python3 scripts/probe_esp32.py --port auto --duration 3
```
Inspect the displayed `buttonMask`. In the idle state, `buttonMask` must equal `0`.

#### Resolution:
1. Rotate each push button $90^\circ$ so that the tied leg pairs straddle the center breadboard trench.
2. Confirm the wiring: one leg to GPIO (`GPIO32`, `GPIO33`, `GPIO25`), the opposite leg to the common `GND` rail.

---

### 4.2 ESP32 USB-Serial Permission & Port Identification
#### Symptoms:
- `PermissionError: [Errno 13] Permission denied: '/dev/ttyUSB0'`
- `ESP32 serial: no frames (timeout; check USB cable/port)`

#### Root Cause:
1. Current Linux user is not a member of the `dialout` group.
2. Micro-USB cable is a charge-only cable lacking data D+/D- lines.

#### Resolution:
1. Add user to `dialout` group and refresh permissions:
   ```bash
   sudo usermod -aG dialout "$USER"
   newgrp dialout
   ```
2. Verify kernel USB enumeration:
   ```bash
   ls -la /dev/serial/by-id/
   dmesg | grep -E "ttyUSB|ttyACM|CP210|ch341"
   ```
3. Target the stable `/dev/serial/by-id/` path when launching:
   ```bash
   python3 scripts/run_live_wearable.py --serial-port /dev/serial/by-id/usb-Silicon_Labs_CP2102...
   ```

---

### 4.3 ESP32 VL53L0X Ranging Lockup (`status: 254`, `raw: 65535`)
#### Symptoms:
- The ESP32 streams data, but reported distance is locked at `8190 mm` or `raw: 65535`, with status `254`.

#### Root Cause:
ST Microelectronics VL53L0X long-range profile requires a **33 ms** measurement timing budget. If Pololu's continuous ranging mode encounters a hardware timeout, the internal interrupt register is left uncleared, latching all future reads into timeout state.

#### Resolution (Implemented in Firmware):
1. In `firmware/drishti_tof/drishti_tof.ino`, `TIMING_BUDGET_MS` is set to `33`.
2. Firmware auto-clears interrupt status: if `sensor.timeoutOccurred()`, it explicitly writes `0x01` to `SYSTEM_INTERRUPT_CLEAR (0x0B)` over I²C.
3. Flash the verified firmware using PlatformIO:
   ```bash
   cd firmware/drishti_tof
   pio run -t upload
   ```

---

## 5. Audio Playback & ALSA Infrastructure

### 5.1 ALSA Buffer Underruns (`xrun`) & Audio Stutter
#### Symptoms:
- Crackling, popping, or console warnings:
  ```text
  ALSA lib pcm.c: underrun occurred
  ```

#### Root Cause:
The real-time computer vision loop (YOLO inference on CPU) saturates all CPU cores, starving the audio playback thread of time slices needed to feed the sound card DMA buffer.

#### Resolution:
1. `ALSAAudioSink` utilizes a bounded queue of four PCM frames; if full, stale ambient pings are dropped to prevent latency accumulation.
2. Ensure the process is granted standard scheduling priority:
   ```bash
   # Check real-time thread priority
   chrt -p $(pgrep -f "run_live_wearable")
   ```
3. If running over Bluetooth headphones (e.g. JBL TUNE500BT), verify PipeWire/PulseAudio is using A2DP codec without packet dropouts.

---

### 5.2 Direct Hardware Audio Routing (`Host is down`)
#### Symptoms:
- `audio open error: Host is down` when opening `default` ALSA PCM.

#### Root Cause:
Running from background daemons or SSH sessions without active PulseAudio/PipeWire user session sockets.

#### Resolution:
Target the physical ALSA card directly via hardware plug:
```bash
aplay -l
# Specify card and device index directly in configuration:
export DRISHTI_AUDIO_DEVICE="plughw:1,0"
```

---

## 6. Camera Subsystem & V4L2

### 6.1 Camera Device Busy or Locked
#### Symptoms:
- `cv2.error: OpenCV(4.x) Can't open camera by index 0`
- `/dev/video0: Device or resource busy`

#### Root Cause:
A previous background python session, motion daemon, or browser tab holds an open file descriptor on `/dev/video0`.

#### Resolution:
Identify and kill the blocking process:
```bash
# Locate holding process PID
sudo fuser -v /dev/video0
# Terminate orphaned process
sudo fuser -k /dev/video0
```

---

## 7. Automated System Health Check

Run the comprehensive non-destructive peripheral audit script before deploying the system:
```bash
python3 scripts/check_hardware_health.py
```

### Expected Output for Healthy System:
```text
================================================================================
                    DRISHTI HARDWARE & SUBSYSTEM HEALTH CHECK
================================================================================
[✓] Camera Subsystem: OpenCV backend acquired device index 0 (640x480 @ 30 FPS).
[✓] ESP32 Link: Connected on /dev/serial/by-id/... (Firmware 1.0.0, 33 Hz).
[✓] ToF Sensor: VL53L0X healthy (Live Range: 842 mm).
[✓] Tactile Inputs: Buttons verified idle (Mask: 0).
[✓] Neural Weights: YOLOv8n ONNX model verified (models/yolov8n.onnx).
[✓] Speech Engine: Piper neural voice verified (models/piper/en_US-lessac-medium.onnx).
[✓] Vision API: Gemini credentials active and provider online.
================================================================================
  >>> ALL CRITICAL SUBSYSTEMS OPERATIONAL: READY FOR LIVE SYSTEM RUNNER <<<
================================================================================
```
