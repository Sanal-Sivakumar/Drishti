# Drishti: Technical Specifications & System Architecture

This document specifies the technical design, mathematical formulations, sensor fusion algorithms, safety contracts, and communication protocols implemented across the Drishti platform.

Last updated: 2026-11-09.

---

## 1. System Architecture & Concurrency Model

Drishti implements a dual-stream concurrency architecture that strictly isolates safety-critical hazard perception from cloud-assisted visual cognition:

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            MAIN PERCEPTION THREAD (30 FPS)                       │
│                                                                                  │
│   Camera V4L2/OpenCV          ESP32 Serial Link (921600 baud)                    │
│         │                                    │                                   │
│         ▼                                    ▼                                   │
│   BGR Frame (640x480)               Filtered ToF Range (mm)                      │
│         │                                    │                                   │
│         ├──► [ToF Emergency Check <= 0.8m] ──┼──► Immediate Audio Warning         │
│         │                                    │                                   │
│         ▼                                    │                                   │
│   YOLOv8n ONNX Detector                      │                                   │
│   (Letterbox, RGB, NMS)                      │                                   │
│         │                                    │                                   │
│         ▼                                    ▼                                   │
│   Bounding Boxes ────────────────► [Spatial Depth Fuser]                         │
│                                              │                                   │
│                                              ▼                                   │
│                                 Fused 3D Hazards [X, Y, Z]                        │
│                                              │                                   │
│                                              ▼                                   │
│                                 [Binaural Spatial Synthesizer]                   │
│                                 (Woodworth ITD / ILD Cues)                       │
│                                              │                                   │
│                                              ▼                                   │
│                                 ALSA Non-Blocking Audio Sink                     │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │ (Non-blocking queue submission)
┌──────────────────────────────────────▼───────────────────────────────────────────┐
│                           COGNITIVE WORKER THREAD POOL                           │
│                                                                                  │
│   Tactile Button Events (GPIO 32: Narrate | GPIO 33: Currency | GPIO 25: Text)   │
│                                       │                                          │
│                                       ▼                                          │
│                         [Edge Privacy Anonymizer]                                │
│                         (Haar Cascade + Bounding Box Gaussian Redaction)         │
│                                       │                                          │
│                                       ▼                                          │
│                         [Online Vision Service]                                  │
│                         (Gemini 3.1 Flash-Lite / GPT-4o-mini via HTTPS)          │
│                                       │                                          │
│                                       ▼                                          │
│                         [Strict JSON Schema Validator]                           │
│                         (Movement Filter & Spoken Lookup Translation)            │
│                                       │                                          │
│                                       ▼                                          │
│                         [Piper Neural Speech Synthesizer]                        │
│                         (Local ONNX TTS Playback)                                │
└──────────────────────────────────────────────────────────────────────────────────┘
```

The fast perception loop executes on the main thread, completing a full cycle in $<35\text{ ms}$ without network blocking. Cognitive queries are offloaded to an asynchronous, bounded worker thread (`CognitiveWorker`) to prevent queue saturation.

---

## 2. Safety & Fail-Closed Readiness Contract

Every perception cycle evaluates a multi-point hardware and software readiness check (`safety/readiness.py`). The state machine yields one of four readiness levels:

- **`AUTHORITATIVE`**: All physical sensors active, YOLO weights validated, geometry calibration confirmed, and fresh physical ToF range present. The system is permitted to clear paths and synthesize full navigation audio.
- **`DEGRADED`**: Non-critical issues present (e.g. unverified audio profile or minor dropouts). System synthesizes warnings but flags reduced confidence.
- **`NOT_READY`**: One or more critical requirements failed.
- **`SIMULATION`**: Explicit synthetic camera or distance simulator in use; production navigation is strictly disarmed.

### Critical Failure Rules
When any of the following conditions occur:
1. Physical camera is disconnected or returning empty frames (`CAMERA_UNAVAILABLE`).
2. Hardware ToF distance sensor is missing, unpowered, or has not produced a valid reading within $100\text{ ms}$ (`RANGE_NO_VALID_SAMPLE`).
3. YOLO ONNX model file is missing or fails tensor dimension verification (`MODEL_CONTRACT_INVALID`).
4. Camera-to-ToF baseline and field-of-view calibration are unconfirmed (`CALIBRATION_UNVERIFIED`).

The pipeline enforces:
- `fast_res["success"] = False`
- `fast_res["path_state"] = "unknown"` (Drishti **never** outputs `"path clear"` without authoritative physical sensors).
- Inferred-distance audio pings (from 2D bounding box priors) are **suppressed**.
- System emits localized alert: `"Navigation not ready"`.

---

## 3. Computer Vision & 3D Spatial Fusion

### 3.1 YOLOv8n ONNX Input Pipeline
1. **Letterboxing**: Incoming $640 \times 480$ BGR images are padded symmetrically to $640 \times 640$ while maintaining aspect ratio:
   $$\text{scale} = \min\left(\frac{640}{W}, \frac{640}{H}\right), \quad \text{pad}_x = \frac{640 - W \cdot \text{scale}}{2}, \quad \text{pad}_y = \frac{640 - H \cdot \text{scale}}{2}$$
2. **Channel Reordering & Normalization**: Converted from BGR to RGB, normalized to $[0.0, 1.0]$, and permuted to NCHW format:
   $$X \in \mathbb{R}^{1 \times 3 \times 640 \times 640}, \quad X_{c, y, x} = \frac{\text{RGB}_{y, x, c}}{255.0}$$
3. **Inference & NMS**: Post-processed using Non-Maximum Suppression (IoU threshold $0.45$, confidence threshold $0.35$).
4. **Adaptive Denormalization**: Coordinates are dynamically inspected; if outputs are normalized ($x_{\max} \le 1.05$), coordinates are scaled by target dimensions before inverse-letterboxing to native camera coordinates $[x_1, y_1, x_2, y_2] \in [0, W] \times [0, H]$.

### 3.2 Optical Geometry & Camera Intrinsics
The horizontal focal length $f_x$ (in pixels) is derived from the verified horizontal field of view $\theta_h$:
$$f_x = \frac{W}{2 \cdot \tan\left(\frac{\theta_h}{2}\right)}$$
For a $640\text{ px}$ frame with an $80^\circ$ FOV:
$$f_x = \frac{640}{2 \cdot \tan(40^\circ)} \approx 381.37\text{ px}$$
The optical principal point is assumed at the geometric center $(c_x, c_y) = \left(\frac{W}{2}, \frac{H}{2}\right) = (320, 240)$.

### 3.3 Spatial Fusion & Metric Coordinate Derivation
For each detected obstacle:
1. **Azimuth Angle ($\theta_{az}$)**:
   $$\theta_{az} = \arctan\left(\frac{x_{center} - c_x}{f_x}\right)$$
2. **Elevation Angle ($\phi_{el}$)**:
   $$\phi_{el} = \arctan\left(\frac{c_y - y_{center}}{f_y}\right)$$
3. **Distance Allocation ($Z$)**:
   - If the bounding box intersects the central ToF beam cone ($\pm 12.5^\circ$ FOV) and a fresh physical reading $d_{tof}$ exists:
     $$Z = d_{tof} + \Delta z_{offset}$$
     The distance is tagged as `source: "tof"`.
   - If the obstacle lies outside the ToF cone, distance is estimated from geometric priors based on known physical heights $H_{real}$ (e.g., person $1.7\text{ m}$, chair $0.8\text{ m}$):
     $$Z_{prior} = \frac{f_y \cdot H_{real}}{h_{bbox\_px}}$$
     The distance is tagged as `source: "prior"`.
4. **Metric 3D Coordinates $[X, Y, Z]$**:
   $$X = Z \cdot \tan(\theta_{az}), \quad Y = Z \cdot \tan(\phi_{el}), \quad Z = Z$$

---

## 4. Binaural Spatial Audio Synthesis

Spatial sound cues are synthesized in real-time as dual-channel 16-bit PCM waveforms at $44.1\text{ kHz}$ (`perception/spatial_audio.py`):

### 4.1 Interaural Time Difference (ITD)
ITD is modeled using Woodworth’s spherical head equation:
$$\text{ITD} = \begin{cases} 
\frac{r_{head}}{c} (\theta + \sin\theta) & \text{for } \theta \ge 0 \\
\frac{r_{head}}{c} (\theta - \sin|\theta|) & \text{for } \theta < 0 
\end{cases}$$
where average head radius $r_{head} = 0.0875\text{ m}$, speed of sound $c = 343\text{ m/s}$, yielding maximum lateral delay $|\text{ITD}_{\max}| \approx 650\ \mu\text{s}$. Sub-sample fractional delays are applied via linear interpolation.

### 4.2 Interaural Level Difference (ILD)
High-frequency shadowing by the head is modeled as:
$$\text{ILD}(\text{dB}) = 12.0 \cdot \sin(\theta_{az})$$
The contralateral ear signal is attenuated accordingly:
$$A_{ipsi} = 1.0, \quad A_{contra} = 10^{-\frac{|\text{ILD}|}{20}}$$

### 4.3 Sonification Mapping
- **Carrier Frequency**: Inverse function of distance $Z$:
  $$f_{pitch}(Z) = f_{\min} + (f_{\max} - f_{\min}) \cdot \left(1.0 - \frac{Z}{Z_{\max}}\right)$$
  Ranging from $400\text{ Hz}$ at $3.0\text{ m}$ to $950\text{ Hz}$ at $0.5\text{ m}$.
- **Repetition Rate**: Escalates from $1.5\text{ Hz}$ (distant) up to $8.0\text{ Hz}$ (critical proximity $<0.8\text{ m}$).
- **Proximity Preemption**: When physical ToF detects an obstacle $\le 0.8\text{ m}$, an immediate dual-ear emergency warning pulse ($880\text{ Hz}$) is synthesized immediately, bypassing detector delay.

---

## 5. Edge Privacy Redaction Engine

Before any image is dispatched to cloud vision APIs, `FaceAnonymizer` (`safety/anonymizer.py`) executes client-side privacy filtering:

### 5.1 2D Gaussian Kernel Derivation
The 2D Gaussian filter is derived discretely from first principles:
$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
The discrete kernel ($15 \times 15$, $\sigma = 7.0$) is normalized:
$$\sum_{x=-r}^{r} \sum_{y=-r}^{r} G(x, y) = 1.0$$
Convolutions are executed across all three color channels with border reflection padding.

### 5.2 Scoped Face & Plate Redaction
To prevent destructive false positives on user reading materials, detection heuristics are strictly scoped:
1. **Real Human Faces**: Detected via OpenCV's `haarcascade_frontalface_default.xml` alongside candidate head regions extracted from the upper $32\%$ of YOLO `person` bounding boxes.
2. **Synthetic Marker Isolation**: Synthetic test markers (`B > 150, G > 180, R > 220`) are executed *only* when the frame matches synthetic camera background signatures (`frame[0, 0, 1] == 40 & frame[0, 0, 2] == 40`). Real white books, paper, walls, and currency are **never** flagged as synthetic faces.
3. **Vehicle License Plates**: OpenCV's license plate Haar cascade and lower-body geometric priors ($65\%\text{--}100\%$ vertical height) execute **only when vehicles (`car`, `bus`, `truck`, `motorcycle`) are present in the frame**. Text lines on books, newspapers, and medicine packaging are completely immune to plate blurring.

---

## 6. Cloud Visual Cognition Architecture

Cloud visual inference (`cognitive/online_vision.py`) provides contextual intelligence on-demand:

### 6.1 Provider Abstraction & Session Management
- **HTTP Transport**: Utilizes a persistent `requests.Session()` with HTTP Keep-Alive, default connection pooling, and explicit timeouts ($5.0\text{ s}$ connect, $15.0\text{ s}$ read).
- **Gemini 3.1 Flash-Lite**: Uses Google's `gemini-3.1-flash-lite-preview` with `thinkingConfig: {"thinkingBudget": 0}`. This eliminates reasoning overhead and slashes cloud inference latency from $>20\text{ s}$ down to $1.5\text{--}1.8\text{ s}$.
- **OpenAI GPT-4o-mini**: Supported as an alternate backend via standard OpenAI Chat Completions API with base64 data URLs.
- **Precedence**: Environment variables (`DRISHTI_VISION_PROVIDER`, `DRISHTI_VISION_MODEL`) take absolute precedence over YAML configuration defaults.

### 6.2 Cognitive Modalities & JSON Schemas

#### 1. Scene Narration (Button 1 / GPIO 32)
Grounded by local facts (confirmed YOLO obstacles + ToF distance).
```json
{
  "summary": "1-2 sentence spatial description",
  "key_objects": ["chair", "table"],
  "ground_surface": "flat tiled floor"
}
```
*Safety Guard*: Response is filtered through strict regex sanitizers (`MOVE_FILTER`). Any directional instruction ("turn left", "step forward", "walk straight") is stripped before speech playback. Navigational commands belong exclusively to local sensors.

#### 2. Banknote Identification (Button 2 / GPIO 33)
Specialized for Indian Rupee Mahatma Gandhi New Series:
```json
{
  "is_banknote": true,
  "denomination_inr": 100,
  "confidence": "high",
  "uncertain": false,
  "details": "Mahatma Gandhi series lavender 100 rupee note"
}
```
*Safety Guard*:
- Strict confidence requirement: `confidence == "high"` and `uncertain == false`.
- The spoken denomination is generated exclusively from the validated lookup dictionary (`VALID_INR_DENOMINATIONS = {10, 20, 50, 100, 200, 500}`); arbitrary model text strings are never spoken.
- Partial, blurred, or non-banknote items return: *"No banknote clearly detected."*

#### 3. Multilingual Text Reading (Button 3 / GPIO 25)
High-fidelity literal transcription:
```json
{
  "text": "Extracted transcription verbatim",
  "spoken_name": "Normalized pronunciation string",
  "languages_detected": ["en", "hi"]
}
```
*Formatting*: Text is normalized to Unicode NFC and announced through Piper neural TTS.

---

## 7. ESP32 Hardware Link & Serial Protocol

The ESP32 DevKit V1 communicates with the host system via USB-UART bridge at **921,600 baud, 8N1** (`hal/esp32_link.py`).

### 7.1 Protocol Frame Types (ASCII CSV, Newline Terminated)
- **Banner Frame (`I`)**: Emitted on startup or reset:
  ```text
  I,1.0.0,VL53L0X\n
  ```
- **Data Frame (`D`)**: Streamed at ~33 Hz:
  ```text
  D,<seq>,<timestamp_ms>,<dist_filtered_mm>,<dist_raw_mm>,<status>,<button_mask>\n
  ```
  - `status == 0`: Valid measurement.
  - `button_mask`: Bit 0 = Button 1 (GPIO 32), Bit 1 = Button 2 (GPIO 33), Bit 2 = Button 3 (GPIO 25).
- **Button Edge Event (`B`)**: Emitted immediately upon state transition:
  ```text
  B,<button_id>,<timestamp_ms>,<edge>\n
  ```
  - `edge`: `1` = Press, `0` = Release.
- **System Health Frame (`S`)**: Emitted at 1 Hz:
  ```text
  S,<actual_hz>,<invalid_pct>,<i2c_ok>,<link_ok>\n
  ```

### 7.2 Sensor Firmware Protections
1. **Measurement Budget**: Fixed at $33\text{ ms}$ for ST VL53L0X high-accuracy long-range profile (`PreRange=18`, `FinalRange=14`).
2. **Interrupt Auto-Recovery**: If `sensor.timeoutOccurred()`, firmware writes `0x01` to `SYSTEM_INTERRUPT_CLEAR (0x0B)` over I²C to prevent state-machine lockup.
3. **Hardware Debouncing**: Tactile switches use active-LOW configuration with `INPUT_PULLUP` and 30 ms state latching.

---

## 8. Telemetry & Remote Monitoring

Companion caregiver connectivity is managed via `TelemetryClient` (`safety/telemetry.py`):
- **Transport**: MQTT 3.1.1 over TLS (port 8883) with device-scoped topic hierarchy:
  - `drishti/device/{device_id}/heartbeat` (1 Hz cadence)
  - `drishti/device/{device_id}/events` (QoS 1 with PUBACK tracking)
  - `drishti/device/{device_id}/emergency` (Immediate SOS)
  - `drishti/device/{device_id}/status` (Retained Last-Will and Testament)
- **Offline Persistence**: If broker is unreachable, telemetry records are persisted to an atomic, bounded JSONL buffer (`var/telemetry-buffer.jsonl`). Buffered events are replayed automatically upon reconnect.
- **Privacy Boundary**: Telemetry transmits *only* structured numeric metadata, battery states, and system health codes. **Raw camera frames and transcribed text strings are never transmitted over telemetry.**
