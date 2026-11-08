# Drishti: Team Handover & Contributor Engineering Guide

Welcome to the Drishti engineering repository. This document defines the architectural standards, development protocols, verification checkpoints, and onboarding instructions for all contributors.

---

## 1. Engineering Principles & Mission

Drishti is an offline-first assistive wearable for visually impaired individuals. When designing and maintaining code in this repository, adhere strictly to these core tenets:

1. **Safety-Critical Real-Time Performance**: Obstacle detection and spatial audio alerts must complete within $< 150 \text{ ms}$ end-to-end. Latency spikes can lead directly to physical user injury.
2. **Headless & Production-Grade**:
   - Zero blocking GUI calls (`cv2.imshow`, `cv2.waitKey` must never be called in headless runtime or test pipelines).
   - Zero hardcoded local paths or fixed test image filenames. Everything is dynamically configured via configuration files or command-line arguments.
   - Robust fallbacks: If physical sensors (Camera Module 3, VL53L1X ToF) are detached, the Hardware Abstraction Layer (HAL) must switch gracefully to synthetic streams without crashing.
3. **Ergonomic Audio Channel Use**: The human ear is a low-bandwidth, serial sensory channel. Never overwhelm the user with redundant speech. Prioritize directional spatial sound cues over verbose natural language.
4. **Offline Autonomy**: All essential navigation, OCR, currency recognition, and basic scene narration features must execute purely on local edge compute without internet access.

---

## 2. Git & Branching Conventions

To maintain strict traceability across distributed engineering teams, all branches, commits, and pull requests must follow standard domain-driven naming schemes:

### 2.1 Branch Naming Scheme
- `feature/<subsystem>-<feature-description>`
  - *Example*: `feature/perception-tof-fusion`
  - *Example*: `feature/cognitive-indic-ocr`
  - *Example*: `feature/audio-binaural-panning`
- `fix/<subsystem>-<issue-description>`
  - *Example*: `fix/hal-i2c-bus-recovery`
  - *Example*: `fix/quantization-scale-overflow`
- `perf/<subsystem>-<optimization-target>`
  - *Example*: `perf/detector-int8-neon-pipeline`

### 2.2 Commit Message Convention
Commits must follow Conventional Commits formatting:
```
<type>(<scope>): <short imperative summary>

[Optional detailed rationale, math derivation, or root cause]

[Optional issue reference]
```
- Allowed types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`.
- Scopes: `hal`, `perception`, `cognitive`, `audio`, `safety`, `telemetry`, `config`.

---

## 3. Hardware Abstraction Layer (HAL) Pattern

Every physical input or output device must implement an abstract base class with both a **Hardware Driver** and a **Synthetic Mock Driver**:

```
                 +-------------------------------+
                 | Abstract Base Class (e.g. HAL)|
                 +---------------+---------------+
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
+---------------------------------+  +---------------------------------+
|     Physical Hardware Driver    |  |     Headless Synthetic Mock     |
| (V4L2 / CSI / I2C / ALSA Sink)  |  | (Pattern Generator / FIFO Sink) |
+---------------------------------+  +---------------------------------+
```

- When running on development PCs without physical sensors, the pipeline seamlessly selects mock drivers.
- In CI/CD pipelines, tests run completely headless with zero physical hardware dependencies.

---

## 4. Verification & Quality Gates

Before merging any code into `main`, every pull request must pass three verification gates:

### Gate 1: Static Analysis & Type Checking
```bash
ruff check . tests/
mypy .
```
All configuration is centralized in `pyproject.toml`. All functions must have comprehensive type hints (`typing.Optional`, `typing.Tuple`, `dataclass`).

### Gate 2: Headless Unit & Integration Tests
```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```
All 92 unit and integration tests must pass cleanly. Minimum acceptable line coverage across core modules is $85\%$.

### Gate 3: Quantitative Latency Budget Check
Every perception and audio commit must be benchmarked against latency limits:

| Subsystem Component | Maximum Permissible Latency (Pi 5) | Maximum Permissible Latency (PC) |
| :--- | :--- | :--- |
| **Image Preprocessing & Resizing** | $\le 8 \text{ ms}$ | $\le 2 \text{ ms}$ |
| **YOLO Nano INT8 Inference** | $\le 45 \text{ ms}$ | $\le 10 \text{ ms}$ |
| **ToF Spatial Fusion & Raycast** | $\le 2 \text{ ms}$ | $\le 0.5 \text{ ms}$ |
| **Binaural Spatial Audio Generation** | $\le 5 \text{ ms}$ | $\le 1 \text{ ms}$ |
| **End-to-End Fast Obstacle Alert** | $\mathbf{\le 80 \text{ ms}}$ | $\mathbf{\le 20 \text{ ms}}$ |
| **Multilingual OCR Recognition** | $\le 650 \text{ ms}$ | $\le 180 \text{ ms}$ |
| **VLM Scene Caption (On-Demand)** | $\le 2500 \text{ ms}$ | $\le 600 \text{ ms}$ |

### Gate 4: Hardware Health Diagnostics
Before running physical hardware on target kits (Raspberry Pi 5 or development laptop), run the non-destructive diagnostic check:
```bash
python3 scripts/check_hardware_health.py
```

---

## 5. Contributor Alignment Prompt

When extending or reviewing modules in Drishti, keep this mental model active:

```
"You are a core systems engineer on the Drishti Assistive Wearable team.
Every line of code you write directly impacts the physical safety and independence
of a visually impaired individual.
Ensure code is strictly modular, non-blocking, headless-compatible, thoroughly tested,
and optimized for resource-constrained edge hardware."
```

---

## 6. Current Implementation Status & Handover Ledger

As of the current milestone, the engineering baseline stands as follows:

| Subsystem / Phase | Current State | Verification Passed |
| :--- | :--- | :--- |
| **Phase 0: HAL** | Abstract Base Classes + Synthetic Drivers + V4L2/OpenCV physical camera with process lifecycle management (`__enter__`/`__exit__`, `kill()` on timeout) + ESP32 DevKit V1 USB-serial 921600 baud deterministic link (`hal/esp32_link.py`) with sequence-gap detection and button latching + ALSA / PipeWire Bluetooth A2DP audio routing (`hal/audio_route.py`) | `tests/unit/test_hal.py`, `tests/unit/test_esp32_link.py`, `tests/unit/test_audio_route.py` (20/20 PASSED) |
| **Phase 1: Fast Perception** | YOLOv8n ONNX Object Detector with adaptive coordinate denormalization and COCO alias normalization, 3D Spatial Audio (Continuous Fractional Delay interpolation via `np.interp`, ITD + ILD + PRF), Parallax-Aware Ray-Cone Depth Fusion with Point-in-Bbox Hit Testing, BGR INR Banknote Classifier with Anisotropy / Solidity Gating (`identify_currency`), Opt-in Currency Mode | `tests/unit/test_spatial_audio.py`, `test_depth_fusion.py`, `test_currency.py`, `test_fast_pipeline.py`, `test_detector.py` (27/27 PASSED) |
| **Phase 2: Cognitive (OCR & VLM)** | Multilingual Indic OCR (Eng, Hin, Tam, Mal NFC), Multimodal Spatial Scene Narrator & ONNX VLM runner, Local Speech with Offline Piper Neural Voices (`en_US-lessac-medium`) and Async Worker Queue, Natural Pacing Fallbacks (`spd-say`, `espeak-ng`) | `tests/unit/test_ocr.py`, `tests/unit/test_vlm.py`, `tests/unit/test_speech.py` (15/15 PASSED) |
| **Phase 3: Safety, Telemetry & Tools** | Local 2D Vectorized Gaussian Face Redaction (`cv2.GaussianBlur` priority with scipy/numpy fallbacks), Mosquitto MQTT Telemetry (TLS 1.3, QoS 1, paho v2), Headless Browser MJPEG Debug Viewer (`perception/debug_viewer.py` on `:8080`), PTY Fake ESP32 Emulator (`tools/fake_esp32.py`) | `tests/unit/test_anonymizer.py`, `tests/unit/test_telemetry.py`, `tests/unit/test_debug_viewer.py`, `tests/integration/test_pipeline.py` (17/17 PASSED) |
| **Phase 4: Hardware Assembly & Deploy** | Physical ESP32 DevKit V1 + VL53L0X breadboard & soldered assembly verified live; firmware updated with 33 ms long-range timing budget, 0.1 MCPS signal rate limit, and auto-clearing ST interrupt recovery (`0x0B`); verified 32.6 Hz stream with 0.0% packet errors and zero sequence drops; optical zero-offset calibration tool (`scripts/calibrate_tof.py`); real-time ASCII distance monitor (`scripts/monitor_tof.py`); automated Pi 5 provisioning script (`scripts/setup_raspberry_pi.sh`) and systemd daemon (`config/drishti.service`) | `scripts/probe_esp32.py`, `scripts/monitor_tof.py`, `tests/unit/test_calibrate_tof.py`, `tests/unit/test_monitor_tof.py` (8/8 PASSED) |
| **Config & Tooling** | Strict schema validation (`validate_pipeline_config`, `validate_hardware_config` including `esp32_serial`), `pyproject.toml` standardized tooling, Hardware health diagnostic CLI with `--serial-port` support (`scripts/check_hardware_health.py`), ESP32 probe CLI (`scripts/probe_esp32.py`) | `tests/unit/test_config_loader.py` (4/4 PASSED) |
| **Total Automated Tests** | **92/92 Headless Tests Passing Cleanly** across unit and end-to-end integration suites | `pytest tests/ -v` (100% PASSED) |
| **Next Tasks (Phase 5)** | Hailo-8L NPU pipeline acceleration, ROS2 message bridge, and outdoor navigation field trials | Scheduled |

---

## 7. External Software Interfacing & Inter-Process Tuning Guide

When integrating Drishti with external software stacks (e.g. ROS2 nodes, mobile companion apps, or custom autonomous navigation frameworks):

### 7.1 Python API Interface
External scripts can import the core pipeline cleanly:
```python
from perception.pipeline_fast import FastPerceptionPipeline
from config.config_loader import load_pipeline_config

cfg = load_pipeline_config("config/pipeline_config.yaml")
pipeline = FastPerceptionPipeline(config=cfg)
# Feed BGR image array + ToF distance in meters
result = pipeline.process_frame(frame, current_distance_m=1.45)
# Access result.obstacles (list of 3D vectors), result.binaural_audio (float32 array)
```

### 7.2 Hardware Telemetry Serial Tap
Connect to the ESP32 CP2102 UART bridge (`/dev/ttyUSB0` or `/dev/drishti-tof`) at **921600 baud 8N1**:
* Line: `D,<seq>,<t_ms>,<mm_filtered>,<mm_raw>,<status>,<button_mask>`
* Status: `0` = valid reflection, `1` = out of range (> 2.0 m), `2` = < 30 mm, `254` = timeout.
* Buttons: Bit 0 = Button 1 (GPIO 32, Narration), Bit 1 = Button 2 (GPIO 33, Currency/OCR).

### 7.3 MQTT Telemetry Bus
External processes can subscribe to JSON events published locally:
* `drishti/telemetry/heartbeat`: periodic status, fps, primary hazard distance.
* `drishti/telemetry/events`: button clicks, recognized currency notes, extracted text.
* `drishti/telemetry/alerts`: urgent proximity collision warnings.

### 7.4 Live MJPEG Video Stream
When running with `--viewer`, the debug stream is exposed at:
* Video Stream: `http://127.0.0.1:8080/stream`
* Frame Snapshot: `http://127.0.0.1:8080/snapshot`
* Metrics JSON: `http://127.0.0.1:8080/metrics`





