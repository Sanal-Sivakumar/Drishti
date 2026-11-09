# Drishti: Intelligent Assistive Wearable Perception

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](pyproject.toml)
[![Tests: Passing](https://img.shields.io/badge/Tests-109%20Passing-brightgreen.svg)](tests/)

Drishti is an offline-first assistive wearable perception platform designed for visually impaired users. It executes real-time 3D spatial obstacle avoidance locally at the edge (<35 ms latency) while providing on-demand visual cognition—scene understanding, Indian banknote identification, and multilingual text reading—via low-latency cloud vision models with edge privacy redaction.

> **Safety Notice:** Research prototype. Drishti is not a certified medical device or mobility aid. It must not replace a primary mobility tool (white cane, guide dog, or orientation-and-mobility instruction). Autonomous navigation, road crossings, stair navigation, and outdoor unguided use are strictly out of scope.

---

## Architectural Overview

Drishti decouples time-critical hazard avoidance from asynchronous cognitive queries through a dual-loop design:

```text
┌────────────────────────────────────────────────────────────────────────────────┐
│                       FAST PERCEPTION STREAM (OFFLINE EDGE)                     │
│                                                                                │
│  [Camera: 640x480] ──► [YOLOv8n ONNX] ──► [Camera/ToF Calibration Fusion]       │
│                                                          │                     │
│  [ESP32: VL53L0X]  ──► [33ms Distance] ──────────────────┘                     │
│                                                          │                     │
│  [Fail-Closed Gate] ◄── [SystemReadiness]                ▼                     │
│                                              [Binaural Spatial Audio (ITD/ILD)]│
│                                              30 FPS Sonar Pings (<35 ms loop)  │
└────────────────────────────────────────────────────────────────────────────────┘
                                        │ (Independent Execution)
┌────────────────────────────────────────────────────────────────────────────────┐
│                   VISUAL COGNITION STREAM (ON-DEMAND MULTIMODAL)                │
│                                                                                │
│  Tactile Inputs (ESP32):                                                       │
│    • Button 1 (GPIO 32): Scene Narration                                       │
│    • Button 2 (GPIO 33): Banknote Identification (INR ₹10–₹500)                │
│    • Button 3 (GPIO 25): Multilingual Text Reading                             │
│                                 │                                              │
│                                 ▼                                              │
│                   [Edge Privacy Anonymizer]                                    │
│                   (Vectorized Gaussian Face & Vehicle Plate Redaction)         │
│                                 │                                              │
│                                 ▼                                              │
│                   [Online Vision Service]                                      │
│                   (Gemini 3.1 Flash-Lite / GPT-4o-mini | ~1.5s Response)       │
│                                 │                                              │
│                                 ▼                                              │
│                   [Local Piper Neural Text-to-Speech]                          │
└────────────────────────────────────────────────────────────────────────────────┘
```

1. **Continuous Fast Perception Loop (100% Offline, Edge)**:
   - Evaluates a strict **fail-closed readiness contract** every cycle.
   - Captures synchronized frames from the camera and distance readings from the hardware Time-of-Flight (VL53L0X) sensor.
   - Runs local YOLOv8n ONNX inference for obstacle classification (`person`, `chair`, `table`, `car`, `bicycle`).
   - Fuses detected bounding boxes with the central ToF cone and calibrated optical geometry.
   - Synthesizes real-time binaural spatial audio pings with interaural time (ITD) and level (ILD) differences.
   - Detects close proximity (<0.8 m) directly from ToF hardware to emit immediate hazard warnings without waiting for detector inference.
   - **Zero network dependencies.** Runs deterministically at 30 FPS on host CPU.

2. **On-Demand Visual Cognition Loop (Cloud Multimodal, Asynchronous)**:
   - Triggered exclusively by explicit physical tactile button inputs:
     - **Button 1 (`GPIO 32`)**: **Scene Narration** — Synthesizes concise situational descriptions grounded strictly by local sensor facts. Model output is automatically filtered against movement or navigational commands.
     - **Button 2 (`GPIO 33`)**: **Banknote Identification** — Specialized verification for Indian Rupee Mahatma Gandhi New Series (₹10, ₹20, ₹50, ₹100, ₹200, ₹500). Enforces strict confidence thresholds and spoken lookup verification to eliminate false positives.
     - **Button 3 (`GPIO 25`)**: **Multilingual Text Reading** — Verbatim transcription across Latin and Indic scripts (Hindi, Tamil, Malayalam, Bengali, etc.).
   - Prior to network transmission, the **Edge Privacy Filter** detects and Gaussian-blurs bystander faces and vehicle license plates. Text and currency notes are preserved.
   - Dispatches requests asynchronously to Google Gemini 3.1 Flash-Lite (configured with zero thinking budget for ~1.5 s latency) or OpenAI GPT-4o-mini.
   - Synthesizes vocal responses locally via offline Piper Neural Text-to-Speech (TTS).

---

## Hardware Specifications & Pinout

The standard development prototype interfaces an **ESP32 DevKit V1** over USB serial (921600 baud) with an **I²C VL53L0X ToF rangefinder** and three tactile switches:

| ESP32 DevKit V1 Pin | Peripheral Connection | Function | Signal Characteristics |
| :--- | :--- | :--- | :--- |
| **3V3** | VL53L0X `VIN` & `XSHUT` | Power Rail & Sensor Enable | 3.3V DC (Do not connect to 5V) |
| **GND** | VL53L0X `GND` & Button Rails | Ground Reference | Common ground |
| **GPIO 21 (D21)** | VL53L0X `SDA` | I²C Data Line | 400 kHz Fast-Mode I²C |
| **GPIO 22 (D22)** | VL53L0X `SCL` | I²C Clock Line | 400 kHz Fast-Mode I²C |
| **GPIO 32 (D32)** | Button 1 (to GND) | **Scene Narration** | Active-LOW, internal pull-up, 30 ms debounced |
| **GPIO 33 (D33)** | Button 2 (to GND) | **Banknote Identification** | Active-LOW, internal pull-up, 30 ms debounced |
| **GPIO 25 (D25)** | Button 3 (to GND) | **Multilingual Text Reading**| Active-LOW, internal pull-up, 30 ms debounced |
| **Micro-USB** | Workstation / Raspberry Pi | Data & Power | USB Serial CDC at 921600 baud (`/dev/ttyUSB0`) |

Detailed schematics and breadboard wiring guides are in [`firmware/drishti_tof/WIRING.md`](firmware/drishti_tof/WIRING.md).

---

## Production Readiness Contract

Drishti enforces an uncompromising **fail-closed** safety architecture. A navigational cycle is authoritative *only* when all conditions are satisfied:

1. **Physical Camera Active**: Camera device open, yielding valid 640×480 BGR frames without frame drops.
2. **Validated ONNX Model**: `models/yolov8n.onnx` loaded, verified against expected NCHW tensor dimensions, and operating via ONNX Runtime.
3. **Fresh Physical ToF Range**: Range data must originate from the physical hardware sensor and be $<100\text{ ms}$ old. If the sensor is disconnected, unpowered, or reporting stale data, readiness transitions to `not_ready`, `path_state` is forced to `unknown`, and navigation audio is suppressed.
4. **Verified Calibration**: Camera focal length and optical-to-ToF baseline offsets must be measured and confirmed (`calibration_verified: true` in configuration).

If any critical requirement fails, the system outputs:
```text
Cycle 12    | NAVIGATION NOT READY | RANGE_NO_VALID_SAMPLE, CALIBRATION_UNVERIFIED
```
Drishti **never** outputs `"path clear"` when physical sensors are missing or degraded.

---

## Quickstart & Installation

### 1. Prerequisites
- Linux OS (Ubuntu 22.04+, Debian Bookworm, or Raspberry Pi OS 64-bit).
- Python 3.10 to 3.14.
- Git, ALSA audio utilities (`libasound2-dev`, `aplay`).

### 2. Environment Setup & Model Provisioning
```bash
# Clone repository
git clone https://github.com/Sanal-Sivakumar/Drishti.git
cd Drishti

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package dependencies
pip install --upgrade pip
pip install -e '.[dev]'

# Provision offline neural weights (YOLOv8n ONNX & Piper neural voice)
python3 scripts/download_models.py

# Configure cloud vision credentials
cp .env.example .env
# Edit .env and supply your GEMINI_API_KEY or OPENAI_API_KEY
```

### 3. Verification Suite
Execute the automated test suite covering safety readiness, depth fusion, anonymization, spatial audio, and cloud vision contracts:
```bash
python3 -m pytest tests/ -q
# Output: 109 passed in ~8s
```

---

## Execution Modes

### Development & Workstation (Webcam + ESP32)
Launch live perception with automatic hardware probing and the local browser debug viewer:
```bash
python3 scripts/run_live_wearable.py --viewer
```
- Open **`http://127.0.0.1:8080`** to view real-time annotated camera streams, ToF tracking spots, and system diagnostics.
- To enable keyboard controls alongside hardware buttons, append `--enable-keyboard` (`N` = Narrate, `C` = Currency, `T` = Text, `Q` = Quit).

### Raspberry Pi 5 Target Deployment
For native Raspberry Pi deployments using Camera Module 3 (libcamera) and hardware audio:
```bash
sudo bash scripts/setup_raspberry_pi.sh
python3 scripts/check_hardware_health.py
python3 scripts/run_live_wearable.py --hardware rpi --viewer
```

### Headless Deterministic Simulation
For CI pipelines or software development without physical hardware attached:
```bash
python3 scripts/run_live_wearable.py --hardware synthetic --max-cycles 100
```

---

## Documentation Index

- **[Technical Architecture & Mathematics](technical_details.md)**: Mathematical models for camera-ToF fusion, Woodworth ITD spatialization, and JSON contract specifications.
- **[Operational Troubleshooting Ledger](troubleshoot.md)**: Exhaustive root-cause guides for hardware timeouts, false-positive blur resolution, USB-serial debugging, and ALSA xruns.
- **[Hardware Wiring & Schematics](firmware/drishti_tof/WIRING.md)**: Breadboard wiring schematics, pinout guides, and firmware flashing steps for ESP32.
- **[Workstation & Laptop Workflow](docs/LAPTOP_WORKFLOW.md)**: Daily developer workflow, device path stabilization, and AI pair-programming procedures.
- **[Latency Profile & Benchmarks](docs/LATENCY.md)**: Monotonic timing budgets, software stage breakdowns, and performance optimization roadmaps.
- **[Engineering Team Handover](TEAM_HANDOVER.md)**: Deployment boundaries, compliance considerations, and supervised indoor testing protocols.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
