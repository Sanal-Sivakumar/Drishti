# Drishti – Assistive Scene Narrator

> **An offline-first, low-latency assistive wearable architecture engineered for visually impaired individuals, delivering real-time spatial awareness, multilingual reading, currency identification, and contextual scene narration on edge hardware.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI Pipeline](https://github.com/Sanal-Sivakumar/Drishti/actions/workflows/ci.yml/badge.svg)](https://github.com/Sanal-Sivakumar/Drishti/actions/workflows/ci.yml)
[![Target Platform](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red.svg)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![AI Accelerator](https://img.shields.io/badge/NPU-Hailo--8L%20(13%20TOPS)-brightgreen.svg)](https://www.raspberrypi.com/products/raspberry-pi-ai-kit/)
[![Offline Capability](https://img.shields.io/badge/Compute-100%25%20Offline-success.svg)](#1-core-philosophy--what-drishti-does)

---

## 1. Project Overview: What Drishti Does

For over 285 million visually impaired individuals worldwide, independent navigation through dynamic, unstructured environments presents significant hazards. Traditional mobility aids like white canes detect only immediate physical obstacles at ground level (within 1–1.5 meters), leaving users vulnerable to overhead obstructions, moving obstacles (pedestrians, vehicles), unrecognized currency, and unreadable signage or printed documents. Cloud-reliant smart glasses frequently fail in real-world scenarios due to cellular latency spikes, dead zones, and privacy concerns.

**Drishti** (meaning *vision* in Sanskrit) is an offline-first wearable edge-AI platform that acts as a continuous sensory surrogate. Mounted unobtrusively on a chest harness, cap, or glasses frame, Drishti pairs real-time optical sensing with Time-of-Flight (ToF) laser ranging, processing every perception stream entirely on-device (Raspberry Pi 5 with optional Hailo-8L NPU, or Ubuntu Linux workstations during development).

### Complete Feature Set

1. **Sub-100ms Real-Time Obstacle Avoidance & 3D Binaural Soundscape**:
   - Detects pedestrians, doors, stairs, chairs, vehicles, and drop-offs at 30 FPS.
   - Combines 2D visual bounding boxes with 1D/multi-zone Time-of-Flight infrared laser ranging to accurately pinpoint metric distance ($0.1 \text{ m}$ to $4.0 \text{ m}$) and 3D angle.
   - Emits non-verbal binaural acoustic cues through bone-conduction headphones using Interaural Time Difference (ITD) and Interaural Level Difference (ILD), allowing the user to naturally "hear" where obstacles are in 3D space without blocking their natural ambient hearing.

2. **Instant Indian Rupee (INR) Banknote Identification**:
   - Real-time classification of Indian currency notes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500).
   - Operates under varied lighting, folds, creases, and partial orientations, providing instant spoken denomination verification ("One hundred rupees") to prevent financial exploitation.

3. **Multilingual Indic Optical Character Recognition (OCR)**:
   - On-demand extraction and vocalization of printed text from medicine strips, grocery packaging, street boards, and documents.
   - Native support for **English**, **Hindi (हिन्दी)**, **Tamil (தமிழ்)**, and **Malayalam (മലയാളം)** with accurate Unicode NFC grapheme cluster processing.

4. **Contextual Scene Narration via On-Device Vision-Language Models (VLM)**:
   - Triggered via a tactile button push or voice keyword.
   - Compact quantized Vision-Language Models (SmolVLM / Moondream2) analyze the full visual scene and generate concise, 2-sentence navigational summaries focusing on spatial relationships, room layout, and navigational hazards (e.g., *"There is a clear path straight ahead. A closed wooden door is located 3 meters to your right."*).

5. **Local Privacy-Preserving Telemetry & Caregiver Companion App**:
   - Zero biometric leakage: all camera frames remain local.
   - In emergency situations (SOS button trigger or persistent obstacle collision warning), an on-device face-redaction filter irreversibly blurs detected human faces and license plates before transmitting an opt-in snapshot and GPS telemetry to a caregiver dashboard via Mosquitto MQTT over TLS.

---

## 2. System Architecture

Drishti divides sensory processing into a **Fast Perception Stream** (continuous, sub-100ms, non-verbal spatial cues) and an **On-Demand Cognitive Stream** (event-driven, natural language narration). This separation prevents cognitive auditory fatigue while guaranteeing rapid physical hazard notification.

```
                      +-----------------------------------+
                      |      Physical Input Devices       |
                      |  - CSI Camera Module 3 (60/30 FPS)|
                      |  - VL53L1X ToF Sensor (I2C)       |
                      |  - Tactile Push Buttons (GPIO)    |
                      |  - USB/MEMS Microphone            |
                      +-----------------+-----------------+
                                        |
       +--------------------------------+--------------------------------+
       |                                                                 |
       v [Continuous Fast Stream: 30 FPS]                                v [On-Demand Tactile Trigger]
+------------------------------------+                         +------------------------------------+
|       Fast Perception Engine       |                         |    On-Demand Cognitive Engine      |
| - Preprocessing & Letterboxing     |                         | - Scene Snapshot Freezing          |
| - YOLO Nano (INT8 TensorRT/ONNX)   |                         | - Multilingual Indic OCR           |
| - Bounding Box Raycasting          |                         |   (English, Hindi, Tamil, Mal)     |
| - Conical ToF Depth Fusion         |                         | - Banknote Classifier (INR)        |
| - 3D Obstacle Metric Vectorizer    |                         | - Visual Language Model (VLM)      |
| - Binaural Synthesizer (ITD/ILD)   |                         |   (SmolVLM / Moondream2)           |
+-----------------+------------------+                         +-----------------+------------------+
                  |                                                              |
                  | (Sub-100ms Spatial Tones)                                    | (Synthesized Speech)
                  +--------------------------------+-----------------------------+
                                                   |
                                                   v
                                 +------------------------------------+
                                 |    Unified Audio Synthesis Hub     |
                                 | - Non-blocking Audio Mixer (ALSA)  |
                                 | - Priority Preemption Arbiter      |
                                 | - Binaural 3D Spatial Sound Cues   |
                                 | - Local Neural TTS (Piper/MMS)     |
                                 +-----------------+------------------+
                                                   |
                                                   v
                                      [ Bone Conduction Headset ]

                                                   : (Opt-in Emergency Event)
                                                   v
                                 +------------------------------------+
                                 | Privacy Mask & Telemetry Broadcast |
                                 | - Face & PII Gaussian Redaction    |
                                 | - Mosquitto MQTT Client (TLS/QoS1) |
                                 | - Flutter Companion App Dashboard  |
                                 +------------------------------------+
```

---

## 3. Real-World User Workflows

```
Scenario A: Daily Walking Navigation
User Walks Forward ──► YOLO + ToF detects chair 1.5m to the left ──► Left ear receives 650Hz clicks ──► User adjusts heading right smoothly.

Scenario B: Cash Payment / Currency Verification
User holds note in front of camera ──► Presses Tactile Button 2 ──► Classifier verifies note ──► Audio: "Five hundred rupees".

Scenario C: Reading Multilingual Signboard or Prescription
User points camera at document ──► Long-presses Button 2 ──► Indic OCR scans script ──► Audio: "നീല ഡ്രഗ് സ്റ്റോർ (Neela Drug Store)".

Scenario D: Unfamiliar Environment Scene Narration
User enters a new lobby ──► Presses Tactile Button 1 ──► VLM evaluates scene snapshot ──► Audio: "You are in an open hallway. Reception desk is 4 meters ahead on your left."
```

---

## 4. Hardware Specifications & Raspberry Pi 5 Wiring

| Component | Hardware Specification | Interface / Pin | Pin Number (Pi Header) | Electrical Specs | Functional Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Wide Camera** | Raspberry Pi Camera Module 3 | MIPI CSI-2 (15-pin FFC) | Dedicated CAM port | 3.3V, Sony IMX708 sensor, 120° Wide FoV | Real-time scene capture |
| **ToF Range Sensor** | STMicroelectronics VL53L1X | I2C Bus 1 | Pins 1, 3, 5, 6 | 3.3V DC, 400 kHz Fast Mode, 940nm VCSEL | True metric range ($0.04\text{m} - 4.0\text{m}$) |
| **ToF Power (VIN)** | 3.3V DC Rail | DC Power | Pin 1 | 3.3V ($\le 20 \text{ mA}$) | Sensor logic power |
| **ToF Ground (GND)**| Ground | Power Ground | Pin 6 | 0V Reference | Common electrical ground |
| **ToF SCL** | I2C Clock | GPIO 3 / SCL | Pin 5 | 3.3V Open-drain, internal pull-up | Serial clock line |
| **ToF SDA** | I2C Data | GPIO 2 / SDA | Pin 3 | 3.3V Open-drain, internal pull-up | Serial bidirectional data |
| **Tactile Button 1**| Momentary Push Button | Digital GPIO | Pin 11 (GPIO 17) | Internal pull-up to 3.3V, active low | Trigger Scene Narration (VLM) |
| **Tactile Button 2**| Momentary Push Button | Digital GPIO | Pin 13 (GPIO 27) | Internal pull-up to 3.3V, active low | Trigger Banknote / Multilingual OCR |
| **Audio Headset** | Open-Ear Bone Conduction | USB / Bluetooth 5.0 | USB Port or BLE | Stereo 44.1/48 kHz PCM | Spatial audio & speech output |
| **AI Accelerator** | Raspberry Pi AI Kit (Hailo-8L) | M.2 PCIe Gen 2/3 | PCIe HAT+ connector | 13 TOPS INT8 compute, 2.5W typical | Sub-15ms neural model inference |

---

## 5. Repository Structure

```
Drishti/
├── .gitignore                # Comprehensive ignore rules for edge AI, models, logs
├── LICENSE                   # GNU General Public License v3.0 (GPL-3.0)
├── README.md                 # Project overview, capabilities, pinout, quickstart (this file)
├── TEAM_HANDOVER.md          # Contributor standards, latency budgets, headless testing rules
├── technical_details.md      # Beginner-to-advanced first-principles theory & math derivations
├── troubleshoot.md           # Operational failure modes, diagnostics, and permanent fixes
├── config/                   # Hardware and pipeline configuration schemas
│   ├── hardware_dev.yaml     # Workstation webcam and synthetic sensor setup
│   ├── hardware_pi5.yaml     # Raspberry Pi 5 pinout, I2C bus, CSI configs
│   └── pipeline_config.yaml  # Detection thresholds, audio frequencies, priority rules
├── hal/                      # Hardware Abstraction Layer (Drivers & Synthetic Mocks)
│   ├── camera.py             # Abstract camera (V4L2, Libcamera, Synthetic Pattern)
│   ├── distance_sensor.py    # Abstract range sensor (VL53L1X I2C, Synthetic Sim)
│   ├── trigger_input.py      # Abstract tactile button (GPIO pull-up, Synthetic Event)
│   └── audio_output.py       # Abstract audio sink (ALSA/Pulse, Headless PCM buffer)
├── perception/               # Fast Perception Subsystem
│   ├── detector.py           # Headless INT8 YOLO object detector
│   ├── depth_fusion.py       # 2D bounding box to 3D spatial ray-cone fusion
│   ├── currency.py           # MobileNetV3-based Indian currency note recognizer
│   └── spatial_audio.py      # Binaural acoustic synthesizer (ITD + ILD)
├── cognitive/                # On-Demand Semantic Subsystem
│   ├── ocr_engine.py         # Multilingual Indic OCR (Eng, Hin, Tam, Mal)
│   ├── vlm_narrator.py       # Local Vision-Language Model scene describer
│   └── speech_service.py     # Piper / MMS-TTS neural speech synthesizer
├── safety/                   # Privacy Masking & Telemetry
│   ├── anonymizer.py         # Edge-side Gaussian face & PII redaction
│   └── telemetry.py          # Mosquitto MQTT client (TLS 1.3, QoS 1)
├── pipeline.py               # Master headless pipeline orchestrator
├── requirements.txt          # Python runtime dependencies
└── tests/                    # Headless automated unit and integration tests
    ├── unit/                 # Subsystem unit tests with synthetic mocks
    └── integration/          # End-to-end headless pipeline tests
```

---

## 6. Comprehensive Documentation Index

All core documentation files reside directly in the root directory:

| Document | Purpose & Target Audience |
| :--- | :--- |
| 📖 **[technical_details.md](technical_details.md)** | **Deep-Dive Theory & Mathematics**: Written from first principles so that anyone with zero prior knowledge can master: pinhole camera projection, Brown-Conrady lens distortion, Time-of-Flight photon physics, ray-cone metric fusion, INT8 affine quantization math, Roofline compute models, binaural ITD/ILD psychoacoustics, DBNet binarization, CTC loss for Indic scripts, and Transformer VLM architectures. |
| 🛠️ **[troubleshoot.md](troubleshoot.md)** | **Diagnostic & Recovery Ledger**: Step-by-step diagnostic trees, commands, root cause analyses, and verified fixes for CSI-2 camera busy errors, I2C bus lockup and recovery, ALSA audio buffer underruns (`xrun`), ONNX runtime provider fallbacks, and Raspberry Pi 5 voltage/thermal throttling. |
| 🤝 **[TEAM_HANDOVER.md](TEAM_HANDOVER.md)** | **Contributor Guide & Standards**: Onboarding protocols, git branching and conventional commit standards, headless test execution rules, and strict quantitative latency budgets ($\le 20 \text{ ms}$ PC, $\le 80 \text{ ms}$ Pi 5). |

---

## 7. Quickstart & Headless Verification

### 1. Prerequisites
- **Operating System**: Ubuntu Linux 22.04 / 24.04 LTS (x86_64 or ARM64) or Raspberry Pi OS Lite 64-bit (Debian Bookworm).
- **Python**: Version 3.10, 3.11, or 3.12 (recommended).
- **System Packages**:
  ```bash
  sudo apt-get update && sudo apt-get install -y \
      build-essential git libasound2-dev libportaudio2 \
      i2c-tools tesseract-ocr tesseract-ocr-hin tesseract-ocr-tam tesseract-ocr-mal \
      mosquitto mosquitto-clients
  ```

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Sanal-Sivakumar/Drishti.git
cd Drishti

# Set up isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade packaging tools and install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Running Headless Verification Tests
The entire test suite runs completely headless without requiring physical cameras, I2C sensors, audio output hardware, or GUI display servers:

```bash
# Run unit and integration tests with coverage report
pytest tests/ -v
```

### 4. Running the Live Fast Perception Pipeline
You can run the end-to-end fast perception loop either with synthetic simulation or on physical hardware:

```bash
# Option A: Headless Synthetic Benchmark (Zero hardware dependencies)
python3 scripts/run_phase1_perception.py --frames 10 --benchmark-frames 50

# Option B: Real Laptop Hardware (Webcam /dev/video0 + Speakers plughw:1,0)
python3 scripts/run_phase1_perception.py --camera real --audio real --frames 10
```

### 5. Running On-Demand Multilingual OCR & VLM Scene Narration
Trigger the on-demand cognitive stream to extract Indic text, narrate the scene, and vocalize speech:

```bash
# Synthetic test mode
python3 scripts/run_phase2_cognitive.py --camera synthetic

# Real laptop webcam snapshot with audible speech synthesis
python3 scripts/run_phase2_cognitive.py --camera real --speak
### 6. Running Master Full-System Orchestrator (Phase 3)
Execute all subsystems concurrently (fast 3D perception, tactile triggers, Gaussian face redaction, and caregiver telemetry):

```bash
# Option A: Headless Simulation Mode (Zero hardware required)
python3 scripts/run_phase3_full_system.py --camera synthetic

# Option B: Live Wearable Mode (Laptop /dev/video0 + plughw:1,0 Audio Sink)
python3 scripts/run_phase3_full_system.py --camera real --audio real
```

### 7. Running Production Autonomous Wearable (Continuous Live Mode)
Runs continuously with active 3D spatial obstacle sonification, automatic banknote detection, and live interactive controls:

```bash
# On Laptop (uses webcam /dev/video0, speakers plughw:1,0, and interactive terminal controls)
python3 scripts/run_live_wearable.py --hardware laptop

# On Raspberry Pi 5 (Camera Module 3, VL53L1X ToF, GPIO 17/27 buttons, and bone conduction)
python3 scripts/run_live_wearable.py --hardware rpi
```

#### Interactive Controls in Live Terminal Mode:
- **Automatic Note Detection**: Hold any Indian Rupee banknote (₹10, ₹20, ₹50, ₹100, ₹200, ₹500) up to the camera—it announces the denomination with debouncing.
- **Type `N` + Enter**: Triggers local VLM scene narration of what lies ahead.
- **Type `C` + Enter**: Triggers manual banknote / Multilingual Indic OCR reading.
- **Type `Q` + Enter** (or `Ctrl+C`): Gracefully powers down the wearable system.

#### Raspberry Pi Autonomous Boot Deployment (systemd):
To launch Drishti automatically as an autonomous background appliance whenever the wearable battery pack powers on:

```bash
# 1. Copy unit service to systemd directory
sudo cp config/drishti.service /etc/systemd/system/

# 2. Reload daemon and enable auto-start on boot
sudo systemctl daemon-reload
sudo systemctl enable --now drishti.service

# 3. Check live service status and journal logs
systemctl status drishti.service
journalctl -u drishti.service -f
```

---

## 8. Implementation Progress & Engineering Milestones

| Phase | Milestone Description | Status | Verification Checkpoint |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Foundations & Hardware Abstraction Layer (HAL)**<br>- Abstract drivers for Camera, ToF, Push Button GPIO, and Audio Sink<br>- Deterministic synthetic mock generators for headless simulation<br>- Hardware and pipeline YAML configuration schemas | ✅ **Completed** | **Checkpoint 0**: All 4 HAL drivers verified via `test_hal.py`. |
| **Phase 1** | **Fast Perception, Depth Fusion & 3D Spatial Audio**<br>- Woodworth ITD + ILD binaural acoustic synthesizer (up to 655 $\mu$s delay)<br>- Camera-ToF ray-cone spatial depth fusion to 3D Cartesian vectors $(X, Y, Z)$<br>- Indian Rupee (₹10–₹500) banknote classifier<br>- V4L2 live webcam streaming and ALSA direct speaker output | ✅ **Completed** | **Checkpoint 1**: 1.28 ms mean PC cycle time ($> 700 \text{ FPS}$), verified via `test_spatial_audio.py`, `test_depth_fusion.py`, `test_currency.py`, `test_fast_pipeline.py`. |
| **Phase 2** | **Multilingual Indic OCR & On-Demand VLM Narration**<br>- Unicode NFC normalized OCR (English, Hindi, Tamil, Malayalam)<br>- Compact local VLM scene describer (SmolVLM / Moondream2)<br>- Local neural TTS audio vocalization (Piper / MMS-TTS / Speech Dispatcher) | ✅ **Completed** | **Checkpoint 2**: Multi-script accuracy and 2-sentence navigational speech verified via `test_ocr.py`, `test_vlm.py`, `test_speech.py`. |
| **Phase 3** | **Edge Privacy, Caregiver Telemetry & Companion App**<br>- Local Gaussian face and PII redaction filter ($G_\sigma(x, y)$ convolution)<br>- Mosquitto MQTT telemetry publisher with TLS 1.3 & QoS 1<br>- Emergency distress notification & master pipeline coordinator | ✅ **Completed** | **Checkpoint 3**: 33/33 tests passing, verified via `test_anonymizer.py`, `test_telemetry.py`, `test_pipeline.py`. |
| **Phase 4** | **Field Deployment, Wearable Packaging & Hailo-8L Acceleration**<br>- Hailo-8L M.2 HAT NPU compilation and ONNX pipeline offload<br>- 3D-printed modular chest harness / spectacles frame integration<br>- Bone-conduction headset low-profile acoustic integration | 🔄 **Scheduled** | **Checkpoint 4**: Real-world field navigation and user trial evaluation. |


---

## 9. License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for complete terms.
