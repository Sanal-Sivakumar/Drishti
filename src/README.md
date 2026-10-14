# Drishti – Assistive Scene Narrator

Drishti is an offline-first, low-latency assistive wearable architecture engineered for visually impaired individuals. It runs locally on edge compute platforms (transitioning from a Linux development workstation to a Raspberry Pi 5 with an optional Hailo-8L AI accelerator). Drishti provides real-time obstacle awareness, Indian currency note recognition, multilingual optical character recognition (OCR), and on-demand scene narration without continuous cloud dependence.

---

## 1. System Architecture Overview

```
                      +-----------------------------------+
                      |      Physical Input Devices       |
                      |  - CSI/USB Camera (60/30 FPS)     |
                      |  - VL53L1X ToF Sensor (I2C)       |
                      |  - Tactile Switches (GPIO Pulled) |
                      |  - USB/MEMS Microphone            |
                      +-----------------+-----------------+
                                        |
       +--------------------------------+--------------------------------+
       |                                                                 |
       v [Continuous Stream]                                             v [On-Demand Trigger]
+------------------------------------+                         +------------------------------------+
|       Fast Perception Engine       |                         |    On-Demand Cognitive Engine      |
| - Preprocessing & Normalization    |                         | - Scene Snapshot Freezing          |
| - YOLO Nano (INT8 TensorRT/ONNX)   |                         | - Multilingual Indic OCR           |
| - Bounding Box Spatial Raycasting  |                         |   (Eng, Hin, Tam, Mal)             |
| - Conical ToF Depth Fusion         |                         | - Banknote Classifier (INR)        |
| - Directional Audio Panning (ITD)  |                         | - Visual Language Model (VLM)      |
| - Obstacle Vector Synthesizer      |                         |   (SmolVLM / Moondream2)           |
+-----------------+------------------+                         +-----------------+------------------+
                  |                                                              |
                  | (Sub-100ms Spatial Cues)                                     | (Speech Payloads)
                  +--------------------------------+-----------------------------+
                                                   |
                                                   v
                                 +------------------------------------+
                                 |    Unified Audio Synthesis Hub     |
                                 | - Non-blocking Audio Mixer (ALSA)  |
                                 | - Priority Preemption Scheduler    |
                                 | - Binaural Spatial Sound Cues      |
                                 | - Local Neural TTS (Piper/MMS)     |
                                 +-----------------+------------------+
                                                   |
                                                   v
                                      [ Bone Conduction Headset ]

                                                   : (Opt-in Event)
                                                   v
                                 +------------------------------------+
                                 | Privacy Mask & Telemetry Broadcast |
                                 | - Face & PII Redaction Engine      |
                                 | - Mosquitto MQTT Client (TLS/QoS1) |
                                 | - Flutter Companion Dashboard      |
                                 +------------------------------------+
```

---

## 2. Key Modules & Directory Layout

```
Drishti/
├── config/                     # Hardware and model runtime configuration YAMLs
│   ├── hardware_pi5.yaml       # Raspberry Pi 5 pinout, I2C bus, camera CSI configs
│   ├── hardware_dev.yaml       # Local workstation camera & simulated sensors
│   └── pipeline_config.yaml    # Confidence thresholds, audio frequencies, priority rules
├── src/                        # Core engineering source and continuous documentation
│   ├── README.md               # Architecture, module reference, and quickstart (this file)
│   ├── technical_details.md    # First-principles theory, mathematical formulations, physics
│   ├── troubleshoot.md         # Diagnostic ledger, root cause analyses, recovery playbooks
│   ├── TEAM_HANDOVER.md        # Team onboarding conventions, testing criteria, contributor guide
│   ├── hal/                    # Hardware Abstraction Layer
│   │   ├── camera.py           # V4L2, OpenCV, and Libcamera abstraction
│   │   ├── distance_sensor.py  # VL53L1X I2C driver and mock sensor simulator
│   │   ├── trigger_input.py    # GPIO pull-up monitor and headless event emitter
│   │   └── audio_output.py     # Headless recorder and ALSA/Pulse audio sink
│   ├── perception/             # Fast perception engine
│   │   ├── detector.py         # Generalized ONNX/TFLite headless object detector
│   │   ├── depth_fusion.py     # Camera-ToF ray-plane intersection and coordinate mapping
│   │   ├── currency.py         # MobileNetV3-based Indian currency note recognizer
│   │   └── spatial_audio.py    # Binaural acoustic synthesizer (ITD + ILD)
│   ├── cognitive/              # On-demand slow perception engine
│   │   ├── ocr_engine.py       # Tesseract/PaddleOCR multilingual Indic text reader
│   │   ├── vlm_narrator.py     # Vision-Language scene describer (SmolVLM / Moondream2)
│   │   └── speech_service.py   # Piper / MMS-TTS local neural voice generator
│   ├── safety/                 # Edge privacy, face blurring, and MQTT telemetry
│   │   ├── anonymizer.py       # Face blurring & license plate redaction
│   │   └── telemetry.py        # Mosquitto MQTT publisher (emergency alerts, GPS)
│   └── pipeline.py             # Headless master pipeline orchestrator
├── tests/                      # Automated unit, integration, and quantitative regression tests
│   ├── unit/                   # Algorithmic tests without physical hardware
│   └── integration/            # Multi-component headless pipeline tests
├── Dockerfile                  # Reproducible development and deployment container
└── requirements.txt            # Python dependencies locked with specific compatible versions
```

---

## 3. Quickstart & Verification

### Prerequisites
- Host: Ubuntu Linux 22.04/24.04 LTS or Raspberry Pi OS Lite 64-bit (Debian Bookworm)
- Python: 3.10 to 3.12 (or isolated Docker container)
- Optional: NVIDIA CUDA 12+ for PC acceleration; Raspberry Pi AI Kit (Hailo-8L) for Pi 5

### Native Environment Setup
```bash
# Clone the repository
git clone <repo-url> Drishti
cd Drishti

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Headless Verification Run
```bash
# Execute headless verification suite across all subcomponents
pytest tests/ -v
```

---

## 4. Hardware Specifications & Raspberry Pi 5 Pinout

| Peripheral | Interface / Pin | Pin Number (Pi Header) | Electrical Specification | Functional Role |
| :--- | :--- | :--- | :--- | :--- |
| **Camera Module 3** | CSI-2 (15-pin FFC) | Dedicated CAM port | MIPI CSI-2 2-lane, Sony IMX708 | Real-time scene capture |
| **VL53L1X ToF VIN** | 3.3V Power | Pin 1 | 3.3V DC (max 20mA active) | Sensor power rail |
| **VL53L1X ToF GND** | Ground | Pin 6 | 0V Reference | Common ground |
| **VL53L1X ToF SCL** | I2C1 SCL | Pin 5 (GPIO 3) | 3.3V I2C, 400 kHz Fast Mode | Clock line for distance queries |
| **VL53L1X ToF SDA** | I2C1 SDA | Pin 3 (GPIO 2) | 3.3V I2C, 400 kHz Fast Mode | Data line for distance queries |
| **Tactile Button 1** | Digital Input | Pin 11 (GPIO 17) | Internal pull-up to 3.3V, active low | Trigger scene narration |
| **Tactile Button 2** | Digital Input | Pin 13 (GPIO 27) | Internal pull-up to 3.3V, active low | Trigger currency / OCR reading |
| **Audio Headset** | USB / Bluetooth | USB Port / BT 5.0 | Open-ear or bone-conduction | Spatial audio and speech output |

