# Drishti – Assistive Scene Narrator

> **An offline-first, low-latency assistive wearable architecture engineered for visually impaired individuals.**

Drishti runs locally on edge compute platforms (transitioning from a Linux development workstation to a Raspberry Pi 5 with an optional Hailo-8L AI accelerator). It provides real-time obstacle avoidance cues, Indian currency note recognition, multilingual optical character recognition (OCR), and on-demand scene narration without depending on continuous cloud connectivity.

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
                                 +-----------------+------------------+
```

---

## 2. Core Documentation Ledger

The complete engineering, theoretical, and operational documentation is maintained directly in the root repository:

| Document | Description |
| :--- | :--- |
| 📖 **[technical_details.md](technical_details.md)** | **Mathematical & Theoretical Foundations**: Pinhole camera models, optical raycasting, Time-of-Flight photon physics, INT8 affine quantization math, ITD/ILD binaural psychoacoustics, and CTC loss for Indic scripts. |
| 🛠️ **[troubleshoot.md](troubleshoot.md)** | **Diagnostic & Recovery Playbook**: Diagnostic decision tree, CSI/V4L2 camera lockups, I2C bus unwedge procedures, ALSA audio buffer underruns, and Pi 5 thermal/voltage throttling. |
| 🤝 **[TEAM_HANDOVER.md](TEAM_HANDOVER.md)** | **Contributor Standards & Handover**: Branching/commit standards, headless testing criteria, and quantitative latency budgets ($\le 80 \text{ ms}$ fast path). |

---

## 3. Hardware Specifications & Raspberry Pi 5 Pinout

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

---

## 4. Quickstart & Headless Verification

### Prerequisites
- Host: Ubuntu Linux 22.04/24.04 LTS or Raspberry Pi OS Lite 64-bit (Debian Bookworm)
- Python: 3.10 to 3.12 (or isolated virtual environment)
- Optional: NVIDIA CUDA 12+ for PC acceleration; Raspberry Pi AI Kit (Hailo-8L) for Pi 5

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/Sanal-Sivakumar/Drishti.git
cd Drishti

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Headless Verification
```bash
# Execute headless verification suite across all subcomponents
pytest tests/ -v
```

---

## 5. License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See [LICENSE](LICENSE) for the full license text.
