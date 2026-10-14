# Drishti – Assistive Scene Narrator

> **An offline-first, low-latency assistive wearable architecture engineered for visually impaired individuals.**

Drishti provides real-time obstacle avoidance cues, Indian currency note recognition, multilingual optical character recognition (OCR), and on-demand scene narration without depending on continuous cloud connectivity. It runs locally on edge compute platforms (transitioning from Linux workstations to Raspberry Pi 5 with an optional Hailo-8L AI accelerator).

---

## Architecture At A Glance

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
```

---

## Core Documentation in `src/`

The primary engineering documentation is actively maintained in the `src/` directory:

- 📖 **[Project Overview & Quickstart](src/README.md)**: Hardware specifications, Raspberry Pi 5 pinouts, quickstart instructions.
- 📐 **[Technical Theory & Math Foundations](src/technical_details.md)**: First-principles physics, optical raycasting math, ToF photon transit equations, INT8 affine quantization, and binaural ITD/ILD psychoacoustics.
- 🛠️ **[Troubleshooting Guide & Diagnostics](src/troubleshoot.md)**: Comprehensive diagnostic ledger, hardware failure modes (CSI, I2C bus lockup, ALSA buffer underruns, power throttling), and verified fixes.
- 🤝 **[Team Handover & Standards](src/TEAM_HANDOVER.md)**: Contributor onboarding, branch/commit naming conventions, headless testing criteria, and latency budgets.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

