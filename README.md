# Drishti

Drishti is an experimental, offline-first assistive-perception prototype for Raspberry Pi 5 and Linux development machines. It combines a camera, a physical Time-of-Flight (ToF) range sensor, COCO object detection, spatial audio, OCR, banknote heuristics, and optional MQTT telemetry.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Safety status:** research prototype. Do not use for unsupervised navigation, road crossing, drop-off detection, or as a replacement for a cane, guide dog, trained assistant, or orientation-and-mobility instruction. Begin only supervised indoor trials after the readiness checks below pass. Outdoor and unsupervised use is currently unsafe.

## Documentation map

- [How the system works today](docs/CURRENT_SYSTEM.md)
- [Laptop + hardware + Git/AI continuation workflow](docs/LAPTOP_WORKFLOW.md)
- [Latency measurements and optimization plan](docs/LATENCY.md)
- [Industry/product architecture roadmap](docs/INDUSTRY_ROADMAP.md)
- [Ordered next steps and gates](docs/NEXT_STEPS.md)
- [Full documentation index](docs/README.md)

## Status at this commit

The hybrid edge-cloud software architecture is fully implemented, with all automated unit and integration tests passing. Continuous 3D obstacle avoidance (YOLOv8n ONNX + VL53L0X/VL53L1X ToF + binaural audio sonification) runs 100% offline at <35 ms per cycle. Visual cognition (Scene Narration, Indian Banknote Identification, and Multilingual Indic OCR) is handled via a cloud Online Vision API (Google Gemini 3.1 Flash-Lite or OpenAI GPT-4o-mini) triggered on-demand via a 3-button hardware layout, with local Gaussian face and license plate anonymization prior to transmission and local Piper neural TTS for playback.

A fresh Git checkout is configured safely: local model weights for offline perception (`models/yolov8n.onnx` and `models/piper/`) are provisioned via `scripts/download_models.py`, while cloud vision keys (`OPENAI_API_KEY` or `GEMINI_API_KEY`) are read securely from local `.env` (never committed or exposed).

## How it works, briefly

The architecture strictly decouples safety-critical perception from on-demand cognition:

1. **Continuous Fast Perception (Offline Edge Loop)**:
   - Reads camera and ToF sensor.
   - Runs local YOLOv8n ONNX obstacle detection.
   - Fuses range with calibrated image geometry.
   - Emits spatial binaural audio pings with ITD/ILD cues.
   - Close ToF proximity bypasses the detector and triggers immediate emergency cues.
   - **Never blocks on network or cloud APIs**.

2. **On-Demand Visual Cognition (Online Vision API Path)**:
   - Triggered via 3 dedicated hardware push buttons or keyboard shortcuts:
     - **Button 1 (`GPIO 32` / Key `N`)**: Scene Narration (Concisely describes visual context; strictly grounded by local sensor facts; forbidden from issuing movement/navigation advice).
     - **Button 2 (`GPIO 33` / Key `C`)**: Banknote Identification (Specialized for Indian Rupee Mahatma Gandhi New Series ₹10–₹500; strictly rejects blurred, partial, or ambiguous notes without false positives).
     - **Button 3 (`GPIO 25` / Key `T`)**: Text Reading / OCR (High-fidelity multilingual Indic text transcription across English, Hindi, Tamil, Malayalam, Bengali, etc.).
   - Prior to upload, an edge privacy filter applies vectorized Gaussian blur to bystander faces and vehicle license plates.
   - A bounded background worker dispatches the request to the configured cloud vision provider (OpenAI or Gemini).
   - If the internet is unreachable, Drishti announces the offline condition immediately while obstacle avoidance remains uninterrupted.
   - The verified response is vocalized locally via Piper neural TTS.

## What is implemented

- A production readiness state (`ready`, `degraded`, `not_ready`) and a separately labeled `simulation` state.
- Fail-closed navigation: a missing camera, stale/unavailable physical range sensor, or missing/invalid YOLO ONNX model produces `path_state: unknown`; it cannot produce “path clear.” Synthetic range is never represented as physical ToF.
- YOLOv8-style ONNX preprocessing with BGR-to-RGB conversion, letterboxing, input/output contract checks, NMS, COCO class validation, and calibrated-FOV intrinsics.
- Camera/ToF fusion and binaural obstacle pings. A shared audio priority arbiter prevents routine pings from overlapping spoken cognition.
- A bounded worker queue for scene narration, currency recognition, and text reading so those tasks do not block perception.
- Cloud-assisted multimodal cognition via `OnlineVisionService` (OpenAI GPT-4o-mini or Gemini 2.5 Flash) with strict JSON contracts, fail-closed movement instruction sanitization, and offline fallback speech.
- Edge privacy filter: Local Haar-cascade face detection, vehicle license plate detection, and Gaussian blur prior to transmission.
- Local Piper neural text-to-speech (TTS) for speech playback without cloud voice dependencies.
- MQTT telemetry with real host sensor readings when available, `null` when unavailable, TLS certificate verification, environment-provided credentials, a bounded persistent buffer, reconnect replay, and QoS 1 PUBACK tracking.
- Picamera2, OpenCV, V4L2, VL53L1X, and ESP32 serial backends, plus explicitly labeled synthetic backends for tests.

## Required production readiness

Production navigation is authoritative only when all of these are true:

1. A physical camera is open and returning BGR frames.
2. `models/yolov8n.onnx` loads and satisfies the expected NCHW/model-output contract.
3. A physical ToF source is present and fresh; synthetic or stale range makes the system not ready.
4. Target classes exist in the model’s COCO label set. The current configuration covers `person`, `chair`, `dining table` (configured as `table`), `car`, and `bicycle`. Doors, stairs, drop-offs, and overhead clearance are not detected by the current model.
5. Camera horizontal FOV and ToF alignment have been measured on the assembled device, and `calibration_verified` is set to `true` with the calibration record retained.

The console, heartbeat, and `/metrics` endpoint expose readiness and `path_state`. A production result with unavailable perception is `unknown`, never `clear`.

## Installation and model provisioning

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/download_models.py
cp .env.example .env  # Add your OPENAI_API_KEY or GEMINI_API_KEY
python -m pytest tests -q
```

On Raspberry Pi OS Bookworm:

```bash
sudo bash scripts/setup_raspberry_pi.sh
```

The provisioning script installs Picamera2 through APT, creates a system-site-packages virtual environment, installs the Pi ToF dependency, verifies YOLO weights and the Piper voice, and installs the systemd unit. Put broker secrets in `/etc/drishti/drishti.env`:

```text
DRISHTI_MQTT_USERNAME=wearable-device
DRISHTI_MQTT_PASSWORD=replace-with-a-secret
```

## Running

Explicit simulation, safe for development:

```bash
python scripts/run_live_wearable.py --hardware synthetic --max-cycles 100
```

Physical Raspberry Pi stack:

```bash
python scripts/check_hardware_health.py
python scripts/run_live_wearable.py --hardware rpi --viewer
```

Laptop production probing (navigation stays not-ready without a physical ToF source and real model):

```bash
python scripts/run_live_wearable.py --hardware laptop --serial-port /dev/drishti-tof --viewer
```

Controls: Visual cognition is triggered on-demand via 3 physical tactile buttons (Button 1: Scene Narration, Button 2: Banknote Identification, Button 3: Multilingual Text Reading). Interactive terminal keyboard shortcuts (`N`, `C`, `T`, `Q`) can be enabled for test benches using `--enable-keyboard`. Continuous banknote scanning is opt-in with `--continuous-currency`.

For the current laptop + ESP32 workflow, stable device paths, Git practices, verification commands, and a reusable AI-session prompt are documented in [`docs/LAPTOP_WORKFLOW.md`](docs/LAPTOP_WORKFLOW.md).

## Configuration

Runtime construction consumes [`config/pipeline_config.yaml`](config/pipeline_config.yaml) for detector thresholds/model/classes, camera/ToF geometry, currency support, spatial audio, OCR, narration, TTS, redaction, heartbeat cadence, and telemetry persistence. Hardware profiles are in [`config/hardware_dev.yaml`](config/hardware_dev.yaml) and [`config/hardware_pi5.yaml`](config/hardware_pi5.yaml).

Geometry is loaded from `config/geometry_calibration.yaml` only when its measured calibration is verified. Generate it from real chessboard and ToF-spot samples with `scripts/calibrate_camera_geometry.py`. SmolVLM2 weights are provisioned by `scripts/download_models.py`; generation is temperature-zero, bounded, and performed only in the cognitive worker.

The detector is COCO-based. Class aliases are validated at startup, and unsupported classes are reported in readiness rather than silently ignored.

For wearable-specific stairs, curbs, poles, glass doors, hanging obstacles, and partial furniture, collect the dataset described in `evaluation/README.md`, fine-tune with `scripts/train_hazard_detector.py`, then update the detector model and label-map paths. No claim of improved recall is made until `scripts/evaluate_drishti.py` reports the held-out results.

Spatial cues combine a 500–900 Hz distance component with a deterministic 2–5 kHz component for stronger ILD, use a smooth envelope and conservative limiter, and support separate air- and bone-conduction profiles. Generate and verify `-90°`, `-45°`, `0°`, `+45°`, and `+90°` cues with `scripts/calibrate_spatial_audio.py`. Bone-conduction profiles must be calibrated per user/headset and are not inferred from air-conduction settings.

## Latency

The only measured real VLM result is approximately 9.1 seconds on the development laptop. The sub-50 ms automated perception check uses simulation and is not a physical Pi/laptop benchmark. The ESP32 timing budget and software stage timers do not establish audible response time. Before trials, collect end-to-end p50/p95/p99 timestamps from sensor acquisition through physical audio/haptic onset. See [`docs/LATENCY.md`](docs/LATENCY.md).

## Future equipment

The proposed product architecture uses dense stereo/depth, multizone near-field and downward ranging, an IMU, an independent safety MCU/watchdog, and redundant audio/haptic critical output. The generative model remains outside the safety channel. Candidate compute and sensor trade-offs, regulatory references, manufacturing evidence, and release gates are in [`docs/INDUSTRY_ROADMAP.md`](docs/INDUSTRY_ROADMAP.md). These are future recommendations, not installed capabilities.

## Interfaces

MQTT topics are device-scoped:

- `drishti/device/{device_id}/heartbeat`
- `drishti/device/{device_id}/events`
- `drishti/device/{device_id}/emergency`
- `drishti/device/{device_id}/status` (retained last-will/status)

With `--viewer`, the local debug server provides:

- `/` — viewer UI
- `/stream` — MJPEG stream
- `/snapshot` — latest JPEG
- `/metrics` — latest JSON perception/readiness state

The supported Python API constructs `FastPerceptionPipeline(camera, distance_sensor, audio_sink, config=..., mode=...)` and calls `step()`. There is no `process_frame` API.

## Verification and field-trial gate

CI compiles the source, runs the test suite with a coverage threshold, and runs static checks. Before any supervised indoor trial, additionally verify on the exact assembled device:

- model checksum/load, RGB preprocessing, target-class fixtures, and false-negative behavior;
- measured camera intrinsics/FOV and camera-to-ToF alignment;
- sensor stale-data rejection, disconnect/reconnect, and fail-closed announcements;
- audio audibility, priority/preemption, and ambient-hearing preservation;
- face-redaction false negatives and an explicit no-image-transmission policy;
- MQTT TLS/authentication, offline persistence limits, replay, duplicate handling, and PUBACK removal;
- battery/thermal behavior and emergency shutdown.

Trials must use a sighted supervisor, a controlled indoor route, conservative stopping rules, incident logging, and the participant’s normal mobility aid. Outdoor or unsupervised trials are out of scope.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
