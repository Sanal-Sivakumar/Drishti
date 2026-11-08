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

The software architecture and local VLM path are implemented, and 105 automated tests pass. A real SmolVLM2 smoke run on the development laptop produced grounded JSON in about 9.1 seconds. That latency is acceptable only for optional on-demand context, not navigation.

A fresh Git checkout is **not production-ready**: model weights are intentionally not committed, geometry and audio profiles are uncalibrated, no wearable hazard dataset/model has been validated, and no real sensor-to-ear latency distribution has been recorded. Critical audio preemption and stale-narration revalidation remain P0 work; see [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md).

## How it works, briefly

The fast loop reads the physical camera and ToF sensor, runs YOLO, fuses a range with calibrated image geometry, and emits a horizontal stereo cue. A close ToF measurement bypasses YOLO and the VLM. Button-triggered OCR/currency/narration work runs in a separate bounded worker. SmolVLM2 receives the image plus confirmed sensor facts; its strict JSON is spoken only when it agrees with those facts.

The VLM is explanatory. It never decides immediate hazard output. A single ToF ray and COCO detector also cannot establish a generally safe or traversable path.

## What is implemented

- A production readiness state (`ready`, `degraded`, `not_ready`) and a separately labeled `simulation` state.
- Fail-closed navigation: a missing camera, stale/unavailable physical range sensor, or missing/invalid YOLO ONNX model produces `path_state: unknown`; it cannot produce “path clear.” Synthetic range is never represented as physical ToF.
- YOLOv8-style ONNX preprocessing with BGR-to-RGB conversion, letterboxing, input/output contract checks, NMS, COCO class validation, and calibrated-FOV intrinsics.
- Camera/ToF fusion and binaural obstacle pings. A shared audio priority arbiter prevents routine pings from overlapping spoken cognition.
- A bounded worker queue for OCR, banknote recognition, and scene narration so those tasks do not block perception.
- Real local SmolVLM2-500M image inference for on-demand scene description. Its deterministic JSON output is checked against confirmed YOLO/ToF facts before speech; rejected output is never spoken.
- Local Haar-cascade face detection, detector-assisted person-region redaction, and Gaussian blur. This is best-effort and is not a privacy guarantee; false negatives are possible.
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
python -m pip install -e '.[dev,vlm]'
python scripts/download_models.py
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

Controls: `N` requests grounded SmolVLM2 scene narration; `C` requests banknote recognition then OCR; `Q` exits. Continuous banknote scanning is opt-in with `--continuous-currency`.

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
