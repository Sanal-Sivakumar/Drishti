# Drishti engineering handover

Last reviewed: 2026-11-08. Read [`docs/README.md`](docs/README.md) before making architectural changes.

## Current status

The repository is a research prototype, not a certified mobility aid. Production mode fails closed: if the camera, physical range source, validated ONNX detector, geometry calibration, or audio-profile calibration is unavailable, navigation is `not_ready` and `path_state` is `unknown`. Simulation is explicitly labeled and must never be used as hardware evidence.

Do not begin outdoor, unsupervised, or target-user trials. Complete the P0/P1 gates in [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md), then use a separately approved supervised indoor protocol.

The offline fast perception stream requires local model weights (`models/yolov8n.onnx` and `models/piper/`), provisioned via `python scripts/download_models.py`. Cloud vision APIs (OpenAI or Gemini) require API keys configured in `.env` (copied from `.env.example`).

## Runtime architecture

- `perception/pipeline_fast.py`: the time-critical camera → detector → ToF fusion → spatial cue loop.
- `cognitive/online_vision.py`: provider-agnostic cloud multimodal service for Scene Narration, Indian Banknote Identification, and Multilingual Indic OCR.
- `cognitive/worker.py`: bounded background queue for narration, currency, and text reading.
- `safety/anonymizer.py`: edge privacy engine performing 2D Gaussian blur of bystander faces and vehicle license plates before cloud upload.
- `hal/audio_priority.py`: shared arbitration state for routine pings and speech.
- `safety/readiness.py`: authoritative readiness contract.
- `pipeline.py`: constructs every subsystem from the loaded configuration and publishes real telemetry.
- `scripts/run_live_wearable.py`: production/simulation entry point. It must not introduce synthetic hardware into production mode.

`OnlineVisionService` performs cloud multimodal inference for on-demand cognitive queries. It operates under strict JSON contracts and safety rules: it receives temporally confirmed detections and physical-ToF facts in its prompt, but is strictly sanitized against emitting directional or movement advice. Cloud vision output never controls immediate cues or navigation.

### Hardware Controls

1. **Button 1 (`GPIO 32` / Key `N`)**: Scene Narration.
2. **Button 2 (`GPIO 33` / Key `C`)**: Indian Banknote Identification (Mahatma Gandhi New Series ₹10–₹500).
3. **Button 3 (`GPIO 25` / Key `T`)**: Multilingual Text Reading / Indic OCR.

## Known capability boundaries

- The COCO detector covers obstacle detection (`person`, `chair`, `dining table`, `car`, `bicycle`). It does not detect doors, stairs, drop-offs, potholes, glass, or overhead clearance.
- A single ToF ray does not provide dense depth and must be aligned/calibrated to the camera.
- Haar/person/vehicle-assisted redaction reduces privacy risks; it is not a guarantee against all edge cases.
- GPS is present only when supplied by a real integration (`DRISHTI_GPS_LAT/LON` currently provides the adapter boundary). Missing readings remain `null`.
- MQTT QoS 1 can deliver duplicates. Consumers must deduplicate by event semantics; the client retains records until PUBACK.

## Provisioning

Run `sudo bash scripts/setup_raspberry_pi.sh`. It installs Picamera2 from Raspberry Pi OS packages, creates the virtual environment with system packages visible, installs `.[pi]`, downloads/verifies the provisioned models, and installs the service.

Broker credentials and API keys belong in `.env` or `/etc/drishti/drishti.env`; do not commit them.

## Verification commands

```bash
python -m compileall -q cognitive config hal perception safety scripts pipeline.py
python -m pytest tests -q
python scripts/download_models.py
python scripts/check_hardware_health.py --hardware rpi
```

Use the local viewer endpoints `/`, `/stream`, `/snapshot`, and `/metrics`. MQTT topics are `drishti/device/{id}/{heartbeat,events,emergency,status}`.

For laptop hardware setup, Git workflow, privacy rules, and an AI-session starter prompt, use [`docs/LAPTOP_WORKFLOW.md`](docs/LAPTOP_WORKFLOW.md).

## Before supervised indoor trials

Record the exact camera module, resolution, measured FOV/intrinsics, ToF offset/alignment, model checksum, test route, observer, weather/lighting, software commit, battery state, and every readiness transition. Deliberately disconnect each required sensor and confirm that neither the UI nor audio says “clear.” Test stale ToF data, model corruption, broker outage/replay, face-redaction misses, and audio preemption.

Stop the trial on any unknown readiness, missed hazard, misleading range, audio conflict, thermal/battery alarm, or participant discomfort. Retain the participant’s normal mobility aid throughout.
