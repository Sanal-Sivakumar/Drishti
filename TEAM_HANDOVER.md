# Drishti engineering handover

Last reviewed: 2026-11-08. Read [`docs/README.md`](docs/README.md) before making architectural changes.

## Current status

The repository is a research prototype, not a certified mobility aid. Production mode fails closed: if the camera, physical range source, validated ONNX detector, geometry calibration, or audio-profile calibration is unavailable, navigation is `not_ready` and `path_state` is `unknown`. Simulation is explicitly labeled and must never be used as hardware evidence.

Do not begin outdoor, unsupervised, or target-user trials. Complete the P0/P1 gates in [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md), then use a separately approved supervised indoor protocol.

The current development laptop has locally downloaded SmolVLM2 weights, but Git intentionally excludes model binaries. A fresh clone must run `python scripts/download_models.py`. At the time of this handover the Git tree itself does not contain YOLO or Piper weights.

## Runtime architecture

- `perception/pipeline_fast.py`: the time-critical camera → detector → ToF fusion → spatial cue loop.
- `cognitive/worker.py`: bounded background queue for narration, currency, and OCR.
- `hal/audio_priority.py`: shared arbitration state for routine pings and speech.
- `safety/readiness.py`: authoritative readiness contract.
- `pipeline.py`: constructs every subsystem from the loaded configuration and publishes real telemetry.
- `scripts/run_live_wearable.py`: production/simulation entry point. It must not introduce synthetic hardware into production mode.

`GroundedSmolVLMNarrator` performs real local SmolVLM2 image inference in the cognitive worker. It receives temporally confirmed detections and physical-ToF facts in its prompt, emits strict JSON at temperature zero, and is rejected if path state, hazard, direction, distance, wording, or supported object claims conflict with perception. VLM output never controls immediate cues.

Measured laptop VLM latency is about 9.1 seconds. Request-time facts can therefore be stale by completion. The current code does not yet perform completion-time sensor revalidation or task expiry. The cognition queue accepts four jobs, so repeated requests can create a stale backlog.

### Open P0 defects

- Close-ToF audio is synthesized before YOLO, but its call does not explicitly carry `CRITICAL` priority or flush speech/older PCM.
- The main loop sleeps a full frame interval after processing, reducing achieved FPS.
- Accepted VLM narration is not revalidated against the newest sensor state before speech.
- User-facing `path_state: clear` is stronger language than a single-ray ToF plus limited COCO detector can justify; migrate to “no tracked hazard.”

## Known capability boundaries

- The COCO detector does not cover doors, stairs, drop-offs, potholes, glass, or overhead clearance.
- A single ToF ray does not provide dense depth and must be aligned/calibrated to the camera.
- Currency recognition is heuristic, not a trained/authenticated banknote verifier.
- Haar/person-assisted redaction can miss faces. Never promise anonymity and do not transmit images by default.
- GPS is present only when supplied by a real integration (`DRISHTI_GPS_LAT/LON` currently provides the adapter boundary). Missing readings remain `null`.
- MQTT QoS 1 can deliver duplicates. Consumers must deduplicate by event semantics; the client retains records until PUBACK.
- The hazard fine-tuning and evaluation tools are implemented, but no wearable dataset is committed and no improved-recall claim is valid until a reviewed dataset is collected and held-out metrics are recorded.

## Provisioning

Run `sudo bash scripts/setup_raspberry_pi.sh`. It installs Picamera2 from Raspberry Pi OS packages, creates the virtual environment with system packages visible, installs `.[pi,vlm]`, downloads/verifies the provisioned models, and installs the service.

Broker credentials belong in `/etc/drishti/drishti.env`; do not commit them. Telemetry persistence defaults to `var/telemetry-buffer.jsonl` and is bounded by `buffer_limit`.

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
