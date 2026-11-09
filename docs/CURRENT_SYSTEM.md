# Current system: how Drishti works today

Last reviewed: 2026-11-08.

## Purpose and safety boundary

Drishti is an offline-first assistive-perception prototype. It detects a small configured set of objects, combines detections with a physical range sensor, produces stereo direction cues, and can answer an on-demand scene-description request with a local vision-language model (VLM).

It is not a certified mobility aid. It must not replace a cane, guide dog, sighted assistant, or orientation-and-mobility training. Outdoor, road-crossing, drop-off, and unsupervised use are outside the present scope.

## Runtime data flow

```text
Physical camera ──► BGR frame ──► YOLO ONNX ──► object boxes/classes
      │                                      │
      │                                      ▼
      │                         camera/ToF geometry fusion
      │                                      ▲
Physical ToF/ESP32 ──► fresh range sample ───┘
                                             │
                                             ├─► immediate stereo obstacle cue
                                             ├─► readiness/path state + viewer
                                             └─► confirmed facts for cognition

Button 1 (GPIO 32 / N) ─► bounded worker ─► edge blur (face/plate) ─► Online Vision API (OpenAI/Gemini)
Button 2 (GPIO 33 / C) ─► bounded worker ─► edge blur             ─► Banknote Recognition (INR)
Button 3 (GPIO 25 / T) ─► bounded worker ─► edge blur             ─► Multilingual Indic OCR
                                          trusted facts           ─► Strict JSON validator & Sanitizer
                                                                  ─► Local Piper TTS playback
```

### 1. Startup and construction

`scripts/run_live_wearable.py` loads `config/pipeline_config.yaml` and one hardware profile. Laptop mode opens a real colour camera, attempts to connect to the ESP32/VL53L0X serial stream, configures audio, telemetry, triggers, and constructs `DrishtiMasterPipeline`.

Production mode never silently substitutes a synthetic camera or range. Missing required hardware produces `not_ready` and `path_state: unknown`. Synthetic devices are available only in explicit simulation.

### 2. Safety/readiness gate

`safety/readiness.py` evaluates the physical camera, real YOLO session, fresh non-synthetic range, detector class support, measured geometry calibration, and selected audio-profile calibration. Navigation is authoritative only when every critical requirement passes.

The current checkout is intentionally **not ready** because the repository does not contain the YOLO ONNX weight, the geometry file is unverified, and the audio profiles are uncalibrated. Model binaries are provisioned locally and are not committed to Git.

### 3. Fast perception loop

`perception/pipeline_fast.py` performs one cycle:

1. Check readiness.
2. Capture the camera frame.
3. obtain a fresh physical ToF sample and apply the calibrated offset.
4. If ToF is at or below `0.8 m`, synthesize an immediate centre warning before YOLO.
5. Run YOLO ONNX on RGB-normalized, letterboxed input.
6. Fuse detections with the ToF cone and calibrated camera geometry.
7. Select the nearest hazard and create a stereo ITD/ILD cue.
8. Publish per-stage timing and readiness.

The current COCO model covers only configured COCO labels: person, chair, dining table, car, and bicycle. It does not detect stairs, curbs, drop-offs, potholes, glass doors, poles, or hanging obstacles.

### 4. Direction and distance

Horizontal direction comes from the bounding-box position and camera intrinsics. A metric distance is labeled as physical only when a detection intersects the calibrated ToF cone, or when the close-range warning comes directly from ToF. Bounding-box size may provide a geometric prior, but that value is not represented to the VLM as a physical ToF measurement.

A single forward ToF sensor is not a depth camera. It cannot establish floor continuity, overhead clearance, or a globally clear walking corridor.

### 5. Audio

`perception/spatial_audio.py` creates stereo PCM using Woodworth ITD, fractional delay, ILD, a low-frequency component, a 2–5 kHz component, a smooth envelope, and a conservative limiter. Separate air- and bone-conduction profiles exist, but both are uncalibrated by default.

`hal/audio_priority.py` shares an arbitration state between routine pings and speech. A known P0 limitation remains: the urgent ToF call currently does not explicitly submit `CRITICAL` priority or flush in-flight output. Critical audible preemption must be fixed and measured before trials.

### 6. Temporal confirmation and on-demand cognition

`perception/temporal_confirmation.py` requires two consistent observations before detections are sent to narration. This confirmation does not delay the immediate close-ToF warning.

The button-triggered cognition path runs in `cognitive/worker.py` via `OnlineVisionService`, away from the fast perception loop. The queue is bounded to reject task buildup. Visual queries are dispatched to cloud multimodal models (OpenAI GPT-4o-mini or Google Gemini 2.5 Flash):

- **Scene Narration (Button 1 / N)**: Receives the privacy-filtered snapshot alongside confirmed local YOLO/ToF facts. Output must strictly describe context without giving movement or navigation advice.
- **Banknote Identification (Button 2 / C)**: Inspects Indian Rupee notes (Mahatma Gandhi New Series ₹10–₹500). Strictly rejects blurred, partial, or ambiguous frames without emitting false-positive denominations.
- **Multilingual Text Reading (Button 3 / T)**: Transcribes printed or written text across Latin and Indic scripts (Devanagari/Hindi, Tamil, Malayalam, Bengali, etc.).

All cloud output is validated and sanitized against directional instructions before being vocalized via local Piper neural TTS.

### 7. Banknotes, privacy, and telemetry

- Banknote recognition uses high-detail visual examination with strict ambiguity rejection to eliminate false positives.
- Multilingual OCR transcribes verbatim printed and written text across Indic scripts.
- Edge privacy redaction uses Haar cascades and vehicle/person priors to apply 2D Gaussian blur to bystander faces and vehicle license plates before cloud upload.
- MQTT telemetry sends metadata through TLS/authenticated configuration, stores a bounded JSONL buffer while disconnected, replays after reconnect, and tracks QoS 1 acknowledgements. QoS 1 consumers must tolerate duplicates.

## What automated tests prove—and do not prove

The current suite has 105 passing tests. It checks software contracts, fail-closed behavior, parsers, queues, geometry math, audio synthesis, telemetry persistence, and simulation integration.

It does not prove real-camera recall, real-ToF accuracy, audible warning latency, VLM latency on Raspberry Pi, face-redaction recall, outdoor robustness, human-factor safety, or regulatory compliance.

