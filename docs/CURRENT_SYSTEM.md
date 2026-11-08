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

Button N / Button 1 ─► bounded cognition queue ─► redaction ─► SmolVLM2
                                              trusted facts ─► JSON validator
                                                            ─► speech if accepted
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

The button-triggered cognition path runs in `cognitive/worker.py`, away from the perception loop. The queue is bounded at four tasks. SmolVLM2 receives the frame and structured trusted facts, then must return exactly:

```json
{
  "path_clear": false,
  "primary_hazard": "chair",
  "direction": "right",
  "distance_m": 1.3,
  "speech": "Chair 1.3 metres to your right."
}
```

The validator rejects contradictory path state, object, direction or distance; untrusted hazard words; invented metric claims; unsafe movement instructions; schema errors; and empty/placeholder speech. Rejected output is not spoken.

The VLM does not control collision warnings. However, its accepted facts currently refer to the request-time snapshot. Completion-time freshness revalidation is still required because laptop inference measured about 9.1 seconds.

### 7. OCR, banknotes, privacy, and telemetry

- OCR uses Tesseract language packs and Unicode NFC normalization.
- Banknote recognition is a heuristic colour/shape/texture classifier, not a trained or authenticated currency verifier.
- Face redaction uses a Haar cascade plus person regions. False negatives are possible; it is risk reduction, not an anonymity guarantee.
- MQTT telemetry sends metadata through TLS/authenticated configuration, stores a bounded JSONL buffer while disconnected, replays after reconnect, and tracks QoS 1 acknowledgements. QoS 1 consumers must tolerate duplicates.

## What automated tests prove—and do not prove

The current suite has 105 passing tests. It checks software contracts, fail-closed behavior, parsers, queues, geometry math, audio synthesis, telemetry persistence, and simulation integration.

It does not prove real-camera recall, real-ToF accuracy, audible warning latency, VLM latency on Raspberry Pi, face-redaction recall, outdoor robustness, human-factor safety, or regulatory compliance.

