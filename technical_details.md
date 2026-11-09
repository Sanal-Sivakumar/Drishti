# Technical details

This document describes the code that exists today. It deliberately separates measured sensor data, model-derived estimates, and simulation.

Last reviewed: 2026-11-08. Future product design is kept separately in [`docs/INDUSTRY_ROADMAP.md`](docs/INDUSTRY_ROADMAP.md).

## Safety and readiness contract

`SystemReadiness` is evaluated on every fast-perception cycle. Production navigation is authoritative only when the physical camera is available, the configured ONNX detector has loaded and passed its input contract, and physical ToF data is available/fresh. Any critical issue yields `not_ready`. A result without authoritative perception uses `path_state: unknown`; only an authoritative, successful cycle with no fused obstacle may use `clear`.

Simulation returns `level: simulation`. Synthetic camera detections and synthetic distance values are useful for deterministic tests but are never labeled as physical ToF.

## Image and model contract

Camera sources return `uint8` BGR arrays. The detector letterboxes while preserving aspect ratio, changes BGR to RGB, normalizes to `[0,1]`, and creates an NCHW float32 tensor. Model construction rejects a non-rank-4 input or a known channel count other than three. Runtime rejects invalid output ranks/types before parsing.

The default detector label space is COCO. Configuration aliases `table` to `dining table`; unsupported configured classes are reported and omitted. Consequently, this implementation does not detect doors, stairs, drop-offs, potholes, glass walls, or overhead clearance.

Horizontal focal length is derived from configured width `w` and measured horizontal field of view `theta`:

```text
fx = w / (2 tan(theta / 2))
```

The configured principal point and camera-to-ToF baselines then feed `SpatialDepthFuser`. These values require calibration on the final assembly; configuration values are not a substitute for measurement.

## ToF fusion

The physical sensor provides a range along a limited cone, not a per-pixel depth map. A detection overlapping that cone may use the measured range. Other detections use clearly labeled geometric/class-prior estimates. When the physical reading is absent or stale, production readiness becomes not-ready; no synthetic number is substituted.

## Spatial audio and arbitration

Azimuth is represented with interaural time and level differences. Distance selects configured pitch and pulse-rate bounds. `PriorityAudioArbiter` is the sole sink passed to real-time sonification. Speech obtains a higher-priority lease, causing ambient pings to be suppressed during speech. The sink remains bounded/non-blocking.

Current limitation: the urgent pre-YOLO ToF call uses the default ambient priority and the underlying ALSA queue is FIFO. It therefore does not yet guarantee that a critical cue interrupts speech or older PCM. This is a documented P0 defect, not a product guarantee.

## Cognitive worker

Button-triggered narration, currency identification, and text reading are copied into a bounded queue serviced by `CognitiveWorker`. Queue saturation rejects new work instead of accumulating stale tasks. Results are returned asynchronously in `cognitive_results` or through `wait_for_cognition()`.

`OnlineVisionService` integrates cloud multimodal models (OpenAI GPT-4o-mini or Google Gemini 2.5 Flash) via HTTPS requests. Strict JSON schemas govern all three cognitive actions:
1. **Scene Narration (Button 1)**: Supplies a privacy-filtered snapshot alongside confirmed local YOLO/ToF facts. Output is strictly sanitized against directional/movement commands; navigation safety belongs exclusively to local sensors.
2. **Indian Banknote Identification (Button 2)**: Targets Mahatma Gandhi New Series ₹10–₹500. Rejects blurred, folded, or ambiguous frames by returning `uncertain: true` and omitting false-positive announcements.
3. **Multilingual Text Reading (Button 3)**: High-fidelity literal transcription across Latin and Indic scripts (Hindi, Tamil, Malayalam, Bengali, etc.).

Visual cognition tasks never block or interrupt the fast perception loop. If internet connectivity drops, Drishti immediately emits local audio informing the user while obstacle avoidance sonification continues without disruption.

## Edge privacy & redaction

`FaceAnonymizer` runs OpenCV's Haar cascades for frontal faces and vehicle license plates, augmented by YOLO person and vehicle bounding box priors. Vectorized 2D Gaussian blur is applied on CPU before frames are compressed and transmitted to cloud vision APIs. Telemetry sends metadata only; raw camera frames are never transmitted via telemetry.

## Telemetry

The client publishes device-scoped MQTT topics with TLS certificate verification and optional username/password loaded from environment variables. Battery percentage and CPU temperature come from Linux sysfs. GPS is `null` unless a real adapter supplies it; placeholder coordinates are forbidden.

Every event enters a bounded deque and optional JSONL persistence file before publish. On reconnect, retained records are replayed. QoS 1 message IDs map to event IDs, and a record is removed only after PUBACK. Atomic file replacement keeps persistence consistent. QoS 1 is at-least-once, so consumers must tolerate duplicates.

## Interfaces

The main in-process interface is:

```python
pipeline = FastPerceptionPipeline(camera, distance_sensor, audio_sink, config=cfg, mode="production")
result = pipeline.step()
```

Viewer endpoints are `/`, `/stream`, `/snapshot`, and `/metrics`. MQTT topics are:

```text
drishti/device/{id}/heartbeat
drishti/device/{id}/events
drishti/device/{id}/emergency
drishti/device/{id}/status
```

## Validation boundary

Unit tests validate contracts, parsing, bounded queues, fail-closed behavior, and synthetic geometry. They do not establish clinical safety, outdoor performance, camera calibration, model recall, face-redaction recall, or usability. Those require controlled, supervised indoor studies and formal hazard analysis before scope can expand.

## Scheduling and latency instrumentation

Each fast step records camera, ToF, detection, fusion, audio-synthesis, and total software time. This excludes physical camera exposure history, serial transit before host receipt, audio-driver buffering, acoustic onset, and human response. The master run loop currently sleeps `1 / target_fps` after every completed step; the effective period is therefore processing time plus that sleep.

Use [`docs/LATENCY.md`](docs/LATENCY.md) for the required monotonic timestamp chain and optimization order.
