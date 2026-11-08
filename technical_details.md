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

Button-triggered narration, currency recognition, and OCR are copied into a bounded queue serviced by `CognitiveWorker`. Queue saturation rejects new work instead of accumulating stale tasks. Results are returned asynchronously in `cognitive_results` or through `wait_for_cognition()`.

`GroundedSmolVLMNarrator` loads `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` through Transformers and passes the real RGB camera image plus structured, temporally confirmed sensor facts. Generation uses `do_sample=False`, temperature zero, a token bound, and a generation time bound. The parser requires an exact JSON schema. It rejects contradictory path state, hazard, direction or ToF distance; unsupported hazard words; invented metric values; and movement instructions. Rejected or unavailable model output is not spoken. Immediate perception and audio continue independently.

One local CPU smoke test completed in approximately 9.1 seconds. The model stays loaded within a running process, but warm p50/p95/p99 performance has not been recorded. Sensor facts are captured when the task is queued and are not yet revalidated at completion; accepted output can therefore be stale. The four-item cognitive queue also needs latest-only coalescing.

Narration confirmation requires consecutive class/direction observations. Critical close-range physical ToF readings bypass both confirmation and the VLM and immediately produce a centre warning with class `unclassified obstacle`.

OCR expects the camera’s BGR contract and applies BT.601 grayscale coefficients in B, G, R order. Currency recognition remains a color/shape heuristic and should not be treated as an authenticated denomination classifier.

## Face redaction

`FaceAnonymizer` runs OpenCV’s bundled frontal-face Haar cascade and can also use detector-provided person regions. Regions are Gaussian-blurred locally. Detection has false negatives, so this is risk reduction—not an anonymity guarantee. Telemetry sends metadata only in current flows; an emergency payload without explicit redaction metadata reports `privacy_filtered: false`.

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
