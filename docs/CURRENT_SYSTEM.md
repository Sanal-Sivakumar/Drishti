# Current System: How Drishti Works Today

Last updated: 2026-11-09.

## 1. Purpose and Safety Boundary

Drishti is an offline-first assistive-perception prototype. It detects obstacles locally using YOLOv8n ONNX and a Time-of-Flight (VL53L0X) laser rangefinder, synthesizes 3D spatial audio cues at 30 FPS (<35 ms loop), and answers on-demand visual queries (Scene Narration, Banknote Identification, Multilingual Text Reading) via cloud vision models with local edge privacy redaction.

It is not a certified mobility aid. It must not replace a cane, guide dog, sighted assistant, or orientation-and-mobility training. Outdoor, road-crossing, drop-off, and unsupervised use are strictly outside the present scope.

## 2. Runtime Data Flow

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

Button 1 (GPIO 32 / N) ─► bounded worker ─► edge blur (face/plate) ─► Online Vision API (Gemini 3.1 / GPT-4o)
Button 2 (GPIO 33 / C) ─► bounded worker ─► edge blur             ─► Banknote Recognition (INR ₹10–₹500)
Button 3 (GPIO 25 / T) ─► bounded worker ─► edge blur             ─► Multilingual Indic OCR
                                          trusted facts           ─► Strict JSON validator & Sanitizer
                                                                  ─► Local Piper TTS playback
```

### 1. Startup and Construction
`scripts/run_live_wearable.py` loads `config/pipeline_config.yaml` and one hardware profile. Laptop mode opens a real colour camera, attempts to connect to the ESP32/VL53L0X serial stream, configures audio, telemetry, triggers, and constructs `DrishtiMasterPipeline`.

Production mode never silently substitutes a synthetic camera or range. Missing required hardware produces `not_ready` and `path_state: unknown`. Synthetic devices are available only in explicit simulation (`--hardware synthetic`).

### 2. Safety/Readiness Gate
`safety/readiness.py` evaluates the physical camera, real YOLO session, fresh non-synthetic range, detector class support, measured geometry calibration, and selected audio-profile calibration. Navigation is authoritative only when every critical requirement passes.

Missing or stale physical range (`RANGE_NO_VALID_SAMPLE`) is treated as a critical fault, forcing `path_state = "unknown"` and suppressing routine spatial pings.

### 3. Fast Perception Loop
`perception/pipeline_fast.py` executes each cycle in $<35\text{ ms}$:
1. Verify system readiness.
2. Capture the camera frame.
3. Obtain a fresh physical ToF range sample and apply the calibrated baseline offset.
4. If ToF detects an object at or below $0.8\text{ m}$, synthesize an immediate emergency cue before detector inference.
5. Run YOLO ONNX on RGB-normalized, letterboxed input.
6. Fuse detections with the ToF cone and calibrated optical geometry.
7. Select the nearest hazard and synthesize a binaural ITD/ILD spatial cue.
8. Output status and metrics to the local debug viewer (`http://127.0.0.1:8080`).

### 4. Visual Cognition & Button Triggering
Cognition runs asynchronously in `cognitive/worker.py` via `OnlineVisionService`:
- **Button 1 (`GPIO 32` / Key `N`)**: **Scene Narration** — Grounded by local YOLO/ToF facts. Directional or movement advice is stripped by regex filters.
- **Button 2 (`GPIO 33` / Key `C`)**: **Banknote Identification** — Evaluates Mahatma Gandhi New Series ₹10–₹500. Requires `confidence == "high"` and spoken denomination mapped strictly from verified lookup tables.
- **Button 3 (`GPIO 25` / Key `T`)**: **Multilingual Text Reading** — Literal transcription across Latin and Indic scripts.

### 5. Edge Privacy Filter
Before cloud upload, `FaceAnonymizer` applies vectorized 2D Gaussian blur to bystander faces (Haar cascade + upper 32% of YOLO person boxes) and vehicle license plates (inside detected vehicle boxes). Synthetic test face heuristics are strictly scoped to simulation frames to prevent blurring of books, paper, and banknotes.

### 6. Automated Testing Boundary
The test suite consists of **109 passing unit and integration tests** verifying contracts, parsing, fail-closed mechanics, geometry math, spatial synthesis, and simulation integration.
