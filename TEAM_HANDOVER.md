# Drishti: Team Handover & Contributor Engineering Guide

Welcome to the Drishti engineering repository. This document defines the architectural standards, development protocols, verification checkpoints, and onboarding instructions for all contributors.

---

## 1. Engineering Principles & Mission

Drishti is an offline-first assistive wearable for visually impaired individuals. When designing and maintaining code in this repository, adhere strictly to these core tenets:

1. **Safety-Critical Real-Time Performance**: Obstacle detection and spatial audio alerts must complete within $< 150 \text{ ms}$ end-to-end. Latency spikes can lead directly to physical user injury.
2. **Headless & Production-Grade**:
   - Zero blocking GUI calls (`cv2.imshow`, `cv2.waitKey` must never be called in headless runtime or test pipelines).
   - Zero hardcoded local paths or fixed test image filenames. Everything is dynamically configured via configuration files or command-line arguments.
   - Robust fallbacks: If physical sensors (Camera Module 3, VL53L1X ToF) are detached, the Hardware Abstraction Layer (HAL) must switch gracefully to synthetic streams without crashing.
3. **Ergonomic Audio Channel Use**: The human ear is a low-bandwidth, serial sensory channel. Never overwhelm the user with redundant speech. Prioritize directional spatial sound cues over verbose natural language.
4. **Offline Autonomy**: All essential navigation, OCR, currency recognition, and basic scene narration features must execute purely on local edge compute without internet access.

---

## 2. Git & Branching Conventions

To maintain strict traceability across distributed engineering teams, all branches, commits, and pull requests must follow standard domain-driven naming schemes:

### 2.1 Branch Naming Scheme
- `feature/<subsystem>-<feature-description>`
  - *Example*: `feature/perception-tof-fusion`
  - *Example*: `feature/cognitive-indic-ocr`
  - *Example*: `feature/audio-binaural-panning`
- `fix/<subsystem>-<issue-description>`
  - *Example*: `fix/hal-i2c-bus-recovery`
  - *Example*: `fix/quantization-scale-overflow`
- `perf/<subsystem>-<optimization-target>`
  - *Example*: `perf/detector-int8-neon-pipeline`

### 2.2 Commit Message Convention
Commits must follow Conventional Commits formatting:
```
<type>(<scope>): <short imperative summary>

[Optional detailed rationale, math derivation, or root cause]

[Optional issue reference]
```
- Allowed types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`.
- Scopes: `hal`, `perception`, `cognitive`, `audio`, `safety`, `telemetry`, `config`.

---

## 3. Hardware Abstraction Layer (HAL) Pattern

Every physical input or output device must implement an abstract base class with both a **Hardware Driver** and a **Synthetic Mock Driver**:

```
                 +-------------------------------+
                 | Abstract Base Class (e.g. HAL)|
                 +---------------+---------------+
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
+---------------------------------+  +---------------------------------+
|     Physical Hardware Driver    |  |     Headless Synthetic Mock     |
| (V4L2 / CSI / I2C / ALSA Sink)  |  | (Pattern Generator / FIFO Sink) |
+---------------------------------+  +---------------------------------+
```

- When running on development PCs without physical sensors, the pipeline seamlessly selects mock drivers.
- In CI/CD pipelines, tests run completely headless with zero physical hardware dependencies.

---

## 4. Verification & Quality Gates

Before merging any code into `main`, every pull request must pass three verification gates:

### Gate 1: Static Analysis & Type Checking
```bash
ruff check . tests/
mypy .
```
All configuration is centralized in `pyproject.toml`. All functions must have comprehensive type hints (`typing.Optional`, `typing.Tuple`, `dataclass`).

### Gate 2: Headless Unit & Integration Tests
```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```
All 50 unit and integration tests must pass cleanly. Minimum acceptable line coverage across core modules is $85\%$.

### Gate 3: Quantitative Latency Budget Check
Every perception and audio commit must be benchmarked against latency limits:

| Subsystem Component | Maximum Permissible Latency (Pi 5) | Maximum Permissible Latency (PC) |
| :--- | :--- | :--- |
| **Image Preprocessing & Resizing** | $\le 8 \text{ ms}$ | $\le 2 \text{ ms}$ |
| **YOLO Nano INT8 Inference** | $\le 45 \text{ ms}$ | $\le 10 \text{ ms}$ |
| **ToF Spatial Fusion & Raycast** | $\le 2 \text{ ms}$ | $\le 0.5 \text{ ms}$ |
| **Binaural Spatial Audio Generation** | $\le 5 \text{ ms}$ | $\le 1 \text{ ms}$ |
| **End-to-End Fast Obstacle Alert** | $\mathbf{\le 80 \text{ ms}}$ | $\mathbf{\le 20 \text{ ms}}$ |
| **Multilingual OCR Recognition** | $\le 650 \text{ ms}$ | $\le 180 \text{ ms}$ |
| **VLM Scene Caption (On-Demand)** | $\le 2500 \text{ ms}$ | $\le 600 \text{ ms}$ |

### Gate 4: Hardware Health Diagnostics
Before running physical hardware on target kits (Raspberry Pi 5 or development laptop), run the non-destructive diagnostic check:
```bash
python3 scripts/check_hardware_health.py
```

---

## 5. Contributor Alignment Prompt

When extending or reviewing modules in Drishti, keep this mental model active:

```
"You are a core systems engineer on the Drishti Assistive Wearable team.
Every line of code you write directly impacts the physical safety and independence
of a visually impaired individual.
Ensure code is strictly modular, non-blocking, headless-compatible, thoroughly tested,
and optimized for resource-constrained edge hardware."
```

---

## 6. Current Implementation Status & Handover Ledger

As of the current milestone, the engineering baseline stands as follows:

| Subsystem / Phase | Current State | Verification Passed |
| :--- | :--- | :--- |
| **Phase 0: HAL** | Abstract Base Classes + Synthetic Drivers + V4L2/OpenCV physical camera with process lifecycle management (`__enter__`/`__exit__`, `kill()` on timeout) + ALSA non-blocking worker thread | `tests/unit/test_hal.py` (4/4 PASSED) |
| **Phase 1: Fast Perception** | YOLOv8n ONNX Object Detector with direct OpenCV letterboxing, 3D Spatial Audio (Continuous Fractional Delay interpolation via `np.interp`, ITD + ILD + PRF), Ray-Cone Depth Fusion, BGR INR Banknote Classifier with Multi-channel Canny Edge Contour Segmentation (`identify_currency`), Aspect Ratio Gating | `tests/unit/test_spatial_audio.py`, `test_depth_fusion.py`, `test_currency.py`, `test_fast_pipeline.py` (18/18 PASSED) |
| **Phase 2: Cognitive (OCR & VLM)** | Multilingual Indic OCR (Eng, Hin, Tam, Mal NFC), Multimodal Spatial Scene Narrator & ONNX VLM runner, Local Speech with Natural Voice Styling (`female1`, 0.95 conversational rate, non-overlapping audio sink gating) | `tests/unit/test_ocr.py`, `tests/unit/test_vlm.py`, `tests/unit/test_speech.py` (12/12 PASSED) |
| **Phase 3: Safety & Telemetry** | Local 2D Vectorized Gaussian Face Redaction (`cv2.GaussianBlur` priority with scipy/numpy fallbacks), Mosquitto MQTT Telemetry (TLS 1.3, QoS 1, paho v2), Master Orchestrator with default background continuous currency | `tests/unit/test_anonymizer.py`, `tests/unit/test_telemetry.py`, `tests/integration/test_pipeline.py` (12/12 PASSED) |
| **Config & Tooling** | Strict schema validation (`validate_pipeline_config`, `validate_hardware_config`), `pyproject.toml` standardized tooling, Hardware health diagnostic CLI (`scripts/check_hardware_health.py`) | `tests/unit/test_config_loader.py` (4/4 PASSED) |
| **Hardware Verification** | Laptop webcam (`/dev/video0`), laptop speakers (`plughw:1,0` / `default`), and synthetic fallback tested live; non-destructive hardware health auditor | `scripts/check_hardware_health.py`, `scripts/run_phase1_perception.py`, `scripts/run_phase2_cognitive.py`, `scripts/run_phase3_full_system.py`, `scripts/run_live_wearable.py` |
| **CI / CD** | GitHub Actions multi-version matrix (Python 3.10, 3.11, 3.12) with model SHA-256 verification and `pyproject.toml` packaging | `.github/workflows/ci.yml` (50/50 Tests Passing) |
| **Next Tasks (Phase 4)** | Hailo-8L NPU pipeline acceleration, 3D wearable enclosure, and field navigation trials | Scheduled |





