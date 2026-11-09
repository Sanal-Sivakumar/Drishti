# Drishti Engineering Handover

Last updated: 2026-11-09. Read [`docs/README.md`](docs/README.md) before making architectural changes.

## 1. Current System Status

Drishti is an offline-first assistive wearable research prototype. The platform operates on a strict **fail-closed** safety standard: if the camera, physical range source, validated YOLO ONNX model, optical geometry calibration, or audio profile is unverified or missing, navigation reports `not_ready` and `path_state` is forced to `unknown`. Synthetic simulation is strictly isolated and explicitly labeled.

The automated test suite verifies 109 unit and integration tests across safety readiness, depth fusion, anonymization, spatial audio, and cloud vision contracts (`python3 -m pytest tests/ -q`).

## 2. Runtime Architecture

The codebase separates real-time obstacle avoidance from on-demand visual cognition:

- `perception/pipeline_fast.py`: Real-time camera → YOLOv8n ONNX → ToF depth fusion → binaural spatial audio cue loop. Executes 100% offline at 30 FPS (<35 ms per cycle).
- `cognitive/online_vision.py`: Provider-agnostic cloud multimodal vision engine supporting Google Gemini 3.1 Flash-Lite (`thinkingBudget: 0`, ~1.5 s response) and OpenAI GPT-4o-mini with HTTP keep-alive connection pooling.
- `cognitive/worker.py`: Bounded background queue for scene narration, banknote recognition, and text reading.
- `safety/anonymizer.py`: Client-side edge privacy filter applying vectorized 2D Gaussian blur to bystander faces and vehicle license plates before cloud upload.
- `safety/readiness.py`: Authoritative fail-closed readiness contract.
- `hal/esp32_link.py`: Asynchronous 921600 baud serial driver interfacing the ESP32 DevKit V1 ranging loop and tactile button events.
- `scripts/run_live_wearable.py`: Master live runner with automatic peripheral probing and headless browser viewer (`http://127.0.0.1:8080`).

## 3. Hardware Controls & Peripherals

The hardware stack connects an ESP32 DevKit V1 and VL53L0X Time-of-Flight sensor over USB serial with 3 tactile push buttons:

1. **Button 1 (`GPIO 32` / Key `N`)**: **Scene Narration** — Synthesizes concise situational descriptions grounded by local sensor facts. Model output is automatically sanitized against directional/movement commands.
2. **Button 2 (`GPIO 33` / Key `C`)**: **Banknote Identification** — Specialized verification for Indian Rupee Mahatma Gandhi New Series (₹10, ₹20, ₹50, ₹100, ₹200, ₹500). Enforces high-confidence thresholds and dictionary lookup.
3. **Button 3 (`GPIO 25` / Key `T`)**: **Multilingual Text Reading** — Literal verbatim transcription across Latin and Indic scripts (Hindi, Tamil, Malayalam, Bengali, etc.).

Terminal keyboard shortcuts (`N`, `C`, `T`, `Q`) are disabled by default and can be enabled for test benches using `--enable-keyboard`.

## 4. Key Architectural Safeguards

1. **Fail-Closed Navigation**: Missing or stale physical range (`RANGE_NO_VALID_SAMPLE`) is marked as critical, forcing `path_state = "unknown"` and suppressing all inferred-distance audio pings.
2. **Edge Privacy Scoping**: Synthetic test face markers are restricted to simulation frames (`frame[0, 0, 1] == 40 and frame[0, 0, 2] == 40`), preventing white books, documents, and currency notes from being blurred as faces. License plate detection executes only when vehicles are detected in the frame.
3. **Gemini Latency Optimization**: All Gemini generation calls specify `thinkingConfig: {"thinkingBudget": 0}` in `generationConfig`, slashing cognitive response latency from $>20\text{ s}$ down to $1.5\text{--}1.8\text{ s}$.
4. **Local Neural Speech**: Vocal output is generated entirely offline via Piper neural TTS (`models/piper/en_US-lessac-medium.onnx`).

## 5. Verification Commands

```bash
# Compile check
python3 -m compileall -q cognitive config hal perception safety scripts pipeline.py

# Test suite (all 109 tests must pass)
python3 -m pytest tests/ -q

# Hardware peripheral audit
python3 scripts/check_hardware_health.py

# Model verification
python3 scripts/download_models.py
```

## 6. Protocols Before Supervised Field Trials

Record the exact camera module, resolution, measured FOV/intrinsics, ToF offset/alignment, model checksum, test route, observer, lighting, software commit, battery state, and every readiness transition. Deliberately disconnect each sensor to confirm that the UI and audio output fail closed (`unknown`). Stop trials immediately on any unverified readiness transition, missed obstacle, or audio conflict.
