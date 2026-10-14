# Drishti: Troubleshooting Guide & Operational Diagnostic Ledger

This ledger documents known failure modes, root causes, reproducible diagnostics, and permanent fixes across both development workstations and target Raspberry Pi 5 edge hardware.

---

## 1. Hardware & Physical Sensor Layer

### 1.1 Camera Subsystem: CSI-2 / V4L2 Device Failures

#### Symptom: `Camera index out of range` or `/dev/video0: Device or resource busy`
- **Root Cause**:
  1. Another daemon (such as `libcamera-vid`, `motion`, or a dangling OpenCV background process) has locked the V4L2 file descriptor.
  2. On Raspberry Pi 5, Camera Module 3 requires `libcamera` backend drivers. Standard legacy V4L2 calls fail if `camera_auto_detect=1` in `/boot/firmware/config.txt` is not active or if the ribbon cable is misaligned.
- **Diagnostic Command**:
  ```bash
  # Check if video devices exist
  ls -la /dev/video*
  # Identify process holding the file descriptor
  fuser -v /dev/video* || lsof /dev/video0
  # On Raspberry Pi 5: verify sensor detection via libcamera
  rpicam-hello --list-cameras
  ```
- **Fix**:
  1. Kill orphaned streaming processes: `kill -9 $(fuser /dev/video0 2>/dev/null)`
  2. For Raspberry Pi 5, verify physical orientation: the blue tape on the 15-pin FFC ribbon cable must face towards the Ethernet/USB ports. Ensure `dtoverlay=imx708` is loaded.
  3. Drishti's Hardware Abstraction Layer (`src/hal/camera.py`) implements automatic fallback to mock/synthetic test video loops during headless validation.

---

### 1.2 I2C Bus & VL53L1X ToF Sensor Failures

#### Symptom: `OSError: [Errno 121] Remote I/O error` or `Sensor not found at 0x29`
- **Root Cause**:
  1. Loose physical wiring on SDA (GPIO 2) or SCL (GPIO 3).
  2. I2C clock stretching timeout: the default 100 kHz bus clock can experience transmission errors if the ribbon cable exceeds 15 cm without pull-up resistors (Raspberry Pi internal pull-ups are 1.8 kΩ to 3.3V).
  3. Hardware bus lockup: the sensor was interrupted mid-byte, leaving the SDA line clamped low.
- **Diagnostic Command**:
  ```bash
  # Scan I2C bus 1 for the default VL53L1X address (0x29)
  i2cdetect -y 1
  ```
- **Fix**:
  1. If `0x29` does not appear, power cycle the 3.3V rail (Pin 1) or run an I2C clock unwedge sequence (toggling SCL 9 times to force the slave to release SDA).
  2. Drishti's distance sensor driver (`src/hal/distance_sensor.py`) checks for hardware presence at initialization; if absent or disabled in dev mode, it switches cleanly to synthetic range simulation.

---

## 2. Audio Subsystem & Spatial Cues

### 2.1 ALSA / PulseAudio Buffer Underrun (`xrun`)

#### Symptom: Audio pops, clicking artifacts, or `ALSA lib pcm.c: underrun occurred`
- **Root Cause**:
  The audio synthesis thread is being starved by heavy computer vision matrix operations executing on the same CPU cores, causing ALSA playback buffers to starve.
- **Diagnostic Command**:
  ```bash
  # Inspect system audio cards
  aplay -l
  # Monitor real-time scheduling priority and thread latency
  chrt -p $(pgrep -f "drishti")
  ```
- **Fix**:
  1. Decouple audio generation into a dedicated high-priority daemon thread with a non-blocking queue (`queue.Queue(maxsize=16)`).
  2. Set the audio processing chunk buffer size to $1024$ samples ($\sim 23 \text{ ms}$ at $44.1 \text{ kHz}$) to absorb jitter without introducing noticeable latency.
  3. Drishti's spatial audio synthesizer uses pre-computed lookup tables for ITD/ILD convolution rather than on-the-fly filtering.

---

## 3. Deep Learning Runtime & Quantization

### 3.1 ONNX Runtime Execution Provider Mismatch

#### Symptom: Slow inference ($>300 \text{ ms}$) on host PC with NVIDIA GPU
- **Root Cause**:
  `onnxruntime` standard package was installed instead of `onnxruntime-gpu`, or CUDA/cuDNN shared libraries (`libcudnn.so`, `libcublas.so`) are missing from `LD_LIBRARY_PATH`.
- **Diagnostic Command**:
  ```python
  import onnxruntime as ort
  print("Available Providers:", ort.get_available_providers())
  ```
- **Fix**:
  1. Install `onnxruntime-gpu` and verify `CUDAExecutionProvider` is in the returned list.
  2. Drishti's detector automatically negotiates the fastest available provider in order: `['HailoRTExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`.

---

### 3.2 INT8 Quantization Clamping / Accuracy Degradation

#### Symptom: False positive detections or missed obstacles after INT8 conversion
- **Root Cause**:
  Poor calibration dataset distribution during Post-Training Quantization (PTQ). If the calibration set does not capture varied indoor/outdoor luminance distributions, dynamic activation ranges are clipped, causing small obstacles to drop below the confidence threshold.
- **Fix**:
  1. Use KL divergence calibration (Entropy calibration) rather than MinMax calibration for activations.
  2. Maintain a minimum confidence threshold of $0.40$ for safety-critical obstacle classes (`person`, `chair`, `door`, `stairs`).

---

## 4. Multilingual OCR & Indic Script Rendering

### 4.1 Missing Indic Language Models / Ligature Glitches

#### Symptom: Tesseract output returns garbled text or unmapped Unicode question marks (`???`)
- **Root Cause**:
  1. Language training packs `tesseract-ocr-hin`, `tesseract-ocr-tam`, `tesseract-ocr-mal` are missing from `/usr/share/tesseract-ocr/4.00/tessdata/` or `TESSDATA_PREFIX`.
  2. Text was not normalized to Unicode NFC, causing decomposed matras to separate from consonant glyphs.
- **Diagnostic Command**:
  ```bash
  # Check installed Tesseract languages
  tesseract --list-langs
  ```
- **Fix**:
  1. Install system Indic training data:
     ```bash
     sudo apt-get install -y tesseract-ocr-hin tesseract-ocr-tam tesseract-ocr-mal
     ```
  2. Normalize all extracted string outputs via `unicodedata.normalize('NFC', text)` before piping into the TTS synthesizer.

---

## 5. System Thermal & Power Throttling (Raspberry Pi 5)

### 5.1 Undervoltage Warning & Frequency Throttling

#### Symptom: Pipeline FPS suddenly drops from 30 FPS to 12 FPS during continuous inference
- **Root Cause**:
  1. Power supply cannot deliver 5V / 5A (25W) required by the Raspberry Pi 5 under combined CPU + CSI + NPU load. Standard phone chargers deliver 5V / 3A (15W), causing brownouts.
  2. SoC thermal throttling kicks in at $80^\circ \text{C}$ ($85^\circ \text{C}$ hard cap), cutting CPU frequency from 2.4 GHz down to 1.5 GHz.
- **Diagnostic Command**:
  ```bash
  # Check throttling status bitmask (0x0 means healthy)
  vcgencmd get_throttled
  # Measure current core temperature
  vcgencmd measure_temp
  ```
- **Bitmask Reference**:
  - `0x1`: Under-voltage detected currently
  - `0x2`: Arm frequency capped currently
  - `0x4`: Currently throttled
  - `0x50000`: Under-voltage occurred in past
  - `0x20000`: Throttling occurred in past
- **Fix**:
  1. Use the official Raspberry Pi 27W USB-C PD power adapter.
  2. Ensure the active cooling fan is installed and functioning. The firmware fan curve will engage at $60^\circ \text{C}$.

---

## 6. Diagnostic Decision Matrix

```
[System Failure Detected]
           |
           +---> Camera stream black/empty?
           |        +---> Check: `fuser -v /dev/video0` -> kill zombie holders
           |        +---> Check: Camera Module 3 ribbon polarity -> blue tape to ports
           |        +---> Fallback: Drishti synthetic test pattern
           |
           +---> Distance always reports 0 or timeout?
           |        +---> Check: `i2cdetect -y 1` -> verify address 0x29
           |        +---> Check: I2C wiring (SDA Pin 3, SCL Pin 5)
           |        +---> Fallback: Drishti bounding-box geometric depth estimator
           |
           +---> Audio stuttering or high latency?
           |        +---> Check: CPU core saturation -> isolate audio to dedicated core
           |        +---> Check: Buffer size -> increase to 1024 frames
           |
           +---> Low inference FPS (<15 FPS on Pi 5)?
                    +---> Check: `vcgencmd get_throttled` -> check power adapter (needs 5V/5A)
                    +---> Check: Verify INT8 model is loaded rather than FP32
```

