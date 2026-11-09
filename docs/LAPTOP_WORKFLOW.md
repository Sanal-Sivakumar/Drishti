# Continue Development on a Laptop Connected to Hardware

This is the operational workflow for development on a workstation or laptop connected to the Drishti hardware stack.

Last updated: 2026-11-09.

---

## 1. Repository Setup & Model Provisioning

```bash
git clone https://github.com/Sanal-Sivakumar/Drishti.git
cd Drishti
git pull --ff-only origin main

# Virtual environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'

# Download local ONNX models (YOLOv8n and Piper neural voice)
python3 scripts/download_models.py

# Configure cloud credentials
cp .env.example .env
# Edit .env and supply GEMINI_API_KEY or OPENAI_API_KEY
```

Verify the environment:
```bash
python3 -m pytest tests/ -q
# Output: 109 passed in ~8s
```

---

## 2. Hardware Connections

The laptop development setup connects:
1. **Camera**: Built-in webcam or external USB camera.
2. **ESP32 DevKit V1**: Running `firmware/drishti_tof/drishti_tof.ino` connected via Micro-USB.
3. **VL53L0X ToF Rangefinder**: Connected via I²C (`SDA` to GPIO 21, `SCL` to GPIO 22, `VIN` to 3.3V, `GND` to GND).
4. **Three Tactile Buttons**:
   - **Button 1 (`GPIO 32`)**: Scene Narration
   - **Button 2 (`GPIO 33`)**: Banknote Identification (INR ₹10–₹500)
   - **Button 3 (`GPIO 25`)**: Multilingual Text Reading
5. **Headphones / Audio Output**: Stereo headphones for spatial binaural audio verification.

### Stable Serial Permissions
Grant serial access to your user account:
```bash
sudo usermod -aG dialout "$USER"
```

Locate the persistent hardware device path:
```bash
ls -l /dev/serial/by-id/
# Example: /dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0
```

Verify the ESP32 connection before starting the full system:
```bash
python3 scripts/probe_esp32.py --port auto --duration 5
python3 scripts/monitor_tof.py
python3 scripts/check_hardware_health.py
```

---

## 3. Running the System

### Live Hardware Mode (Webcam + ESP32)
```bash
python3 scripts/run_live_wearable.py --viewer
```
- Open `http://127.0.0.1:8080/` in Chrome or Firefox to monitor real-time annotated perception, 3D coordinates, and ToF tracking.
- Test visual cognition using the physical tactile buttons on the breadboard.
- To enable interactive keyboard shortcuts in the terminal, append `--enable-keyboard` (`N` = Narrate, `C` = Currency, `T` = Text, `Q` = Quit).

### Synthetic Simulation Mode (No Hardware Required)
```bash
python3 scripts/run_live_wearable.py --hardware synthetic --max-cycles 100
```

---

## 4. Development Best Practices

- Always run `python3 -m pytest tests/ -q` before committing code.
- Never commit `.env`, API keys, or raw sensor recording files.
- Keep the fast perception loop ($<35\text{ ms}$) strictly isolated from blocking network calls.
