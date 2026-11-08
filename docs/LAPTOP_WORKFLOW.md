# Continue development on a laptop connected to hardware

This is the working handoff for the current office-account/laptop workflow. Keep credentials, personal data, model weights, calibration captures, and participant recordings out of Git.

## 1. Clone or update safely

```bash
git clone https://github.com/Sanal-Sivakumar/Drishti.git
cd Drishti
git pull --ff-only origin main
git status
```

Use Git Credential Manager, SSH, or `gh auth login` for the office GitHub account. Never place a personal access token in a remote URL, shell script, Markdown file, `.env`, or committed configuration.

For each unit of work:

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/<short-task-name>
```

Run tests, review `git diff`, commit, and push the branch. Merge only after CI passes.

## 2. Environment

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,vlm]'
python scripts/download_models.py
```

Training is optional and heavier:

```bash
python -m pip install -e '.[training]'
```

The virtual environment and model files are ignored by Git. A complete local setup needs approximately 2 GB for SmolVLM2 plus the Python/PyTorch environment and other model files.

## 3. Connect the current hardware

The present laptop stack expects:

- a real USB or built-in colour camera;
- ESP32 DevKit V1 running `firmware/drishti_tof/drishti_tof.ino`;
- VL53L0X connected according to `firmware/drishti_tof/WIRING.md`;
- two ESP32 buttons, or keyboard `N`/`C` input;
- wired stereo headphones for initial spatial-audio work.

Find stable device names:

```bash
ls -l /dev/serial/by-id/
v4l2-ctl --list-devices
aplay -l
```

Grant serial access once, then log out/in:

```bash
sudo usermod -a -G dialout "$USER"
```

Probe the ESP32 before starting the full pipeline:

```bash
python scripts/probe_esp32.py --port auto --duration 10
python scripts/monitor_tof.py
python scripts/check_hardware_health.py --serial-port auto
```

## 4. Run modes

Safe deterministic simulation:

```bash
python scripts/run_live_wearable.py --hardware synthetic --max-cycles 100
```

Laptop with real camera and auto-detected ESP32:

```bash
python scripts/run_live_wearable.py --hardware laptop --serial-port auto --viewer
```

Explicit stable serial path is preferred:

```bash
python scripts/run_live_wearable.py \
  --hardware laptop \
  --serial-port /dev/serial/by-id/<your-esp32-id> \
  --camera-index 0 \
  --viewer
```

Open `http://127.0.0.1:8080/`. Use `N` for narration, `C` for banknote/OCR, and `Q` to stop. If required hardware or calibration is missing, expect `NOT_READY`; do not bypass it.

## 5. Verify every change

```bash
python -m compileall -q cognitive config hal perception safety scripts pipeline.py tests
python -m pytest tests -q
python -m pytest tests/unit/test_spatial_audio.py -v
python scripts/smoke_test_vlm.py
git diff --check
```

Before committing hardware behavior, save a benchmark/experiment record containing the Git commit, configuration checksum, device identifiers, firmware version, lighting, route, power/thermal state, and raw metric output. Do not commit identifiable camera images.

## 6. Starting a new AI coding session

Ask the AI to read these files first:

```text
README.md
docs/README.md
docs/CURRENT_SYSTEM.md
docs/LATENCY.md
docs/NEXT_STEPS.md
TEAM_HANDOVER.md
```

Suggested opening prompt:

```text
Work in the Drishti repository. Read the documentation index and current-system
handover first. Preserve fail-closed production behavior and do not treat
simulation as hardware evidence. Show me git status before editing. Implement
only the next unchecked P0 item, add tests, run the relevant hardware/software
checks, update the documentation, and leave a reviewable Git diff.
```

An AI model can help implement and review code; it cannot certify the device, replace measured hardware evidence, approve risk acceptance, or authorize human trials.

## 7. Git hygiene and office-account privacy

Do not commit:

- `.venv/`, model weights, caches, generated audio, or build output;
- MQTT passwords, broker certificates/private keys, GitHub tokens, or office credentials;
- participant names, faces, voices, routes, GPS coordinates, or consent records;
- raw calibration/evaluation captures unless an approved data-governance process explicitly permits it.

Before every push:

```bash
git status --short
git diff --check
git diff --cached
```

