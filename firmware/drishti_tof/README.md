# Drishti ESP32 sensor MCU

Firmware for the ESP32 DevKit V1 that owns the VL53L0X ranging loop and the
two tactile buttons, and streams them to the laptop/Pi over USB serial.

This is the **current prototype sensor MCU**, not the proposed product depth architecture. VL53L0X is a single-zone, forward range source; it cannot detect a complete corridor, curbs/drop-offs outside its ray, or overhead hazards. The future roadmap evaluates multizone/dense depth, a downward channel, IMU synchronization, and an independent safety watchdog; see [`../../docs/INDUSTRY_ROADMAP.md`](../../docs/INDUSTRY_ROADMAP.md).

- **Wiring:** see [`WIRING.md`](WIRING.md)
- **Host driver:** `hal/esp32_link.py`
- **Emulator (no hardware):** `tools/fake_esp32.py`

---

## Protocol

ASCII CSV, newline terminated, **921600 8N1**.

### Host → MCU

| Line | Meaning |
|---|---|
| `V\n` | Request the firmware banner (the host sends this on every connect) |
| `P\n` | Heartbeat (every 500 ms). Missing for 2 s sets `link_ok=0` |
| `R\n` | Soft reset |

### MCU → host

| Line | Meaning |
|---|---|
| `I,<fw>,<model>` | Boot banner, e.g. `I,1.0.0,VL53L0X` |
| `D,<seq>,<t_ms>,<mm_f>,<mm_raw>,<st>,<btn>` | Range sample |
| `B,<id>,<t_ms>,<edge>` | Button edge (`1` press, `0` release) |
| `S,<hz>,<invalid_pct>,<i2c_ok>,<link_ok>` | 1 Hz health |
| `E,<code>,<msg>` | Error |

- `st = 0` means a valid VL53L0X measurement. Any other value must be
  discarded by the host.
- `btn` bit 0 = GPIO32 (narration), bit 1 = GPIO33 (currency/OCR).
- `mm_f` is the filtered value; `mm_raw` is retained for calibration.

---

## Ranging profile

| Setting | Value | Why |
|---|---|---|
| Timing budget | 33 ms | ~30–33 Hz, required for long-range VCSEL pulse periods |
| Signal rate limit | 0.10 MCPS | High sensitivity long-range profile up to 2.0 m |
| VCSEL pre-range | 18 | Longer laser pulse for maximum distance |
| VCSEL final-range | 14 | Longer laser pulse for maximum distance |
| Validity window | 30–2000 mm | Reject floor/no-target artefacts |
| Timeout recovery | Auto-clear reg `0x0B` | Prevents hardware interrupt state machine lockup |
| Filter | median-of-5 + EMA (α 0.3) | Suppress speckle without lag |

Range is roughly **1.2–2.0 m** depending on target reflectivity and ambient
light. Black fabric and direct sunlight reduce it.

---

## Build & Diagnostic Tools

PlatformIO:

```bash
pio run -t upload
pio device monitor -b 921600
```

Arduino IDE: board **ESP32 Dev Module**, library **VL53L0X by Pololu**,
upload speed 921600.

### Host Diagnostic Utilities:

- **Live Distance Gauge**: `python3 scripts/monitor_tof.py` (real-time ASCII distance bar)
- **Link & Button Probe**: `python3 scripts/probe_esp32.py --duration 10`
- **Zero-Offset Calibration**: `python3 scripts/calibrate_tof.py --distance-mm 500 --samples 100 --save`

---

## Tuning

Constants live at the top of `drishti_tof.ino`:

- `TIMING_BUDGET_MS` — 20 ms default; 33 ms improves accuracy, 10 ms improves rate.
- `DEBOUNCE_MS` — 30 ms default.
- `HOST_TIMEOUT_MS` — 2000 ms; sets `link_ok`.
- `MIN_VALID_MM` / `MAX_VALID_MM` — validity window.

---

## Notes

- WiFi and Bluetooth are never started, so the ranging loop stays jitter-free.
- The MCU keeps ranging even if the host disappears; `link_ok` reports the
  heartbeat state.
- **Single sensor:** the VL53L0X is a single-ray rangefinder. For a second
  sensor (e.g. downward for drop-offs) use `XSHUT` to assign a different I2C
  address at boot, then extend the `D` line with a sensor id.
- The 33 ms timing budget is not an end-to-end latency guarantee. Measure MCU timestamp to physical audio/haptic onset on the assembled system.
- Product firmware will require authenticated/rollback-capable update design, watchdog/fault latching, production test, version traceability, and a documented safe state.
