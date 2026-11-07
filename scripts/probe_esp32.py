#!/usr/bin/env python3
"""Probe the Drishti ESP32 sensor MCU over USB serial.

Prints live range samples, button events, link rate and health counters.
Use this before running the full pipeline.

Examples::

    python scripts/probe_esp32.py
    python scripts/probe_esp32.py --port /dev/ttyUSB0 --duration 5
    python scripts/probe_esp32.py --port /tmp/drishti-tof   # fake ESP32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hal.esp32_link import Esp32Link  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the Drishti ESP32 sensor MCU")
    parser.add_argument("--port", default="auto", help="serial port (default: auto-detect)")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--duration", type=float, default=5.0, help="seconds to observe")
    parser.add_argument("--stale-ms", type=float, default=120.0)
    args = parser.parse_args()

    link = Esp32Link(port=args.port, baud=args.baud, stale_ms=args.stale_ms)
    print(f"[*] Opening ESP32 link (port={args.port}, baud={args.baud}) ...")
    link.start()

    start = time.monotonic()
    last_samples = 0
    try:
        while time.monotonic() - start < args.duration:
            time.sleep(0.5)
            status = link.status()
            sample = link.latest()
            if sample is None:
                print(
                    f"  waiting... connected={status.connected} port={status.port or '-'} "
                    f"error={status.last_error or '-'}"
                )
                continue

            distance_m = sample.mm_filtered / 1000.0
            age = link.age_ms()
            print(
                f"  seq={sample.seq:<6} {distance_m:5.3f} m  raw={sample.mm_raw:4d} mm  "
                f"status={sample.status}  age={age:5.1f} ms  "
                f"rate={status.hz:5.1f} Hz  drops={status.seq_drops}"
            )

            for event in link.pop_button_events():
                print(f"    >>> button {event.button_id} pressed")

            if status.samples < last_samples:
                print("    [!] sample counter reset (reconnect)")
            last_samples = status.samples

            if sample.status != 0:
                print(f"    [!] invalid range status {sample.status}")

    except KeyboardInterrupt:
        pass
    finally:
        link.stop()
        status = link.status()

    print("\n[*] Summary")
    print(f"    port        : {status.port or '-'}")
    print(f"    firmware    : {status.firmware or '-'}  sensor: {status.sensor_model or '-'}")
    print(f"    connected   : {status.connected}")
    print(f"    samples     : {status.samples}")
    print(f"    rate        : {status.hz:.1f} Hz")
    print(f"    invalid     : {status.invalid_pct:.1f} %")
    print(f"    seq drops   : {status.seq_drops}")
    print(f"    reconnects  : {status.reconnects}")
    if status.last_error:
        print(f"    last error  : {status.last_error}")

    if status.samples == 0:
        print("\n[!] No samples received. Check wiring, firmware, port and baud rate.")
        return 1
    print("\n[✓] ESP32 link is streaming.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
