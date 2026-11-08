#!/usr/bin/env python3
"""Interactive Real-Time Time-of-Flight (VL53L0X) Sensor Monitor.

Provides a live, high-refresh-rate visual dashboard in the terminal
showing distance (mm / cm / m / inches), a dynamic distance bar,
target reflection status, and ranging statistics.

Usage:
    python scripts/monitor_tof.py
    python scripts/monitor_tof.py --port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hal.esp32_link import Esp32Link


def build_bar(distance_m: float, max_m: float = 2.0, width: int = 40) -> str:
    """Render a dynamic ASCII visual distance bar."""
    clamped = max(0.0, min(distance_m, max_m))
    fill = int(round((clamped / max_m) * width))
    bar = "█" * fill + "░" * (width - fill)
    return f"[{bar}] {distance_m:4.2f} m"


def main() -> int:
    parser = argparse.ArgumentParser(description="Live ToF Distance Sensor Monitor")
    parser.add_argument("--port", default="auto", help="serial port (default: auto-detect)")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--max-range", type=float, default=2.0, help="maximum display range in meters")
    args = parser.parse_args()

    link = Esp32Link(port=args.port, baud=args.baud)
    print(f"[*] Connecting to Time-of-Flight sensor on port={args.port}...")
    link.start()

    # Wait for first packet
    t0 = time.time()
    while time.time() - t0 < 3.0 and link.latest() is None:
        time.sleep(0.05)

    if link.latest() is None:
        link.stop()
        print("\n[!] ERROR: No data received from sensor. Check cable and power.")
        return 1

    print("\033[2J\033[H", end="")  # Clear screen and move to top
    print("=" * 72)
    print("              DRISHTI ToF (VL53L0X) REAL-TIME SENSOR MONITOR")
    print("=" * 72)
    print(" Move your hand or target in front of the sensor to see live response.")
    print(" Press Ctrl+C to exit.\n")

    min_mm = 99999
    max_mm = 0
    valid_samples = 0
    start_time = time.time()
    last_seq = None

    status_labels = {
        0: "\033[92m● TARGET DETECTED (VALID REFLECTION)\033[0m",
        1: "\033[93m○ OUT OF RANGE / NO REFLECTION (> 2.0 m)\033[0m",
        2: "\033[91m▲ TOO CLOSE (< 30 mm)\033[0m",
        254: "\033[91m✖ SENSOR TIMEOUT\033[0m",
    }

    try:
        while True:
            time.sleep(0.04)  # ~25 Hz terminal update rate
            sample = link.latest()
            status = link.status()

            if sample is None:
                continue

            if sample.seq != last_seq:
                last_seq = sample.seq
                if sample.status == 0 and sample.mm_raw > 0:
                    valid_samples += 1
                    min_mm = min(min_mm, sample.mm_raw)
                    max_mm = max(max_mm, sample.mm_raw)

            dist_m = sample.mm_filtered / 1000.0
            dist_cm = sample.mm_filtered / 10.0
            dist_in = sample.mm_filtered / 25.4
            bar = build_bar(dist_m, max_m=args.max_range, width=36)
            status_text = status_labels.get(sample.status, f"Status: {sample.status}")

            # Overwrite in-place using ANSI escape sequences
            sys.stdout.write("\033[7;1H")  # Move cursor to line 7
            sys.stdout.write(f"  \033[1mTarget Status   :\033[0m  {status_text}\033[K\n")
            sys.stdout.write(f"  \033[1mRaw Millimeters :\033[0m  {sample.mm_raw:5d} mm\033[K\n")
            sys.stdout.write(f"  \033[1mFiltered Metric :\033[0m  {sample.mm_filtered:5d} mm  ({dist_cm:5.1f} cm  /  {dist_m:4.2f} m)\033[K\n")
            sys.stdout.write(f"  \033[1mImperial Units  :\033[0m  {dist_in:5.1f} inches\033[K\n")
            sys.stdout.write(f"  \033[1mVisual Gauge    :\033[0m  {bar}\033[K\n\n")

            sys.stdout.write(f"  \033[90mLink Rate: {status.hz:4.1f} Hz | Samples: {status.samples} | Seq: {sample.seq} | Drops: {status.seq_drops}\033[0m\033[K\n")
            if min_mm < 99999:
                sys.stdout.write(f"  \033[90mSession Stats: Min = {min_mm} mm ({min_mm/10:.1f} cm), Max = {max_mm} mm ({max_mm/10:.1f} cm)\033[0m\033[K\n")
            sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        link.stop()
        print("\n\n" + "=" * 72)
        print("                         MONITOR SUMMARY")
        print("=" * 72)
        elapsed = time.time() - start_time
        print(f"  Duration         : {elapsed:.1f} seconds")
        print(f"  Valid Samples    : {valid_samples}")
        if min_mm < 99999:
            print(f"  Closest Reading  : {min_mm} mm ({min_mm / 10.0:.1f} cm)")
            print(f"  Furthest Reading : {max_mm} mm ({max_mm / 10.0:.1f} cm)")
        print("=" * 72 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
