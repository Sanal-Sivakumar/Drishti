#!/usr/bin/env python3
"""Virtual Drishti ESP32 sensor MCU over a pseudo-terminal.

This lets the entire software stack (link, distance sensor, buttons,
fusion, viewer, pipeline) be built and tested **before** the physical
ESP32 breadboard exists. It emits exactly the same CSV protocol as
``firmware/drishti_tof/drishti_tof.ino``.

Usage::

    python tools/fake_esp32.py
    # -> /dev/pts/7
    python scripts/run_live_wearable.py --hardware laptop --serial-port /dev/pts/7

    # or create a stable symlink:
    python tools/fake_esp32.py --link /tmp/drishti-tof
    python scripts/run_live_wearable.py --hardware laptop --serial-port /tmp/drishti-tof

Scenarios:
    walk       slow approach/recede around 1.1 m (default)
    approach   steady approach from 2.0 m to 0.3 m, then repeats
    static     fixed distance (--distance)
"""

from __future__ import annotations

import argparse
import math
import os
import pty
import random
import select
import signal
import sys
import time
from typing import Optional


def _distance_at(scenario: str, t: float, static_distance: float) -> float:
    if scenario == "static":
        value = static_distance
    elif scenario == "approach":
        cycle = t % 20.0
        value = max(0.30, 2.0 - 0.085 * cycle)
    else:  # walk
        value = 1.10 + 0.70 * math.sin(2.0 * math.pi * 0.07 * t)
    return max(0.03, min(2.0, value))


def _write(master_fd: int, line: str) -> bool:
    try:
        os.write(master_fd, line.encode("ascii"))
        return True
    except OSError:
        # EIO until the host opens the slave side; harmless.
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Virtual ESP32 VL53L0X + buttons over a PTY")
    parser.add_argument("--hz", type=float, default=30.0, help="sample rate (default 30)")
    parser.add_argument("--scenario", choices=["walk", "approach", "static"], default="walk")
    parser.add_argument("--distance", type=float, default=1.0, help="static distance in metres")
    parser.add_argument("--press-every", type=float, default=5.0, help="seconds between fake button presses (0=off)")
    parser.add_argument("--invalid-every", type=float, default=20.0, help="seconds between invalid-status bursts (0=off)")
    parser.add_argument("--dropout-at", type=float, default=0.0, help="seconds before a simulated sample dropout (0=off)")
    parser.add_argument("--dropout-for", type=float, default=2.0, help="dropout duration in seconds")
    parser.add_argument("--link", default=None, help="optional stable symlink path, e.g. /tmp/drishti-tof")
    parser.add_argument("--quiet", action="store_true", help="suppress the stderr legend")
    args = parser.parse_args()

    master_fd, slave_fd = pty.openpty()
    slave_name = os.ttyname(slave_fd)

    if args.link:
        try:
            if os.path.islink(args.link) or os.path.exists(args.link):
                os.remove(args.link)
            os.symlink(slave_name, args.link)
        except OSError as exc:
            print(f"warning: could not create symlink {args.link}: {exc}", file=sys.stderr)

    # The path goes to stdout so shell scripts can capture it.
    print(slave_name, flush=True)
    if not args.quiet:
        print(
            f"[fake-esp32] {slave_name} @ {args.hz:.0f} Hz, scenario={args.scenario}. "
            f"Ctrl+C to stop.",
            file=sys.stderr,
            flush=True,
        )

    sequence = 0
    samples_since_status = 0
    last_status = time.monotonic()
    start = time.monotonic()
    next_sample = start
    next_press = start + args.press_every if args.press_every > 0 else float("inf")
    next_invalid = start + args.invalid_every if args.invalid_every > 0 else float("inf")
    next_dropout = start + args.dropout_at if args.dropout_at > 0 else float("inf")
    invalid_until = 0.0
    dropout_until = 0.0
    press_button = 1
    rng = random.Random(7)

    _write(master_fd, "I,1.0.0,VL53L0X-FAKE\n")
    announced = False

    running = True

    def _handle_sigint(signum, frame):  # noqa: ARG001
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        while running:
            now = time.monotonic()
            t = now - start

            # Drain host heartbeat bytes so the PTY never fills up. A late
            # host still gets the boot banner on its first heartbeat.
            try:
                readable, _, _ = select.select([master_fd], [], [], 0)
                if readable:
                    host_data = os.read(master_fd, 256)
                    if b"V" in host_data or b"v" in host_data:
                        _write(master_fd, "I,1.0.0,VL53L0X-FAKE\n")
                        announced = True
                    elif not announced and (b"P" in host_data or b"p" in host_data):
                        _write(master_fd, "I,1.0.0,VL53L0X-FAKE\n")
                        announced = True
            except OSError:
                pass

            if now >= next_dropout:
                dropout_until = now + args.dropout_for
                next_dropout = float("inf")  # one-shot dropout
            if dropout_until and now >= dropout_until:
                dropout_until = 0.0

            if now >= next_invalid:
                invalid_until = now + 0.15
                next_invalid = now + args.invalid_every
            if invalid_until and now >= invalid_until:
                invalid_until = 0.0

            if args.press_every > 0 and now >= next_press:
                press_line = f"B,{press_button},{int(t * 1000)},{1}\n"
                release_line = f"B,{press_button},{int(t * 1000)},{0}\n"
                if _write(master_fd, press_line) and _write(master_fd, release_line):
                    press_button = 2 if press_button == 1 else 1
                next_press = now + args.press_every

            if now >= next_sample and not dropout_until:
                raw = _distance_at(args.scenario, t, args.distance)
                mm = int(round(raw * 1000.0 + rng.uniform(-6.0, 6.0)))
                status = 4 if invalid_until else 0
                filtered = mm if status == 0 else 0
                next_sequence = sequence + 1
                line = f"D,{next_sequence},{int(t * 1000)},{filtered},{mm},{status},0\n"
                if _write(master_fd, line):
                    sequence = next_sequence
                    samples_since_status += 1
                next_sample += 1.0 / max(args.hz, 1.0)

                # If the scheduler slept through several slots, resync instead
                # of emitting a catch-up burst (real sensors do not buffer).
                if next_sample < now - 2.0 / max(args.hz, 1.0):
                    next_sample = now

            if now - last_status >= 1.0:
                elapsed = max(now - last_status, 1e-6)
                measured_hz = samples_since_status / elapsed
                invalid_pct = 20.0 if invalid_until else 0.0
                _write(master_fd, f"S,{measured_hz:.1f},{invalid_pct:.1f},1,1\n")
                samples_since_status = 0
                last_status = now

            sleep_for = min(next_sample - time.monotonic(), 0.05)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass
        try:
            os.close(slave_fd)
        except OSError:
            pass
        if args.link:
            try:
                os.remove(args.link)
            except OSError:
                pass
        print("\n[fake-esp32] stopped.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
