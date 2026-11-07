#!/usr/bin/env python3
"""Zero-Offset Calibration Tool for Drishti Time-of-Flight (VL53L0X / VL53L1X) Sensor.

This script guides the user through real-world zero-offset calibration:
1. Place the sensor at a precisely measured distance (e.g., 500 mm) perpendicular to a flat, matte white wall or target.
2. The tool captures N valid ranging samples from the ESP32 link.
3. Computes the systematic error:
       offset_mm = ground_truth_mm - mean_raw_mm
4. Optionally writes the calibrated offset directly to `config/hardware_dev.yaml`
   and `config/hardware_pi5.yaml`.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import argparse
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import yaml

from hal.esp32_link import Esp32Link, TofSample


def collect_calibration_samples(
    link: Esp32Link,
    target_count: int = 100,
    timeout_s: float = 10.0,
) -> Tuple[List[int], int]:
    """Collect raw millimeter samples from the ESP32 link.

    Returns:
        Tuple[List[int], int]: (valid_raw_samples, invalid_status_count)
    """
    samples: List[int] = []
    invalid_count = 0
    start_time = time.time()
    last_seq: Optional[int] = None

    while len(samples) < target_count and (time.time() - start_time) < timeout_s:
        frame = link.latest()
        if frame is None or frame.seq == last_seq:
            time.sleep(0.01)
            continue

        last_seq = frame.seq
        if frame.status == 0 and frame.mm_raw > 0:
            samples.append(frame.mm_raw)
            pct = int((len(samples) / target_count) * 100)
            sys.stdout.write(f"\r[*] Collecting samples: {len(samples)}/{target_count} [{pct}%] (raw: {frame.mm_raw} mm)")
            sys.stdout.flush()
        else:
            invalid_count += 1

    sys.stdout.write("\n")
    return samples, invalid_count


def compute_calibration_stats(
    samples: List[int],
    ground_truth_mm: int,
) -> Dict[str, float]:
    """Compute statistical distribution and recommended calibration offset."""
    if not samples:
        raise ValueError("No valid calibration samples collected.")

    arr = np.array(samples, dtype=np.float64)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    recommended_offset = int(round(ground_truth_mm - mean_val))

    return {
        "ground_truth_mm": float(ground_truth_mm),
        "sample_count": float(len(samples)),
        "mean_raw_mm": round(mean_val, 2),
        "median_raw_mm": round(median_val, 2),
        "std_dev_mm": round(std_val, 2),
        "min_raw_mm": round(min_val, 2),
        "max_raw_mm": round(max_val, 2),
        "recommended_offset_mm": float(recommended_offset),
    }


def update_hardware_config_offset(config_path: Path, offset_mm: int) -> bool:
    """Update offset_mm under distance_sensor in a YAML configuration file."""
    if not config_path.exists():
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if "distance_sensor" not in cfg:
            cfg["distance_sensor"] = {}

        cfg["distance_sensor"]["offset_mm"] = int(offset_mm)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        return True
    except Exception as e:
        print(f"[!] Error updating {config_path}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-Offset Calibration Tool for Drishti ToF Sensor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        default="auto",
        help="ESP32 serial port (/dev/ttyUSB0, /dev/ttyACM0, /dev/drishti-tof, or auto)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
        help="Serial baud rate matching drishti_tof.ino",
    )
    parser.add_argument(
        "--distance-mm",
        type=int,
        default=500,
        help="Physical ground truth distance from sensor lens to target wall in mm",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of valid ranging samples to collect for statistical averaging",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Automatically save recommended offset_mm to config/hardware_dev.yaml",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save recommended offset_mm to both hardware_dev.yaml and hardware_pi5.yaml",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate calibration run with synthetic noisy readings for testing",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("      DRISHTI ToF ZERO-OFFSET CALIBRATION UTILITY")
    print("=" * 70)
    print(f"Target Ground Truth Distance: {args.distance_mm} mm ({args.distance_mm / 1000.0:.2f} m)")
    print(f"Sample Count Budget:         {args.samples} frames")
    print(f"Target Port:                 {args.port}")
    print("-" * 70)

    if args.simulate:
        print("[*] Running in SIMULATION mode (generating synthetic noisy readings)...")
        # Simulate sensor with a +18 mm physical offset and +/- 6mm jitter
        sim_offset = 18
        raw_samples = [
            int(round(args.distance_mm - sim_offset + np.random.normal(0, 5)))
            for _ in range(args.samples)
        ]
        invalid_count = 0
    else:
        print("[*] Connecting to ESP32 sensor link...")
        link = Esp32Link(port=args.port, baud=args.baud)
        link.start()

        # Wait for link to stabilize
        print("[*] Waiting for active ranging stream from ESP32...")
        t0 = time.time()
        while time.time() - t0 < 3.0 and link.latest() is None:
            time.sleep(0.05)

        if link.latest() is None:
            link.stop()
            print("[!] ERROR: No ranging data received from ESP32.")
            print("    Check: Is the ESP32 connected via a USB data cable?")
            print("    Check: Did you flash firmware/drishti_tof/drishti_tof.ino?")
            print("    Try:   python3 scripts/probe_esp32.py --port " + args.port)
            sys.exit(1)

        print("[✓] ESP32 stream active. Hold sensor steady pointing perpendicular to target.")
        time.sleep(0.5)

        try:
            raw_samples, invalid_count = collect_calibration_samples(
                link, target_count=args.samples, timeout_s=12.0
            )
        finally:
            link.stop()

    if not raw_samples:
        print("[!] ERROR: Failed to collect any valid calibration samples.")
        sys.exit(1)

    # Compute statistics
    stats = compute_calibration_stats(raw_samples, ground_truth_mm=args.distance_mm)

    print("\n" + "=" * 70)
    print("                 CALIBRATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Ground Truth:          {stats['ground_truth_mm']:>8.1f} mm")
    print(f"  Samples Analyzed:      {int(stats['sample_count']):>8d} frames")
    print(f"  Invalid / Dropped:     {invalid_count:>8d} frames")
    print(f"  Mean Raw Measurement:  {stats['mean_raw_mm']:>8.2f} mm")
    print(f"  Median Raw Measurement:{stats['median_raw_mm']:>8.2f} mm")
    print(f"  Std Deviation (Noise): {stats['std_dev_mm']:>8.2f} mm")
    print(f"  Min / Max Measured:    {stats['min_raw_mm']:>8.1f} / {stats['max_raw_mm']:>8.1f} mm")
    print("-" * 70)

    recommended_offset = int(stats["recommended_offset_mm"])
    sign = "+" if recommended_offset >= 0 else ""
    print(f"  >>> RECOMMENDED OFFSET:  {sign}{recommended_offset} mm")
    print("=" * 70)

    if args.save or args.save_all:
        dev_cfg = root_dir / "config" / "hardware_dev.yaml"
        if update_hardware_config_offset(dev_cfg, recommended_offset):
            print(f"[✓] Saved offset_mm: {recommended_offset} to {dev_cfg.relative_to(root_dir)}")

    if args.save_all:
        pi_cfg = root_dir / "config" / "hardware_pi5.yaml"
        if update_hardware_config_offset(pi_cfg, recommended_offset):
            print(f"[✓] Saved offset_mm: {recommended_offset} to {pi_cfg.relative_to(root_dir)}")

    if not (args.save or args.save_all):
        print("\nTo apply this offset, add it to your configuration file:")
        print(f"  distance_sensor:\n    offset_mm: {recommended_offset}")
        print("Or re-run with --save to write automatically.")


if __name__ == "__main__":
    main()
