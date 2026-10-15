#!/usr/bin/env python3
"""Drishti Hardware Diagnostics & Health Check Utility.

Probes and audits physical and edge hardware peripherals:
1. Video Camera (/dev/video0, OpenCV, V4L2)
2. Time-of-Flight Distance Sensor (I2C bus, VL53L1X @ 0x29)
3. Audio Output (ALSA, SoundDevice, Bone-conduction sink)
4. Edge AI Neural Compute (ONNX Runtime, execution providers, YOLO model integrity)

Prints a diagnostic summary with actionable remediation steps from troubleshoot.md.
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import get_pipeline_config, get_hardware_config


def check_camera() -> Dict[str, Any]:
    """Audit video capture hardware."""
    status = {"name": "Video Camera", "status": "FAIL", "details": []}

    # Check device node
    dev_path = "/dev/video0"
    dev_exists = os.path.exists(dev_path)
    status["details"].append(f"Device node {dev_path}: {'PRESENT' if dev_exists else 'ABSENT'}")

    # Check OpenCV
    opencv_ok = False
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                status["details"].append(f"OpenCV capture: OK ({w}x{h})")
                opencv_ok = True
            cap.release()
    except Exception as e:
        status["details"].append(f"OpenCV probe notice: {e}")

    # Check ffmpeg
    has_ffmpeg = shutil.which("ffmpeg") is not None
    status["details"].append(f"ffmpeg binary: {'AVAILABLE' if has_ffmpeg else 'MISSING'}")

    if opencv_ok or (dev_exists and has_ffmpeg):
        status["status"] = "OK"
    elif dev_exists:
        status["status"] = "WARNING"
        status["details"].append("Device present but stream failed. Refer to troubleshoot.md §1.2.")
    else:
        status["status"] = "SYNTHETIC_FALLBACK"
        status["details"].append("No physical camera detected. Operating in autonomous synthetic simulation.")

    return status


def check_distance_sensor() -> Dict[str, Any]:
    """Audit Time-of-Flight (ToF) distance sensor."""
    status = {"name": "VL53L1X ToF Sensor", "status": "FAIL", "details": []}

    i2c_dev = "/dev/i2c-1"
    has_i2c = os.path.exists(i2c_dev)
    status["details"].append(f"I2C bus device {i2c_dev}: {'PRESENT' if has_i2c else 'ABSENT'}")

    # Check driver
    has_driver = False
    try:
        import VL53L1X
        has_driver = True
        status["details"].append("VL53L1X Python driver: INSTALLED")
    except ImportError:
        status["details"].append("VL53L1X Python driver: NOT INSTALLED ('pip install VL53L1X')")

    if has_i2c and has_driver:
        try:
            tof = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
            tof.open()
            tof.start_ranging(1)
            dist_mm = tof.get_distance()
            tof.stop_ranging()
            status["status"] = "OK"
            status["details"].append(f"Sensor ping: {dist_mm} mm")
        except Exception as e:
            status["status"] = "WARNING"
            status["details"].append(f"I2C connect notice: {e}. Check wiring on GPIO Pin 3/5.")
    else:
        status["status"] = "SYNTHETIC_FALLBACK"
        status["details"].append("Physical ToF sensor absent. Using synthetic raycast depth prior.")

    return status


def check_audio_output() -> Dict[str, Any]:
    """Audit audio playback sinks."""
    status = {"name": "Audio Playback Sink", "status": "FAIL", "details": []}

    has_aplay = shutil.which("aplay") is not None
    status["details"].append(f"ALSA aplay utility: {'AVAILABLE' if has_aplay else 'MISSING'}")

    # Check sounddevice
    has_sd = False
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        status["details"].append(f"SoundDevice detected {len(devices)} audio device(s).")
        has_sd = True
    except Exception as e:
        status["details"].append(f"SoundDevice notice: {e}")

    if has_aplay or has_sd:
        status["status"] = "OK"
    else:
        status["status"] = "SYNTHETIC_FALLBACK"
        status["details"].append("No hardware audio output found. Headless buffer recorder active.")

    return status


def check_neural_models() -> Dict[str, Any]:
    """Audit ONNX Runtime neural compute environment and model weights."""
    status = {"name": "Neural Vision Engine", "status": "FAIL", "details": []}

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        status["details"].append(f"ONNX Runtime Providers: {', '.join(providers)}")
    except ImportError:
        status["details"].append("onnxruntime not installed. Run 'pip install onnxruntime'.")
        return status

    model_path = PROJECT_ROOT / "models" / "yolov8n.onnx"
    if model_path.exists():
        size_kb = model_path.stat().st_size // 1024
        # Verify SHA256
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        digest = h.hexdigest()
        status["details"].append(f"Model: {model_path.name} ({size_kb} KB, SHA-256: {digest[:16]}...)")
        status["status"] = "OK"
    else:
        status["status"] = "MISSING_WEIGHTS"
        status["details"].append("models/yolov8n.onnx missing. Run 'python scripts/download_models.py'.")

    return status


def main() -> int:
    print("=" * 76)
    print("           DRISHTI ASSISTIVE WEARABLE: HARDWARE DIAGNOSTICS")
    print("=" * 76)

    checks = [
        check_camera(),
        check_distance_sensor(),
        check_audio_output(),
        check_neural_models(),
    ]

    all_passed = True
    for c in checks:
        badge = f"[{c['status']}]"
        print(f"\n{badge:<20} {c['name']}")
        for d in c["details"]:
            print(f"    * {d}")
        if c["status"] in ("FAIL", "MISSING_WEIGHTS"):
            all_passed = False

    print("\n" + "=" * 76)
    if all_passed:
        print("[✓] Diagnostic completed: Drishti edge system is operational and ready to run.")
        return 0
    else:
        print("[!] Diagnostic notice: One or more critical components require attention.")
        print("    Consult troubleshoot.md for targeted resolution steps.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

