#!/usr/bin/env python3
"""Drishti Live Wearable Runner: Autonomous Edge Operation.

Production entrypoint for continuous wearable execution on Raspberry Pi 5 / Laptop.
Runs all capabilities concurrently:
1. Continuous sub-100ms 3D spatial obstacle avoidance via binaural audio.
2. Continuous Indian currency note recognition (auto-announces banknotes in view).
3. On-demand multilingual Indic OCR & local VLM scene narration via tactile triggers / keys.
4. Local 2D Gaussian face redaction privacy filter.
5. Caregiver MQTT telemetry with QoS 1 and Last Will and Testament.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import os
import time
import signal
import argparse
import threading
from typing import Optional, Dict, Any

from hal.camera import BaseCameraSource, V4L2CameraSource, SyntheticCameraSource, OpenCVCameraSource, create_camera_source
from hal.distance_sensor import BaseDistanceSensor, VL53L1XSensor, SyntheticDistanceSensor
from hal.trigger_input import BaseTriggerInput, GPIOPushButtonTrigger, SyntheticTriggerInput
from hal.audio_output import BaseAudioSink, ALSAAudioSink, HeadlessAudioRecorder
from safety.telemetry import TelemetryClient
from pipeline import DrishtiMasterPipeline
from config.config_loader import get_pipeline_config, get_hardware_config


class KeyboardTriggerThread(threading.Thread):
    """Listens for keyboard commands in terminal to trigger on-demand features on PC."""

    def __init__(self, trigger_input: SyntheticTriggerInput) -> None:
        super().__init__(daemon=True)
        self.trigger_input = trigger_input
        self.stop_requested = False

    def run(self) -> None:
        while not self.stop_requested:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                if cmd in ("n", "narrate", "1"):
                    print("\n[KEYBOARD TRIGGER] Injected Button 1 (Scene Narration)")
                    self.trigger_input.inject_trigger(button_id=1)
                elif cmd in ("c", "ocr", "currency", "2"):
                    print("\n[KEYBOARD TRIGGER] Injected Button 2 (Currency / OCR)")
                    self.trigger_input.inject_trigger(button_id=2)
                elif cmd in ("q", "quit", "exit"):
                    print("\n[KEYBOARD] Exit requested by user.")
                    os.kill(os.getpid(), signal.SIGINT)
            except Exception:
                break


def is_raspberry_pi() -> bool:
    """Detect whether host environment is a Raspberry Pi."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        return ("Raspberry Pi" in cpuinfo) or ("BCM" in cpuinfo)
    except Exception:
        return False


def build_hardware_stack(hardware_mode: str, hw_cfg: Optional[Dict[str, Any]] = None):
    """Instantiate appropriate HAL drivers based on host environment and config files."""
    if hardware_mode == "auto":
        hardware_mode = "rpi" if is_raspberry_pi() else "laptop"

    hw_cfg = hw_cfg or get_hardware_config(profile=hardware_mode)
    print(f"[*] Target Architecture Selected: [{hardware_mode.upper()}]")

    cam_cfg = hw_cfg.get("camera", {})
    dist_cfg = hw_cfg.get("distance_sensor", {})
    trig_cfg = hw_cfg.get("trigger_input", {})
    audio_cfg = hw_cfg.get("audio_output", {})

    width = cam_cfg.get("capture_width", 640)
    height = cam_cfg.get("capture_height", 480)
    fps = cam_cfg.get("fps", 30)

    if hardware_mode == "rpi":
        # Raspberry Pi 5 physical peripheral stack
        print("    -> Camera: Initializing Camera Module (with automatic fallback)")
        camera = create_camera_source(backend="auto", device_path="/dev/video0", width=width, height=height, fps=fps)

        i2c_bus = dist_cfg.get("i2c_bus", 1)
        i2c_addr = dist_cfg.get("i2c_address", 0x29)
        print(f"    -> Distance: ST VL53L1X Time-of-Flight (I2C Bus {i2c_bus}, Address 0x{i2c_addr:02X})")
        try:
            distance_sensor: BaseDistanceSensor = VL53L1XSensor(i2c_bus=i2c_bus, i2c_address=i2c_addr)
        except Exception as e:
            print(f"    [!] VL53L1X I2C notice ({e}). Falling back to synthetic depth prior.")
            distance_sensor = SyntheticDistanceSensor(base_distance_m=1.75)

        pin_b1 = trig_cfg.get("pin_button_narration", 17)
        pin_b2 = trig_cfg.get("pin_button_currency_ocr", 27)
        print(f"    -> Tactile Input: Push Buttons on GPIO {pin_b1} (Narration) and GPIO {pin_b2} (Currency/OCR)")
        try:
            trigger_input: BaseTriggerInput = GPIOPushButtonTrigger(pin_button_1=pin_b1, pin_button_2=pin_b2)
        except Exception:
            print("    [!] GPIO library unavailable. Falling back to synthetic trigger input.")
            trigger_input = SyntheticTriggerInput()

        alsa_dev = audio_cfg.get("device_name", "default")
        print(f"    -> Audio Sink: ALSA direct device {alsa_dev} (Bone-Conduction Transducer)")
        try:
            audio_sink: BaseAudioSink = ALSAAudioSink(device=alsa_dev)
        except Exception:
            print("    [!] ALSA audio device unavailable. Falling back to default audio sink.")
            audio_sink = ALSAAudioSink(device="default")

    elif hardware_mode == "laptop":
        # Workstation / Laptop developer stack
        print("    -> Camera: Workstation webcam on /dev/video0 (with automatic fallback)")
        camera = create_camera_source(backend="auto", device_path="/dev/video0", width=width, height=height, fps=fps)

        print("    -> Distance: Synthetic raycast distance simulator")
        distance_sensor = SyntheticDistanceSensor(base_distance_m=1.75)

        print("    -> Tactile Input: Interactive Terminal Keyboard (N: Narration, C: Currency/OCR, Q: Quit)")
        synthetic_trigger = SyntheticTriggerInput()
        trigger_input = synthetic_trigger
        kb_thread = KeyboardTriggerThread(synthetic_trigger)
        kb_thread.start()

        alsa_dev = audio_cfg.get("device_name", "default")
        print(f"    -> Audio Sink: Audio output on {alsa_dev}")
        try:
            audio_sink = ALSAAudioSink(device=alsa_dev)
        except Exception:
            print(f"    [!] ALSA {alsa_dev} unavailable. Using headless recorder.")
            audio_sink = HeadlessAudioRecorder()

    else:
        # Pure headless synthetic simulation mode
        print("    -> Operating in deterministic headless simulation mode.")
        camera = SyntheticCameraSource(width=width, height=height, fps=fps)
        distance_sensor = SyntheticDistanceSensor(base_distance_m=1.75)
        trigger_input = SyntheticTriggerInput()
        audio_sink = HeadlessAudioRecorder()

    return camera, distance_sensor, trigger_input, audio_sink


def main() -> None:
    parser = argparse.ArgumentParser(description="Drishti Live Continuous Wearable Runner")
    parser.add_argument(
        "--hardware",
        choices=["auto", "laptop", "rpi", "synthetic"],
        default="auto",
        help="Target hardware execution profile (default: auto)",
    )
    parser.add_argument(
        "--continuous-currency",
        action="store_true",
        default=False,
        help="Enable continuous automatic banknote recognition in background",
    )
    parser.add_argument(
        "--currency-cooldown",
        type=float,
        default=3.5,
        help="Seconds to wait between repeated announcements of the same banknote",
    )
    parser.add_argument(
        "--currency-interval",
        type=int,
        default=8,
        help="Frames between continuous currency evaluations (default: 8 frames, ~3-4 Hz)",
    )
    parser.add_argument(
        "--no-speech",
        action="store_true",
        help="Disable verbal speech announcements",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional maximum number of perception cycles to execute before exiting (default: infinite)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("        DRISHTI ASSISTIVE WEARABLE: AUTONOMOUS LIVE SYSTEM")
    print("=" * 80)

    # 1. Load Centralized Configurations
    pipeline_cfg = get_pipeline_config()
    hw_cfg = get_hardware_config(profile=args.hardware)

    # 2. Initialize HAL stack from config
    camera, distance_sensor, trigger_input, audio_sink = build_hardware_stack(args.hardware, hw_cfg=hw_cfg)
    
    # 3. Configure and Connect MQTT Telemetry
    telem_cfg = pipeline_cfg.get("safety", {}).get("telemetry", {})
    telemetry = TelemetryClient(
        device_id="drishti-live-wearable",
        broker_host=telem_cfg.get("broker_host", "localhost"),
        broker_port=telem_cfg.get("broker_port", 1883),
        use_tls=telem_cfg.get("use_tls", False),
        qos=telem_cfg.get("qos", 1),
    )
    is_telem_connected = telemetry.connect()
    if is_telem_connected:
        print(f"[*] Telemetry: Connected to Caregiver Gateway at {telemetry.broker_host}:{telemetry.broker_port}")
    else:
        print(f"[*] Telemetry: Running offline buffer (broker at {telemetry.broker_host}:{telemetry.broker_port} unavailable)")

    # 4. Build Master Pipeline
    pipeline = DrishtiMasterPipeline(
        camera=camera,
        distance_sensor=distance_sensor,
        trigger_input=trigger_input,
        audio_sink=audio_sink,
        telemetry_client=telemetry,
        continuous_currency=args.continuous_currency,
        currency_interval=args.currency_interval,
        currency_cooldown_s=args.currency_cooldown,
    )

    # 5. Setup Graceful Shutdown Handler
    running = True

    def sigint_handler(signum, frame):
        nonlocal running
        print("\n\n[*] Interrupt signal received. Initiating graceful shutdown...")
        running = False

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    # 4. Startup Vocal Announcement
    if not args.no_speech:
        print("[*] Broadcasting startup chime...")
        greeting = "Drishti online. Continuous obstacle guidance active. Press C or Button 2 to identify currency."
        if args.continuous_currency:
            greeting = "Drishti online. Continuous obstacle guidance and currency detection active."
        pipeline.speech_service.speak(greeting, play_audio=True)

    print("\n" + "=" * 80)
    print("  SYSTEM RUNNING CONTINUOUSLY (Press Ctrl+C to terminate)")
    print("  Controls: [N + Enter] = Narrate Scene | [C + Enter] = Identify Note/OCR | [Q + Enter] = Quit")
    print("=" * 80 + "\n")


    cycle_count = 0
    t_start = time.time()

    try:
        while running and (args.max_cycles is None or cycle_count < args.max_cycles):
            cycle_count += 1
            res = pipeline.step()
            fast = res.get("fast_perception", {})

            # Format real-time hazard status
            if fast.get("success"):
                hazard = fast.get("primary_hazard")
                lat = fast["latency_ms"]["total_cycle"]
                fps = 1000.0 / max(lat, 1.0)

                if hazard:
                    cls_name = hazard["class_name"]
                    dist = f"{hazard['distance_m']:.2f} m"
                    az = f"{hazard['azimuth_deg']:+.1f}°"
                    coords = f"[{hazard['coords_3d_m'][0]:+.2f}, {hazard['coords_3d_m'][1]:+.2f}, {hazard['coords_3d_m'][2]:.2f}]"
                    line = f"Cycle {cycle_count:<5} | {cls_name:<9} | {dist:<8} | {az:<7} | 3D: {coords:<20} | {lat:5.1f} ms ({fps:4.1f} FPS)"
                else:
                    line = f"Cycle {cycle_count:<5} | Path Clear| --       | --      | 3D: --                   | {lat:5.1f} ms ({fps:4.1f} FPS)"

                print(line)

            # Continuous currency detected
            if "continuous_currency" in res:
                c = res["continuous_currency"]
                print(f"  >>> [CURRENCY ANNOUNCED] Spoken: \"{c['spoken_name']}\" (Confidence: {c['confidence']:.2f})")

            # Tactile Button 1 Narration event
            if "narration" in res:
                print(f"  >>> [VLM NARRATION] Spoken: \"{res['narration']}\"")

            # Tactile Button 2 Currency / OCR event
            if "currency" in res:
                c = res["currency"]
                print(f"  >>> [TACTILE CURRENCY] Spoken: \"{c['spoken_name']}\" (Confidence: {c['confidence']:.2f})")
            elif "ocr" in res:
                print(f"  >>> [TACTILE OCR] Spoken: \"{res['ocr']['text']}\"")

            # 30 FPS pacing yield
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[*] Releasing camera, sensors, and telemetry sockets...")
        if not args.no_speech:
            pipeline.speech_service.speak("Drishti powering down.", play_audio=True)
        pipeline.shutdown()
        total_time = time.time() - t_start
        mean_fps = cycle_count / max(total_time, 0.001)
        print(f"[✓] Shutdown complete. Executed {cycle_count} cycles in {total_time:.1f}s (Average {mean_fps:.1f} FPS).")


if __name__ == "__main__":
    main()

