#!/usr/bin/env python3
"""Drishti Live Wearable Runner: Autonomous Edge Operation.

Production entrypoint for continuous wearable execution on Raspberry Pi 5 / Laptop.
Runs all capabilities concurrently:
1. Continuous sub-100ms 3D spatial obstacle avoidance via binaural audio.
2. On-demand Indian currency recognition / Indic OCR via tactile Button 2 or C
   (continuous currency scanning is opt-in, and off by default on laptop).
3. On-demand multilingual Indic OCR and sensor-grounded SmolVLM2 narration.
4. Local Haar-cascade face detection and Gaussian redaction.
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
from hal.distance_sensor import (
    BaseDistanceSensor,
    VL53L1XSensor,
    SyntheticDistanceSensor,
    ESP32DistanceSensor,
    UnavailableDistanceSensor,
)
from hal.trigger_input import (
    BaseTriggerInput,
    GPIOPushButtonTrigger,
    SyntheticTriggerInput,
    ESP32ButtonTrigger,
    CompositeTriggerInput,
)
from hal.audio_output import BaseAudioSink, ALSAAudioSink, HeadlessAudioRecorder
from perception.debug_viewer import DebugViewer
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
                elif cmd in ("c", "currency", "2"):
                    print("\n[KEYBOARD TRIGGER] Injected Button 2 (Currency Identification)")
                    self.trigger_input.inject_trigger(button_id=2)
                elif cmd in ("t", "text", "ocr", "read", "3"):
                    print("\n[KEYBOARD TRIGGER] Injected Button 3 (Text Reading)")
                    self.trigger_input.inject_trigger(button_id=3)
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


def build_esp32_distance_sensor(
    dist_cfg: Dict[str, Any], serial_port: Optional[str] = None
) -> ESP32DistanceSensor:
    """Build the ESP32/VL53L0X sensor from the hardware profile."""
    return ESP32DistanceSensor(
        port=serial_port or str(dist_cfg.get("serial_port", "auto")),
        baud=int(dist_cfg.get("baud", 921600)),
        stale_ms=float(dist_cfg.get("stale_ms", 120)),
        min_valid_mm=int(dist_cfg.get("min_valid_mm", 30)),
        max_valid_mm=int(dist_cfg.get("max_valid_mm", 2000)),
        offset_mm=int(dist_cfg.get("offset_mm", 0)),
    )


def build_hardware_stack(
    hardware_mode: str,
    hw_cfg: Optional[Dict[str, Any]] = None,
    serial_port: Optional[str] = None,
    camera_index: Optional[int] = None,
    no_esp32: bool = False,
    enable_keyboard: bool = False,
):
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
        print("    -> Camera: Initializing configured physical camera")
        camera = create_camera_source(
            backend=str(cam_cfg.get("backend", "libcamera")), device_path="/dev/video0",
            width=width, height=height, fps=fps, allow_synthetic_fallback=False,
        )

        dist_backend = str(dist_cfg.get("backend", "vl53l1x"))
        if serial_port is not None or dist_backend == "esp32_serial":
            requested_port = serial_port or str(dist_cfg.get("serial_port", "auto"))
            print(f"    -> Distance: ESP32 + VL53L0X over USB serial (port={requested_port})")
            esp_sensor = build_esp32_distance_sensor(dist_cfg, serial_port=serial_port)
            esp_sensor.start_ranging()

            deadline = time.time() + 2.5
            while time.time() < deadline and esp_sensor.link.latest() is None:
                time.sleep(0.05)

            if esp_sensor.link.latest() is None:
                status = esp_sensor.get_link_status()
                print(
                    f"    [!] No ToF frames from ESP32 ({status.last_error or 'timeout'}). "
                    "Navigation will remain NOT READY."
                )
                esp_sensor.stop_ranging()
                distance_sensor = UnavailableDistanceSensor()
                esp_trigger: Optional[ESP32ButtonTrigger] = None
            else:
                time.sleep(0.1)
                link_status = esp_sensor.get_link_status()
                print(
                    f"    [✓] ESP32 link on {link_status.port} | fw {link_status.firmware or '?'} "
                    f"| {link_status.hz:.1f} Hz | drops {link_status.seq_drops}"
                )
                esp_sensor.link.pop_button_events()
                distance_sensor = esp_sensor
                esp_trigger = ESP32ButtonTrigger(esp_sensor.link)

            if esp_trigger is not None:
                print("    -> Tactile Input: Physical ESP32 push buttons (Button 1: GPIO32, Button 2: GPIO33, Button 3: GPIO25)")
                trigger_input = esp_trigger
            else:
                print("    -> Tactile Input: Synthetic trigger (no physical buttons detected)")
                trigger_input = SyntheticTriggerInput()
        else:
            i2c_bus = dist_cfg.get("i2c_bus", 1)
            i2c_addr = dist_cfg.get("i2c_address", 0x29)
            print(f"    -> Distance: ST VL53L1X Time-of-Flight (I2C Bus {i2c_bus}, Address 0x{i2c_addr:02X})")
            try:
                candidate = VL53L1XSensor(i2c_bus=i2c_bus, i2c_address=i2c_addr)
                distance_sensor = candidate if candidate.is_available else UnavailableDistanceSensor()
            except Exception as e:
                print(f"    [!] VL53L1X I2C unavailable ({e}). Navigation will remain NOT READY.")
                distance_sensor = UnavailableDistanceSensor()

            pin_b1 = trig_cfg.get("pin_button_1", trig_cfg.get("pin_button_narration", 17))
            pin_b2 = trig_cfg.get("pin_button_2", trig_cfg.get("pin_button_currency", trig_cfg.get("pin_button_currency_ocr", 27)))
            pin_b3 = trig_cfg.get("pin_button_3", trig_cfg.get("pin_button_reading", 22))
            print(f"    -> Tactile Input: Push Buttons on GPIO {pin_b1} (Narration), GPIO {pin_b2} (Currency), GPIO {pin_b3} (Text)")
            try:
                trigger_input = GPIOPushButtonTrigger(
                    pin_button_1=pin_b1, pin_button_2=pin_b2, pin_button_3=pin_b3
                )
            except Exception:
                print("    [!] GPIO library unavailable. Falling back to synthetic trigger input.")
                trigger_input = SyntheticTriggerInput()

        if enable_keyboard:
            print("    -> Keyboard shortcuts enabled (N: Narration, C: Currency, T: Text, Q: Quit)")
            synthetic_trigger = SyntheticTriggerInput()
            kb_thread = KeyboardTriggerThread(synthetic_trigger)
            kb_thread.start()
            trigger_input = CompositeTriggerInput([trigger_input, synthetic_trigger])

        alsa_dev = audio_cfg.get("device_name", "default")
        print(f"    -> Audio Sink: ALSA direct device {alsa_dev} (Bone-Conduction Transducer)")
        try:
            audio_sink = ALSAAudioSink(device=alsa_dev)
        except Exception:
            print("    [!] ALSA audio device unavailable. Falling back to default audio sink.")
            audio_sink = ALSAAudioSink(device="default")

    elif hardware_mode == "laptop":
        # Workstation / Laptop developer stack
        requested_index = camera_index
        if requested_index is None:
            cfg_index = int(cam_cfg.get("device_index", -1))
            requested_index = cfg_index if cfg_index >= 0 else None

        print("    -> Camera: Workstation webcam (auto-probing for a colour camera)")
        camera = create_camera_source(
            backend=str(cam_cfg.get("backend", "auto")),
            device_path="/dev/video0",
            width=width,
            height=height,
            fps=fps,
            device_index=requested_index,
            allow_synthetic_fallback=False,
        )

        dist_backend = str(dist_cfg.get("backend", "synthetic"))
        if no_esp32:
            dist_backend = "synthetic"

        if (serial_port is not None or dist_backend == "esp32_serial") and not no_esp32:
            requested_port = serial_port or str(dist_cfg.get("serial_port", "auto"))
            print(f"    -> Distance: ESP32 + VL53L0X over USB serial (port={requested_port})")
            esp_sensor = build_esp32_distance_sensor(dist_cfg, serial_port=serial_port)
            esp_sensor.start_ranging()

            # Give the reader thread time to enumerate the port and receive frames.
            deadline = time.time() + 2.5
            while time.time() < deadline and esp_sensor.link.latest() is None:
                time.sleep(0.05)

            if esp_sensor.link.latest() is None:
                status = esp_sensor.get_link_status()
                print(
                    f"    [!] No ToF frames from ESP32 ({status.last_error or 'timeout'}). "
                    "Navigation will remain NOT READY."
                )
                esp_sensor.stop_ranging()
                distance_sensor: BaseDistanceSensor = UnavailableDistanceSensor()
                esp_trigger: Optional[ESP32ButtonTrigger] = None
            else:
                # If sensor arrived in timeout state (status=254), send soft reset to recover I2C
                latest = esp_sensor.link.latest()
                if latest is not None and latest.status == 254:
                    print("    [*] ESP32 sensor in timeout state; issuing auto-recovery reset...")
                    esp_sensor.link.request_reset()
                    time.sleep(0.5)

                # Let the boot banner arrive on the first heartbeat before printing.
                time.sleep(0.2)
                link_status = esp_sensor.get_link_status()
                print(
                    f"    [✓] ESP32 link on {link_status.port} | fw {link_status.firmware or '?'} "
                    f"| {link_status.hz:.1f} Hz | drops {link_status.seq_drops}"
                )
                # Discard presses that landed during the startup probe so the
                # first pipeline cycle does not fire a stale narration.
                esp_sensor.link.pop_button_events()
                distance_sensor = esp_sensor
                esp_trigger = ESP32ButtonTrigger(esp_sensor.link)
        else:
            print("    -> Distance: unavailable (use --hardware synthetic for labeled simulation)")
            distance_sensor = UnavailableDistanceSensor()
            esp_trigger = None

        if esp_trigger is not None:
            print("    -> Tactile Input: Physical ESP32 push buttons (Button 1: GPIO32, Button 2: GPIO33, Button 3: GPIO25)")
            trigger_input: BaseTriggerInput = esp_trigger
        else:
            print("    -> Tactile Input: Synthetic trigger (no physical buttons detected)")
            trigger_input = SyntheticTriggerInput()

        if enable_keyboard:
            print("    -> Keyboard shortcuts enabled (N: Narration, C: Currency, T: Text, Q: Quit)")
            synthetic_trigger = SyntheticTriggerInput()
            kb_thread = KeyboardTriggerThread(synthetic_trigger)
            kb_thread.start()
            trigger_input = CompositeTriggerInput([trigger_input, synthetic_trigger])

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


def build_viewer_status_lines(
    cycle_count: int,
    fast_result: Dict[str, Any],
    distance_sensor: BaseDistanceSensor,
) -> list:
    """Compose the HUD text for the debug viewer."""
    latency = fast_result.get("latency_ms", {}).get("total_cycle", 0.0)
    fps = 1000.0 / max(float(latency), 1.0)
    lines = [f"cycle {cycle_count} | {fps:.1f} FPS | {float(latency):.1f} ms"]

    if isinstance(distance_sensor, (ESP32DistanceSensor, VL53L1XSensor)):
        distance = distance_sensor.get_distance_meters()
        lines.append(f"ToF: {distance:.2f} m" if distance is not None else "ToF: -- (stale/invalid)")

    if isinstance(distance_sensor, ESP32DistanceSensor):
        link = distance_sensor.get_link_status()
        age = "inf" if link.age_ms == float("inf") else f"{link.age_ms:.0f}"
        lines.append(
            f"ESP32: {link.port or '-'} | {link.hz:.0f} Hz | age {age} ms | "
            f"drops {link.seq_drops} | invalid {link.invalid_pct:.1f}%"
        )

    hazard = fast_result.get("primary_hazard")
    if hazard:
        lines.append(
            f"PRIMARY: {hazard['class_name']} {hazard['distance_m']:.2f} m "
            f"az {hazard['azimuth_deg']:+.0f} deg | source {hazard['fusion_source']}"
        )
    else:
        state = fast_result.get("path_state", "unknown")
        lines.append("PRIMARY: path clear" if state == "clear" else f"NAVIGATION: {state.upper()}")
    return lines


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
        dest="continuous_currency",
        action="store_true",
        default=None,
        help="Enable continuous automatic banknote recognition in background (default: off on laptop)",
    )
    parser.add_argument(
        "--no-continuous-currency",
        dest="continuous_currency",
        action="store_false",
        help="Disable continuous automatic banknote recognition (default on laptop)",
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
    parser.add_argument(
        "--serial-port",
        default=None,
        help="ESP32 serial port (default: auto-detect /dev/ttyUSB*, /dev/ttyACM*)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Explicit OpenCV camera index (default: auto-probe for a colour camera)",
    )
    parser.add_argument(
        "--no-esp32",
        action="store_true",
        help="Force synthetic distance even if an ESP32 is configured",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Serve an annotated MJPEG debug view at http://127.0.0.1:8080/",
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=8080,
        help="Port for the debug viewer (default: 8080)",
    )
    parser.add_argument(
        "--viewer-host",
        default="127.0.0.1",
        help="Bind address for the debug viewer (default: localhost; use 0.0.0.0 for phone access)",
    )
    parser.add_argument(
        "--enable-keyboard",
        action="store_true",
        default=False,
        help="Enable optional interactive terminal keyboard shortcuts for testing (default: disabled, physical tactile buttons only)",
    )
    args = parser.parse_args()

    if args.continuous_currency is None:
        target = args.hardware
        if target == "auto":
            target = "rpi" if is_raspberry_pi() else "laptop"
        args.continuous_currency = False

    print("=" * 80)
    print("        DRISHTI ASSISTIVE WEARABLE: AUTONOMOUS LIVE SYSTEM")
    print("=" * 80)

    # 1. Load Centralized Configurations
    pipeline_cfg = get_pipeline_config()
    hw_cfg = get_hardware_config(profile=args.hardware)

    # 2. Initialize HAL stack from config
    camera, distance_sensor, trigger_input, audio_sink = build_hardware_stack(
        args.hardware,
        hw_cfg=hw_cfg,
        serial_port=args.serial_port,
        camera_index=args.camera_index,
        no_esp32=args.no_esp32,
        enable_keyboard=args.enable_keyboard,
    )
    
    # 3. Configure and Connect MQTT Telemetry
    telem_cfg = pipeline_cfg.get("safety", {}).get("telemetry", {})
    telemetry = TelemetryClient(
        device_id="drishti-live-wearable",
        broker_host=telem_cfg.get("broker_host", "localhost"),
        broker_port=telem_cfg.get("broker_port", 1883),
        use_tls=telem_cfg.get("use_tls", False),
        ca_certs=telem_cfg.get("ca_certs"),
        qos=telem_cfg.get("qos", 1),
        username=os.environ.get(telem_cfg.get("username_env", "DRISHTI_MQTT_USERNAME")),
        password=os.environ.get(telem_cfg.get("password_env", "DRISHTI_MQTT_PASSWORD")),
        buffer_limit=telem_cfg.get("buffer_limit", 500),
        persistence_path=telem_cfg.get("persistence_path"),
        require_auth=telem_cfg.get("require_auth", True),
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
        pipeline_config=pipeline_cfg,
        mode="simulation" if args.hardware == "synthetic" else "production",
        speech_enabled=not args.no_speech,
    )

    # 5. Setup Graceful Shutdown Handler
    running = True

    # 5a. Report the speech backend and Bluetooth route before the loop starts.
    speech = pipeline.speech_service
    backends = speech.available_backends() or ["none"]
    print(f"[*] Speech backend: {', '.join(backends)}")
    if speech.has_piper:
        print(f"    -> Piper voice: {speech.piper_model}")
    else:
        print("    [!] Piper not found; falling back to system speech (more robotic).")
    audio_route = speech.get_audio_route()
    print(f"[*] Audio route: {audio_route.summary()}")
    for warning in audio_route.warnings:
        print(f"    [!] {warning}")

    viewer: Optional[DebugViewer] = None
    if args.viewer:
        viewer = DebugViewer(host=args.viewer_host, port=args.viewer_port)
        viewer.start()
        print(f"[*] Debug viewer: {viewer.url}  (Ctrl+C to stop)")

    def sigint_handler(signum, frame):
        nonlocal running
        print("\n\n[*] Interrupt signal received. Initiating graceful shutdown...")
        running = False

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    # 4. Startup Vocal Announcement
    if not args.no_speech:
        print("[*] Broadcasting startup chime...")
        greeting = (
            "Drishti online. Continuous obstacle guidance active. "
            "Ready for button input: Button 1 for narration, Button 2 for currency, Button 3 for text."
        )
        pipeline.speech_service.speak(greeting, play_audio=True)

    print("\n" + "=" * 80)
    print("  SYSTEM RUNNING CONTINUOUSLY (Press Ctrl+C to terminate)")
    print("  Hardware Tactile Buttons:")
    print("    [Button 1] = Scene Narration (Gemini Vision)")
    print("    [Button 2] = Banknote Identification (Gemini Vision)")
    print("    [Button 3] = Multilingual Text Reading (Gemini Vision)")
    if args.enable_keyboard:
        print("  Keyboard Shortcuts: [N] = Narrate | [C] = Currency | [T] = Text | [Q] = Quit")
    print("=" * 80 + "\n")


    cycle_count = 0
    t_start = time.time()

    try:
        while running and (args.max_cycles is None or cycle_count < args.max_cycles):
            cycle_count += 1
            res = pipeline.step()
            fast = res.get("fast_perception", {})

            if viewer is not None:
                viewer.update(
                    getattr(pipeline.fast_pipeline, "last_frame", None),
                    fast,
                    build_viewer_status_lines(cycle_count, fast, distance_sensor),
                )

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
                elif fast.get("path_state") == "clear":
                    line = f"Cycle {cycle_count:<5} | Path Clear| --       | --      | 3D: --                   | {lat:5.1f} ms ({fps:4.1f} FPS)"
                else:
                    line = f"Cycle {cycle_count:<5} | NAV NOT READY | {fast.get('path_state', 'unknown').upper()} | {lat:5.1f} ms"

                print(line)
            else:
                issues = fast.get("readiness", {}).get("issues", [])
                codes = ", ".join(x.get("code", "UNKNOWN") for x in issues) or fast.get("error", "unknown")
                print(f"Cycle {cycle_count:<5} | NAVIGATION NOT READY | {codes}")

            # Tactile Button Trigger Events (instant terminal confirmation on press)
            for ev in res.get("events", []):
                ev_type = ev.get("type")
                if ev_type == "SCENE_NARRATION_TRIGGERED":
                    print("\n  >>> [BUTTON 1 PRESSED] Scene Narration queued for Gemini Vision...")
                elif ev_type == "CURRENCY_TRIGGERED":
                    print("\n  >>> [BUTTON 2 PRESSED] Banknote Identification queued for Gemini Vision...")
                elif ev_type == "TEXT_READING_TRIGGERED":
                    print("\n  >>> [BUTTON 3 PRESSED] Multilingual Text Reading queued for Gemini Vision...")

            # Cognitive results (Scene narration, Banknote, OCR)
            for cognitive in res.get("cognitive_results", []):
                if "narration" in cognitive:
                    print(f"  >>> [SCENE NARRATION] Spoken: \"{cognitive['narration']}\"")
                elif cognitive.get("currency", {}).get("denomination") is not None:
                    c = cognitive["currency"]
                    print(f"  >>> [CURRENCY] Spoken: \"{c['spoken_name']}\" (Confidence: {c.get('confidence', 'high')})")
                elif "ocr" in cognitive or "text" in cognitive:
                    txt = cognitive.get("text") or cognitive.get("ocr", {}).get("text")
                    print(f"  >>> [TEXT READING] Spoken: \"{txt or 'No clear text detected.'}\"")
                elif cognitive.get("spoken_name"):
                    print(f"  >>> [COGNITION] Spoken: \"{cognitive['spoken_name']}\"")

            # 30 FPS pacing yield
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[*] Releasing camera, sensors, and telemetry sockets...")
        if viewer is not None:
            viewer.stop()
        if not args.no_speech:
            pipeline.speech_service.speak("Drishti powering down.", play_audio=True)
            pipeline.speech_service.wait_until_idle(timeout=3.0)
        pipeline.shutdown()
        total_time = time.time() - t_start
        mean_fps = cycle_count / max(total_time, 0.001)
        print(f"[✓] Shutdown complete. Executed {cycle_count} cycles in {total_time:.1f}s (Average {mean_fps:.1f} FPS).")


if __name__ == "__main__":
    main()
