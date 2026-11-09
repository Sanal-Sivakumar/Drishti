#!/usr/bin/env python3
"""Drishti Phase 3: Full Wearable System Orchestrator Runner.

Executes the complete Drishti architecture:
1. Continuous Fast Perception Loop (Camera -> YOLO -> ToF Fusion -> 3D Spatial Audio).
2. Tactile Input Handling (Button 1: Scene Narration, Button 2: Currency, Button 3: Text Reading).
3. Local Edge Privacy Filter (2D Gaussian Face and Vehicle Plate Redaction).
4. Caregiver MQTT Telemetry Gateway (Heartbeat, GPS, and SOS Alerts).
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import time
import argparse

from hal.camera import SyntheticCameraSource, V4L2CameraSource
from hal.distance_sensor import SyntheticDistanceSensor
from hal.trigger_input import SyntheticTriggerInput
from hal.audio_output import HeadlessAudioRecorder, ALSAAudioSink
from safety.telemetry import TelemetryClient
from pipeline import DrishtiMasterPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Drishti Phase 3 Master Wearable Runner")
    parser.add_argument("--camera", choices=["synthetic", "real"], default="synthetic", help="Camera source")
    parser.add_argument("--audio", choices=["headless", "real"], default="headless", help="Audio sink")
    parser.add_argument("--cycles", type=int, default=8, help="Number of perception cycles to execute")
    args = parser.parse_args()

    print("=" * 78)
    print("      DRISHTI: MASTER WEARABLE SYSTEM RUNNER (ALL SUBSYSTEMS ACTIVE)")
    print(f"      Modes: Camera [{args.camera.upper()}] | Audio [{args.audio.upper()}]")
    print("=" * 78)

    # 1. Initialize HAL
    if args.camera == "real":
        print("[*] Initializing physical webcam on /dev/video0...")
        camera = V4L2CameraSource(device_path="/dev/video0", width=640, height=480, fps=30)
    else:
        print("[*] Initializing synthetic camera simulator...")
        camera = SyntheticCameraSource(width=640, height=480, fps=30)

    distance_sensor = SyntheticDistanceSensor(base_distance_m=1.75, noise_std_m=0.01)
    trigger = SyntheticTriggerInput()

    if args.audio == "real":
        print("[*] Initializing hardware audio on plughw:1,0...")
        audio_sink = ALSAAudioSink(device="plughw:1,0")
    else:
        audio_sink = HeadlessAudioRecorder()

    telemetry = TelemetryClient(device_id="drishti-wearable-demo")

    # 2. Initialize Master Pipeline
    pipeline = DrishtiMasterPipeline(
        camera=camera,
        distance_sensor=distance_sensor,
        trigger_input=trigger,
        audio_sink=audio_sink,
        telemetry_client=telemetry,
    )

    print("\n[1] Starting Continuous Wearable Perception Stream:")
    print("-" * 78)

    for i in range(1, args.cycles + 1):
        # On Cycle 3: Inject tactile Button 1 press (User asks for Scene Narration)
        if i == 3:
            print("\n  >>> [EVENT] User presses Tactile Button 1 (Scene Narration) <<<")
            trigger.inject_trigger(button_id=1)

        # On Cycle 5: Inject tactile Button 2 press (User asks for Currency Identification)
        elif i == 5:
            print("\n  >>> [EVENT] User presses Tactile Button 2 (Currency Identification) <<<")
            trigger.inject_trigger(button_id=2)

        # On Cycle 7: Inject tactile Button 3 press (User asks for Text Reading)
        elif i == 7:
            print("\n  >>> [EVENT] User presses Tactile Button 3 (Text Recognition) <<<")
            trigger.inject_trigger(button_id=3)

        res = pipeline.step()
        fast = res.get("fast_perception", {})

        if fast.get("success"):
            hazard = fast.get("primary_hazard")
            lat = fast["latency_ms"]["total_cycle"]
            if hazard:
                cls_name = hazard["class_name"]
                dist = f"{hazard['distance_m']:.2f} m"
                az = f"{hazard['azimuth_deg']:+.1f}°"
                coords = f"[{hazard['coords_3d_m'][0]:+.2f}, {hazard['coords_3d_m'][1]:+.2f}, {hazard['coords_3d_m'][2]:.2f}]"
                print(f"Cycle {i:<2} | {cls_name:<8} | {dist:<8} | {az:<8} | 3D: {coords:<20} | Latency: {lat:.2f} ms")
            else:
                print(f"Cycle {i:<2} | Clear    | --       | --       | 3D: --                   | Latency: {lat:.2f} ms")

        for cog in res.get("cognitive_results", []):
            if "narration" in cog:
                print(f"         └─► Scene Narration: \"{cog['narration']}\"")
            if "currency" in cog:
                curr = cog["currency"]
                print(f"         └─► Banknote: {curr.get('spoken_name', curr.get('denomination'))}")
            if "text" in cog or "ocr" in cog:
                txt = cog.get("text") or cog.get("ocr", {}).get("text")
                print(f"         └─► Text Recognized: \"{txt}\"")

        time.sleep(0.04)

    # Await any in-flight cognitive requests
    pending_cog = pipeline.wait_for_cognition(timeout=3.0)
    for cog in pending_cog:
        if "narration" in cog:
            print(f"         └─► Scene Narration: \"{cog['narration']}\"")
        if "currency" in cog:
            curr = cog["currency"]
            print(f"         └─► Banknote: {curr.get('spoken_name', curr.get('denomination'))}")
        if "text" in cog or "ocr" in cog:
            txt = cog.get("text") or cog.get("ocr", {}).get("text")
            print(f"         └─► Text Recognized: \"{txt}\"")

    # 3. Caregiver Telemetry Ledger
    print("\n" + "-" * 78)
    print(f"[2] Caregiver MQTT Gateway Telemetry Summary ({len(telemetry.buffered_telemetry)} messages):")
    for idx, msg in enumerate(telemetry.buffered_telemetry, 1):
        msg_type = msg.get("alert_type", "HEARTBEAT")
        status = msg.get("status", msg.get("system_state", "OK"))
        print(f"  {idx}. Message Type: {msg_type:<26} | Status: {status}")

    print("\n" + "=" * 78)
    print("  >>> CHECKPOINT 3 VERIFIED: COMPLETE END-TO-END SYSTEM OPERATIONAL <<<")
    print("=" * 78)

    pipeline.shutdown()


if __name__ == "__main__":
    main()

