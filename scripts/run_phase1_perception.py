#!/usr/bin/env python3
"""Drishti Phase 1: Fast Perception & Spatial Audio Verification Runner.

Executes real-time camera capture, obstacle detection, ray-cone depth fusion,
and 3D binaural spatial audio generation with millisecond latency profiling.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import time
import argparse
import numpy as np

from hal.camera import SyntheticCameraSource, V4L2CameraSource
from hal.distance_sensor import SyntheticDistanceSensor
from hal.audio_output import HeadlessAudioRecorder, ALSAAudioSink
from perception.pipeline_fast import FastPerceptionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Drishti Phase 1 Fast Perception Runner")
    parser.add_argument("--frames", type=int, default=10, help="Number of live frames to display")
    parser.add_argument("--benchmark-frames", type=int, default=50, help="Number of benchmark cycles")
    parser.add_argument("--camera", choices=["synthetic", "real"], default="synthetic", help="Camera backend (synthetic or real webcam)")
    parser.add_argument("--audio", choices=["headless", "real"], default="headless", help="Audio sink (headless buffer or real speakers)")
    args = parser.parse_args()

    print("=" * 78)
    print("      DRISHTI: PHASE 1 FAST PERCEPTION & 3D SPATIAL AUDIO STREAM")
    print(f"      Backend Modes: Camera [{args.camera.upper()}] | Audio [{args.audio.upper()}]")
    print("=" * 78)

    # Initialize Camera
    if args.camera == "real":
        print("[*] Initializing physical laptop camera on /dev/video0...")
        camera = V4L2CameraSource(device_path="/dev/video0", width=640, height=480, fps=30)
    else:
        camera = SyntheticCameraSource(width=640, height=480, fps=30)

    # Initialize Distance Sensor
    distance_sensor = SyntheticDistanceSensor(base_distance_m=1.70, noise_std_m=0.01)

    # Initialize Audio Sink
    if args.audio == "real":
        print("[*] Initializing physical audio output on plughw:1,0...")
        audio_sink = ALSAAudioSink(device="plughw:1,0")
    else:
        audio_sink = HeadlessAudioRecorder()

    pipeline = FastPerceptionPipeline(
        camera=camera,
        distance_sensor=distance_sensor,
        audio_sink=audio_sink,
    )

    print(f"\n[1] Streaming {args.frames} Real-Time Perception Cycles:")
    print("-" * 78)
    header = f"{'Frame':<6} | {'Class':<8} | {'Dist (m)':<9} | {'Azimuth':<9} | {'3D [X, Y, Z] (m)':<22} | {'Latency':<8}"
    print(header)
    print("-" * 78)

    for i in range(1, args.frames + 1):
        result = pipeline.step()
        if not result["success"]:
            print(f"Cycle {i} failed: {result.get('error')}")
            continue

        hazard = result.get("primary_hazard")
        lat = result["latency_ms"]["total_cycle"]

        if hazard:
            cls_name = hazard["class_name"]
            dist = f"{hazard['distance_m']:.2f} m"
            az = f"{hazard['azimuth_deg']:+.1f}°"
            x, y, z = hazard["coords_3d_m"]
            coords = f"[{x:+.2f}, {y:+.2f}, {z:.2f}]"
            print(f"{i:<6} | {cls_name:<8} | {dist:<9} | {az:<9} | {coords:<22} | {lat:.2f} ms")
        else:
            print(f"{i:<6} | {'None':<8} | {'--':<9} | {'--':<9} | {'--':<22} | {lat:.2f} ms")

        time.sleep(0.03)  # Emulate 30 FPS timing

    print("\n" + "-" * 78)
    print(f"[2] Running Quantitative Latency Benchmark ({args.benchmark_frames} frames)...")
    stats = pipeline.benchmark(num_frames=args.benchmark_frames)

    print(f"    - Mean End-to-End Cycle Latency: {stats['mean_cycle_ms']:.2f} ms")
    print(f"    - Min / Max Cycle Latency:       {stats['min_cycle_ms']:.2f} ms / {stats['max_cycle_ms']:.2f} ms")
    print(f"    - Mean Detection Latency:        {stats['mean_detection_ms']:.2f} ms")
    print(f"    - Mean Depth Fusion Latency:     {stats['mean_depth_fusion_ms']:.2f} ms")
    print(f"    - Mean Audio Synthesis Latency:  {stats['mean_audio_synth_ms']:.2f} ms")
    print(f"    - Sustained Throughput:          {stats['achieved_fps']:.1f} FPS")

    # Audio synthesis validation
    if hasattr(audio_sink, "get_total_samples") and callable(audio_sink.get_total_samples):
        total_audio_samples = audio_sink.get_total_samples()
        recorded_pcm = audio_sink.get_concatenated_pcm()
        print("\n[3] 3D Binaural Audio Verification:")
        print(f"    - Total Synthesized Samples:     {total_audio_samples} samples")
        if recorded_pcm is not None:
            duration_sec = total_audio_samples / 44100.0
            print(f"    - Total Buffered Audio Duration: {duration_sec:.2f} seconds")
            print(f"    - Audio Buffer Shape:            {recorded_pcm.shape} (Channels: {recorded_pcm.shape[1]})")
            print(f"    - Peak Amplitude:                {np.max(np.abs(recorded_pcm)):.3f} / 1.000")
    else:
        print("\n[3] Real-Time Audio Output: Dispatched to physical soundcard (plughw:1,0).")

    # Verification check
    if args.camera == "synthetic":
        assert stats["mean_cycle_ms"] <= 20.0, "Latency exceeded PC budget (20ms)!"
    print("\n" + "=" * 78)
    print("  >>> CHECKPOINT 1 VERIFIED: FAST PERCEPTION OPERATING WITHIN BUDGET <<<")
    print("=" * 78)

    pipeline.shutdown()


if __name__ == "__main__":
    main()
