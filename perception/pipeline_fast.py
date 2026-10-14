"""Perception Subsystem: Integrated Fast Perception & Spatial Audio Pipeline.

Coordinates Camera, ToF Distance Sensor, Object Detector, Ray-Cone Depth Fuser,
and 3D Binaural Audio Synthesizer into a sub-100ms continuous hazard loop.
"""

from typing import Dict, Any, Optional, List
import time
import numpy as np

from hal.camera import BaseCameraSource
from hal.distance_sensor import BaseDistanceSensor
from hal.audio_output import BaseAudioSink
from perception.detector import HeadlessObjectDetector
from perception.depth_fusion import SpatialDepthFuser
from perception.spatial_audio import SpatialAudioSynthesizer


class FastPerceptionPipeline:
    """Real-time fast perception loop coordinating vision, depth, and spatial audio."""

    def __init__(
        self,
        camera: BaseCameraSource,
        distance_sensor: BaseDistanceSensor,
        audio_sink: BaseAudioSink,
        detector: Optional[HeadlessObjectDetector] = None,
        depth_fuser: Optional[SpatialDepthFuser] = None,
        audio_synthesizer: Optional[SpatialAudioSynthesizer] = None,
    ) -> None:
        self.camera = camera
        self.distance_sensor = distance_sensor
        self.audio_sink = audio_sink
        self.detector = detector or HeadlessObjectDetector()
        self.depth_fuser = depth_fuser or SpatialDepthFuser()
        self.audio_synthesizer = audio_synthesizer or SpatialAudioSynthesizer()

        # Start active range sensing
        self.distance_sensor.start_ranging()

    def step(self) -> Dict[str, Any]:
        """Execute a single perception, fusion, and spatial audio cycle.
        
        Returns:
            Dict[str, Any]: Telemetry containing detected obstacles, fused 3D coordinates,
                            selected primary hazard, audio metadata, and latency breakdown.
        """
        t_start = time.perf_counter()

        # 1. Acquire Camera Frame
        t0 = time.perf_counter()
        success, frame = self.camera.read_frame()
        t_cam = (time.perf_counter() - t0) * 1000.0

        if not success or frame is None:
            return {"success": False, "error": "Camera frame acquisition failed"}

        # 2. Acquire ToF Distance Reading
        t0 = time.perf_counter()
        tof_dist_m = self.distance_sensor.get_distance_meters()
        t_tof = (time.perf_counter() - t0) * 1000.0

        # 3. Detect 2D Obstacles
        t0 = time.perf_counter()
        raw_detections = self.detector.detect(frame)
        t_detect = (time.perf_counter() - t0) * 1000.0

        # 4. Ray-Cone Spatial Depth Fusion
        t0 = time.perf_counter()
        fused_obstacles: List[Dict[str, Any]] = []
        for det in raw_detections:
            fused = self.depth_fuser.fuse_detection(
                bbox=tuple(det["bbox"]),
                class_name=det["class_name"],
                confidence=det["confidence"],
                tof_distance_m=tof_dist_m,
            )
            fused_obstacles.append(fused)
        t_fuse = (time.perf_counter() - t0) * 1000.0

        # 5. Select Primary Hazard & Synthesize 3D Binaural Cue
        t0 = time.perf_counter()
        primary_hazard: Optional[Dict[str, Any]] = None
        audio_synthesized = False

        if fused_obstacles:
            # Sort by distance (closest obstacle takes highest priority)
            fused_obstacles.sort(key=lambda o: o["distance_m"])
            primary_hazard = fused_obstacles[0]

            # Generate spatial audio ping
            azimuth_rad = primary_hazard["azimuth_rad"]
            dist_m = primary_hazard["distance_m"]
            stereo_pcm = self.audio_synthesizer.synthesize_obstacle_ping(
                azimuth_rad=azimuth_rad, distance_m=dist_m, duration_s=0.06
            )
            self.audio_sink.play_pcm(stereo_pcm, sample_rate=self.audio_synthesizer.sample_rate)
            audio_synthesized = True

        t_audio = (time.perf_counter() - t0) * 1000.0
        t_total = (time.perf_counter() - t_start) * 1000.0

        return {
            "success": True,
            "obstacles_count": len(fused_obstacles),
            "obstacles": fused_obstacles,
            "primary_hazard": primary_hazard,
            "audio_synthesized": audio_synthesized,
            "latency_ms": {
                "camera_read": round(t_cam, 2),
                "tof_read": round(t_tof, 2),
                "detection": round(t_detect, 2),
                "depth_fusion": round(t_fuse, 2),
                "audio_synthesis": round(t_audio, 2),
                "total_cycle": round(t_total, 2),
            },
        }

    def benchmark(self, num_frames: int = 50) -> Dict[str, float]:
        """Execute multiple cycles to quantitatively profile throughput and latency.
        
        Args:
            num_frames: Number of cycles to benchmark.
            
        Returns:
            Dict[str, float]: Mean latencies and achieved FPS.
        """
        cycle_times = []
        detect_times = []
        fusion_times = []
        audio_times = []

        for _ in range(num_frames):
            res = self.step()
            if res.get("success"):
                lat = res["latency_ms"]
                cycle_times.append(lat["total_cycle"])
                detect_times.append(lat["detection"])
                fusion_times.append(lat["depth_fusion"])
                audio_times.append(lat["audio_synthesis"])

        mean_cycle = float(np.mean(cycle_times)) if cycle_times else 0.0
        return {
            "num_frames": num_frames,
            "mean_cycle_ms": round(mean_cycle, 2),
            "max_cycle_ms": round(float(np.max(cycle_times)), 2) if cycle_times else 0.0,
            "min_cycle_ms": round(float(np.min(cycle_times)), 2) if cycle_times else 0.0,
            "mean_detection_ms": round(float(np.mean(detect_times)), 2) if detect_times else 0.0,
            "mean_depth_fusion_ms": round(float(np.mean(fusion_times)), 2) if fusion_times else 0.0,
            "mean_audio_synth_ms": round(float(np.mean(audio_times)), 2) if audio_times else 0.0,
            "achieved_fps": round(1000.0 / mean_cycle, 1) if mean_cycle > 0 else 0.0,
        }

    def shutdown(self) -> None:
        """Release all active sensor and audio resources."""
        self.distance_sensor.stop_ranging()
        self.camera.release()
        self.audio_sink.stop()
