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
from safety.readiness import ReadinessIssue, SystemReadiness


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
        config: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
    ) -> None:
        self.camera = camera
        self.distance_sensor = distance_sensor
        self.audio_sink = audio_sink
        self.config = config or {}
        self.mode = mode or ("simulation" if getattr(camera, "is_synthetic", False) else "production")
        perception_cfg = self.config.get("perception", {})
        detector_cfg = perception_cfg.get("detector", {})
        fusion_cfg = perception_cfg.get("tof_fusion", {})
        self.calibration_verified = bool(fusion_cfg.get("calibration_verified", False))
        self.tof_offset_m = float(fusion_cfg.get("tof_offset_m", 0.0))
        audio_cfg = perception_cfg.get("spatial_audio", {})
        if detector is None:
            # The amber heuristic is only valid for SyntheticCameraSource; on a
            # real frame it can mistake wood, skin or clothing for an obstacle.
            detector = HeadlessObjectDetector(
                confidence_threshold=float(detector_cfg.get("confidence_threshold", 0.35)),
                iou_threshold=float(detector_cfg.get("iou_threshold", 0.45)),
                target_classes=detector_cfg.get("target_classes"),
                model_path=detector_cfg.get("model_path"),
                label_map_path=detector_cfg.get("label_map_path"),
                allow_synthetic_fallback=self.mode == "simulation",
            )
        self.detector = detector
        frame_width = float(self.config.get("pipeline", {}).get("frame_width", 640))
        fov_deg = float(fusion_cfg.get("camera_fov_h_deg", 0.0) or 0.0)
        derived_fx = frame_width / (2.0 * np.tan(np.radians(fov_deg) / 2.0)) if 0.0 < fov_deg < 179.0 else 500.0
        explicit_fx = fusion_cfg.get("camera_focal_length_px")
        calibrated_fx = (
            float(explicit_fx)
            if self.calibration_verified and explicit_fx is not None
            else derived_fx if 0.0 < fov_deg < 179.0 else 500.0
        )
        self.depth_fuser = depth_fuser or SpatialDepthFuser(
            focal_length_px=calibrated_fx,
            focal_length_y_px=float(fusion_cfg.get("camera_focal_length_y_px", calibrated_fx)),
            principal_point_x=float(fusion_cfg.get("principal_point_x", frame_width / 2.0)),
            principal_point_y=float(fusion_cfg.get("principal_point_y", self.config.get("pipeline", {}).get("frame_height", 480) / 2.0)),
            tof_cone_half_angle_deg=float(fusion_cfg.get("cone_half_angle_deg", 13.5)),
            class_priors_m=fusion_cfg.get("class_priors_m"),
            tof_baseline_x_m=float(fusion_cfg.get("baseline_x_m", 0.0)),
            tof_baseline_y_m=float(fusion_cfg.get("baseline_y_m", 0.0)),
            tof_hit_margin_px=float(fusion_cfg.get("tof_hit_margin_px", 8.0)),
            min_distance_m=float(fusion_cfg.get("min_distance_m", 0.05)),
            max_distance_m=float(fusion_cfg.get("max_distance_m", 4.0)),
        )
        self.audio_synthesizer = audio_synthesizer or SpatialAudioSynthesizer(
            sample_rate_hz=int(audio_cfg.get("sample_rate_hz", 44100)),
            head_radius_m=float(audio_cfg.get("head_radius_m", 0.0875)),
            speed_of_sound_mps=float(audio_cfg.get("speed_of_sound_mps", 343.0)),
            ild_attenuation_db_per_rad=float(audio_cfg.get("ild_attenuation_db_per_rad", 12.0)),
            caution_pitch_hz=float(audio_cfg.get("caution_pitch_hz", 440.0)),
            warning_pitch_hz=float(audio_cfg.get("warning_pitch_hz", 660.0)),
            critical_pitch_hz=float(audio_cfg.get("critical_pitch_hz", 880.0)),
            pulse_min_rate_hz=float(audio_cfg.get("pulse_min_rate_hz", 1.0)),
            pulse_max_rate_hz=float(audio_cfg.get("pulse_max_rate_hz", 10.0)),
            audio_profile=str(audio_cfg.get("audio_profile", "air_conduction")),
            profiles=audio_cfg.get("profiles"),
            high_component_hz=float(audio_cfg.get("high_component_hz", 3200.0)),
            high_component_mix=float(audio_cfg.get("high_component_mix", 0.28)),
            limiter_peak=float(audio_cfg.get("limiter_peak", 0.35)),
        )
        self.critical_tof_distance_m = float(audio_cfg.get("critical_tof_distance_m", 0.8))

        # Start active range sensing
        self.distance_sensor.start_ranging()
        self.last_frame: Optional[np.ndarray] = None
        self.last_ping_time: float = 0.0

    def _readiness(self) -> SystemReadiness:
        readiness = SystemReadiness(mode=self.mode)
        if self.mode == "simulation":
            return readiness
        if getattr(self.camera, "is_synthetic", False) or not getattr(self.camera, "is_available", True):
            readiness.issues.append(ReadinessIssue("CAMERA_UNAVAILABLE", "Physical camera is unavailable."))
        if not self.detector.is_real_model_available:
            readiness.issues.append(ReadinessIssue("DETECTOR_UNAVAILABLE", "A validated YOLO model is not loaded."))
        if getattr(self.distance_sensor, "is_synthetic", False):
            readiness.issues.append(ReadinessIssue("SYNTHETIC_RANGE", "Synthetic range is forbidden in production."))
        elif not getattr(self.distance_sensor, "is_available", True):
            readiness.issues.append(ReadinessIssue("RANGE_UNAVAILABLE", "Physical range data is unavailable or stale."))
        if self.detector.unsupported_target_classes:
            readiness.issues.append(
                ReadinessIssue(
                    "UNSUPPORTED_CLASSES",
                    "Unsupported detector classes: " + ", ".join(self.detector.unsupported_target_classes),
                    critical=False,
                )
            )
        if not self.calibration_verified:
            readiness.issues.append(
                ReadinessIssue(
                    "CALIBRATION_UNVERIFIED",
                    "Camera intrinsics and camera-to-ToF alignment are not marked verified.",
                )
            )
        if not self.audio_synthesizer.profile_calibrated:
            readiness.issues.append(
                ReadinessIssue(
                    "AUDIO_PROFILE_UNCALIBRATED",
                    f"Audio profile '{self.audio_synthesizer.audio_profile}' is not calibrated.",
                )
            )
        return readiness

    def step(self) -> Dict[str, Any]:
        """Execute a single perception, fusion, and spatial audio cycle.
        
        Returns:
            Dict[str, Any]: Telemetry containing detected obstacles, fused 3D coordinates,
                            selected primary hazard, audio metadata, and latency breakdown.
        """
        t_start = time.perf_counter()
        readiness = self._readiness()

        if self.mode != "simulation" and not self.detector.is_real_model_available:
            return {
                "success": False,
                "error": "Detector unavailable; navigation output suppressed",
                "path_state": "unknown",
                "readiness": readiness.as_dict(),
                "obstacles": [],
                "obstacles_count": 0,
                "primary_hazard": None,
            }

        # 1. Acquire Camera Frame
        t0 = time.perf_counter()
        success, frame = self.camera.read_frame()
        t_cam = (time.perf_counter() - t0) * 1000.0

        if not success or frame is None:
            return {"success": False, "error": "Camera frame acquisition failed", "path_state": "unknown", "readiness": readiness.as_dict()}

        self.last_frame = frame

        # 2. Acquire ToF Distance Reading
        t0 = time.perf_counter()
        tof_dist_m = self.distance_sensor.get_distance_meters()
        if tof_dist_m is not None:
            tof_dist_m = float(tof_dist_m) + self.tof_offset_m
        t_tof = (time.perf_counter() - t0) * 1000.0
        if self.mode != "simulation" and tof_dist_m is None:
            readiness.issues.append(
                ReadinessIssue("RANGE_NO_VALID_SAMPLE", "No fresh, valid physical ToF sample is available.", critical=False)
            )

        # Issue a centre warning from physical ToF before waiting for YOLO.
        urgent_audio_synthesized = False
        if tof_dist_m is not None and tof_dist_m <= self.critical_tof_distance_m:
            _, urgent_rate = self.audio_synthesizer.compute_pitch_and_pulse_rate(tof_dist_m)
            now = time.time()
            if now - self.last_ping_time >= 1.0 / max(urgent_rate, 0.5):
                urgent_pcm = self.audio_synthesizer.synthesize_obstacle_ping(0.0, tof_dist_m, 0.06)
                self.audio_sink.play_pcm(urgent_pcm, sample_rate=self.audio_synthesizer.sample_rate)
                self.last_ping_time = now
                urgent_audio_synthesized = True

        # 3. Detect 2D Obstacles
        t0 = time.perf_counter()
        raw_detections = self.detector.detect(frame)
        t_detect = (time.perf_counter() - t0) * 1000.0

        # 4. Ray-Cone Spatial Depth Fusion with Multi-Obstacle Disambiguation
        t0 = time.perf_counter()
        fused_obstacles = self.depth_fuser.fuse_all_detections(
            raw_detections=raw_detections,
            tof_distance_m=tof_dist_m,
        )
        # Physical close-range ToF bypasses detector/VLM latency. The class is
        # deliberately unknown; only range and centre direction are asserted.
        if tof_dist_m is not None and tof_dist_m <= self.critical_tof_distance_m:
            central_measured = any(
                item.get("fusion_source") == "tof_laser_fusion" for item in fused_obstacles
            )
            if not central_measured:
                fused_obstacles.append({
                    "class_name": "unclassified obstacle",
                    "confidence": 1.0,
                    "bbox": [],
                    "azimuth_rad": 0.0,
                    "azimuth_deg": 0.0,
                    "distance_m": round(float(tof_dist_m), 3),
                    "coords_3d_m": [0.0, 0.0, round(float(tof_dist_m), 3)],
                    "fusion_source": "tof_only",
                    "critical": True,
                })
        t_fuse = (time.perf_counter() - t0) * 1000.0

        # 5. Select Primary Hazard & Synthesize 3D Binaural Cue with PRF Scheduling
        t0 = time.perf_counter()
        primary_hazard: Optional[Dict[str, Any]] = None
        audio_synthesized = urgent_audio_synthesized

        if fused_obstacles:
            # Sort by distance (closest obstacle takes highest priority)
            fused_obstacles.sort(key=lambda o: o["distance_m"])
            primary_hazard = fused_obstacles[0]

            dist_m = primary_hazard["distance_m"]
            azimuth_rad = primary_hazard["azimuth_rad"]

            # Obey biological sonar Pulse Repetition Frequency (PRF)
            _, pulse_rate_hz = self.audio_synthesizer.compute_pitch_and_pulse_rate(dist_m)
            min_interval_s = 1.0 / max(pulse_rate_hz, 0.5)
            now = time.time()

            if (now - self.last_ping_time) >= min_interval_s:
                stereo_pcm = self.audio_synthesizer.synthesize_obstacle_ping(
                    azimuth_rad=azimuth_rad, distance_m=dist_m, duration_s=0.06
                )
                self.audio_sink.play_pcm(stereo_pcm, sample_rate=self.audio_synthesizer.sample_rate)
                audio_synthesized = True
                self.last_ping_time = now


        t_audio = (time.perf_counter() - t0) * 1000.0
        t_total = (time.perf_counter() - t_start) * 1000.0

        authoritative = readiness.navigation_authoritative or self.mode == "simulation"
        path_state = "obstructed" if fused_obstacles else ("clear" if authoritative else "unknown")
        return {
            "success": True,
            "path_state": path_state,
            "readiness": readiness.as_dict(),
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
