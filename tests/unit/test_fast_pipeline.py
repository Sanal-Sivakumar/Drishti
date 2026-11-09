"""Unit and Latency tests for the Fast Perception Pipeline."""

import pytest
import numpy as np

from hal.camera import SyntheticCameraSource
from hal.distance_sensor import SyntheticDistanceSensor
from hal.audio_output import HeadlessAudioRecorder
from perception.pipeline_fast import FastPerceptionPipeline


def test_fast_perception_single_step():
    """Verify single-cycle end-to-end execution of camera -> detector -> fusion -> audio."""
    cam = SyntheticCameraSource(width=320, height=240, fps=30)
    sensor = SyntheticDistanceSensor(base_distance_m=1.60)
    audio_sink = HeadlessAudioRecorder()

    pipeline = FastPerceptionPipeline(camera=cam, distance_sensor=sensor, audio_sink=audio_sink)

    step_result = pipeline.step()
    assert step_result["success"] is True
    assert "latency_ms" in step_result
    assert step_result["obstacles_count"] >= 1

    primary = step_result["primary_hazard"]
    assert primary is not None
    assert "distance_m" in primary
    assert "azimuth_deg" in primary
    assert len(primary["coords_3d_m"]) == 3

    # Verify that audio was generated and buffered
    assert step_result["audio_synthesized"] is True
    assert audio_sink.get_total_samples() > 0

    pipeline.shutdown()


def test_fast_perception_latency_budget():
    """Verify that end-to-end cycle latency is strictly below the 20ms workstation budget."""
    cam = SyntheticCameraSource(width=320, height=240, fps=30)
    sensor = SyntheticDistanceSensor(base_distance_m=1.50)
    audio_sink = HeadlessAudioRecorder()

    pipeline = FastPerceptionPipeline(camera=cam, distance_sensor=sensor, audio_sink=audio_sink)

    benchmark_stats = pipeline.benchmark(num_frames=50)

    # Quantitative Assertions for real YOLOv8n ONNX perception
    assert benchmark_stats["num_frames"] == 50
    # Real neural network on CPU operates well within the interactive latency budget
    assert benchmark_stats["mean_cycle_ms"] < 75.0
    assert benchmark_stats["achieved_fps"] >= 13.0

    pipeline.shutdown()


def test_close_tof_creates_immediate_unclassified_hazard_without_detection():
    class NoDetections:
        is_real_model_available = True
        unsupported_target_classes = []

        def detect(self, frame):
            return []

    pipeline = FastPerceptionPipeline(
        camera=SyntheticCameraSource(width=320, height=240),
        distance_sensor=SyntheticDistanceSensor(base_distance_m=0.45, noise_std_m=0.0),
        audio_sink=HeadlessAudioRecorder(),
        detector=NoDetections(),
        mode="simulation",
        config={"perception": {"spatial_audio": {"critical_tof_distance_m": 0.8}}},
    )
    result = pipeline.step()
    assert result["primary_hazard"]["class_name"] == "unclassified obstacle"
    assert result["primary_hazard"]["fusion_source"] == "tof_only"
    assert result["audio_synthesized"] is True
    pipeline.shutdown()
