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

    # Quantitative Assertions
    assert benchmark_stats["num_frames"] == 50
    # Must be well under the 20ms PC latency budget
    assert benchmark_stats["mean_cycle_ms"] < 20.0
    # Achieved throughput must exceed 30 FPS
    assert benchmark_stats["achieved_fps"] >= 30.0

    pipeline.shutdown()
