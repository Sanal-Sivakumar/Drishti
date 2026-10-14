"""Unit tests for the Drishti Hardware Abstraction Layer (HAL)."""

import pytest
import numpy as np
from hal.camera import SyntheticCameraSource
from hal.distance_sensor import SyntheticDistanceSensor
from hal.trigger_input import SyntheticTriggerInput
from hal.audio_output import HeadlessAudioRecorder


def test_synthetic_camera_source():
    """Verify synthetic camera produces valid BGR frames and animates over time."""
    cam = SyntheticCameraSource(width=320, height=240, fps=30)
    props = cam.get_properties()
    assert props["backend"] == "synthetic"
    assert props["width"] == 320
    assert props["height"] == 240

    success, frame = cam.read_frame()
    assert success is True
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8
    assert np.mean(frame) > 0  # Frame is not completely black

    # Read another frame and verify frame counter increments
    success2, frame2 = cam.read_frame()
    assert success2 is True
    assert cam.get_properties()["frame_count"] == 2

    # Verify release
    cam.release()
    success_closed, frame_closed = cam.read_frame()
    assert success_closed is False
    assert frame_closed is None


def test_synthetic_distance_sensor():
    """Verify synthetic distance sensor produces valid metric distance and multi-zone depth."""
    sensor = SyntheticDistanceSensor(base_distance_m=1.80, noise_std_m=0.01)
    
    # Before starting, distance query should return None
    assert sensor.get_distance_meters() is None

    sensor.start_ranging()
    dist = sensor.get_distance_meters()
    assert dist is not None
    assert 0.5 <= dist <= 3.5  # Sensible range for indoor testing

    zones = sensor.get_zone_distances()
    assert "left" in zones and "center" in zones and "right" in zones
    assert zones["center"] is not None

    sensor.stop_ranging()
    assert sensor.get_distance_meters() is None


def test_synthetic_trigger_input():
    """Verify synthetic trigger input handles programmatic events and callbacks."""
    trigger = SyntheticTriggerInput()
    callback_fired = []

    def on_button_1():
        callback_fired.append(1)

    trigger.register_callback(1, on_button_1)

    # Initial state should be untriggered
    assert trigger.poll_trigger(1) is False
    assert trigger.poll_trigger(2) is False

    # Inject button 1 press
    trigger.inject_trigger(1)
    assert len(callback_fired) == 1
    assert trigger.poll_trigger(1) is True
    # Poll should consume the event
    assert trigger.poll_trigger(1) is False


def test_headless_audio_recorder():
    """Verify headless audio recorder buffers PCM stereo samples and maintains metrics."""
    recorder = HeadlessAudioRecorder()
    sample_rate = 44100
    
    # Synthesize 0.1s stereo test tone
    duration_s = 0.1
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    test_pcm = np.column_stack([np.sin(2 * np.pi * 440 * t), np.cos(2 * np.pi * 440 * t)])

    recorder.play_pcm(test_pcm, sample_rate=sample_rate)
    assert recorder.get_total_samples() == num_samples

    concat = recorder.get_concatenated_pcm()
    assert concat is not None
    assert concat.shape == (num_samples, 2)
    assert concat.dtype == np.float32

    # Clear recorder
    recorder.clear()
    assert recorder.get_total_samples() == 0
    assert recorder.get_concatenated_pcm() is None

