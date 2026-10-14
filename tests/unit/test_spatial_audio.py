"""Unit tests for the 3D Binaural Spatial Audio Synthesizer."""

import pytest
import numpy as np
from perception.spatial_audio import SpatialAudioSynthesizer


def test_woodworth_itd_formulation():
    """Verify ITD matches Woodworth spherical head model values within 1%."""
    synth = SpatialAudioSynthesizer(sample_rate_hz=44100, head_radius_m=0.0875, speed_of_sound_mps=343.0)

    # 1. Directly ahead (azimuth = 0) -> ITD must be 0
    itd_0 = synth.compute_itd_seconds(0.0)
    assert np.isclose(itd_0, 0.0, atol=1e-6)

    # 2. Maximum lateral azimuth (+90 degrees = pi/2 rad)
    # Expected: (0.0875 / 343.0) * (pi/2 + 1) = 0.0006554s (655.4 us)
    itd_90 = synth.compute_itd_seconds(np.pi / 2)
    expected_90 = (0.0875 / 343.0) * (np.pi / 2 + 1.0)
    assert np.isclose(itd_90, expected_90, rtol=1e-3)
    assert 0.00065 < itd_90 < 0.00066

    # 3. Symmetry: ITD(-theta) = -ITD(theta)
    itd_neg_45 = synth.compute_itd_seconds(-np.pi / 4)
    itd_pos_45 = synth.compute_itd_seconds(np.pi / 4)
    assert np.isclose(itd_neg_45, -itd_pos_45, atol=1e-6)


def test_ild_gain_calculations():
    """Verify Interaural Level Difference (ILD) gain behavior across azimuths."""
    synth = SpatialAudioSynthesizer()

    # Directly ahead -> gains must be equal (0 dB difference)
    gain_l, gain_r = synth.compute_ild_gains(0.0)
    assert np.isclose(gain_l, gain_r, atol=1e-4)

    # Sound from right (+45 degrees) -> right ear louder than left ear
    gain_l_r, gain_r_r = synth.compute_ild_gains(np.pi / 4)
    assert gain_r_r > gain_l_r

    # Sound from left (-45 degrees) -> left ear louder than right ear
    gain_l_l, gain_r_l = synth.compute_ild_gains(-np.pi / 4)
    assert gain_l_l > gain_r_l


def test_distance_pitch_and_pulse_rate():
    """Verify distance maps correctly to auditory warning pitch and pulse rate."""
    synth = SpatialAudioSynthesizer()

    # Far distance (3.0m) -> 440 Hz, low pulse rate (~1.6 Hz)
    pitch_far, rate_far = synth.compute_pitch_and_pulse_rate(3.0)
    assert pitch_far == 440.0
    assert 1.0 <= rate_far <= 2.5

    # Critical distance (0.5m) -> 880 Hz, high pulse rate (10.0 Hz)
    pitch_close, rate_close = synth.compute_pitch_and_pulse_rate(0.5)
    assert pitch_close == 880.0
    assert rate_close == 10.0


def test_stereo_waveform_synthesis():
    """Verify synthesized stereo PCM waveform characteristics."""
    synth = SpatialAudioSynthesizer(sample_rate_hz=44100)
    pcm = synth.synthesize_obstacle_ping(azimuth_rad=np.pi / 3, distance_m=1.2, duration_s=0.05)

    assert pcm.ndim == 2
    assert pcm.shape[1] == 2  # 2 channels (stereo)
    assert pcm.dtype == np.float32
    assert np.max(np.abs(pcm)) <= 1.0  # Zero clipping
    # Right channel should have higher energy when sound is from right
    energy_left = np.sum(pcm[:, 0] ** 2)
    energy_right = np.sum(pcm[:, 1] ** 2)
    assert energy_right > energy_left

