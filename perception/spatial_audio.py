"""Perception Subsystem: 3D Binaural Spatial Audio Synthesizer.

Synthesizes real-time 3D directional acoustic cues from scratch using the
Duplex Theory of sound localization:
- Woodworth's Spherical Head Model for Interaural Time Difference (ITD)
- Frequency-dependent Head Acoustic Shadowing for Interaural Level Difference (ILD)
- Proximity Pulse Repetition Frequency (PRF) and Harmonic Pitch Modulation
"""

from typing import Tuple
import numpy as np


class SpatialAudioSynthesizer:
    """Synthesizes stereo binaural audio cues for 3D obstacle perception."""

    def __init__(
        self,
        sample_rate_hz: int = 44100,
        head_radius_m: float = 0.0875,
        speed_of_sound_mps: float = 343.0,
        ild_attenuation_db_per_rad: float = 12.0,
    ) -> None:
        self.sample_rate = sample_rate_hz
        self.head_radius = head_radius_m
        self.speed_of_sound = speed_of_sound_mps
        self.ild_attenuation_db_per_rad = ild_attenuation_db_per_rad

    def compute_itd_seconds(self, azimuth_rad: float) -> float:
        """Compute Interaural Time Difference (ITD) using Woodworth's spherical head model.
        
        Args:
            azimuth_rad: Horizontal angle in radians. 
                         0 = directly ahead, >0 = right, <0 = left.
                         Range: [-pi/2, +pi/2].
        
        Returns:
            float: Time delay in seconds. Positive means sound reaches right ear first.
        """
        # Clamp to [-pi/2, +pi/2]
        theta = float(np.clip(azimuth_rad, -np.pi / 2, np.pi / 2))
        # Woodworth formula: Delta t = (r / c) * (theta + sin(theta))
        itd = (self.head_radius / self.speed_of_sound) * (theta + np.sin(theta))
        return itd

    def compute_ild_gains(self, azimuth_rad: float) -> Tuple[float, float]:
        """Compute Interaural Level Difference (ILD) linear gain multipliers for left & right ears.
        
        Args:
            azimuth_rad: Horizontal angle in radians.
        
        Returns:
            Tuple[float, float]: (left_linear_gain, right_linear_gain)
        """
        theta = float(np.clip(azimuth_rad, -np.pi / 2, np.pi / 2))
        # Level difference in decibels proportional to sin(theta)
        delta_db = self.ild_attenuation_db_per_rad * np.sin(theta)

        # Right ear is ipsilateral for theta > 0
        right_db = delta_db / 2.0
        left_db = -delta_db / 2.0

        gain_left = float(10.0 ** (left_db / 20.0))
        gain_right = float(10.0 ** (right_db / 20.0))
        return gain_left, gain_right

    def compute_pitch_and_pulse_rate(self, distance_m: float) -> Tuple[float, float]:
        """Map distance to acoustic pitch and pulse repetition frequency.
        
        Args:
            distance_m: Metric distance to nearest obstacle.
        
        Returns:
            Tuple[float, float]: (fundamental_frequency_hz, pulse_repetition_rate_hz)
        """
        d = float(np.clip(distance_m, 0.2, 4.0))

        # Pitch shifting: 440 Hz (far) -> 660 Hz (warning) -> 880 Hz (critical)
        if d >= 2.0:
            pitch_hz = 440.0
        elif d >= 1.0:
            pitch_hz = 660.0
        else:
            pitch_hz = 880.0

        # Pulse repetition rate: 1.0 Hz (at 4m) to 10.0 Hz (at 0.5m)
        # Inversely proportional to distance
        rate_hz = float(np.clip(5.0 / d, 1.0, 10.0))
        return pitch_hz, rate_hz

    def synthesize_obstacle_ping(
        self,
        azimuth_rad: float,
        distance_m: float,
        duration_s: float = 0.08,
    ) -> np.ndarray:
        """Synthesize a single spatialized binaural audio ping.
        
        Args:
            azimuth_rad: Horizontal obstacle angle (-pi/2 to +pi/2).
            distance_m: Metric distance to obstacle.
            duration_s: Duration of the acoustic ping pulse.
        
        Returns:
            np.ndarray: Stereo audio array of shape (N, 2), dtype float32.
        """
        pitch_hz, _ = self.compute_pitch_and_pulse_rate(distance_m)
        num_samples = int(self.sample_rate * duration_s)
        t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)

        # Generate base tone with smooth Tukey / Hann amplitude envelope to prevent speaker click
        envelope = np.sin(np.pi * np.linspace(0, 1, num_samples, dtype=np.float32)) ** 2
        base_tone = np.sin(2.0 * np.pi * pitch_hz * t) * envelope

        # Compute ITD time delay
        itd_seconds = self.compute_itd_seconds(azimuth_rad)
        sample_shift = int(round(abs(itd_seconds) * self.sample_rate))

        # Compute ILD gains
        gain_left, gain_right = self.compute_ild_gains(azimuth_rad)

        left_channel = np.copy(base_tone) * gain_left
        right_channel = np.copy(base_tone) * gain_right

        # Apply delay via zero-padding (fractional delay approximation)
        if sample_shift > 0:
            if itd_seconds > 0:
                # Sound comes from right -> left ear is delayed (contralateral)
                left_channel = np.pad(left_channel, (sample_shift, 0), mode="constant")[:num_samples]
            else:
                # Sound comes from left -> right ear is delayed (contralateral)
                right_channel = np.pad(right_channel, (sample_shift, 0), mode="constant")[:num_samples]

        # Stack into (N, 2) stereo buffer
        stereo_pcm = np.column_stack([left_channel, right_channel]).astype(np.float32)

        # Normalize to prevent digital clipping
        max_val = np.max(np.abs(stereo_pcm))
        if max_val > 1.0:
            stereo_pcm /= max_val

        return stereo_pcm

