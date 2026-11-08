"""Perception Subsystem: 3D Binaural Spatial Audio Synthesizer.

Synthesizes real-time 3D directional acoustic cues from scratch using the
Duplex Theory of sound localization:
- Woodworth's Spherical Head Model for Interaural Time Difference (ITD)
- Frequency-dependent Head Acoustic Shadowing for Interaural Level Difference (ILD)
- Proximity Pulse Repetition Frequency (PRF) and Harmonic Pitch Modulation
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


class SpatialAudioSynthesizer:
    """Synthesizes stereo binaural audio cues for 3D obstacle perception."""

    def __init__(
        self,
        sample_rate_hz: int = 44100,
        head_radius_m: float = 0.0875,
        speed_of_sound_mps: float = 343.0,
        ild_attenuation_db_per_rad: float = 12.0,
        caution_pitch_hz: float = 440.0,
        warning_pitch_hz: float = 660.0,
        critical_pitch_hz: float = 880.0,
        pulse_min_rate_hz: float = 1.0,
        pulse_max_rate_hz: float = 10.0,
        audio_profile: str = "air_conduction",
        profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        high_component_hz: float = 3200.0,
        high_component_mix: float = 0.28,
        limiter_peak: float = 0.35,
    ) -> None:
        self.sample_rate = sample_rate_hz
        self.head_radius = head_radius_m
        self.speed_of_sound = speed_of_sound_mps
        self.ild_attenuation_db_per_rad = ild_attenuation_db_per_rad
        self.caution_pitch_hz = float(caution_pitch_hz)
        self.warning_pitch_hz = float(warning_pitch_hz)
        self.critical_pitch_hz = float(critical_pitch_hz)
        self.pulse_min_rate_hz = float(pulse_min_rate_hz)
        self.pulse_max_rate_hz = float(pulse_max_rate_hz)
        self.audio_profile = audio_profile
        profile = (profiles or {}).get(audio_profile, {})
        self.itd_scale = float(profile.get("itd_scale", 1.0))
        self.ild_scale = float(profile.get("ild_scale", 1.0))
        self.left_gain = float(profile.get("left_gain", 1.0))
        self.right_gain = float(profile.get("right_gain", 1.0))
        self.profile_calibrated = bool(profile.get("calibrated", False))
        self.high_component_hz = float(high_component_hz)
        self.high_component_mix = float(np.clip(high_component_mix, 0.0, 0.5))
        self.limiter_peak = float(np.clip(limiter_peak, 0.05, 0.95))

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
        itd = (self.head_radius / self.speed_of_sound) * (theta + np.sin(theta)) * self.itd_scale
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
        delta_db = self.ild_attenuation_db_per_rad * self.ild_scale * np.sin(theta)

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
            pitch_hz = self.caution_pitch_hz
        elif d >= 1.0:
            pitch_hz = self.warning_pitch_hz
        else:
            pitch_hz = self.critical_pitch_hz

        # Pulse repetition rate: 1.0 Hz (at 4m) to 10.0 Hz (at 0.5m)
        # Inversely proportional to distance
        rate_hz = float(np.clip(5.0 / d, self.pulse_min_rate_hz, self.pulse_max_rate_hz))
        return pitch_hz, rate_hz

    @staticmethod
    def apply_fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
        """Apply fractional delay to 1D signal using linear sub-sample interpolation.
        
        Args:
            signal: 1D input waveform array of shape (N,).
            delay_samples: Non-negative delay in fractional samples.
            
        Returns:
            np.ndarray: Delayed waveform of shape (N,), zero-padded at onset.
        """
        if delay_samples <= 0.0:
            return signal.copy()
        
        n = len(signal)
        orig_indices = np.arange(n, dtype=np.float32)
        delayed_indices = orig_indices - delay_samples
        return np.interp(delayed_indices, orig_indices, signal, left=0.0, right=0.0).astype(np.float32)

    def synthesize_obstacle_ping(
        self,
        azimuth_rad: float,
        distance_m: float,
        duration_s: float = 0.06,
    ) -> np.ndarray:
        """Synthesize a binaural acoustic ping representing an obstacle's spatial position.
        
        Args:
            azimuth_rad: Obstacle bearing in radians [-pi/2 to +pi/2].
            distance_m: Obstacle distance in meters.
            duration_s: Pulse duration in seconds (default 60ms).
            
        Returns:
            np.ndarray: Stereo float32 PCM array of shape (N, 2), normalized to [-1.0, 1.0].
        """
        num_samples = int(self.sample_rate * duration_s)
        t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)

        pitch_hz, _ = self.compute_pitch_and_pulse_rate(distance_m)

        # Generate base tone with smooth Tukey / Hann amplitude envelope to prevent speaker click
        envelope = np.sin(np.pi * np.linspace(0, 1, num_samples, dtype=np.float32)) ** 2
        low = np.sin(2.0 * np.pi * pitch_hz * t)
        # Two close upper-band components provide a robust 2-5 kHz ILD cue
        # without random output, preserving deterministic tests/calibration.
        high = 0.5 * (
            np.sin(2.0 * np.pi * self.high_component_hz * t + 0.31)
            + np.sin(2.0 * np.pi * (self.high_component_hz * 1.19) * t + 1.07)
        )
        base_tone = ((1.0 - self.high_component_mix) * low + self.high_component_mix * high) * envelope

        # Compute ITD time delay in fractional samples
        itd_seconds = self.compute_itd_seconds(azimuth_rad)
        delay_samples = abs(itd_seconds) * self.sample_rate

        # Compute ILD gains
        gain_left, gain_right = self.compute_ild_gains(azimuth_rad)

        left_channel = np.copy(base_tone) * gain_left * self.left_gain
        right_channel = np.copy(base_tone) * gain_right * self.right_gain

        # Apply continuous fractional delay for sub-sample spatial localization precision
        if delay_samples > 0.001:
            if itd_seconds > 0:
                # Sound comes from right -> left ear is delayed (contralateral)
                left_channel = self.apply_fractional_delay(left_channel, delay_samples)
            else:
                # Sound comes from left -> right ear is delayed (contralateral)
                right_channel = self.apply_fractional_delay(right_channel, delay_samples)

        # Stack into (N, 2) stereo buffer
        stereo_pcm = np.column_stack([left_channel, right_channel]).astype(np.float32)

        # Conservative digital limiter; acoustic SPL still requires per-device
        # calibration and must not be inferred from this sample peak.
        max_val = np.max(np.abs(stereo_pcm))
        if max_val > self.limiter_peak:
            stereo_pcm *= self.limiter_peak / max_val

        return stereo_pcm
