"""Hardware Abstraction Layer: Audio Output Subsystem.

Provides unified interface for physical audio playback (ALSA/PulseAudio/sounddevice)
and headless in-memory PCM buffers for automated testing.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import threading


class BaseAudioSink(ABC):
    """Abstract base class for audio playback sinks."""

    @abstractmethod
    def play_pcm(self, pcm_data: np.ndarray, sample_rate: int = 44100) -> None:
        """Play or buffer stereo PCM audio data.
        
        Args:
            pcm_data: Array of shape (N, 2) or (2, N) containing float32 or int16 samples.
            sample_rate: Sample rate in Hz (default 44100).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Halt playback immediately."""
        pass


class HeadlessAudioRecorder(BaseAudioSink):
    """Thread-safe in-memory PCM buffer for headless testing and quantitative audio analysis."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.buffered_chunks: List[np.ndarray] = []
        self.sample_rates: List[int] = []
        self.is_active = True

    def play_pcm(self, pcm_data: np.ndarray, sample_rate: int = 44100) -> None:
        """Record the PCM chunk into memory."""
        with self.lock:
            if not self.is_active:
                return
            # Ensure float32 representation
            arr = np.array(pcm_data, dtype=np.float32)
            self.buffered_chunks.append(arr)
            self.sample_rates.append(sample_rate)

    def stop(self) -> None:
        with self.lock:
            self.is_active = False

    def clear(self) -> None:
        with self.lock:
            self.buffered_chunks.clear()
            self.sample_rates.clear()
            self.is_active = True

    def get_total_samples(self) -> int:
        with self.lock:
            total = 0
            for chunk in self.buffered_chunks:
                if chunk.ndim == 1:
                    total += len(chunk)
                else:
                    total += chunk.shape[0] if chunk.shape[0] > chunk.shape[1] else chunk.shape[1]
            return total

    def get_concatenated_pcm(self) -> Optional[np.ndarray]:
        with self.lock:
            if not self.buffered_chunks:
                return None
            # Standardize to (N, 2)
            standardized = []
            for chunk in self.buffered_chunks:
                if chunk.ndim == 1:
                    # Mono to stereo
                    s = np.column_stack([chunk, chunk])
                elif chunk.shape[1] == 2:
                    s = chunk
                elif chunk.shape[0] == 2:
                    s = chunk.T
                else:
                    s = chunk
                standardized.append(s)
            return np.vstack(standardized)


class ALSAAudioSink(BaseAudioSink):
    """Real-time audio output sink using sounddevice or ALSA."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        self.sd = None
        try:
            import sounddevice as sd
            self.sd = sd
        except ImportError:
            self.sd = None

    def play_pcm(self, pcm_data: np.ndarray, sample_rate: int = 44100) -> None:
        if self.sd is None:
            return
        try:
            # Non-blocking playback
            self.sd.play(pcm_data, samplerate=sample_rate, blocking=False)
        except Exception:
            pass

    def stop(self) -> None:
        if self.sd is not None:
            try:
                self.sd.stop()
            except Exception:
                pass

