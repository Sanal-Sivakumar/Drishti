"""Hardware Abstraction Layer: Audio Output Subsystem.

Provides unified interface for physical audio playback (ALSA/PulseAudio/sounddevice)
and headless in-memory PCM buffers for automated testing.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import threading
from config.logging_config import get_logger

logger = get_logger("hal.audio")


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


from collections import deque
import queue
import subprocess


class HeadlessAudioRecorder(BaseAudioSink):
    """Thread-safe bounded PCM ring buffer for headless testing and quantitative audio analysis."""

    def __init__(self, max_chunks: int = 200) -> None:
        self.lock = threading.Lock()
        self.buffered_chunks: deque = deque(maxlen=max_chunks)
        self.sample_rates: deque = deque(maxlen=max_chunks)
        self.is_active = True

    def play_pcm(self, pcm_data: np.ndarray, sample_rate: int = 44100) -> None:
        """Record the PCM chunk into bounded memory."""
        with self.lock:
            if not self.is_active:
                return
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
            standardized = []
            for chunk in self.buffered_chunks:
                if chunk.ndim == 1:
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
    """Real-time audio output sink with a persistent non-blocking worker thread.
    
    Prevents process zombie accumulation and audio buffer underruns by streaming
    frames sequentially through a dedicated thread worker.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or "plughw:1,0"
        self.audio_queue: queue.Queue = queue.Queue(maxsize=4)
        self.is_running = True
        self.sd = None

        try:
            import sounddevice as sd
            self.sd = sd
        except ImportError:
            self.sd = None

        # Start persistent playback worker
        self.worker = threading.Thread(target=self._playback_loop, daemon=True)
        self.worker.start()

    def _playback_loop(self) -> None:
        """Sequential audio consumer running in background thread."""
        while self.is_running:
            try:
                item = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            pcm_data, sample_rate = item
            played = False

            # Option A: SoundDevice output
            if self.sd is not None:
                try:
                    self.sd.play(pcm_data, samplerate=sample_rate, blocking=True)
                    played = True
                except Exception as e:
                    logger.debug("sounddevice playback failed (%s); attempting aplay fallback", e)
                    played = False

            # Option B: Direct ALSA hardware aplay execution with wait() to prevent zombies
            if not played:
                try:
                    int16_samples = (np.clip(pcm_data, -1.0, 1.0) * 32767).astype(np.int16)
                    cmd = ["aplay", "-D", self.device, "-r", str(sample_rate), "-f", "S16_LE", "-c", "2", "-q"]
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    duration_s = len(pcm_data) / max(sample_rate, 1)
                    timeout = max(0.5, duration_s * 2.0)
                    try:
                        proc.communicate(input=int16_samples.tobytes(), timeout=timeout)
                    except subprocess.TimeoutExpired:
                        logger.warning("aplay playback timed out after %.2fs; killing PID %d", timeout, proc.pid)
                        proc.kill()
                        proc.wait(timeout=0.5)
                except Exception as e:
                    logger.debug("aplay direct execution failed for device %s: %s", self.device, e)

            self.audio_queue.task_done()

    def play_pcm(self, pcm_data: np.ndarray, sample_rate: int = 44100) -> None:
        """Enqueue PCM chunk for playback without blocking the caller."""
        if not self.is_running:
            return
        # Drop oldest frame if queue full to avoid audio latency lag
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                pass
        try:
            self.audio_queue.put_nowait((pcm_data, sample_rate))
        except queue.Full:
            pass

    def stop(self) -> None:
        self.is_running = False
        try:
            self.audio_queue.put_nowait(None)
        except Exception:
            pass
        if self.sd is not None:
            try:
                self.sd.stop()
            except Exception:
                pass


