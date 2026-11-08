"""Shared audio priority arbiter for hazard cues and spoken cognition."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from enum import IntEnum
from typing import Iterator

import numpy as np

from hal.audio_output import BaseAudioSink


class AudioPriority(IntEnum):
    AMBIENT = 10
    SPEECH = 50
    CRITICAL = 100


class PriorityAudioArbiter(BaseAudioSink):
    """Serialize audio categories and suppress lower-priority overlap.

    Spatial PCM and TTS share this state even when TTS ultimately uses a system
    player.  A speech/critical lease prevents routine pings from playing over a
    spoken instruction; urgent callers may request ``CRITICAL`` and preempt it.
    """

    def __init__(self, sink: BaseAudioSink) -> None:
        self.sink = sink
        self._lock = threading.RLock()
        self._active_priority = AudioPriority.AMBIENT
        self._leases = 0
        self.dropped_low_priority = 0

    @contextmanager
    def lease(self, priority: AudioPriority) -> Iterator[bool]:
        acquired = False
        with self._lock:
            if priority >= self._active_priority:
                self._active_priority = priority
                self._leases += 1
                acquired = True
        try:
            yield acquired
        finally:
            if acquired:
                with self._lock:
                    self._leases = max(0, self._leases - 1)
                    if self._leases == 0:
                        self._active_priority = AudioPriority.AMBIENT

    def play_pcm(
        self,
        pcm_data: np.ndarray,
        sample_rate: int = 44100,
        priority: AudioPriority = AudioPriority.AMBIENT,
    ) -> None:
        with self._lock:
            if priority < self._active_priority:
                self.dropped_low_priority += 1
                return
        self.sink.play_pcm(pcm_data, sample_rate=sample_rate)

    def stop(self) -> None:
        self.sink.stop()

