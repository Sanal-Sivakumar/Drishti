"""Cognitive Subsystem: Local Speech Synthesis (TTS) Engine.

Converts machine-extracted text, currency denominations, and VLM descriptions
into natural spoken audio. The preferred backend is Piper neural TTS (natural,
offline, 22 kHz); system speech dispatcher and espeak-ng remain as fallbacks.
Playback is queued on a worker thread so a multi-second announcement never
stalls the 20-30 FPS obstacle-avoidance loop.

Headless tests still get a deterministic synthetic PCM buffer via
``speak(..., play_audio=False)``; that buffer is never sent to real speakers.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.logging_config import get_logger
from hal.audio_output import BaseAudioSink, HeadlessAudioRecorder
from hal.audio_route import AudioRoute, detect_audio_route, playback_commands

logger = get_logger("cognitive.speech")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SpeechService:
    """Local text-to-speech service with Piper and system fallbacks.

    Backend priority:
      1. Piper neural TTS (best quality, offline)
      2. speech-dispatcher (``spd-say``; espeak-ng voice, slower pacing)
      3. ``espeak-ng`` directly
      4. Headless PCM buffer (tests only)

    ``speak(..., play_audio=True)`` enqueues playback and returns immediately.
    ``speak(..., play_audio=False)`` stays synchronous for tests and synthesis
    inspection.
    """

    def __init__(
        self,
        voice: str = "en_US",
        speaking_rate: float = 0.95,
        voice_type: str = "female1",
        pitch: int = 0,
        sample_rate_hz: int = 22050,
        audio_sink: Optional[BaseAudioSink] = None,
        piper_model: Optional[str] = None,
        piper_bin: Optional[str] = None,
        piper_length_scale: Optional[float] = None,
        piper_sentence_silence: float = 0.25,
        async_playback: bool = True,
    ) -> None:
        self.voice = voice
        self.speaking_rate = float(speaking_rate)
        self.voice_type = voice_type
        self.pitch = pitch
        self.sample_rate = int(sample_rate_hz)
        self.audio_sink = audio_sink
        self.async_playback = bool(async_playback)

        # Natural pacing: >1.0 is slower. A small base slowdown avoids the
        # clipped, rushed cadence of the raw model.
        self.piper_length_scale = (
            float(piper_length_scale)
            if piper_length_scale is not None
            else max(0.85, 1.06 / max(self.speaking_rate, 0.5))
        )
        self.piper_sentence_silence = float(piper_sentence_silence)

        self.piper_bin = self._find_piper_binary(piper_bin)
        self.piper_model = self._find_piper_model(piper_model)
        self.has_piper = bool(
            self.piper_bin and self.piper_model and Path(self.piper_model).is_file()
        )
        self.has_spdsay = shutil.which("spd-say") is not None
        self.has_espeak = (
            shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None
        )

        self._audio_route: Optional[AudioRoute] = None
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._worker: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()
        self._current_process: Optional[subprocess.Popen] = None
        self._current_lock = threading.Lock()
        self._stopping = threading.Event()
        self._warned_no_backend = False

        logger.info(
            "SpeechService initialized (piper=%s, spd-say=%s, espeak=%s, "
            "rate=%.2f, length_scale=%.2f)",
            self.has_piper,
            self.has_spdsay,
            self.has_espeak,
            self.speaking_rate,
            self.piper_length_scale,
        )
        if self.has_piper:
            logger.info("Piper neural voice: %s", self.piper_model)

    # ------------------------------------------------------------------ discovery

    @staticmethod
    def _find_piper_binary(explicit: Optional[str]) -> Optional[str]:
        candidates: List[str] = []
        if explicit:
            candidates.append(explicit)
        env_bin = os.environ.get("DRISHTI_PIPER_BIN")
        if env_bin:
            candidates.append(env_bin)
        found = shutil.which("piper")
        if found:
            candidates.append(found)
        venv_piper = Path(sys.executable).with_name("piper")
        candidates.append(str(venv_piper))

        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        return None

    @staticmethod
    def _find_piper_model(explicit: Optional[str]) -> Optional[str]:
        candidates: List[str] = []
        if explicit:
            candidates.append(explicit)
        env_model = os.environ.get("DRISHTI_PIPER_MODEL")
        if env_model:
            candidates.append(env_model)

        search_dirs = [
            _PROJECT_ROOT / "models" / "piper",
            Path.home() / ".local" / "share" / "piper" / "voices",
            Path.home() / "piper",
        ]
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            models = sorted(directory.glob("*.onnx"))
            # Prefer the bundled US English medium voice, then any English
            # voice, then whatever is installed.
            preferred = [
                model
                for model in models
                if "en_US-lessac-medium" in model.name
            ]
            english = [model for model in models if model.name.startswith("en_")]
            ordered = preferred or english or models
            candidates.extend(str(model) for model in ordered)

        for candidate in candidates:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = _PROJECT_ROOT / path
            if path.is_file():
                return str(path)
        return None

    def available_backends(self) -> List[str]:
        backends = []
        if self.has_piper:
            backends.append("piper")
        if self.has_spdsay:
            backends.append("spd-say")
        if self.has_espeak:
            backends.append("espeak-ng")
        return backends

    def get_audio_route(self, refresh: bool = False) -> AudioRoute:
        if self._audio_route is None or refresh:
            self._audio_route = detect_audio_route()
        return self._audio_route

    def describe_audio_route(self) -> str:
        route = self.get_audio_route()
        return route.summary()

    # ------------------------------------------------------------------ public API

    def speak(
        self,
        text: str,
        play_audio: bool = True,
        interrupt: bool = False,
    ) -> Dict[str, Any]:
        """Convert text to speech, optionally queueing physical playback.

        Args:
            text: Text to speak.
            play_audio: Queue playback on the audio backend.
            interrupt: Drop queued announcements and stop the current one
                before speaking (use for urgent safety messages).

        Returns:
            Metadata including duration, backend, synthetic PCM buffer, and
            whether playback was queued. The synthetic PCM buffer is used for
            headless tests and is never sent to real hardware.
        """
        clean_text = text.strip()
        if not clean_text:
            return {"success": False, "error": "Empty text string"}

        words = clean_text.split()
        estimated_duration_s = max(0.5, len(words) / (2.33 * max(self.speaking_rate, 0.1)))
        pcm_samples = self._synthesize_test_pcm(estimated_duration_s)

        result: Dict[str, Any] = {
            "success": True,
            "text": clean_text,
            "word_count": len(words),
            "duration_s": round(estimated_duration_s, 2),
            "backend": self._preferred_backend_name(),
            "audio_played": False,
            "queued": False,
            "pcm_samples": pcm_samples,
            "sample_rate": self.sample_rate,
        }

        if not play_audio:
            return result

        if not self.async_playback:
            backend, played = self._play_text(clean_text)
            result["backend"] = backend
            result["audio_played"] = played
            return result

        if interrupt:
            self.stop_current()
            self._clear_queue()

        self._ensure_worker()
        try:
            self._queue.put_nowait((clean_text, interrupt))
            result["queued"] = True
        except queue.Full:
            # Keep the newest message: drop the oldest queued announcement.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((clean_text, interrupt))
                result["queued"] = True
            except queue.Full:
                logger.warning("Speech queue full; dropping announcement: %s", clean_text)

        return result

    def stop(self) -> None:
        """Stop playback, drain the queue, and join the worker thread."""
        self._stopping.set()
        self.stop_current()
        self._clear_queue()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        self._worker = None

    def stop_current(self) -> None:
        """Terminate the in-flight player process (if any)."""
        with self._current_lock:
            process = self._current_process
            self._current_process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

    def wait_until_idle(self, timeout: float = 10.0) -> bool:
        """Block until queued speech finishes (useful for shutdown/tests)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._current_lock:
                active = self._current_process is not None
            if self._queue.empty() and not active:
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------ worker

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stopping.clear()
            self._worker = threading.Thread(
                target=self._playback_loop,
                name="drishti-speech",
                daemon=True,
            )
            self._worker.start()

    def _playback_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                text, interrupt = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if interrupt:
                    self.stop_current()
                self._play_text(text)
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                logger.warning("Speech playback failed: %s", exc)
            finally:
                self._queue.task_done()

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return

    # ------------------------------------------------------------------ backends

    def _preferred_backend_name(self) -> str:
        if self.has_piper:
            return "piper"
        if self.has_spdsay:
            return "spd_say"
        if self.has_espeak:
            return "espeak_ng"
        return "none"

    def _play_text(self, text: str) -> Tuple[str, bool]:
        """Run the best available backend synchronously (worker thread)."""
        if self.has_piper:
            backend, played = self._speak_piper(text)
            if played:
                return backend, True
            logger.warning("Piper playback failed; falling back to system speech.")

        if self.has_spdsay:
            if self._speak_spd_say(text):
                return "spd_say", True

        if self.has_espeak:
            if self._speak_espeak(text):
                return "espeak_ng", True

        # Never play the synthetic test tone on real hardware. It is only
        # recorded by the headless sink so automated tests can assert audio.
        if isinstance(self.audio_sink, HeadlessAudioRecorder):
            pcm = self._synthesize_test_pcm(max(0.5, len(text.split()) / 2.33))
            stereo = np.column_stack((pcm, pcm))
            self.audio_sink.play_pcm(stereo, sample_rate=self.sample_rate)
            return "headless_pcm", True

        if not self._warned_no_backend:
            logger.warning(
                "No speech backend available. Install Piper (pip install piper-tts) "
                "or espeak-ng for offline speech."
            )
            self._warned_no_backend = True
        return "none", False

    def _speak_piper(self, text: str) -> Tuple[str, bool]:
        """Synthesize with Piper then play the WAV on the default sink."""
        if not self.piper_bin or not self.piper_model:
            return "piper", False

        route = self.get_audio_route()
        if route.is_bluetooth and route.is_hfp:
            logger.warning(
                "Bluetooth route is still HFP; speech will be mono 16 kHz until "
                "the earbuds switch to A2DP."
            )

        fd, wav_path = tempfile.mkstemp(prefix="drishti_tts_", suffix=".wav")
        os.close(fd)
        timeout = max(20.0, min(120.0, 0.16 * len(text) + 15.0))
        try:
            command = [
                self.piper_bin,
                "-m",
                self.piper_model,
                "-f",
                wav_path,
                "--length-scale",
                f"{self.piper_length_scale:.3f}",
                "--sentence-silence",
                f"{self.piper_sentence_silence:.2f}",
            ]
            returncode, _, stderr = self._run_process(
                command,
                input_bytes=text.encode("utf-8"),
                timeout=timeout,
            )
            if returncode != 0 or not Path(wav_path).is_file() or Path(wav_path).stat().st_size == 0:
                logger.warning("Piper synthesis failed (rc=%s): %s", returncode, stderr.strip()[:200])
                return "piper", False

            for player in playback_commands(wav_path):
                if not shutil.which(player[0]):
                    continue
                play_timeout = max(20.0, min(120.0, 0.25 * len(text) + 20.0))
                returncode, _, stderr = self._run_process(player, timeout=play_timeout)
                if returncode == 0:
                    return "piper", True
                logger.debug("Playback command %s failed: %s", player[0], stderr.strip()[:120])
            return "piper", False
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _speak_spd_say(self, text: str) -> bool:
        try:
            # -w waits for completion so queued announcements do not overlap.
            # -r -13 is a natural, slightly slower pace; -p -5 softens pitch.
            rate = int(np.clip((self.speaking_rate - 1.0) * 100.0 - 8.0, -30.0, 0.0))
            command = ["spd-say", "-w", "-r", str(rate), "-p", "-5"]
            if self.voice_type:
                command.extend(["-t", self.voice_type])
            command.append(text)
            env = os.environ.copy()
            env["LC_ALL"] = "C.UTF-8"
            returncode, _, _ = self._run_process(
                command,
                timeout=max(15.0, min(120.0, 0.2 * len(text) + 15.0)),
                env=env,
            )
            return returncode == 0
        except Exception as exc:  # noqa: BLE001
            logger.debug("spd-say playback failed: %s", exc)
            return False

    def _speak_espeak(self, text: str) -> bool:
        espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if not espeak_bin:
            return False
        try:
            wpm = int(np.clip(150.0 * self.speaking_rate, 100.0, 190.0))
            voice = "en+f3" if "female" in self.voice_type.lower() else "en"
            command = [
                espeak_bin,
                "-s",
                str(wpm),
                "-v",
                voice,
                "-p",
                str(int(np.clip(45 + self.pitch, 0, 99))),
                "-g",
                "6",
                text,
            ]
            returncode, _, _ = self._run_process(
                command,
                timeout=max(15.0, min(120.0, 0.2 * len(text) + 15.0)),
            )
            return returncode == 0
        except Exception as exc:  # noqa: BLE001
            logger.debug("espeak playback failed: %s", exc)
            return False

    # ------------------------------------------------------------------ helpers

    def _run_process(
        self,
        command: List[str],
        input_bytes: Optional[bytes] = None,
        timeout: float = 30.0,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str, str]:
        """Run a process while tracking it so ``stop_current`` can cancel it."""
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, OSError) as exc:
            return 127, "", str(exc)

        with self._current_lock:
            self._current_process = process
        try:
            stdout, stderr = process.communicate(input=input_bytes, timeout=timeout)
            return (
                int(process.returncode or 0),
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except subprocess.TimeoutExpired:
            logger.warning("Speech process timed out after %.0fs: %s", timeout, command[0])
            try:
                process.kill()
                process.communicate(timeout=1.0)
            except Exception:
                pass
            return 124, "", "timeout"
        finally:
            with self._current_lock:
                if self._current_process is process:
                    self._current_process = None

    def _synthesize_test_pcm(self, duration_s: float) -> np.ndarray:
        """Deterministic harmonic buffer used only by automated tests."""
        duration_s = max(0.1, float(duration_s))
        num_samples = max(1, int(self.sample_rate * duration_s))
        t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)
        f0 = 160.0
        wave = (
            0.5 * np.sin(2.0 * np.pi * f0 * t)
            + 0.3 * np.sin(2.0 * np.pi * 2.0 * f0 * t)
            + 0.2 * np.sin(2.0 * np.pi * 3.0 * f0 * t)
        )
        modulation = np.abs(np.sin(2.0 * np.pi * 3.0 * t))
        return (wave * modulation * 0.7).astype(np.float32)
