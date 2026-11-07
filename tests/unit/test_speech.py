"""Unit tests for the Speech Synthesis (TTS) Engine."""

import pytest
import numpy as np
from cognitive.speech_service import SpeechService


def test_speech_service_basic_synthesis():
    """Verify text is converted into duration-bounded PCM samples."""
    service = SpeechService()
    test_text = "Caution, obstacle one meter ahead."
    res = service.speak(test_text, play_audio=False)

    assert res["success"] is True
    assert res["word_count"] == 5
    assert res["duration_s"] > 0.0
    assert "pcm_samples" in res
    assert isinstance(res["pcm_samples"], np.ndarray)
    assert len(res["pcm_samples"]) > 0
    assert np.max(np.abs(res["pcm_samples"])) <= 1.0


def test_speech_service_empty_text():
    """Verify empty or whitespace strings are gracefully rejected."""
    service = SpeechService()
    res = service.speak("   ", play_audio=False)
    assert res["success"] is False
    assert "error" in res


def test_speech_service_speaking_rate():
    """Verify higher speaking rate produces shorter estimated audio duration."""
    service_normal = SpeechService(speaking_rate=1.0)
    service_fast = SpeechService(speaking_rate=1.5)

    long_text = "This is an important navigational announcement regarding room layout."
    res_normal = service_normal.speak(long_text, play_audio=False)
    res_fast = service_fast.speak(long_text, play_audio=False)

    assert res_fast["duration_s"] < res_normal["duration_s"]


def test_speech_service_voice_styling():
    """Verify voice_type and pitch configuration parameters."""
    service = SpeechService(speaking_rate=0.95, voice_type="female1", pitch=5)
    assert service.speaking_rate == 0.95
    assert service.voice_type == "female1"
    assert service.pitch == 5

    res = service.speak("Testing clear voice styling.", play_audio=False)
    assert res["success"] is True
    assert res["duration_s"] > 0.0


def test_speech_service_audio_sink_fallback():
    """Verify audio_sink is used when physical speech dispatcher is not invoked."""
    from hal.audio_output import HeadlessAudioRecorder
    recorder = HeadlessAudioRecorder()
    service = SpeechService(audio_sink=recorder)
    # When play_audio is True, if no system speech was dispatched, recorder receives PCM
    res = service.speak("System initialized.", play_audio=True)
    assert res["success"] is True


def test_async_playback_queues_and_runs_worker(monkeypatch):
    """Verify playback is queued off the perception thread."""
    service = SpeechService(async_playback=True)
    service.has_piper = False
    service.has_spdsay = False
    service.has_espeak = False

    played = []
    monkeypatch.setattr(
        service,
        "_play_text",
        lambda text: (played.append(text) or ("test", True)),
    )

    result = service.speak("Chair one meter ahead.", play_audio=True)
    assert result["queued"] is True
    assert service.wait_until_idle(timeout=2.0) is True
    assert played == ["Chair one meter ahead."]
    service.stop()


def test_sync_playback_reports_backend(monkeypatch):
    """Verify synchronous mode (used by scripts/tests) reports the backend."""
    service = SpeechService(async_playback=False)
    monkeypatch.setattr(service, "_play_text", lambda text: ("piper", True))

    result = service.speak("Testing playback.", play_audio=True)
    assert result["audio_played"] is True
    assert result["backend"] == "piper"
    assert result["queued"] is False


def test_piper_model_resolves_relative_to_project_root():
    """The bundled model path must work regardless of the current directory."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    service = SpeechService(piper_model="models/piper/en_US-lessac-medium.onnx")
    if (root / "models/piper/en_US-lessac-medium.onnx").is_file():
        assert service.has_piper is True
        assert Path(service.piper_model).is_absolute()
        assert service.piper_length_scale >= 1.0
    else:
        assert service.piper_model is None or not service.has_piper


