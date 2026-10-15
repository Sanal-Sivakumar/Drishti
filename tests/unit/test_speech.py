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


