import numpy as np

from hal.audio_output import HeadlessAudioRecorder
from hal.audio_priority import AudioPriority, PriorityAudioArbiter
from hal.camera import SyntheticCameraSource
from hal.distance_sensor import SyntheticDistanceSensor
from perception.detector import HeadlessObjectDetector
from perception.pipeline_fast import FastPerceptionPipeline


def test_production_fails_closed_without_real_detector():
    pipeline = FastPerceptionPipeline(
        SyntheticCameraSource(), SyntheticDistanceSensor(), HeadlessAudioRecorder(),
        detector=HeadlessObjectDetector(allow_synthetic_fallback=False), mode="production",
    )
    result = pipeline.step()
    assert result["success"] is False
    assert result["path_state"] == "unknown"
    assert result["readiness"]["navigation_authoritative"] is False
    pipeline.shutdown()


def test_audio_arbiter_suppresses_ping_during_speech():
    sink = HeadlessAudioRecorder()
    arbiter = PriorityAudioArbiter(sink)
    pcm = np.ones((20, 2), dtype=np.float32)
    with arbiter.lease(AudioPriority.SPEECH):
        arbiter.play_pcm(pcm)
    assert sink.get_total_samples() == 0
    assert arbiter.dropped_low_priority == 1
