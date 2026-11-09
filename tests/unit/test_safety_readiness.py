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
        detector=HeadlessObjectDetector(model_path="nonexistent.onnx", allow_synthetic_fallback=False), mode="production",
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


class _MissingToFSensor(SyntheticDistanceSensor):
    def get_distance_meters(self):
        return None


def test_production_suppresses_navigation_when_tof_sample_is_missing():
    sink = HeadlessAudioRecorder()
    pipeline = FastPerceptionPipeline(
        SyntheticCameraSource(), _MissingToFSensor(), sink,
        detector=HeadlessObjectDetector(allow_synthetic_fallback=True), mode="production",
    )
    result = pipeline.step()
    assert result["success"] is False
    assert result["path_state"] == "unknown"
    assert result["audio_synthesized"] is False
    assert result["readiness"]["navigation_authoritative"] is False
    assert any(i["code"] == "RANGE_NO_VALID_SAMPLE" for i in result["readiness"]["issues"])
    assert sink.get_total_samples() == 0
    pipeline.shutdown()


def test_degraded_readiness_suppresses_inferred_distance_audio():
    sink = HeadlessAudioRecorder()
    pipeline = FastPerceptionPipeline(
        SyntheticCameraSource(), SyntheticDistanceSensor(base_distance_m=1.8), sink,
        detector=HeadlessObjectDetector(allow_synthetic_fallback=True), mode="production",
    )
    pipeline.calibration_verified = False
    result = pipeline.step()
    assert result["path_state"] == "unknown"
    pipeline.shutdown()
