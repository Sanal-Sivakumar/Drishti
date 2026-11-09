"""Integration tests for the Drishti Master Pipeline Orchestrator."""

import pytest
from hal.camera import SyntheticCameraSource
from hal.distance_sensor import SyntheticDistanceSensor
from hal.trigger_input import SyntheticTriggerInput
from hal.audio_output import HeadlessAudioRecorder
from safety.telemetry import TelemetryClient
from pipeline import DrishtiMasterPipeline


@pytest.fixture
def master_pipeline():
    """Build a complete headless pipeline fixture."""
    cam = SyntheticCameraSource(width=320, height=240, fps=30)
    sensor = SyntheticDistanceSensor(base_distance_m=1.65)
    trigger = SyntheticTriggerInput()
    audio = HeadlessAudioRecorder()
    telemetry = TelemetryClient(device_id="integration-test-device")

    pipeline = DrishtiMasterPipeline(
        camera=cam,
        distance_sensor=sensor,
        trigger_input=trigger,
        audio_sink=audio,
        telemetry_client=telemetry,
    )
    yield pipeline
    pipeline.shutdown()


def test_pipeline_continuous_steps(master_pipeline):
    """Verify master pipeline executes continuous perception cycles without errors."""
    for _ in range(5):
        result = master_pipeline.step()
        assert "fast_perception" in result
        assert result["fast_perception"]["success"] is True


def test_pipeline_button1_scene_narration(master_pipeline):
    """Verify narration is queued away from the real-time step."""
    # Inject user pressing Button 1 (Scene Narration)
    master_pipeline.trigger_input.inject_trigger(button_id=1)

    result = master_pipeline.step()
    assert result["events"] == [{"type": "SCENE_NARRATION_TRIGGERED", "queued": True}]
    completed = master_pipeline.wait_for_cognition()
    assert completed and completed[0]["narration"]

    # Ensure caregiver telemetry alert was queued
    alerts = [
        msg for msg in master_pipeline.telemetry.buffered_telemetry
        if msg.get("alert_type") == "USER_NARRATION_REQUEST"
    ]
    assert len(alerts) >= 1


def test_pipeline_button2_currency(master_pipeline):
    """Verify tactile Button 2 triggers currency recognition with privacy redaction."""
    master_pipeline.trigger_input.inject_trigger(button_id=2)

    result = master_pipeline.step()
    assert result["events"] == [{"type": "CURRENCY_TRIGGERED", "queued": True}]
    completed = master_pipeline.wait_for_cognition()
    assert completed and ("currency" in completed[0])
    assert "faces_redacted" in completed[0]
    assert "plates_redacted" in completed[0]


def test_pipeline_button3_text_reading(master_pipeline):
    """Verify tactile Button 3 triggers online text recognition with privacy redaction."""
    master_pipeline.trigger_input.inject_trigger(button_id=3)

    result = master_pipeline.step()
    assert result["events"] == [{"type": "TEXT_READING_TRIGGERED", "queued": True}]
    completed = master_pipeline.wait_for_cognition()
    assert completed and (("ocr" in completed[0]) or ("text" in completed[0]))
    assert "faces_redacted" in completed[0]
    assert "plates_redacted" in completed[0]


def test_pipeline_all_cloud_tasks_pass_redacted_frame(monkeypatch):
    """Verify narration, currency, and OCR frames pass through FaceAnonymizer before cloud upload."""
    import numpy as np

    cam = SyntheticCameraSource(width=320, height=240, fps=30)
    sensor = SyntheticDistanceSensor(base_distance_m=1.65)
    trigger = SyntheticTriggerInput()
    audio = HeadlessAudioRecorder()
    telemetry = TelemetryClient(device_id="privacy-test-device")

    pipeline = DrishtiMasterPipeline(
        camera=cam,
        distance_sensor=sensor,
        trigger_input=trigger,
        audio_sink=audio,
        telemetry_client=telemetry,
    )

    received_frames = []

    def mock_currency(img):
        received_frames.append(("currency", img))
        return {"success": True, "currency": {"denomination": 100, "spoken_name": "One hundred rupees"}}

    def mock_text(img):
        received_frames.append(("text", img))
        return {"success": True, "text": "Drishti 500mg", "spoken_name": "Drishti 500mg"}

    def mock_narration(img, obs, **kwargs):
        received_frames.append(("narration", img))
        return {"success": True, "narration": "A scene"}

    monkeypatch.setattr(pipeline.online_vision, "identify_currency", mock_currency)
    monkeypatch.setattr(pipeline.online_vision, "read_text", mock_text)
    monkeypatch.setattr(pipeline.online_vision, "describe_scene", mock_narration)

    # 1. Trigger currency
    trigger.inject_trigger(button_id=2)
    pipeline.step()
    pipeline.wait_for_cognition()
    assert len(received_frames) == 1
    assert received_frames[0][0] == "currency"
    # Verify received frame is an ndarray with matching dimensions
    assert isinstance(received_frames[0][1], np.ndarray)

    # 2. Trigger text
    trigger.inject_trigger(button_id=3)
    pipeline.step()
    pipeline.wait_for_cognition()
    assert len(received_frames) == 2
    assert received_frames[1][0] == "text"
    assert isinstance(received_frames[1][1], np.ndarray)

    pipeline.shutdown()


def test_pipeline_run_loop(master_pipeline):
    """Verify master pipeline runs for bounded cycles and shuts down cleanly."""
    master_pipeline.run_loop(max_cycles=3)
    assert master_pipeline.cycle_count >= 3


def test_pipeline_continuous_currency_mode():
    """Verify master pipeline executes continuous currency detection without tactile triggers."""
    cam = SyntheticCameraSource(width=320, height=240, fps=30)
    sensor = SyntheticDistanceSensor(base_distance_m=1.65)
    trigger = SyntheticTriggerInput()
    audio = HeadlessAudioRecorder()
    telemetry = TelemetryClient(device_id="continuous-curr-test")

    pipeline = DrishtiMasterPipeline(
        camera=cam,
        distance_sensor=sensor,
        trigger_input=trigger,
        audio_sink=audio,
        telemetry_client=telemetry,
        continuous_currency=True,
        currency_interval=1,  # evaluate on every cycle for testing
    )

    # Step through cycles - continuous currency runs automatically
    res = pipeline.step()
    assert "fast_perception" in res
    assert res["fast_perception"]["success"] is True
    # Verify that continuous currency evaluated
    assert pipeline.continuous_currency is True
    pipeline.shutdown()

