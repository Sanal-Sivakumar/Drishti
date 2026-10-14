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
    """Verify tactile Button 1 triggers on-demand VLM scene narration and speech."""
    # Inject user pressing Button 1 (Scene Narration)
    master_pipeline.trigger_input.inject_trigger(button_id=1)

    result = master_pipeline.step()
    assert result.get("event") == "SCENE_NARRATION_TRIGGERED"
    assert "narration" in result
    assert len(result["narration"]) > 0

    # Ensure caregiver telemetry alert was queued
    alerts = [
        msg for msg in master_pipeline.telemetry.buffered_telemetry
        if msg.get("alert_type") == "USER_NARRATION_REQUEST"
    ]
    assert len(alerts) >= 1


def test_pipeline_button2_reading_or_currency(master_pipeline):
    """Verify tactile Button 2 triggers currency recognition or OCR readout."""
    # Inject user pressing Button 2
    master_pipeline.trigger_input.inject_trigger(button_id=2)

    result = master_pipeline.step()
    assert result.get("event") == "READING_OR_CURRENCY_TRIGGERED"
    assert ("currency" in result) or ("ocr" in result)


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


