"""Unit tests for Caregiver MQTT Telemetry."""

import pytest
import time
from safety.telemetry import TelemetryClient


def test_heartbeat_payload_schema():
    """Verify heartbeat JSON payload matches telemetry specifications."""
    client = TelemetryClient(device_id="drishti-unit-test")
    payload = client.build_heartbeat_payload(battery_pct=92, cpu_temp_c=48.5)

    assert payload["device_id"] == "drishti-unit-test"
    assert payload["battery_pct"] == 92
    assert payload["cpu_temp_c"] == 48.5
    assert payload["system_state"] == "ACTIVE_NAVIGATION"
    assert "gps" in payload
    # No GPS hardware/environment means unknown, never a fabricated location.
    assert payload["gps"] is None


def test_telemetry_buffer_is_bounded():
    client = TelemetryClient(device_id="bounded", buffer_limit=3)
    for index in range(10):
        client.publish_event("TEST", {"index": index})
    assert len(client.buffered_telemetry) == 3


def test_persistence_reload_and_puback_removal(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    client = TelemetryClient(device_id="persisted", buffer_limit=3, persistence_path=str(path))
    client.publish_event("ONE")
    reloaded = TelemetryClient(device_id="persisted", buffer_limit=3, persistence_path=str(path))
    assert len(reloaded.buffered_telemetry) == 1

    class Info:
        mid = 42

    class FakeMqtt:
        def publish(self, *args, **kwargs):
            return Info()

    reloaded.mqtt_client = FakeMqtt()
    reloaded.is_connected = True
    reloaded._replay_buffer()
    assert 42 in reloaded.pending_acks
    reloaded._on_publish(None, None, 42)
    assert len(reloaded.buffered_telemetry) == 0
    assert path.read_text(encoding="utf-8") == ""


def test_publish_emergency_alert():
    """Verify emergency alert construction and buffering."""
    client = TelemetryClient(device_id="drishti-unit-test")
    alert = client.publish_emergency_alert(
        alert_type="SOS_FALL_DETECTED",
        obstacle_class="stairs",
        distance_m=0.8,
    )

    assert alert["alert_type"] == "SOS_FALL_DETECTED"
    assert alert["obstacle_class"] == "stairs"
    assert alert["distance_m"] == 0.8
    assert alert["status"] == "REQUIRES_CAREGIVER_ATTENTION"
    # Ensure buffered in memory
    assert len(client.buffered_telemetry) >= 1


def test_publish_event():
    """Verify standard non-emergency event publishing."""
    client = TelemetryClient(device_id="drishti-unit-test")
    event = client.publish_event(
        event_type="USER_NARRATION_REQUEST",
        details={"faces_redacted": 1},
    )

    assert event["event_type"] == "USER_NARRATION_REQUEST"
    assert event["status"] == "INFORMATIONAL"
    assert event["details"]["faces_redacted"] == 1
    assert len(client.buffered_telemetry) >= 1
