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
    assert "latitude" in payload["gps"]
    assert "longitude" in payload["gps"]


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
