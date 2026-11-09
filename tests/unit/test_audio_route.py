"""Unit tests for audio-route detection helpers."""

from hal.audio_route import (
    AudioRoute,
    _find_a2dp_profile,
    _node_id_from_inspect,
    _parse_wpctl_inspect,
    playback_commands,
)


def test_parse_wpctl_inspect_and_node_id():
    text = """id 58, type PipeWire:Interface:Node
    * node.name = "bluez_output.AC_12_2F_00_11_22.a2dp-sink"
      device.api = "bluez5"
      device.description = "WH-1000XM4"
      api.bluez5.profile = "a2dp-sink"
      api.bluez5.codec = "aac"
"""
    props = _parse_wpctl_inspect(text)
    assert props["node.name"].startswith("bluez_output")
    assert props["device.api"] == "bluez5"
    assert props["api.bluez5.profile"] == "a2dp-sink"
    assert _node_id_from_inspect(text) == 58


def test_audio_route_quality_labels():
    a2dp = AudioRoute(
        is_bluetooth=True,
        is_a2dp=True,
        codec="aac",
        description="WH-1000XM4",
    )
    hfp = AudioRoute(
        is_bluetooth=True,
        is_hfp=True,
        description="WH-1000XM4",
    )
    wired = AudioRoute(description="Built-in Audio")

    assert "A2DP" in a2dp.summary()
    assert "aac" in a2dp.summary()
    assert "HFP" in hfp.summary()
    assert "wired" in wired.summary()


def test_find_a2dp_profile_prefers_sink_and_priority():
    entries = [
        {"index": 1, "name": "headset-head-unit", "priority": 100},
        {"index": 2, "name": "a2dp-sink", "priority": 50},
        {"index": 3, "name": "a2dp-sink-codec", "priority": 60},
        {"index": 4, "name": "a2dp-source", "priority": 200},
    ]
    profile = _find_a2dp_profile(entries)
    assert profile is not None
    assert profile["index"] == 3  # highest-priority sink profile, not a source


def test_playback_commands_returns_available_players():
    commands = playback_commands("/tmp/test.wav")
    assert commands
    assert all(isinstance(command, list) for command in commands)
    assert any(command[-1] == "/tmp/test.wav" for command in commands)


def test_alsa_audio_sink_initialization_survives_missing_portaudio():
    from hal.audio_output import ALSAAudioSink
    sink = ALSAAudioSink(device="default")
    assert sink.is_running is True
    sink.stop()
