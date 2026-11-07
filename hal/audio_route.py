"""Audio routing helpers for laptop / Raspberry Pi / Bluetooth earbuds.

Drishti can run on a laptop with wired speakers or on a Pi with Bluetooth
earbuds. This module inspects the Linux audio graph (PipeWire/WirePlumber when
available, ALSA otherwise) so the runtime can:

* report which sink will receive speech and sonification,
* detect Bluetooth sinks and warn when they fall back to HFP/HSP
  (mono, 16 kHz, telephone quality) instead of A2DP (stereo, 44.1/48 kHz),
* attempt to switch a Bluetooth card back to its A2DP sink profile.

Everything is best-effort and non-fatal: if the tools are missing or the user is
on an unusual setup, the caller simply keeps the system default sink.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.logging_config import get_logger

logger = get_logger("hal.audio_route")

_HFP_MARKERS = (
    "headset",
    "handsfree",
    "handsfree-head-unit",
    "hfp",
    "hsp",
    "sco",
    "msbc",
    "cvsd",
)
_A2DP_MARKERS = ("a2dp", "a2dp-sink")


@dataclass
class AudioRoute:
    """Snapshot of the current default playback route."""

    sink_id: Optional[int] = None
    sink_name: str = ""
    description: str = ""
    driver: str = "unknown"
    is_bluetooth: bool = False
    profile_name: str = ""
    profile_description: str = ""
    codec: str = ""
    is_a2dp: bool = False
    is_hfp: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def quality_label(self) -> str:
        if not self.is_bluetooth:
            return "wired/default"
        if self.is_a2dp:
            return f"Bluetooth A2DP ({self.codec or 'unknown codec'})"
        if self.is_hfp:
            return "Bluetooth HFP/HSP (mono 16 kHz)"
        return "Bluetooth (profile unknown)"

    def summary(self) -> str:
        name = self.description or self.sink_name or "system default"
        return f"{name} [{self.quality_label}]"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sink_id": self.sink_id,
            "sink_name": self.sink_name,
            "description": self.description,
            "driver": self.driver,
            "is_bluetooth": self.is_bluetooth,
            "profile_name": self.profile_name,
            "profile_description": self.profile_description,
            "codec": self.codec,
            "is_a2dp": self.is_a2dp,
            "is_hfp": self.is_hfp,
            "warnings": list(self.warnings),
        }


def _run(command: List[str], timeout: float = 4.0) -> Optional[str]:
    """Run a command and return stdout, or None when it is unavailable/fails."""
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _parse_wpctl_inspect(text: str) -> Dict[str, str]:
    """Parse the flat ``key = value`` lines emitted by ``wpctl inspect``."""
    props: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("*").strip()
        value = value.strip().strip('"')
        if key:
            props[key] = value
    return props


def _node_id_from_inspect(text: str) -> Optional[int]:
    """Extract the id from the first line ``id 58, type ...``."""
    first = text.splitlines()[0] if text else ""
    parts = first.replace(",", " ").split()
    for idx, part in enumerate(parts):
        if part == "id" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def _pw_dump_objects() -> List[Dict[str, Any]]:
    raw = _run(["pw-dump"], timeout=6.0)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _profile_entries(device_object: Dict[str, Any]) -> List[Dict[str, Any]]:
    info = device_object.get("info", {})
    params = info.get("params", {})
    entries: List[Dict[str, Any]] = []
    for key in ("EnumProfile", "Profile"):
        value = params.get(key)
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, dict))
    return entries


def _set_default_sink(sink_id: int) -> bool:
    return _run(["wpctl", "set-default", str(sink_id)]) is not None


def _find_a2dp_profile(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = []
    for entry in entries:
        name = str(entry.get("name", "")).lower()
        description = str(entry.get("description", "")).lower()
        if not any(marker in name or marker in description for marker in _A2DP_MARKERS):
            continue
        if "source" in name and "sink" not in name:
            continue
        candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda e: int(e.get("priority", 0) or 0), reverse=True)
    return candidates[0]


def _a2dp_sink_nodes(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes = []
    for obj in objects:
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        props = obj.get("info", {}).get("props", {})
        media_class = str(props.get("media.class", ""))
        node_name = str(props.get("node.name", "")).lower()
        if "Audio/Sink" not in media_class and not node_name.startswith("bluez_output"):
            continue
        if "bluez" in node_name or str(props.get("device.api", "")) == "bluez5":
            nodes.append(obj)
    return nodes


def _try_enable_a2dp(route: AudioRoute) -> AudioRoute:
    """Ask WirePlumber to move the Bluetooth card to an A2DP sink profile."""
    if not shutil.which("wpctl"):
        route.warnings.append(
            "wpctl not found; set the earbuds to A2DP in the system sound settings."
        )
        return route

    objects = _pw_dump_objects()
    device_id: Optional[int] = route.sink_id
    if route.sink_id is not None:
        for obj in objects:
            if obj.get("id") != route.sink_id:
                continue
            props = obj.get("info", {}).get("props", {})
            raw_device_id = props.get("device.id")
            if raw_device_id is not None:
                try:
                    device_id = int(raw_device_id)
                except (TypeError, ValueError):
                    pass

    if device_id is None:
        route.warnings.append(
            "Could not identify the Bluetooth card; choose A2DP in the sound settings."
        )
        return route

    device_object = next((obj for obj in objects if obj.get("id") == device_id), None)
    if device_object is None:
        route.warnings.append("Could not inspect the Bluetooth card; choose A2DP manually.")
        return route

    profile = _find_a2dp_profile(_profile_entries(device_object))
    if profile is None:
        route.warnings.append(
            "No A2DP profile advertised by the Bluetooth card; reconnect the earbuds."
        )
        return route

    logger.info("Switching Bluetooth card %s to A2DP profile '%s'", device_id, profile.get("name"))
    if _run(["wpctl", "set-profile", str(device_id), str(profile.get("index"))]) is None:
        route.warnings.append("Failed to switch the Bluetooth card to A2DP automatically.")
        return route

    # The card re-enumerates its nodes after a profile switch.
    time.sleep(0.8)
    refreshed = detect_audio_route()
    refreshed.warnings = route.warnings + refreshed.warnings

    # Prefer the explicit A2DP output node when WirePlumber did not move the
    # default sink itself.
    if not refreshed.is_a2dp:
        sink_nodes = _a2dp_sink_nodes(_pw_dump_objects())
        if sink_nodes:
            a2dp_sink = sink_nodes[0]
            if _set_default_sink(int(a2dp_sink["id"])):
                time.sleep(0.3)
                refreshed = detect_audio_route()
                refreshed.warnings = route.warnings + refreshed.warnings

    return refreshed


def detect_audio_route() -> AudioRoute:
    """Inspect the current default sink and classify it (Bluetooth/A2DP/HFP)."""
    if shutil.which("wpctl"):
        inspect = _run(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"])
        if inspect:
            props = _parse_wpctl_inspect(inspect)
            route = AudioRoute(
                sink_id=_node_id_from_inspect(inspect),
                sink_name=props.get("node.name", ""),
                description=props.get("device.description", "")
                or props.get("node.description", ""),
                driver="pipewire",
                profile_name=props.get("api.bluez5.profile", "")
                or props.get("device.profile.name", ""),
                profile_description=props.get("device.profile.description", ""),
                codec=props.get("api.bluez5.codec", ""),
            )
            haystack = " ".join(
                [
                    route.sink_name,
                    route.description,
                    route.profile_name,
                    route.profile_description,
                ]
            ).lower()
            route.is_bluetooth = (
                "bluez" in route.sink_name.lower()
                or str(props.get("device.api", "")).lower() == "bluez5"
                or "bluetooth" in route.description.lower()
            )
            route.is_hfp = any(marker in haystack for marker in _HFP_MARKERS)
            route.is_a2dp = route.is_bluetooth and not route.is_hfp
            if route.is_bluetooth and not route.is_hfp:
                # WirePlumber may not expose api.bluez5.profile on the sink
                # node; a bluez sink without an HFP marker is the normal A2DP
                # media route.
                route.is_a2dp = True

            if route.is_bluetooth and route.is_hfp:
                route.warnings.append(
                    "Bluetooth earbuds are in HFP/HSP (mono 16 kHz) mode; speech will "
                    "sound telephone-like. Trying to switch to A2DP."
                )
                route = _try_enable_a2dp(route)
            return route

    # ALSA / non-PipeWire fallback.
    route = AudioRoute(driver="alsa")
    cards = _run(["aplay", "-l"]) or ""
    for line in cards.splitlines():
        if "card " in line and "device " in line:
            route.sink_name = line.strip()
            break
    if route.sink_name:
        route.description = route.sink_name
    return route


def playback_commands(wav_path: str) -> List[List[str]]:
    """Return candidate commands to play a WAV on the current default route.

    PipeWire's ALSA plugin (``aplay -D default``) is the most reliable path on
    both the laptop and the Pi because it automatically follows the configured
    default sink, including Bluetooth A2DP. Native PipeWire/Pulse players are
    tried next, then plain ALSA.
    """
    commands: List[List[str]] = []
    if shutil.which("aplay"):
        commands.append(["aplay", "-D", "default", "-q", wav_path])
        commands.append(["aplay", "-q", wav_path])
    if shutil.which("pw-play"):
        commands.append(["pw-play", "--volume", "1.0", wav_path])
    if shutil.which("paplay"):
        commands.append(["paplay", wav_path])
    if shutil.which("ffplay"):
        commands.append(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path]
        )
    return commands
