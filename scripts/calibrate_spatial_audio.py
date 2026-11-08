#!/usr/bin/env python3
"""Generate lateralization cues and persist a user/headset audio profile."""

from __future__ import annotations

import argparse
import math
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.spatial_audio import SpatialAudioSynthesizer


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    samples = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["air_conduction", "bone_conduction"], required=True)
    parser.add_argument("--itd-scale", type=float, required=True)
    parser.add_argument("--ild-scale", type=float, required=True)
    parser.add_argument("--left-gain", type=float, default=1.0)
    parser.add_argument("--right-gain", type=float, default=1.0)
    parser.add_argument("--output-dir", default="var/audio-calibration")
    args = parser.parse_args()
    if not 0.0 <= args.itd_scale <= 1.5 or not 0.0 <= args.ild_scale <= 1.5:
        raise SystemExit("ITD/ILD scales must be in 0..1.5")
    if not 0.5 <= args.left_gain <= 1.5 or not 0.5 <= args.right_gain <= 1.5:
        raise SystemExit("Channel gains must be in 0.5..1.5")

    profile = {args.profile: {
        "calibrated": True, "itd_scale": args.itd_scale, "ild_scale": args.ild_scale,
        "left_gain": args.left_gain, "right_gain": args.right_gain,
    }}
    synth = SpatialAudioSynthesizer(audio_profile=args.profile, profiles=profile)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    silence = np.zeros((int(synth.sample_rate * 0.15), 2), dtype=np.float32)
    for angle in (-90, -45, 0, 45, 90):
        cue = synth.synthesize_obstacle_ping(math.radians(angle), 1.5, 0.18)
        sequence = np.vstack([cue, silence, cue, silence, cue])
        write_wav(output / f"azimuth_{angle:+d}.wav", sequence, synth.sample_rate)
    (output / f"{args.profile}.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    print(f"Wrote five test cues and profile to {output}")
    print("Verify that + angles are right, - angles are left, and 0 is centred before copying the profile into pipeline_config.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
