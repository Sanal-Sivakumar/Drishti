#!/usr/bin/env python3
"""Run one real, local SmolVLM2 inference through Drishti's safety validator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cognitive.vlm_narrator import GroundedSmolVLMNarrator


def main() -> int:
    # A deterministic synthetic frame keeps this a repeatable runtime smoke test;
    # it is not an accuracy or field-safety evaluation.
    frame = np.full((288, 384, 3), 230, dtype=np.uint8)
    frame[80:250, 235:330] = (45, 90, 150)
    trusted = [{
        "class_name": "chair",
        "azimuth_deg": 32.0,
        "distance_m": 1.3,
        "fusion_source": "tof_laser_fusion",
        "confirmation_frames": 3,
    }]
    narrator = GroundedSmolVLMNarrator(
        model_path=str(PROJECT_ROOT / "models" / "smolvlm2-500m"),
        local_files_only=True,
        max_tokens=96,
        max_image_edge=384,
        timeout_s=60.0,
    )
    result = narrator.describe_scene(
        frame,
        detected_obstacles=trusted,
        navigation_authoritative=True,
        path_state="obstructed",
    )
    print(json.dumps(result, indent=2))
    # Loading and executing the model is the smoke-test objective. A rejected
    # answer is intentionally a successful fail-closed safety outcome.
    return 0 if narrator.is_loaded else 1


if __name__ == "__main__":
    raise SystemExit(main())
