"""Real SmolVLM2 inference with strict sensor-grounded output validation."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from config.logging_config import get_logger

logger = get_logger("cognitive.vlm")
InferenceFunction = Callable[[np.ndarray, str], str]

HAZARD_TERMS = {
    "person", "chair", "table", "car", "bicycle", "stairs", "stair", "curb",
    "pole", "door", "glass", "obstacle", "drop-off", "dropoff", "vehicle",
    "furniture", "barrier", "branch", "hanging obstacle",
}
UNSAFE_INSTRUCTIONS = {"proceed", "cross", "turn", "step", "walk", "safe to", "path is safe"}


def direction_for_azimuth(azimuth_deg: float) -> str:
    if azimuth_deg < -15.0:
        return "left"
    if azimuth_deg > 15.0:
        return "right"
    return "ahead"


class GroundedSmolVLMNarrator:
    """Run SmolVLM2 and accept only claims supported by trusted perception."""

    REQUIRED_KEYS = {"path_clear", "primary_hazard", "direction", "distance_m", "speech"}

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        model_path: Optional[str] = "models/smolvlm2-500m",
        max_tokens: int = 96,
        temperature: float = 0.0,
        timeout_s: float = 12.0,
        max_image_edge: int = 512,
        local_files_only: bool = True,
        system_prompt: Optional[str] = None,
        inference_fn: Optional[InferenceFunction] = None,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.max_tokens = int(max_tokens)
        self.temperature = 0.0
        self.timeout_s = float(timeout_s)
        self.max_image_edge = max(224, int(max_image_edge))
        self.local_files_only = bool(local_files_only)
        self.system_prompt = system_prompt or self._default_prompt()
        self._inference_fn = inference_fn
        self._processor = None
        self._model = None
        self._load_lock = threading.Lock()
        self.is_loaded = inference_fn is not None
        self.backend = "injected_test_backend" if inference_fn else "smolvlm2_transformers"
        self.last_error: Optional[str] = None

    @staticmethod
    def _default_prompt() -> str:
        return (
            "You describe a camera image for a blind user. Trusted sensor facts are supplied below. "
            "Return exactly one JSON object with keys path_clear, primary_hazard, direction, "
            "distance_m, and speech. Never invent or estimate distance. Never add an immediate "
            "hazard absent from trusted facts. Never give movement instructions. Sensor facts "
            "override the image. Keep speech natural, calm, and under 28 words."
        )

    def _load(self) -> None:
        if self._inference_fn is not None or self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                source = self.model_name
                if self.model_path and Path(self.model_path).is_dir():
                    source = self.model_path
                self._processor = AutoProcessor.from_pretrained(
                    source, local_files_only=self.local_files_only
                )
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                self._model = AutoModelForImageTextToText.from_pretrained(
                    source,
                    dtype=dtype,
                    device_map="auto",
                    local_files_only=self.local_files_only,
                    low_cpu_mem_usage=True,
                )
                self._model.eval()
                self.is_loaded = True
                self.last_error = None
            except Exception as exc:
                self.is_loaded = False
                self.last_error = str(exc)
                raise RuntimeError(f"SmolVLM2 unavailable: {exc}") from exc

    def _generate(self, bgr_image: np.ndarray, prompt: str) -> str:
        if self._inference_fn is not None:
            return self._inference_fn(bgr_image, prompt)
        self._load()
        import torch

        assert self._processor is not None and self._model is not None
        rgb_image = Image.fromarray(bgr_image[:, :, ::-1].astype(np.uint8), mode="RGB")
        rgb_image.thumbnail(
            (self.max_image_edge, self.max_image_edge),
            Image.Resampling.LANCZOS,
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": rgb_image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        input_length = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_tokens,
                max_time=self.timeout_s, use_cache=True,
            )
        return self._processor.batch_decode(
            generated[:, input_length:], skip_special_tokens=True
        )[0].strip()

    @staticmethod
    def _trusted_facts(obstacles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        facts = []
        for obstacle in obstacles:
            source = str(obstacle.get("fusion_source", ""))
            measured = source in {"tof_laser_fusion", "tof_only"}
            facts.append({
                "object": str(obstacle.get("class_name", "obstacle")),
                "direction": direction_for_azimuth(float(obstacle.get("azimuth_deg", 0.0))),
                "azimuth_deg": round(float(obstacle.get("azimuth_deg", 0.0)), 1),
                "distance_m": round(float(obstacle["distance_m"]), 2) if measured else None,
                "distance_source": "physical_tof" if measured else "unverified_geometry",
                "confirmation_frames": int(obstacle.get("confirmation_frames", 1)),
            })
        return sorted(
            facts,
            key=lambda item: item["distance_m"] if item["distance_m"] is not None else 999.0,
        )

    def _build_prompt(self, facts: List[Dict[str, Any]], path_clear: bool) -> str:
        if facts:
            primary = facts[0]
            distance = (
                f" {primary['distance_m']:g} metres"
                if primary["distance_m"] is not None
                else ""
            )
            direction_phrase = (
                "ahead"
                if primary["direction"] == "ahead"
                else f"to your {primary['direction']}"
            )
            speech_example = (
                f"{primary['object'].capitalize()}{distance} {direction_phrase}."
            )
        else:
            speech_example = "Describe only visible context; do not claim that it is safe to move."
        schema = {
            "path_clear": path_clear,
            "primary_hazard": facts[0]["object"] if facts else None,
            "direction": facts[0]["direction"] if facts else None,
            "distance_m": facts[0]["distance_m"] if facts else None,
            "speech": speech_example,
        }
        return (
            f"{self.system_prompt}\n\n"
            f"TRUSTED_SENSOR_FACTS={json.dumps(facts, separators=(',', ':'))}\n"
            f"AUTHORITATIVE_PATH_CLEAR={json.dumps(path_clear)}\n"
            f"A minimally valid grounded answer is: {json.dumps(schema, separators=(',', ':'))}. "
            "You may make speech more natural and add one brief visual-context clause, but must "
            "preserve every grounded fact exactly."
        )

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("model did not return a JSON object")
        value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("model output is not an object")
        return value

    def _validate(
        self,
        candidate: Dict[str, Any],
        facts: List[Dict[str, Any]],
        authoritative_clear: bool,
    ) -> Dict[str, Any]:
        if set(candidate) != self.REQUIRED_KEYS:
            raise ValueError("model output has missing or unexpected keys")
        if not isinstance(candidate["path_clear"], bool) or candidate["path_clear"] != authoritative_clear:
            raise ValueError("path_clear contradicts authoritative perception")

        primary = facts[0] if facts else None
        expected = {
            "primary_hazard": primary["object"] if primary else None,
            "direction": primary["direction"] if primary else None,
            "distance_m": primary["distance_m"] if primary else None,
        }
        for key, expected_value in expected.items():
            actual = candidate[key]
            if key == "distance_m" and expected_value is not None:
                if not isinstance(actual, (int, float)) or abs(float(actual) - expected_value) > 0.05:
                    raise ValueError("distance is not supported by physical ToF")
            elif actual != expected_value:
                raise ValueError(f"{key} contradicts trusted perception")

        speech = candidate["speech"]
        if not isinstance(speech, str) or not speech.strip() or len(speech.split()) > 28:
            raise ValueError("speech must contain 1-28 words")
        lowered = speech.lower()
        if lowered in {"short natural description", "natural description", "speech"}:
            raise ValueError("speech copied a schema placeholder")
        if any(phrase in lowered for phrase in UNSAFE_INSTRUCTIONS):
            raise ValueError("speech contains an unsafe movement instruction")
        if not authoritative_clear and re.search(r"\b(path|way)\b.{0,12}\bclear\b", lowered):
            raise ValueError("speech makes an unsupported clear-path claim")
        trusted_names = {fact["object"].lower() for fact in facts}
        for term in HAZARD_TERMS:
            if re.search(rf"\b{re.escape(term)}s?\b", lowered) and not any(
                term in name or name in term for name in trusted_names
            ):
                raise ValueError(f"speech contains unsupported hazard: {term}")
        claims = [
            float(value)
            for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)\b", lowered)
        ]
        distances = {fact["distance_m"] for fact in facts if fact["distance_m"] is not None}
        if any(not any(abs(claim - value) <= 0.05 for value in distances) for claim in claims):
            raise ValueError("speech contains an unsupported metric distance")
        if primary:
            primary_name = primary["object"].lower()
            if primary_name not in lowered:
                raise ValueError("speech omits the trusted primary hazard")
            direction_words = {
                "left": ("left",),
                "right": ("right",),
                "ahead": ("ahead", "front", "straight"),
            }[primary["direction"]]
            if not any(word in lowered for word in direction_words):
                raise ValueError("speech omits the trusted direction")
            if primary["distance_m"] is not None and not any(
                abs(claim - primary["distance_m"]) <= 0.05 for claim in claims
            ):
                raise ValueError("speech omits the physical ToF distance")
        return {**candidate, "speech": speech.strip()}

    def describe_scene(
        self,
        image: Optional[np.ndarray],
        detected_obstacles: Optional[List[Dict[str, Any]]] = None,
        navigation_authoritative: bool = False,
        path_state: str = "unknown",
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {"success": False, "narration": "Scene narration unavailable.", "error": "empty image"}
        facts = self._trusted_facts(detected_obstacles or [])
        authoritative_clear = bool(navigation_authoritative and path_state == "clear" and not facts)
        prompt = self._build_prompt(facts, authoritative_clear)
        raw = ""
        try:
            raw = self._generate(image, prompt)
            validated = self._validate(self._parse_json(raw), facts, authoritative_clear)
            return {
                "success": True,
                "narration": validated["speech"],
                "structured": validated,
                "trusted_facts": facts,
                "model": self.model_name,
                "backend": self.backend,
                "model_output_accepted": True,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            }
        except Exception as exc:
            logger.warning("Grounded VLM narration rejected: %s", exc)
            return {
                "success": False,
                "narration": "Scene narration unavailable. Continue using immediate obstacle cues.",
                "trusted_facts": facts,
                "model": self.model_name,
                "backend": self.backend,
                "model_output_accepted": False,
                "raw_model_output": raw[:2048],
                "error": str(exc),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            }


SpatialSceneNarrator = GroundedSmolVLMNarrator
VisionLanguageNarrator = GroundedSmolVLMNarrator
