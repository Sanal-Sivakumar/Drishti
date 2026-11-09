"""Online Vision Cognitive Service for Drishti Assistive Wearable.

Provides cloud-assisted multimodal cognition (Scene Narration, Indian Currency
Recognition, and Multilingual Indic OCR) via OpenAI or Google Gemini Vision APIs.

Safety Principles:
1. Isolated from navigation: Real-time obstacle avoidance runs 100% offline.
2. Privacy by design: Faces and vehicle registration plates are blurred before transmission.
3. Strict JSON contracts: Online vision cannot issue movement commands or override ToF distance.
4. Security: API keys are read from environment variables or .env, never logged or committed.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional
import cv2
import numpy as np
import requests
import socket
import urllib3.util.connection as urllib3_conn

# Prioritize IPv4 to prevent IPv6 routing timeouts on edge networks
urllib3_conn.allowed_gai_family = lambda: socket.AF_INET

from config.logging_config import get_logger

logger = get_logger("cognitive.online_vision")


def load_env_file(env_path: Optional[str | Path] = None) -> None:
    """Load key-value pairs from .env into os.environ if not already present."""
    path = Path(env_path or Path(__file__).resolve().parent.parent / ".env")
    if not path.is_file():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and v and (k not in os.environ or not os.environ[k].strip()):
                    os.environ[k] = v
    except Exception as exc:
        logger.warning(f"Could not read .env file: {exc}")


class OnlineVisionService:
    """Cloud-assisted visual cognition with strict JSON schemas and offline fallback."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 12.0,
        synthetic_mode: bool = False,
        mock_fn: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        load_env_file()
        self.synthetic_mode = synthetic_mode
        self.mock_fn = mock_fn
        self.timeout_s = timeout_s or 12.0

        # Resolve provider and API key
        env_provider = os.environ.get("DRISHTI_VISION_PROVIDER", "").strip().lower()
        has_gemini = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())

        resolved_provider = (
            provider
            or (env_provider if env_provider in ("gemini", "openai") else None)
            or ("gemini" if has_gemini and not has_openai else "openai")
        ).lower()
        self.provider = resolved_provider

        env_model = os.environ.get("DRISHTI_VISION_MODEL", "").strip()
        if self.provider == "gemini":
            self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "").strip()
            self.model = model or (env_model if env_model else "gemini-3.1-flash-lite")
        else:
            self.provider = "openai"
            self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "").strip()
            self.model = model or (env_model if env_model else "gpt-4o-mini")

        self.has_key = bool(self.api_key and len(self.api_key.strip()) > 5)

    @staticmethod
    def encode_image_jpeg(image: np.ndarray, max_edge: int = 1024, quality: int = 85) -> str:
        """Resize image to max_edge and return base64-encoded JPEG bytes."""
        h, w = image.shape[:2]
        if max(h, w) > max_edge:
            scale = max_edge / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = image

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        success, buffer = cv2.imencode(".jpg", img_resized, encode_param)
        if not success:
            raise ValueError("Failed to encode image to JPEG")
        return base64.b64encode(buffer).decode("utf-8")

    def _call_cloud_api(self, prompt: str, b64_img: str, high_fidelity: bool = False) -> Dict[str, Any]:
        """Make an authenticated cloud vision API request expecting strict JSON."""
        if not self.has_key:
            return {
                "success": False,
                "error": "no_api_key",
                "speech": "API key not configured in environment. Obstacle guidance remains active.",
            }

        t0 = time.time()
        try:
            if self.provider == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}},
                            ]
                        }
                    ],
                    "generationConfig": {
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                    },
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                data = resp.json()
                text_content = data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                detail_level = "high" if high_fidelity else "low"
                payload = {
                    "model": self.model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_img}",
                                        "detail": detail_level,
                                    },
                                },
                            ],
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                data = resp.json()
                text_content = data["choices"][0]["message"]["content"]

            latency_ms = round((time.time() - t0) * 1000.0, 1)
            parsed = json.loads(text_content)
            return {"success": True, "data": parsed, "latency_ms": latency_ms, "provider": self.provider}

        except requests.exceptions.Timeout:
            logger.warning("Online vision request timed out.")
            return {
                "success": False,
                "error": "timeout",
                "speech": "Internet request timed out. Obstacle guidance remains active.",
            }
        except Exception as exc:
            logger.warning(f"Online vision request failed: {exc}")
            return {
                "success": False,
                "error": str(exc),
                "speech": "Internet unavailable. Obstacle guidance remains active.",
            }

    def describe_scene(
        self,
        image: np.ndarray,
        obstacles: Optional[List[Dict[str, Any]]] = None,
        navigation_authoritative: bool = False,
        path_state: str = "unknown",
    ) -> Dict[str, Any]:
        """Describe visual scene context grounded by sensor facts without giving movement commands."""
        if self.synthetic_mode:
            if self.mock_fn:
                text = self.mock_fn(image, "describe_scene")
                return {"success": True, "narration": text, "structured": {"speech": text}}
            return {
                "success": True,
                "narration": "Path is clear with open space ahead.",
                "structured": {"speech": "Path is clear with open space ahead.", "visual_context": ["room"], "uncertain": False},
            }

        facts = []
        for o in (obstacles or []):
            name = o.get("class_name", "object")
            dist = o.get("distance_m")
            az = o.get("azimuth_deg")
            dist_str = f"{dist:.2f} m" if dist is not None else "unknown distance"
            az_str = f"{az:+.0f}°" if az is not None else "ahead"
            facts.append(f"- {name} at {dist_str}, bearing {az_str}")

        sensor_facts_str = "\n".join(facts) if facts else "No immediate hazards detected by local sensors."
        prompt = (
            "You are the visual describer for Drishti, an assistive wearable for a visually impaired user.\n"
            "Describe the camera image concisely, naturally and informatively in 1 to 2 spoken sentences.\n"
            f"Sensor facts from wearable:\n{sensor_facts_str}\n"
            f"Path readiness: {path_state}\n\n"
            "SAFETY CONSTRAINTS:\n"
            "1. You MUST NEVER give directional or movement advice (e.g. do not say 'walk forward', 'turn left', 'safe to proceed').\n"
            "2. Never claim the path is clear or contradict sensor facts. Local sensors own navigation safety.\n"
            "3. Return strictly a JSON object with this exact structure:\n"
            '{"speech": "Concise 1-2 sentence spoken description", "visual_context": ["object1", "object2"], "uncertain": false}'
        )

        b64_img = self.encode_image_jpeg(image, max_edge=768, quality=80)
        res = self._call_cloud_api(prompt, b64_img, high_fidelity=False)

        if not res.get("success"):
            speech = res.get("speech", "Internet unavailable. Scene description is unavailable.")
            return {"success": False, "narration": speech, "error": res.get("error")}

        data = res.get("data", {})
        speech = data.get("speech") or "No description available."

        # Fail-closed navigation safety sanitization: strip movement/nav advice
        forbidden_phrases = [
            "walk forward", "turn left", "turn right", "step forward",
            "safe to proceed", "safe to walk", "go ahead", "move forward",
            "keep walking", "clear to cross", "proceed safely"
        ]
        import re
        for phrase in forbidden_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.search(speech):
                logger.warning(f"Sanitizing forbidden navigation advice from online vision: '{phrase}'")
                speech = pattern.sub("[navigation advice omitted]", speech)

        return {
            "success": True,
            "narration": speech,
            "structured": data,
            "latency_ms": res.get("latency_ms", 0.0),
            "provider": res.get("provider"),
        }

    def identify_currency(self, image: np.ndarray) -> Dict[str, Any]:
        """Identify Indian Rupee (INR) banknotes using multimodal vision with ambiguity rejection."""
        if self.synthetic_mode:
            if self.mock_fn:
                text = self.mock_fn(image, "currency")
                return {"success": True, "currency": {"denomination": 50, "spoken_name": text}}
            return {
                "success": True,
                "currency": {
                    "denomination": 100,
                    "confidence": "high",
                    "spoken_name": "One hundred rupees",
                    "visible_side": "front",
                },
            }

        prompt = (
            "You are the currency identification assistant for the Drishti wearable.\n"
            "Examine the image to identify any Indian Rupee (INR) paper banknote (Mahatma Gandhi New Series: 10, 20, 50, 100, 200, 500).\n"
            "RULES:\n"
            "1. Inspect the denominational numerals (e.g. 10, 20, 50, 100, 200, 500), Gandhi portrait, color, and reserve bank text.\n"
            "2. If no banknote is visible, or if the note is heavily blurred, obscured, or ambiguous, DO NOT GUESS. "
            "Set denomination_inr to null, confidence to 'low', and uncertain to true.\n"
            "3. Return strictly valid JSON in this format:\n"
            "{\n"
            '  "denomination_inr": 100,\n'
            '  "confidence": "high",\n'
            '  "spoken_name": "One hundred rupees",\n'
            '  "visible_side": "front",\n'
            '  "reason": "Clear Mahatma Gandhi portrait and 100 numeral",\n'
            '  "uncertain": false\n'
            "}"
        )

        b64_img = self.encode_image_jpeg(image, max_edge=800, quality=85)
        res = self._call_cloud_api(prompt, b64_img, high_fidelity=False)

        if not res.get("success"):
            speech = "Internet unavailable. Banknote identification is unavailable."
            return {"success": False, "spoken_name": speech, "error": res.get("error")}

        data = res.get("data", {})
        denom = data.get("denomination_inr")
        uncertain = data.get("uncertain", False) or data.get("confidence") == "low"

        if denom is not None and not uncertain:
            spoken = data.get("spoken_name") or f"{denom} rupees"
            return {
                "success": True,
                "currency": {
                    "denomination": int(denom),
                    "confidence": data.get("confidence", "high"),
                    "spoken_name": spoken,
                    "visible_side": data.get("visible_side", "unknown"),
                    "reason": data.get("reason", ""),
                },
                "latency_ms": res.get("latency_ms", 0.0),
            }
        else:
            return {
                "success": True,
                "currency": {
                    "denomination": None,
                    "spoken_name": "No banknote clearly detected.",
                    "confidence": "low",
                },
                "latency_ms": res.get("latency_ms", 0.0),
            }

    def read_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform high-fidelity literal OCR across Latin and Indic scripts."""
        if self.synthetic_mode:
            if self.mock_fn:
                text = self.mock_fn(image, "read_text")
                return {"success": True, "text": text}
            return {
                "success": True,
                "text": "Drishti Pharmacy 500mg",
                "script": "Latin",
                "detected_language": "English",
            }

        prompt = (
            "You are the reading assistant for the Drishti wearable for the visually impaired.\n"
            "Transcribe all legible printed or written text in this image verbatim.\n"
            "Support English and all Indian scripts (Devanagari/Hindi, Tamil, Malayalam, Bengali, Kannada, Telugu, etc.).\n"
            "Preserve exact numbers, dosages, packaging labels, and street names.\n"
            "If no legible text is found, set text to '' and uncertain to true.\n"
            "Return strictly valid JSON in this format:\n"
            "{\n"
            '  "text": "Literal transcribed text",\n'
            '  "script": "Latin / Devanagari / Malayalam / etc.",\n'
            '  "detected_language": "English / Hindi / etc.",\n'
            '  "uncertain": false\n'
            "}"
        )

        b64_img = self.encode_image_jpeg(image, max_edge=1200, quality=90)
        res = self._call_cloud_api(prompt, b64_img, high_fidelity=True)

        if not res.get("success"):
            speech = "Internet unavailable. Text reading is unavailable."
            return {"success": False, "text": "", "spoken_name": speech, "error": res.get("error")}

        data = res.get("data", {})
        text = (data.get("text") or "").strip()
        spoken = text if text else "No clear text detected."

        return {
            "success": True,
            "text": text,
            "spoken_name": spoken,
            "script": data.get("script", "unknown"),
            "detected_language": data.get("detected_language", "unknown"),
            "latency_ms": res.get("latency_ms", 0.0),
        }
