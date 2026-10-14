"""Drishti On-Demand Cognitive Subsystem."""

from cognitive.ocr_engine import MultilingualOCREngine
from cognitive.vlm_narrator import VisionLanguageNarrator
from cognitive.speech_service import SpeechService

__all__ = [
    "MultilingualOCREngine",
    "VisionLanguageNarrator",
    "SpeechService",
]
