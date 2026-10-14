"""Perception Subsystem: Indian Rupee (INR) Banknote Classifier.

Identifies Mahatma Gandhi New Series banknotes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500)
using color chromaticity distribution, aspect ratios, and feature classification.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


class IndianCurrencyRecognizer:
    """Classifies Indian Rupee paper banknotes for visually impaired assistance."""

    # Reference specifications for Mahatma Gandhi New Series
    NOTE_PROFILES = {
        10: {
            "name": "Ten rupees",
            "dimensions_mm": (123, 63),
            "aspect_ratio": 1.95,
            "hue_center_deg": 18.0,  # Chocolate Brown
        },
        20: {
            "name": "Twenty rupees",
            "dimensions_mm": (129, 63),
            "aspect_ratio": 2.05,
            "hue_center_deg": 48.0,  # Greenish Yellow
        },
        50: {
            "name": "Fifty rupees",
            "dimensions_mm": (135, 66),
            "aspect_ratio": 2.04,
            "hue_center_deg": 195.0, # Fluorescent Blue
        },
        100: {
            "name": "One hundred rupees",
            "dimensions_mm": (142, 66),
            "aspect_ratio": 2.15,
            "hue_center_deg": 265.0, # Lavender
        },
        200: {
            "name": "Two hundred rupees",
            "dimensions_mm": (146, 66),
            "aspect_ratio": 2.21,
            "hue_center_deg": 55.0,  # Bright Yellow
        },
        500: {
            "name": "Five hundred rupees",
            "dimensions_mm": (150, 66),
            "aspect_ratio": 2.27,
            "hue_center_deg": 110.0, # Stone Grey / Greenish
        },
    }

    def __init__(self, confidence_threshold: float = 0.70) -> None:
        self.conf_thresh = confidence_threshold

    @staticmethod
    def rgb_to_hsv_fast(rgb_image: np.ndarray) -> np.ndarray:
        """Vectorized conversion of RGB image to HSV without external dependencies."""
        img = rgb_image.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c

        # Value
        v = max_c

        # Saturation
        s = np.zeros_like(max_c)
        nonzero = max_c > 1e-5
        s[nonzero] = delta[nonzero] / max_c[nonzero]

        # Hue in degrees [0, 360]
        h = np.zeros_like(max_c)
        delta_nonzero = delta > 1e-5

        # When R is max
        mask_r = delta_nonzero & (max_c == r)
        h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)

        # When G is max
        mask_g = delta_nonzero & (max_c == g)
        h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)

        # When B is max
        mask_b = delta_nonzero & (max_c == b)
        h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

        return np.stack([h, s, v], axis=-1)

    def classify_banknote(
        self, crop_image: np.ndarray, bbox_aspect_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """Classify a cropped banknote image.
        
        Args:
            crop_image: RGB image containing the banknote.
            bbox_aspect_ratio: Optional width / height ratio of the bounding box.
            
        Returns:
            Dict[str, Any]: Classification result with denomination, spoken name, and confidence.
        """
        if crop_image is None or crop_image.size == 0:
            return {"denomination": None, "spoken_name": "No note detected", "confidence": 0.0}

        hsv = self.rgb_to_hsv_fast(crop_image)
        # Focus on saturated pixels to capture ink pigments
        sat_mask = hsv[:, :, 1] > 0.15
        if np.sum(sat_mask) < 20:
            mean_hue = float(np.mean(hsv[:, :, 0]))
        else:
            mean_hue = float(np.mean(hsv[:, :, 0][sat_mask]))

        # Calculate distances to known note profiles
        best_denomination = None
        min_error = float("inf")

        for denom, profile in self.NOTE_PROFILES.items():
            ref_hue = profile["hue_center_deg"]
            # Circular angular difference on the hue circle
            hue_diff = min(abs(mean_hue - ref_hue), 360.0 - abs(mean_hue - ref_hue))

            # Aspect ratio error if available
            aspect_err = 0.0
            if bbox_aspect_ratio is not None and bbox_aspect_ratio > 0:
                aspect_err = abs(bbox_aspect_ratio - profile["aspect_ratio"]) * 30.0

            total_err = hue_diff + aspect_err
            if total_err < min_error:
                min_error = total_err
                best_denomination = denom

        # Convert error to confidence score
        confidence = float(np.exp(-min_error / 45.0))
        confidence = round(float(np.clip(confidence, 0.1, 0.99)), 3)

        if confidence >= self.conf_thresh and best_denomination is not None:
            spoken = self.NOTE_PROFILES[best_denomination]["name"]
            return {
                "denomination": best_denomination,
                "spoken_name": spoken,
                "confidence": confidence,
            }
        else:
            return {
                "denomination": None,
                "spoken_name": "Unrecognized currency",
                "confidence": confidence,
            }

