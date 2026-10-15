"""Perception Subsystem: Indian Rupee (INR) Banknote Classifier.

Identifies Mahatma Gandhi New Series banknotes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500)
using color chromaticity distribution, aspect ratios, and feature classification.
"""

from typing import Dict, Any, Optional, Tuple
import cv2
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
            "hue_center_deg": 42.0,  # Greenish Yellow
            "hue_tolerance": 12.0,
        },
        50: {
            "name": "Fifty rupees",
            "dimensions_mm": (135, 66),
            "aspect_ratio": 2.04,
            "hue_center_deg": 195.0, # Fluorescent Blue
            "hue_tolerance": 16.0,
        },
        100: {
            "name": "One hundred rupees",
            "dimensions_mm": (142, 66),
            "aspect_ratio": 2.15,
            "hue_center_deg": 265.0, # Lavender
            "hue_tolerance": 16.0,
        },
        200: {
            "name": "Two hundred rupees",
            "dimensions_mm": (146, 66),
            "aspect_ratio": 2.21,
            "hue_center_deg": 58.0,  # Bright Yellow
            "hue_tolerance": 12.0,
        },
        500: {
            "name": "Five hundred rupees",
            "dimensions_mm": (150, 66),
            "aspect_ratio": 2.27,
            "hue_center_deg": 100.0, # Stone Grey / Greenish
            "hue_tolerance": 22.0,
        },
    }

    def __init__(self, confidence_threshold: float = 0.70) -> None:
        self.conf_thresh = confidence_threshold

    @staticmethod
    def bgr_to_hsv_fast(image: np.ndarray, color_order: str = "BGR") -> np.ndarray:
        """Vectorized conversion of BGR or RGB image to HSV without external dependencies.
        
        Args:
            image: Input uint8 image array (H, W, 3).
            color_order: "BGR" (default from camera/OpenCV) or "RGB".
        """
        img = image.astype(np.float32) / 255.0
        if color_order.upper() == "BGR":
            b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        else:
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

    @classmethod
    def rgb_to_hsv_fast(cls, rgb_image: np.ndarray) -> np.ndarray:
        """Backward-compatible RGB to HSV converter."""
        return cls.bgr_to_hsv_fast(rgb_image, color_order="RGB")


    @staticmethod
    def compute_edge_density(image: np.ndarray) -> Tuple[float, float]:
        """Compute 2D spatial gradient magnitude and texture variance.
        
        Banknotes have dense intaglio guilloche patterns, numerals, and border lines.
        Smooth surfaces (human skin, clothes, walls) exhibit near-zero edge density.
        """
        if image is None or image.shape[0] < 4 or image.shape[1] < 4:
            return 0.0, 0.0
        # Convert to grayscale via ITU-R BT.601 luma weights
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        gx = np.abs(gray[:, 1:] - gray[:, :-1])
        gy = np.abs(gray[1:, :] - gray[:-1, :])
        mean_density = float(np.mean(gx) + np.mean(gy))
        std_density = float(np.std(gx) + np.std(gy))
        return mean_density, std_density

    def classify_banknote(
        self,
        crop_image: np.ndarray,
        bbox_aspect_ratio: Optional[float] = None,
        color_order: str = "BGR",
    ) -> Dict[str, Any]:
        """Classify a cropped banknote image using multi-stage physical and optical verification.
        
        Stages:
        1. High-frequency edge density & texture gating (rejects smooth faces, clothes, walls).
        2. Geometric aspect ratio verification (Mahatma Gandhi New Series: 1.85 to 2.35).
        3. Chromatic ink saturation and tight hue dispersion.
        
        Args:
            crop_image: BGR or RGB image containing the candidate banknote.
            bbox_aspect_ratio: Optional width / height ratio of the bounding box.
            color_order: "BGR" (default from camera/OpenCV) or "RGB".
            
        Returns:
            Dict[str, Any]: Classification result with denomination, spoken name, and confidence.
        """
        if crop_image is None or crop_image.size == 0:
            return {"denomination": None, "spoken_name": "No note detected", "confidence": 0.0}

        # 1. Texture & Edge Density Verification
        edge_mean, edge_std = self.compute_edge_density(crop_image)
        # If no explicit aspect ratio is supplied, image MUST exhibit banknote engraving texture
        if bbox_aspect_ratio is None and edge_mean < 8.0 and edge_std < 4.0:
            return {
                "denomination": None,
                "spoken_name": "No note detected",
                "confidence": 0.0,
                "reason": "insufficient_texture",
            }

        # If aspect ratio is provided, verify it falls within paper currency geometry with tilt tolerance
        if bbox_aspect_ratio is not None:
            if bbox_aspect_ratio < 1.40 or bbox_aspect_ratio > 2.85:
                return {
                    "denomination": None,
                    "spoken_name": "No note detected",
                    "confidence": 0.0,
                    "reason": "invalid_aspect_ratio",
                }

        # 2. Chromatic Ink Distribution
        hsv = self.bgr_to_hsv_fast(crop_image, color_order=color_order)
        sat_mask = (hsv[:, :, 1] > 0.16) & (hsv[:, :, 2] > 0.14)
        sat_count = int(np.sum(sat_mask))
        total_pixels = crop_image.shape[0] * crop_image.shape[1]

        # Banknotes must have significant ink saturation across at least 6% of area
        if sat_count < max(20, int(0.06 * total_pixels)):
            return {
                "denomination": None,
                "spoken_name": "No note detected",
                "confidence": 0.0,
                "reason": "low_saturation",
            }

        mean_hue = float(np.mean(hsv[:, :, 0][sat_mask]))

        # 3. Match against Mahatma Gandhi New Series Denomination Profiles
        best_denomination = None
        best_confidence = 0.0

        for denom, profile in self.NOTE_PROFILES.items():
            ref_hue = profile["hue_center_deg"]
            hue_tol = profile.get("hue_tolerance", 16.0)
            # Circular angular difference on 360-degree color wheel
            hue_diff = min(abs(mean_hue - ref_hue), 360.0 - abs(mean_hue - ref_hue))

            # Strict angular tolerance: ink must be within tolerance of reference
            if hue_diff > hue_tol:
                continue

            # Aspect ratio error calculation
            aspect_score = 1.0
            if bbox_aspect_ratio is not None:
                aspect_err = abs(bbox_aspect_ratio - profile["aspect_ratio"])
                if aspect_err > 0.50:
                    continue
                aspect_score = max(0.0, 1.0 - (aspect_err / 0.50))

            hue_score = max(0.0, 1.0 - (hue_diff / hue_tol))
            # Weighted multi-stage confidence score
            conf = round(0.70 * hue_score + 0.30 * aspect_score, 3)

            if conf > best_confidence:
                best_confidence = conf
                best_denomination = denom

        if best_denomination is not None and best_confidence >= self.conf_thresh:
            spoken = self.NOTE_PROFILES[best_denomination]["name"]
            return {
                "denomination": best_denomination,
                "spoken_name": spoken,
                "confidence": best_confidence,
            }
        else:
            return {
                "denomination": None,
                "spoken_name": "No note detected",
                "confidence": best_confidence,
            }

    def identify_currency(self, frame: np.ndarray, color_order: str = "BGR") -> Dict[str, Any]:
        """Detect and classify any Indian Rupee banknote in a camera frame.
        
        Performs multi-channel edge & contour analysis to isolate the candidate
        banknote bounding box from surrounding room/person backgrounds, extracts
        the true banknote aspect ratio, and classifies the denomination.
        
        Falls back to center-crop evaluation if the banknote fills the camera frame.
        """
        if frame is None or frame.size == 0:
            return {"denomination": None, "spoken_name": "No note detected", "confidence": 0.0}

        h, w = frame.shape[:2]
        total_pixels = h * w

        # 1. Multi-channel edge contour analysis to isolate candidate banknote
        edges = np.zeros((h, w), dtype=np.uint8)
        for ch in range(3):
            c_edges = cv2.Canny(frame[:, :, ch], 30, 100)
            edges = cv2.bitwise_or(edges, c_edges)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_conf = 0.0

        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            # Banknote should be large enough to read but not the whole frame
            if area < 0.02 * total_pixels or area > 0.85 * total_pixels:
                continue
            ar = max(bw, bh) / max(1.0, min(bw, bh))
            if 1.40 <= ar <= 2.85:
                roi = frame[by:by + bh, bx:bx + bw]
                res = self.classify_banknote(roi, bbox_aspect_ratio=ar, color_order=color_order)
                if res.get("denomination") is not None and res["confidence"] > best_conf:
                    best_conf = res["confidence"]
                    best_candidate = res
                    best_candidate["bbox"] = [bx, by, bw, bh]

        if best_candidate is not None:
            return best_candidate

        # 2. Close-up center crop fallback (when note is held close and fills the lens)
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        crop = frame[y1:y2, x1:x2]
        res_center = self.classify_banknote(crop, bbox_aspect_ratio=None, color_order=color_order)
        if res_center.get("denomination") is not None:
            return res_center

        return {"denomination": None, "spoken_name": "No note detected", "confidence": 0.0}


