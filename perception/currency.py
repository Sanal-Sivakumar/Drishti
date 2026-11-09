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
            "hue_center_deg": 20.0,  # Chocolate Brown
            "hue_tolerance": 16.0,
        },
        20: {
            "name": "Twenty rupees",
            "dimensions_mm": (129, 63),
            "aspect_ratio": 2.05,
            "hue_center_deg": 42.0,  # Greenish Yellow
            "hue_tolerance": 10.0,
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

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        supported_denominations: Optional[list[int]] = None,
    ) -> None:
        self.conf_thresh = confidence_threshold
        self.supported_denominations = set(supported_denominations or self.NOTE_PROFILES)

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
    def compute_texture_stats(image: np.ndarray) -> Tuple[float, float, float]:
        """Compute edge density, variance and directional anisotropy.

        Banknote intaglio/guilloche engraving is fine and roughly isotropic.
        Smooth skin/clothing has very low edge density; wood grain, table edges
        and fabric seams are strongly directional (one gradient axis dominates).
        """
        if image is None or image.shape[0] < 4 or image.shape[1] < 4:
            return 0.0, 0.0, 1.0
        gray = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
        gx = np.abs(gray[:, 1:] - gray[:, :-1])
        gy = np.abs(gray[1:, :] - gray[:-1, :])
        gx_mean = float(np.mean(gx))
        gy_mean = float(np.mean(gy))
        mean_density = gx_mean + gy_mean
        std_density = float(np.std(gx) + np.std(gy))
        anisotropy = max(gx_mean, gy_mean) / max(1e-3, min(gx_mean, gy_mean))
        return mean_density, std_density, float(anisotropy)

    @staticmethod
    def compute_edge_density(image: np.ndarray) -> Tuple[float, float]:
        """Backward-compatible edge density and variance."""
        mean_density, std_density, _ = IndianCurrencyRecognizer.compute_texture_stats(image)
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
        edge_mean, edge_std, anisotropy = self.compute_texture_stats(crop_image)
        # Reject smooth skin/clothing/walls and strongly directional wood grain.
        # A real note shows fine, roughly isotropic engraving texture.
        if edge_mean < 5.0 and edge_std < 3.0:
            return {
                "denomination": None,
                "spoken_name": "No note detected",
                "confidence": 0.0,
                "reason": "insufficient_texture",
            }
        if anisotropy > 3.2:
            return {
                "denomination": None,
                "spoken_name": "No note detected",
                "confidence": 0.0,
                "reason": "directional_texture",
            }

        # Reject flat, unprinted surfaces (wooden desks, monochrome walls/furniture).
        # Real banknotes feature sharp intaglio contrast, portrait, numerals, and watermarks.
        if color_order.upper() == "BGR":
            gray = 0.114 * crop_image[:, :, 0] + 0.587 * crop_image[:, :, 1] + 0.299 * crop_image[:, :, 2]
        else:
            gray = 0.299 * crop_image[:, :, 0] + 0.587 * crop_image[:, :, 1] + 0.114 * crop_image[:, :, 2]
        gray_std = float(np.std(gray))
        if gray_std < 5.0:
            return {
                "denomination": None,
                "spoken_name": "No note detected",
                "confidence": 0.0,
                "reason": "flat_surface",
            }

        # Verify paper-currency geometry with tilt/perspective tolerance.
        if bbox_aspect_ratio is not None:
            if bbox_aspect_ratio < 1.45 or bbox_aspect_ratio > 2.85:
                return {
                    "denomination": None,
                    "spoken_name": "No note detected",
                    "confidence": 0.0,
                    "reason": "invalid_aspect_ratio",
                }

        # 2. Chromatic Ink Distribution
        hsv = self.bgr_to_hsv_fast(crop_image, color_order=color_order)
        sat_mask = (hsv[:, :, 1] > 0.18) & (hsv[:, :, 2] > 0.15)
        sat_count = int(np.sum(sat_mask))
        total_pixels = crop_image.shape[0] * crop_image.shape[1]

        # Banknotes must have significant ink saturation across at least 10% of area
        if sat_count < max(25, int(0.10 * total_pixels)):
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
            if denom not in self.supported_denominations:
                continue
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

        A note that fills the entire camera frame is deliberately rejected:
        there is no reliable geometry or background to distinguish it from a
        coloured wall or shirt.
        """
        if frame is None or frame.size == 0:
            return {"denomination": None, "spoken_name": "No note detected", "confidence": 0.0}

        h, w = frame.shape[:2]
        total_pixels = h * w

        # 1. Multi-channel edge contour analysis to isolate candidate banknote.
        # External contours only: internal guilloche details are not candidates.
        edges = np.zeros((h, w), dtype=np.uint8)
        for ch in range(3):
            c_edges = cv2.Canny(frame[:, :, ch], 30, 100)
            edges = cv2.bitwise_or(edges, c_edges)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_conf = 0.0

        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            bbox_area = float(bw * bh)
            if bbox_area < 0.02 * total_pixels or bbox_area > 0.85 * total_pixels:
                continue
            if bw < 50 or bh < 25:
                continue

            ar = max(bw, bh) / max(1.0, min(bw, bh))
            if ar < 1.45 or ar > 2.85:
                continue

            # A banknote is a filled, roughly rectangular shape. Reject long thin
            # lines, tiny fragments, and heavily concave artifacts while allowing
            # natural hand-held tilt and fingers holding the note edge.
            contour_area = float(cv2.contourArea(c))
            if contour_area <= 0:
                continue
            rectangularity = contour_area / bbox_area
            hull = cv2.convexHull(c)
            hull_area = float(cv2.contourArea(hull))
            solidity = contour_area / hull_area if hull_area > 0 else 0.0
            if rectangularity < 0.35 or solidity < 0.65:
                continue

            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) > 16:
                continue

            # Boundary contrast verification against surrounding background:
            # A real banknote has distinct contrast against its immediate border.
            # Internal grain/shadows within the same surface have near-zero color delta.
            margin = 6
            y_in_0, y_in_1 = by + 2, by + bh - 2
            x_in_0, x_in_1 = bx + 2, bx + bw - 2
            if y_in_1 > y_in_0 and x_in_1 > x_in_0:
                inner_region = frame[y_in_0:y_in_1, x_in_0:x_in_1]
                outer_strips = []
                if by >= margin:
                    outer_strips.append(frame[by - margin:by, bx:bx + bw].reshape(-1, 3))
                if by + bh + margin <= h:
                    outer_strips.append(frame[by + bh:by + bh + margin, bx:bx + bw].reshape(-1, 3))
                if bx >= margin:
                    outer_strips.append(frame[by:by + bh, max(0, bx - margin):bx].reshape(-1, 3))
                if bx + bw + margin <= w:
                    outer_strips.append(frame[by:by + bh, bx + bw:min(w, bx + bw + margin)].reshape(-1, 3))

                if outer_strips:
                    outer_pixels = np.concatenate(outer_strips, axis=0)
                    mean_inner = np.mean(inner_region.astype(float), axis=(0, 1))
                    mean_outer = np.mean(outer_pixels.astype(float), axis=0)
                    color_delta = float(np.linalg.norm(mean_inner - mean_outer))
                    if color_delta < 12.0:
                        continue

            roi = frame[by:by + bh, bx:bx + bw]
            res = self.classify_banknote(roi, bbox_aspect_ratio=ar, color_order=color_order)
            if res.get("denomination") is not None and res["confidence"] > best_conf:
                best_conf = res["confidence"]
                best_candidate = res
                best_candidate["bbox"] = [bx, by, bw, bh]

        if best_candidate is not None:
            return best_candidate

        # No center-crop fallback: a frame filling the lens cannot be
        # distinguished from a coloured wall/shirt, and was the main source of
        # false positives. Ask the user to hold the note 20-30 cm away instead.
        return {"denomination": None, "spoken_name": "No note detected", "confidence": 0.0}

