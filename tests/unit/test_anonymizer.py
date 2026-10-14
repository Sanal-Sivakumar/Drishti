"""Unit tests for the Edge Face Anonymization Filter."""

import pytest
import numpy as np
from safety.anonymizer import FaceAnonymizer


def test_gaussian_kernel_normalization():
    """Verify 2D Gaussian kernel derivation sums to 1.0."""
    anonymizer = FaceAnonymizer(kernel_size=15, sigma=7.0)
    kernel = anonymizer.gaussian_kernel

    assert kernel.shape == (15, 15)
    assert kernel.dtype == np.float32
    assert np.isclose(np.sum(kernel), 1.0, atol=1e-5)
    # Peak must be at center (7, 7)
    assert kernel[7, 7] == np.max(kernel)


def test_face_blurring_reduces_variance():
    """Verify Gaussian convolution significantly reduces high-frequency pixel variance."""
    anonymizer = FaceAnonymizer(kernel_size=15, sigma=5.0)

    # Create a patch with sharp high-frequency checkerboard pattern
    patch = np.zeros((30, 30, 3), dtype=np.uint8)
    patch[::2, ::2, :] = 255
    patch[1::2, 1::2, :] = 255

    initial_variance = float(np.var(patch))
    blurred = anonymizer.apply_gaussian_blur(patch)
    blurred_variance = float(np.var(blurred))

    assert blurred.shape == patch.shape
    # Blurring should destroy high-frequency variance by at least 70%
    assert blurred_variance < initial_variance * 0.30


def test_anonymize_frame_with_box():
    """Verify bounding box redaction over image frame."""
    anonymizer = FaceAnonymizer()
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    # Inject a face region
    frame[20:50, 20:50, :] = 220

    res = anonymizer.anonymize_frame(frame, extra_face_boxes=[(20, 20, 50, 50)])
    assert res["faces_redacted"] >= 1
    assert "image" in res
    assert res["image"].shape == frame.shape

