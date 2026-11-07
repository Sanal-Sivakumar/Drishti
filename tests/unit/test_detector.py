"""Unit tests for edge object detector parsing and normalization handling."""

import numpy as np
import pytest
from perception.detector import HeadlessObjectDetector, COCO_CLASSES


def test_parse_onnx_outputs_normalized_coordinates():
    """Verify that models outputting normalized box coordinates in [0.0, 1.0] are scaled to pixels."""
    detector = HeadlessObjectDetector(confidence_threshold=0.5, allow_synthetic_fallback=False)

    # Synthetic output: 8400 anchors, 84 channels (4 bbox + 80 classes)
    # Output format is (1, 84, 8400)
    preds = np.zeros((1, 84, 8400), dtype=np.float32)

    # Box 0: Centered person box in normalized coordinates [0.5, 0.5, 0.3, 0.6]
    preds[0, 0, 0] = 0.5  # cx
    preds[0, 1, 0] = 0.5  # cy
    preds[0, 2, 0] = 0.3  # bw
    preds[0, 3, 0] = 0.6  # bh
    preds[0, 4, 0] = 0.9  # class 0 (person) confidence = 0.9

    # Frame: 640x480 letterboxed into 640x640 (scale=1.0, pad_x=0, pad_y=80)
    dets = detector._parse_onnx_outputs(
        preds, scale=1.0, pad_x=0, pad_y=80, orig_w=640, orig_h=480, target_size=(640, 640)
    )

    assert len(dets) == 1
    det = dets[0]
    assert det["class_name"] == "person"
    assert det["confidence"] == 0.9
    x1, y1, x2, y2 = det["bbox"]
    # cx=320, cy=320, bw=192, bh=384
    # x1 = 320 - 96 = 224
    # y1 = 320 - 192 - 80 = 48
    # x2 = 320 + 96 = 416
    # y2 = 320 + 192 - 80 = 432
    assert x1 == pytest.approx(224.0, abs=1.0)
    assert y1 == pytest.approx(48.0, abs=1.0)
    assert x2 == pytest.approx(416.0, abs=1.0)
    assert y2 == pytest.approx(432.0, abs=1.0)
    assert y2 > y1
    assert x2 > x1


def test_parse_onnx_outputs_pixel_coordinates():
    """Verify that models outputting pixel box coordinates in [0, 640] remain unscaled."""
    detector = HeadlessObjectDetector(confidence_threshold=0.5, allow_synthetic_fallback=False)

    preds = np.zeros((1, 84, 8400), dtype=np.float32)

    # Box 0: Person box in pixel coordinates [320, 320, 192, 384]
    preds[0, 0, 0] = 320.0
    preds[0, 1, 0] = 320.0
    preds[0, 2, 0] = 192.0
    preds[0, 3, 0] = 384.0
    preds[0, 4, 0] = 0.85  # class 0 (person) confidence = 0.85

    dets = detector._parse_onnx_outputs(
        preds, scale=1.0, pad_x=0, pad_y=80, orig_w=640, orig_h=480, target_size=(640, 640)
    )

    assert len(dets) == 1
    det = dets[0]
    assert det["class_name"] == "person"
    x1, y1, x2, y2 = det["bbox"]
    assert x1 == pytest.approx(224.0, abs=1.0)
    assert y1 == pytest.approx(48.0, abs=1.0)
    assert x2 == pytest.approx(416.0, abs=1.0)
    assert y2 == pytest.approx(432.0, abs=1.0)


def test_parse_onnx_outputs_filters_low_confidence_and_degenerate():
    """Verify that low-confidence and zero-area boxes are cleanly rejected."""
    detector = HeadlessObjectDetector(confidence_threshold=0.6, allow_synthetic_fallback=False)

    preds = np.zeros((1, 84, 8400), dtype=np.float32)

    # Box 0: High confidence but zero width/height
    preds[0, 0, 0] = 0.5
    preds[0, 1, 0] = 0.5
    preds[0, 2, 0] = 0.0
    preds[0, 3, 0] = 0.0
    preds[0, 4, 0] = 0.95

    # Box 1: Low confidence (0.4 < 0.6)
    preds[0, 0, 1] = 0.5
    preds[0, 1, 1] = 0.5
    preds[0, 2, 1] = 0.2
    preds[0, 3, 1] = 0.2
    preds[0, 4, 1] = 0.40

    dets = detector._parse_onnx_outputs(
        preds, scale=1.0, pad_x=0, pad_y=0, orig_w=640, orig_h=640
    )
    assert len(dets) == 0


def test_non_maximum_suppression_deduplication():
    """Verify that duplicate overlapping detections are suppressed by IoU threshold."""
    detector = HeadlessObjectDetector(confidence_threshold=0.3, iou_threshold=0.5)

    raw_dets = [
        {"bbox": [100.0, 100.0, 200.0, 200.0], "confidence": 0.90, "class_name": "chair", "class_id": 56},
        {"bbox": [105.0, 102.0, 198.0, 202.0], "confidence": 0.80, "class_name": "chair", "class_id": 56},
        {"bbox": [400.0, 300.0, 500.0, 400.0], "confidence": 0.75, "class_name": "tv", "class_id": 62},
    ]

    nms_dets = detector.non_maximum_suppression(raw_dets)
    assert len(nms_dets) == 2
    assert nms_dets[0]["confidence"] == 0.90
    assert nms_dets[1]["class_name"] == "tv"
