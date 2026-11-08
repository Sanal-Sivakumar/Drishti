from scripts.evaluate_drishti import evaluate


def test_safety_metrics():
    metrics = evaluate([{
        "ground_truth": [{"class_name": "curb", "critical": True,
                          "azimuth_deg": -20.0, "distance_m": 1.0}],
        "predictions": [{"class_name": "curb", "azimuth_deg": -15.0, "distance_m": 1.2},
                        {"class_name": "chair", "azimuth_deg": 30.0, "distance_m": 2.0}],
        "narration_claims": ["curb", "stairs"],
        "narration_latency_ms": 800,
    }])
    assert metrics["critical_hazard_recall"] == 1.0
    assert metrics["false_warning_rate"] == 0.5
    assert metrics["mean_direction_error_deg"] == 5.0
    assert abs(metrics["mean_distance_error_m"] - 0.2) < 1e-9
    assert metrics["unsupported_narrated_objects"] == 1
