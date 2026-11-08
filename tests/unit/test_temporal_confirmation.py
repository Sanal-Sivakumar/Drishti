from perception.temporal_confirmation import TemporalDetectionConfirmer


def _obstacle(distance=1.2):
    return [{"class_name": "chair", "azimuth_deg": 25.0, "distance_m": distance}]


def test_requires_consecutive_frames_and_resets_when_missing():
    confirmer = TemporalDetectionConfirmer(required_frames=2)
    assert confirmer.update(_obstacle()) == []
    confirmed = confirmer.update(_obstacle(1.25))
    assert confirmed[0]["confirmation_frames"] == 2
    assert confirmer.update([]) == []
    assert confirmer.update(_obstacle()) == []
