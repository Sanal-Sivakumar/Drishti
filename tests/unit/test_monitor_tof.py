"""Unit tests for the ToF real-time monitor script."""

from scripts.monitor_tof import build_bar


def test_build_bar_zero():
    bar = build_bar(0.0, max_m=2.0, width=10)
    assert "░" * 10 in bar
    assert "0.00 m" in bar


def test_build_bar_half():
    bar = build_bar(1.0, max_m=2.0, width=10)
    assert "█████░░░░░" in bar
    assert "1.00 m" in bar


def test_build_bar_full():
    bar = build_bar(2.0, max_m=2.0, width=10)
    assert "█" * 10 in bar
    assert "2.00 m" in bar


def test_build_bar_clamped_overflow():
    bar = build_bar(3.5, max_m=2.0, width=10)
    assert "█" * 10 in bar
    assert "3.50 m" in bar
