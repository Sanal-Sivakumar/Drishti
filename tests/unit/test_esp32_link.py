"""Unit tests for the ESP32 USB-serial link, ToF sensor and button triggers."""

import time

from hal.esp32_link import Esp32Link, TofSample
from hal.distance_sensor import ESP32DistanceSensor
from hal.trigger_input import CompositeTriggerInput, ESP32ButtonTrigger, SyntheticTriggerInput


class StubLink:
    """Minimal link interface used to test the sensor without a serial port."""

    def __init__(self, sample=None, age_ms=0.0):
        self.sample = sample
        self.age = age_ms
        self.started = False
        self.stopped = False
        self.presses = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def latest(self):
        return self.sample

    def age_ms(self, now=None):
        return self.age

    def take_button_press(self, button_id):
        if button_id in self.presses:
            self.presses.remove(button_id)
            return True
        return False


def _sample(mm=842, status=0):
    return TofSample(
        seq=1,
        device_ms=100,
        mm_filtered=mm,
        mm_raw=mm,
        status=status,
        button_mask=0,
        recv_monotonic=time.monotonic(),
    )


# --------------------------------------------------------------------- parsing


def test_parses_range_frame():
    link = Esp32Link(port="/dev/null")
    link.feed_line(b"D,7,1234,842,845,0,0\n")

    sample = link.latest()
    assert sample is not None
    assert sample.seq == 7
    assert sample.device_ms == 1234
    assert sample.mm_filtered == 842
    assert sample.mm_raw == 845
    assert sample.status == 0


def test_ignores_malformed_frames():
    link = Esp32Link(port="/dev/null")
    link.feed_line("garbage")
    link.feed_line("D,1,2")  # too short
    link.feed_line("D,a,b,c,d,e,f")  # non-numeric

    assert link.latest() is None
    assert link.status().samples == 0


def test_sequence_gap_is_counted():
    link = Esp32Link(port="/dev/null")
    link.feed_line("D,1,10,800,800,0,0")
    link.feed_line("D,2,20,800,800,0,0")
    link.feed_line("D,5,30,800,800,0,0")  # two frames lost

    status = link.status()
    assert status.samples == 3
    assert status.seq_drops == 2


def test_invalid_status_counts_towards_health():
    link = Esp32Link(port="/dev/null")
    link.feed_line("D,1,10,0,8190,4,0")
    link.feed_line("D,2,20,800,801,0,0")

    status = link.status()
    assert status.samples == 2
    assert status.invalid_pct == 50.0


def test_boot_banner_and_status_line():
    link = Esp32Link(port="/dev/null")
    link.feed_line("I,1.0.0,VL53L0X")
    link.feed_line("S,36.2,1.5,1,1")

    status = link.status()
    assert status.firmware == "1.0.0"
    assert status.sensor_model == "VL53L0X"
    assert status.i2c_ok is True
    assert status.device_link_ok is True


# --------------------------------------------------------------------- buttons


def test_button_press_is_latched_and_consumed_once():
    link = Esp32Link(port="/dev/null")
    link.feed_line("B,2,150,1")
    link.feed_line("B,2,180,0")  # release edges are ignored

    assert link.take_button_press(2) is True
    assert link.take_button_press(2) is False
    assert link.take_button_press(1) is False


def test_composite_trigger_polls_every_source():
    link = Esp32Link(port="/dev/null")
    link.feed_line("B,1,150,1")

    synthetic = SyntheticTriggerInput()
    composite = CompositeTriggerInput([ESP32ButtonTrigger(link), synthetic])

    assert composite.poll_trigger(1) is True
    assert composite.poll_trigger(1) is False


# -------------------------------------------------------------------- distance


def test_sensor_returns_distance_from_fresh_valid_sample():
    link = StubLink(sample=_sample(mm=842), age_ms=10.0)
    sensor = ESP32DistanceSensor(link=link)
    sensor.start_ranging()

    assert link.started is True
    assert sensor.get_distance_meters() == 0.842


def test_sensor_rejects_stale_sample():
    link = StubLink(sample=_sample(mm=842), age_ms=500.0)
    sensor = ESP32DistanceSensor(link=link, stale_ms=120)
    sensor.start_ranging()

    assert sensor.get_distance_meters() is None


def test_sensor_rejects_invalid_status():
    link = StubLink(sample=_sample(mm=0, status=4), age_ms=10.0)
    sensor = ESP32DistanceSensor(link=link)
    sensor.start_ranging()

    assert sensor.get_distance_meters() is None


def test_sensor_rejects_out_of_range_distance():
    link = StubLink(sample=_sample(mm=2500), age_ms=10.0)
    sensor = ESP32DistanceSensor(link=link, max_valid_mm=2000)
    sensor.start_ranging()

    assert sensor.get_distance_meters() is None


def test_sensor_applies_calibration_offset():
    link = StubLink(sample=_sample(mm=500), age_ms=10.0)
    sensor = ESP32DistanceSensor(link=link, offset_mm=-20)
    sensor.start_ranging()

    assert sensor.get_distance_meters() == 0.48


def test_sensor_zone_distances_are_honest_for_single_ray():
    link = StubLink(sample=_sample(mm=900), age_ms=10.0)
    sensor = ESP32DistanceSensor(link=link)
    sensor.start_ranging()

    zones = sensor.get_zone_distances()
    assert zones["center"] == 0.9
    assert zones["left"] is None
    assert zones["right"] is None


def test_sensor_returns_none_before_ranging():
    link = StubLink(sample=_sample(mm=900), age_ms=10.0)
    sensor = ESP32DistanceSensor(link=link)

    assert sensor.get_distance_meters() is None
