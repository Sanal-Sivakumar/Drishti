"""Hardware Abstraction Layer: Tactile Trigger & Input Subsystem.

Provides unified interface for physical push buttons (RPi.GPIO / gpiod)
and programmatic/synthetic event triggers for headless testing.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional
import queue
import time


class BaseTriggerInput(ABC):
    """Abstract base class for user tactile input triggers."""

    @abstractmethod
    def poll_trigger(self, button_id: int) -> bool:
        """Poll whether a specific button is currently pressed.
        
        Args:
            button_id: 1 for Scene Narration, 2 for Currency / Multilingual OCR.
        """
        pass

    @abstractmethod
    def register_callback(self, button_id: int, callback: Callable[[], None]) -> None:
        """Register a callback function to be invoked on button press event."""
        pass


class SyntheticTriggerInput(BaseTriggerInput):
    """Programmatic event trigger for headless automated testing."""

    def __init__(self) -> None:
        self.callbacks: Dict[int, List[Callable[[], None]]] = {1: [], 2: []}
        self.event_queue: queue.Queue = queue.Queue()

    def poll_trigger(self, button_id: int) -> bool:
        """Check if an event exists in the queue for this button."""
        # Non-blocking check
        items = []
        triggered = False
        while not self.event_queue.empty():
            item = self.event_queue.get_nowait()
            if item == button_id and not triggered:
                triggered = True
            else:
                items.append(item)
        # Put remaining back
        for it in items:
            self.event_queue.put(it)
        return triggered

    def inject_trigger(self, button_id: int) -> None:
        """Simulate a user pressing a tactile button."""
        self.event_queue.put(button_id)
        if button_id in self.callbacks:
            for cb in self.callbacks[button_id]:
                cb()

    def register_callback(self, button_id: int, callback: Callable[[], None]) -> None:
        if button_id not in self.callbacks:
            self.callbacks[button_id] = []
        self.callbacks[button_id].append(callback)


class GPIOPushButtonTrigger(BaseTriggerInput):
    """Driver for physical momentary push buttons on Raspberry Pi GPIO pins."""

    def __init__(
        self,
        pin_button_1: int = 17,
        pin_button_2: int = 27,
        debounce_ms: int = 50
    ) -> None:
        self.pin_map = {1: pin_button_1, 2: pin_button_2}
        self.debounce_sec = debounce_ms / 1000.0
        self.last_press_time = {1: 0.0, 2: 0.0}
        self.gpio_available = False

        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setmode(GPIO.BCM)
            for pin in self.pin_map.values():
                self.GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self.gpio_available = True
        except (ImportError, RuntimeError):
            self.gpio_available = False

    def poll_trigger(self, button_id: int) -> bool:
        if not self.gpio_available or button_id not in self.pin_map:
            return False

        pin = self.pin_map[button_id]
        # Active low: button pressed pulls GPIO to GND (0)
        is_pressed = (self.GPIO.input(pin) == self.GPIO.LOW)
        current_time = time.time()

        if is_pressed and (current_time - self.last_press_time[button_id] > self.debounce_sec):
            self.last_press_time[button_id] = current_time
            return True
        return False

    def register_callback(self, button_id: int, callback: Callable[[], None]) -> None:
        if not self.gpio_available or button_id not in self.pin_map:
            return

        pin = self.pin_map[button_id]
        self.GPIO.add_event_detect(
            pin,
            self.GPIO.FALLING,
            callback=lambda channel: callback(),
            bouncetime=int(self.debounce_sec * 1000)
        )

