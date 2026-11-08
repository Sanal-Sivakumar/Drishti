"""Bounded worker queue for non-real-time cognitive tasks."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class CognitiveTask:
    kind: str
    payload: Dict[str, Any]
    submitted_at: float


class CognitiveWorker:
    """Run expensive cognition away from the navigation loop."""

    def __init__(self, handler: Callable[[CognitiveTask], Dict[str, Any]], max_queue_size: int = 4) -> None:
        self._handler = handler
        self._tasks: queue.Queue[CognitiveTask] = queue.Queue(maxsize=max_queue_size)
        self._results: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max_queue_size * 2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="drishti-cognition", daemon=True)
        self._thread.start()

    def submit(self, kind: str, payload: Dict[str, Any]) -> bool:
        try:
            self._tasks.put_nowait(CognitiveTask(kind, payload, time.time()))
            return True
        except queue.Full:
            return False

    def poll(self) -> Optional[Dict[str, Any]]:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def wait_until_idle(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._tasks.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        return False

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._tasks.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                result = self._handler(task)
                result.setdefault("kind", task.kind)
                result.setdefault("submitted_at", task.submitted_at)
            except Exception as exc:
                result = {"kind": task.kind, "submitted_at": task.submitted_at, "success": False, "error": str(exc)}
            try:
                self._results.put_nowait(result)
            except queue.Full:
                try:
                    self._results.get_nowait()
                    self._results.put_nowait(result)
                except queue.Empty:
                    pass
            finally:
                self._tasks.task_done()
