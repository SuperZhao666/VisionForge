from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .screen_capture import ScreenCapture
from .log_utils import log


@dataclass(frozen=True)
class CapturedFrame:
    frame: np.ndarray
    seq: int
    timestamp: float


class LatestFrameReader:
    """Continuously reads screen frames and keeps only the latest one.

    This removes capture waiting from the inference/control loop. It also drops
    stale frames by design, which is what low-latency realtime control wants.
    """

    def __init__(self, capture: ScreenCapture, target_fps: float = 120.0, idle_sleep_ms: float = 1.0) -> None:
        self.capture = capture
        self.target_fps = max(1.0, float(target_fps or 120.0))
        self.idle_sleep = max(0.0, float(idle_sleep_ms or 1.0)) / 1000.0
        self._lock = threading.Lock()
        self._latest: Optional[CapturedFrame] = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._errors = 0
        self._last_error = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="LatestFrameReader", daemon=True)
        self._thread.start()
        log("capture thread started: latest-frame mode", "OK")

    def _loop(self) -> None:
        min_interval = 1.0 / self.target_fps
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(min(self.idle_sleep, max(0.0, next_t - now)))
                    continue
                frame = self.capture.read()
                ts = time.perf_counter()
                with self._lock:
                    self._seq += 1
                    # Keep a copy to avoid accidental mutation by downstream visualization.
                    self._latest = CapturedFrame(frame.copy(), self._seq, ts)
                next_t = ts + min_interval
            except Exception as e:  # keep capture thread alive across transient DXGI errors
                self._errors += 1
                self._last_error = str(e)
                if self._errors <= 3 or self._errors % 30 == 0:
                    log(f"capture thread warning: {e}", "WARN")
                time.sleep(0.01)

    def get_latest(self) -> Optional[CapturedFrame]:
        with self._lock:
            return self._latest

    def wait_first(self, timeout: float = 2.0) -> bool:
        end = time.perf_counter() + max(0.0, timeout)
        while time.perf_counter() < end:
            if self.get_latest() is not None:
                return True
            time.sleep(0.005)
        return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.capture.close()
