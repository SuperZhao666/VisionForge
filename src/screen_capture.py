from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import cv2
import numpy as np

from .log_utils import log


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    @property
    def tuple_xyxy(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)

    @property
    def center(self) -> tuple[float, float]:
        return self.left + self.width * 0.5, self.top + self.height * 0.5


class ScreenCapture:
    """Stable center-ROI screen capture.

    Default backend is dxcam in streaming mode, not one-shot grab mode.

    Why this matters:
    - dxcam.grab(region=...) may sporadically return None on Windows.
    - dxcam.start(...)+get_latest_frame() keeps a desktop duplication stream alive and is the pattern used by
      the old project. It is more stable for continuous realtime loops.
    - Empty frames are treated as transient capture misses. The realtime loop will reuse the latest valid frame
      for a short period and then restart dxcam. It will not crash on a single empty frame.

    Supported backends:
    - dxcam: stable streaming mode, recommended.
    - auto: try dxcam streaming first, fall back to mss only if dxcam cannot initialize.
    - mss: slower but simple fallback.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 640,
        backend: str = "dxcam",
        target_fps: int = 120,
        max_reused_frames: int = 2,
        max_reused_frame_ms: float = 20.0,
        restart_after_empty_frames: int = 8,
        fallback_after_restarts: int = 2,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.target_fps = int(target_fps or 120)
        self.max_reused_frames = max(0, int(max_reused_frames))
        self.max_reused_frame_ms = max(0.0, float(max_reused_frame_ms))
        self.restart_after_empty_frames = max(1, int(restart_after_empty_frames))
        self.fallback_after_restarts = max(0, int(fallback_after_restarts))
        self.requested_backend = str(backend or "dxcam").lower()
        if self.requested_backend == "auto":
            self.requested_backend = "dxcam_auto"
        self._backend: Optional[str] = None
        self._dxcam = None
        self._mss = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time = 0.0
        self._empty_count = 0
        self._restart_count = 0
        self.region = self._calc_region(self.width, self.height)
        self._init_backend()

    @staticmethod
    def _screen_size() -> tuple[int, int]:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    user32.SetProcessDPIAware()
                except Exception:
                    pass
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            return 1920, 1080

    @classmethod
    def _calc_region(cls, width: int, height: int) -> CaptureRegion:
        sw, sh = cls._screen_size()
        left = max(0, int(sw / 2 - width / 2))
        top = max(0, int(sh / 2 - height / 2))
        width = max(1, min(int(width), sw - left))
        height = max(1, min(int(height), sh - top))
        return CaptureRegion(left, top, width, height)

    def _init_backend(self) -> None:
        if self.requested_backend in ("dxcam", "dxcam_auto"):
            try:
                self._init_dxcam_stream()
                return
            except Exception as e:
                if self.requested_backend == "dxcam":
                    raise RuntimeError(f"dxcam backend unavailable: {e}") from e
                log(f"dxcam unavailable, switching to mss: {e}", "WARN")
        self._init_mss()

    def _init_dxcam_stream(self) -> None:
        import dxcam
        self._stop_dxcam_silent()
        self._dxcam = dxcam.create(output_color="BGR")
        if self._dxcam is None:
            raise RuntimeError("dxcam.create returned None")
        # dxcam may return a cached camera object. In packaged GUI builds, a previous failed
        # start can leave the cached object in the capturing state. Stop it after create and
        # before start, then retry once on the known "Capture is already running" condition.
        try:
            self._dxcam.stop()
            time.sleep(0.05)
        except Exception:
            pass
        try:
            self._dxcam.start(target_fps=self.target_fps, region=self.region.tuple_xyxy)
        except Exception as e:
            if "already running" in str(e).lower():
                try:
                    self._dxcam.stop()
                    time.sleep(0.12)
                    self._dxcam.start(target_fps=self.target_fps, region=self.region.tuple_xyxy)
                except Exception as e2:
                    raise RuntimeError("屏幕采集已被占用，请关闭其他采集/录屏程序后重试") from e2
            else:
                raise
        self._backend = "dxcam"
        self._empty_count = 0
        log(f"screen capture backend: dxcam_stream, region={self.region}, target_fps={self.target_fps}", "SUCCESS")

        # Warm up the duplication stream. Do not enter realtime loop before the first valid frame.
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            frame = self._dxcam.get_latest_frame()
            if frame is not None and getattr(frame, "size", 0) > 0:
                self._last_frame = frame.copy()
                self._last_frame_time = time.perf_counter()
                log("dxcam first frame acquired", "SUCCESS")
                return
            time.sleep(0.02)
        raise RuntimeError("dxcam stream started but produced no frame during warmup")

    def _stop_dxcam_silent(self) -> None:
        if self._dxcam is not None:
            try:
                self._dxcam.stop()
            except Exception:
                pass
        self._dxcam = None

    def _restart_dxcam(self) -> None:
        self._restart_count += 1
        log(f"dxcam empty frames repeated; restarting stream, count={self._restart_count}", "WARN")
        time.sleep(0.05)
        self._init_dxcam_stream()

    def _init_mss(self) -> None:
        try:
            import mss
            self._mss = mss.mss()
            self._backend = "mss"
            self._empty_count = 0
            log(f"screen capture backend: mss, region={self.region}", "SUCCESS")
        except Exception as e:
            raise RuntimeError(f"no screen capture backend available: {e}") from e

    def _read_mss(self) -> np.ndarray:
        r = self.region
        raw = np.asarray(self._mss.grab({"left": r.left, "top": r.top, "width": r.width, "height": r.height}))
        frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        self._last_frame = frame.copy()
        self._last_frame_time = time.perf_counter()
        return frame

    def _read_dxcam_stream(self) -> np.ndarray:
        frame = self._dxcam.get_latest_frame()
        if frame is not None and getattr(frame, "size", 0) > 0:
            self._empty_count = 0
            self._last_frame = frame.copy()
            self._last_frame_time = time.perf_counter()
            return frame

        self._empty_count += 1

        # For short gaps, keep the realtime loop alive by returning the latest valid frame.
        # This matches the old project behavior: a temporary dxcam miss was logged/ignored, not fatal.
        if self._last_frame is not None:
            age_ms = (time.perf_counter() - self._last_frame_time) * 1000.0 if self._last_frame_time else 999999.0
            if self._empty_count <= self.max_reused_frames and age_ms <= self.max_reused_frame_ms:
                if self._empty_count == 1:
                    log(f"dxcam empty frame miss=1; reusing latest valid frame age={age_ms:.1f}ms", "WARN")
                time.sleep(0.001)
                return self._last_frame.copy()

        # Long no-frame period: restart dxcam stream. V17.8 also stops
        # repeatedly recycling stale frames: after several restarts, fall back to
        # MSS even if the user requested dxcam. A stale capture frame can make a
        # sudden close target look like "no reaction" because inference is seeing
        # an old image.
        if self.fallback_after_restarts >= 0 and self._restart_count >= self.fallback_after_restarts:
            log("dxcam kept producing empty frames; switching capture backend to mss", "WARN")
            self._init_mss()
            return self._read_mss()
        if self._empty_count < self.restart_after_empty_frames:
            # Try once more after a tiny wait. Do not keep returning old content after
            # the strict short bridge above has expired; that was a direct cause of
            # visible late/no reaction when the target had already appeared on screen.
            time.sleep(0.001)
            frame = self._dxcam.get_latest_frame()
            if frame is not None and getattr(frame, "size", 0) > 0:
                self._empty_count = 0
                self._last_frame = frame.copy()
                self._last_frame_time = time.perf_counter()
                return frame

        try:
            self._restart_dxcam()
            if self._last_frame is not None:
                return self._last_frame.copy()
        except Exception as e:
            if self.requested_backend in ("dxcam_auto", "dxcam"):
                log(f"dxcam restart failed, switching to mss: {e}", "WARN")
                self._init_mss()
                return self._read_mss()
            raise RuntimeError(
                "dxcam stream failed repeatedly. Keep the target window visible, avoid remote desktop/minimized "
                "states, and check display scaling/multi-monitor settings."
            ) from e

        raise RuntimeError("dxcam stream failed repeatedly")

    def read(self) -> np.ndarray:
        if self._backend == "mss":
            return self._read_mss()
        if self._backend == "dxcam":
            return self._read_dxcam_stream()
        raise RuntimeError("capture backend not initialized")

    def close(self) -> None:
        self._stop_dxcam_silent()
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img
