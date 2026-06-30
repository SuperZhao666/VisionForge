from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Deque, Dict, Iterator


@dataclass
class StageStats:
    count: int
    avg_ms: float
    max_ms: float


class RollingProfiler:
    """Tiny rolling profiler for realtime loops.

    It intentionally has no heavy dependencies and only stores recent values.
    Use it to locate the bottleneck instead of guessing from Task Manager.
    """

    def __init__(self, enabled: bool = False, window: int = 120) -> None:
        self.enabled = bool(enabled)
        self.window = max(10, int(window))
        self._data: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t0) * 1000.0)

    def add(self, name: str, ms: float) -> None:
        if self.enabled:
            self._data[str(name)].append(float(ms))

    def add_many(self, prefix: str, values: Dict[str, float]) -> None:
        if not self.enabled:
            return
        for k, v in values.items():
            self.add(f"{prefix}.{k}", float(v))

    def stats(self) -> Dict[str, StageStats]:
        out: Dict[str, StageStats] = {}
        for k, vals in self._data.items():
            if not vals:
                continue
            xs = list(vals)
            out[k] = StageStats(len(xs), sum(xs) / len(xs), max(xs))
        return out

    def summary(self, names: list[str] | None = None) -> str:
        if not self.enabled:
            return "profile=off"
        st = self.stats()
        keys = names or sorted(st.keys())
        parts = []
        for k in keys:
            if k in st:
                s = st[k]
                parts.append(f"{k}={s.avg_ms:.2f}ms/{s.max_ms:.2f}max")
        return ", ".join(parts) if parts else "profile=no-data"
