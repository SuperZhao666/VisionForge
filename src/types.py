from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class DetectionBox:
    cls_id: int
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        return self.x1 + self.w * 0.5, self.y1 + self.h * 0.5

    def shifted(self, dx: float, dy: float) -> "DetectionBox":
        return DetectionBox(
            cls_id=self.cls_id,
            cls_name=self.cls_name,
            conf=self.conf,
            x1=self.x1 + dx,
            y1=self.y1 + dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetResult:
    found: bool
    x: float = 0.0
    y: float = 0.0
    source: str = "none"
    confidence: float = 0.0
    reason: str = ""
    head_box: Optional[DetectionBox] = None
    body_box: Optional[DetectionBox] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["head_box"] = self.head_box.to_dict() if self.head_box else None
        d["body_box"] = self.body_box.to_dict() if self.body_box else None
        return d
