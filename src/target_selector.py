from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

from .types import DetectionBox, TargetResult


class TargetSelector:
    """从 body/head 检测框中选最终控制点。

    类别约定：
    - 0: body
    - 1: head

    规则：优先选 head 框中心；没有 head 时可用 body 上方比例点兜底。
    """

    def __init__(
        self,
        body_class_id: int = 0,
        head_class_id: int = 1,
        head_conf: float = 0.25,
        body_conf: float = 0.25,
        body_fallback_y_ratio: float = 0.18,
        prefer_center: Optional[Tuple[float, float]] = None,
        prefer_head: bool = True,
        fallback_to_body: bool = True,
    ) -> None:
        self.body_class_id = int(body_class_id)
        self.head_class_id = int(head_class_id)
        self.head_conf = float(head_conf)
        self.body_conf = float(body_conf)
        self.body_fallback_y_ratio = float(body_fallback_y_ratio)
        self.prefer_center = prefer_center
        self.prefer_head = bool(prefer_head)
        self.fallback_to_body = bool(fallback_to_body)

    def _rank(self, box: DetectionBox) -> tuple[float, float, float]:
        dist_score = 0.0
        if self.prefer_center is not None:
            cx, cy = box.center
            px, py = self.prefer_center
            dist_score = -math.hypot(cx - px, cy - py)
        # 置信度优先，其次离中心更近，最后面积大一点更稳
        return box.conf, dist_score, box.area

    def select(self, boxes: Iterable[DetectionBox]) -> TargetResult:
        boxes = list(boxes)
        heads = [b for b in boxes if b.cls_id == self.head_class_id and b.conf >= self.head_conf]
        bodies = [b for b in boxes if b.cls_id == self.body_class_id and b.conf >= self.body_conf]

        if self.prefer_head and heads:
            head = max(heads, key=self._rank)
            x, y = head.center
            body = self._match_body(head, bodies)
            return TargetResult(True, x, y, "head", head.conf, "selected head box center", head, body)

        if self.fallback_to_body and bodies:
            body = max(bodies, key=self._rank)
            x = body.x1 + body.w * 0.5
            y = body.y1 + body.h * self.body_fallback_y_ratio
            return TargetResult(True, x, y, "body_fallback", body.conf, "no head; used body fallback point", None, body)

        return TargetResult(False, reason="no valid body/head detection")

    @staticmethod
    def _match_body(head: DetectionBox, bodies: Sequence[DetectionBox]) -> Optional[DetectionBox]:
        """Match a head to its plausible body.

        Older versions returned the highest-confidence body even when it was spatially
        unrelated to the head. That made false-positive head+body pairs look valid.
        This version only returns a body that contains or is very near the head center.
        """
        if not bodies:
            return None
        hx, hy = head.center
        containing = [b for b in bodies if b.x1 <= hx <= b.x2 and b.y1 <= hy <= b.y2]
        if containing:
            return max(containing, key=lambda b: (b.conf, b.area))

        candidates = []
        for b in bodies:
            # Tolerate imperfect boxes for small/far targets, but do not pair arbitrary
            # unrelated boxes.
            margin_x = b.w * 0.25
            margin_y = b.h * 0.12
            if (b.x1 - margin_x) <= hx <= (b.x2 + margin_x) and (b.y1 - margin_y) <= hy <= (b.y2 + margin_y):
                # Prefer nearer and higher-confidence body boxes.
                bx, by = b.center
                dist = math.hypot(hx - bx, hy - by)
                candidates.append((b.conf, -dist, b.area, b))
        if candidates:
            return max(candidates, key=lambda t: (t[0], t[1], t[2]))[3]
        return None
