from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .target_selector import TargetSelector
from .types import DetectionBox, TargetResult


@dataclass(frozen=True)
class TargetLockConfig:
    """Temporal target identity lock for multi-target stability.

    v17.6 fixes two opposite failures seen in logs:
    - body IoU alone could keep a lock across different nearby targets, causing multi-target jitter;
    - when the locked target dropped for 1-2 frames, the system returned no target and appeared stuck.

    The new logic therefore uses stricter identity matching while allowing a very short predictive hold
    and a controlled switch after the old locked target is really missing.
    """

    enabled: bool = True
    hold_lost_frames: int = 6
    hold_lost_seconds: float = 0.18
    match_max_distance_px: float = 85.0
    hard_match_max_distance_px: float = 150.0
    body_iou_match_max_distance_px: float = 120.0
    min_lock_head_conf: float = 0.30
    head_without_body_lock_conf: float = 0.86
    head_iou_min: float = 0.015
    body_iou_min: float = 0.015
    allow_switch_while_locked: bool = False
    switch_center_advantage_px: float = 90.0
    switch_conf_advantage: float = 0.20
    # V17.8.14: multi-target arbitration. A target closer to the ROI center can
    # challenge the current lock, but switching requires temporal confirmation.
    # This avoids the left/right indecision seen when several head/body pairs are
    # visible in the same frame.
    switch_confirm_frames: int = 3
    switch_match_px: float = 42.0
    switch_score_advantage: float = 0.10
    switch_max_center_dist_px: float = 210.0
    initial_center_weight: float = 1.0
    initial_conf_weight: float = 1.65
    initial_body_conf_weight: float = 0.65
    reset_on_active_press: bool = True
    reset_on_control_off: bool = False

    # v17.6: if the old lock is gone but a credible new candidate remains visible,
    # avoid freezing in the middle for the whole hold_lost window.
    allow_switch_when_locked_missing: bool = True
    lost_switch_after_frames: int = 2
    lost_switch_min_conf: float = 0.40
    lost_switch_requires_body: bool = True
    lost_switch_center_max_px: float = 180.0
    missing_switch_confirm_frames: int = 2
    missing_switch_match_px: float = 60.0

    # v17.7: bound lock velocity used for predictive hold. V17.6 learned velocity
    # from any accepted match; a detector jitter or identity swap could therefore
    # poison prediction and create oscillation.
    max_lock_velocity_px_s: float = 1600.0
    max_velocity_update_jump_px: float = 55.0

    # v17.6: bridge one or two detector dropouts for the same locked target.
    # This is deliberately short and only applies to previously body-paired locks.
    predict_lost_target: bool = True
    predict_lost_frames: int = 2
    predict_lost_ms: float = 45.0
    predict_lost_min_conf: float = 0.35
    prediction_ms: float = 35.0
    velocity_smoothing: float = 0.65


@dataclass
class _Candidate:
    target: TargetResult
    head: DetectionBox
    body: Optional[DetectionBox]
    center_dist: float
    base_score: float


class TargetLockManager:
    """Sticky target identity manager.

    The lock is conservative about identity. It prefers the old target, but no longer
    lets a different head inherit the same lock only because a large body box overlaps.
    """

    def __init__(self, cfg: TargetLockConfig):
        self.cfg = cfg
        self.reset("init")

    def reset(self, reason: str = "reset") -> None:
        self.locked = False
        self.lock_id = 0
        self.lost_frames = 0
        self.last_seen_time = 0.0
        self.last_point: Optional[tuple[float, float]] = None
        self.last_head: Optional[DetectionBox] = None
        self.last_body: Optional[DetectionBox] = None
        self.last_conf = 0.0
        self.last_reason = reason
        self.last_velocity: tuple[float, float] = (0.0, 0.0)
        self.last_body_paired = False
        self._missing_switch_point: Optional[tuple[float, float]] = None
        self._missing_switch_count = 0
        self._challenger_point: Optional[tuple[float, float]] = None
        self._challenger_count = 0

    def on_active_rising(self) -> None:
        if self.cfg.reset_on_active_press:
            self.reset("active key rising; lock reset for fresh target")

    @staticmethod
    def _iou(a: Optional[DetectionBox], b: Optional[DetectionBox]) -> float:
        if a is None or b is None:
            return 0.0
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = max(a.area + b.area - inter, 1e-6)
        return inter / union

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _predicted_point(self) -> Optional[tuple[float, float]]:
        if self.last_point is None:
            return None
        now = time.perf_counter()
        dt = max(0.0, min(now - self.last_seen_time, float(self.cfg.prediction_ms) / 1000.0))
        vx, vy = self.last_velocity
        return (self.last_point[0] + vx * dt, self.last_point[1] + vy * dt)

    def _build_head_candidates(
        self,
        boxes: Iterable[DetectionBox],
        selector: TargetSelector,
        center: tuple[float, float],
    ) -> list[_Candidate]:
        boxes = list(boxes)
        heads = [
            b for b in boxes
            if b.cls_id == selector.head_class_id
            and b.conf >= selector.head_conf
            and b.conf >= self.cfg.min_lock_head_conf
        ]
        bodies = [
            b for b in boxes
            if b.cls_id == selector.body_class_id and b.conf >= selector.body_conf
        ]
        out: list[_Candidate] = []
        for head in heads:
            hx, hy = head.center
            body = selector._match_body(head, bodies)
            if body is None and float(head.conf) < float(self.cfg.head_without_body_lock_conf):
                continue
            center_dist = self._dist((hx, hy), center)
            body_bonus = 0.35 if body is not None else 0.0
            body_conf = float(body.conf) if body is not None else 0.0
            # V17.8.14: initial acquisition should be decisive in multi-target
            # scenes. Earlier scoring let a far high-confidence target beat a
            # near, sufficiently credible target, which looked like indecision
            # between the left and right opponent. Center distance is now a
            # first-class term; confidence remains important but no longer
            # dominates the reticle-centered target.
            score = (
                float(head.conf) * float(self.cfg.initial_conf_weight)
                + body_conf * float(self.cfg.initial_body_conf_weight)
                + body_bonus
                - center_dist / max(float(self.cfg.switch_center_advantage_px) * 1.15, 55.0) * float(self.cfg.initial_center_weight)
                + min(head.area, 1600.0) / 12000.0
            )
            target = TargetResult(
                True,
                hx,
                hy,
                "head",
                float(head.conf),
                "head candidate",
                head,
                body,
            )
            out.append(_Candidate(target, head, body, center_dist, score))
        return out

    def _choose_initial(self, candidates: Sequence[_Candidate]) -> Optional[_Candidate]:
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.base_score)

    def _match_locked(self, candidates: Sequence[_Candidate]) -> Optional[_Candidate]:
        if not self.locked or self.last_point is None or not candidates:
            return None
        predicted = self._predicted_point() or self.last_point
        best: Optional[_Candidate] = None
        best_score = -1e18
        for cand in candidates:
            point = (float(cand.target.x), float(cand.target.y))
            dist_last = self._dist(point, self.last_point)
            dist_pred = self._dist(point, predicted)
            dist = min(dist_last, dist_pred)
            hiou = self._iou(self.last_head, cand.head)
            biou = self._iou(self.last_body, cand.body)

            distance_match = dist <= float(self.cfg.match_max_distance_px)
            head_iou_match = hiou >= float(self.cfg.head_iou_min)
            # v17.6: body IoU by itself is too weak in multi-target scenes because
            # large torso boxes can overlap even when the selected head has changed.
            body_iou_match = (
                biou >= float(self.cfg.body_iou_min)
                and dist <= float(self.cfg.body_iou_match_max_distance_px)
            )
            hard_ok = dist <= float(self.cfg.hard_match_max_distance_px) or head_iou_match
            if not hard_ok or not (distance_match or head_iou_match or body_iou_match):
                continue

            score = -dist + hiou * 260.0 + biou * 65.0 + float(cand.head.conf) * 10.0 - cand.center_dist * 0.01
            if score > best_score:
                best = cand
                best_score = score
        return best

    def _should_switch(self, locked: _Candidate, challenger: _Candidate) -> bool:
        if not self.cfg.allow_switch_while_locked:
            return False
        if challenger is locked:
            return False
        if challenger.body is None:
            return False
        if challenger.center_dist > float(self.cfg.switch_max_center_dist_px):
            return False
        center_gain = locked.center_dist - challenger.center_dist
        conf_gain = float(challenger.target.confidence) - float(locked.target.confidence)
        score_gain = float(challenger.base_score) - float(locked.base_score)
        return (
            center_gain >= float(self.cfg.switch_center_advantage_px)
            and conf_gain >= float(self.cfg.switch_conf_advantage)
            and score_gain >= float(self.cfg.switch_score_advantage)
        )

    def _confirm_switch_candidate(self, challenger: _Candidate) -> tuple[bool, str]:
        p = (float(challenger.target.x), float(challenger.target.y))
        if self._challenger_point is None or self._dist(p, self._challenger_point) > float(self.cfg.switch_match_px):
            self._challenger_point = p
            self._challenger_count = 1
        else:
            self._challenger_count += 1
            self._challenger_point = p
        needed = max(1, int(self.cfg.switch_confirm_frames))
        return self._challenger_count >= needed, f"challenger pending {self._challenger_count}/{needed}"

    def _clear_switch_candidate(self) -> None:
        self._challenger_point = None
        self._challenger_count = 0

    def _commit(self, cand: _Candidate, reason: str) -> TargetResult:
        now = time.perf_counter()
        point = (float(cand.target.x), float(cand.target.y))
        if not self.locked:
            self.lock_id += 1
            self.last_velocity = (0.0, 0.0)
        elif self.last_point is not None and self.last_seen_time > 0:
            dt = max(1e-3, min(now - self.last_seen_time, 0.25))
            jump = self._dist(point, self.last_point)
            raw_vx = (point[0] - self.last_point[0]) / dt
            raw_vy = (point[1] - self.last_point[1]) / dt
            raw_speed = math.hypot(raw_vx, raw_vy)
            if jump <= float(self.cfg.max_velocity_update_jump_px) and raw_speed <= float(self.cfg.max_lock_velocity_px_s):
                a = max(0.0, min(1.0, float(self.cfg.velocity_smoothing)))
                old_vx, old_vy = self.last_velocity
                self.last_velocity = (old_vx * a + raw_vx * (1.0 - a), old_vy * a + raw_vy * (1.0 - a))
            else:
                # Keep the lock but do not let a suspicious jump poison prediction.
                self.last_velocity = (0.0, 0.0)

        self._missing_switch_point = None
        self._missing_switch_count = 0
        self.locked = True
        self.lost_frames = 0
        self.last_seen_time = now
        self.last_point = point
        self.last_head = cand.head
        self.last_body = cand.body
        self.last_body_paired = cand.body is not None
        self.last_conf = float(cand.target.confidence)
        self.last_reason = reason
        cand.target.reason = f"{cand.target.reason}; lock_id={self.lock_id}; {reason}"
        return cand.target

    def _predict_lost_target(self) -> Optional[TargetResult]:
        if not self.cfg.predict_lost_target or self.last_point is None:
            return None
        if not self.last_body_paired or self.last_body is None or self.last_head is None:
            return None
        now = time.perf_counter()
        age_ms = (now - self.last_seen_time) * 1000.0 if self.last_seen_time else 999999.0
        if self.lost_frames > int(self.cfg.predict_lost_frames):
            return None
        if age_ms > float(self.cfg.predict_lost_ms):
            return None
        if self.last_conf < float(self.cfg.predict_lost_min_conf):
            return None
        pred = self._predicted_point() or self.last_point
        self.last_reason = f"locked target grace prediction: lost={self.lost_frames}/{self.cfg.predict_lost_frames}, age={age_ms:.1f}ms"
        return TargetResult(
            True,
            float(pred[0]),
            float(pred[1]),
            "head",
            float(self.last_conf),
            self.last_reason + f"; lock_id={self.lock_id}",
            self.last_head,
            self.last_body,
        )

    def _locked_lost(self, candidates: Sequence[_Candidate], center: tuple[float, float]) -> TargetResult:
        self.lost_frames += 1
        now = time.perf_counter()
        age = now - self.last_seen_time if self.last_seen_time else 999.0

        if candidates and self.cfg.allow_switch_when_locked_missing and self.lost_frames >= int(self.cfg.lost_switch_after_frames):
            best = self._choose_initial(candidates)
            if best is not None:
                has_body = best.body is not None
                credible = (
                    float(best.head.conf) >= float(self.cfg.lost_switch_min_conf)
                    and best.center_dist <= float(self.cfg.lost_switch_center_max_px)
                    and (has_body or not bool(self.cfg.lost_switch_requires_body))
                )
                if credible:
                    p = (float(best.target.x), float(best.target.y))
                    if self._missing_switch_point is None or self._dist(p, self._missing_switch_point) > float(self.cfg.missing_switch_match_px):
                        self._missing_switch_point = p
                        self._missing_switch_count = 1
                    else:
                        self._missing_switch_count += 1
                        # Track the current point so a moving but stable challenger can confirm.
                        self._missing_switch_point = p
                    if self._missing_switch_count >= int(self.cfg.missing_switch_confirm_frames):
                        self.reset("locked target missing; acquiring confirmed credible new target")
                        return self._commit(best, "switched after locked target missing with confirmation")

        predicted = self._predict_lost_target()
        if predicted is not None:
            return predicted

        if self.lost_frames <= int(self.cfg.hold_lost_frames) and age <= float(self.cfg.hold_lost_seconds):
            self.last_reason = (
                f"locked target temporarily lost: {self.lost_frames}/{self.cfg.hold_lost_frames}; "
                "switch suppressed"
            )
            return TargetResult(False, reason=self.last_reason)
        self.reset("locked target expired; ready to acquire new target")
        return TargetResult(False, reason="locked target expired")

    def select(
        self,
        boxes: Iterable[DetectionBox],
        selector: TargetSelector,
        center: tuple[float, float],
        *,
        active: bool = True,
    ) -> TargetResult:
        if not self.cfg.enabled:
            self.last_reason = "target lock disabled"
            return selector.select(boxes)

        if self.cfg.reset_on_control_off and not active:
            self.reset("inactive; lock reset")
            return selector.select(boxes)

        candidates = self._build_head_candidates(boxes, selector, center)

        if self.locked:
            locked = self._match_locked(candidates)
            if locked is not None:
                chosen = locked
                if len(candidates) > 1 and self.cfg.allow_switch_while_locked:
                    challengers = [c for c in candidates if c is not locked]
                    challenger = max(challengers, key=lambda c: c.base_score, default=None)
                    if challenger is not None and self._should_switch(locked, challenger):
                        ready, pending_reason = self._confirm_switch_candidate(challenger)
                        if ready:
                            self.lock_id += 1
                            self._clear_switch_candidate()
                            return self._commit(challenger, "switched: center-stable challenger confirmed")
                        return self._commit(locked, f"kept locked target; {pending_reason}")
                    self._clear_switch_candidate()
                return self._commit(chosen, "kept locked target")

            lost = self._locked_lost(candidates, center)
            if not lost.found:
                if not self.locked and candidates:
                    new_cand = self._choose_initial(candidates)
                    if new_cand is not None:
                        return self._commit(new_cand, "new target after lock expired")
                return lost
            return lost

        initial = self._choose_initial(candidates)
        if initial is not None:
            return self._commit(initial, "new target lock")

        # V17.8.10 no-ghost fix: when the target-lock layer cannot build a
        # credible head candidate, do not fall back to the raw selector. The raw
        # selector may choose a low-confidence isolated head-like map light, which
        # later becomes a one-frame motor pulse when Shift is held.
        self.last_reason = "no credible locked head candidate"
        return TargetResult(False, reason="no credible locked head candidate")
