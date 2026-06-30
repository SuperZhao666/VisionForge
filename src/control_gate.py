from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .types import TargetResult


@dataclass
class ControlGateConfig:
    """Stateful movement gate with hysteresis.

    v16 used one hard confidence threshold. Logs showed that real small/far heads
    often oscillate around that threshold, causing control_allowed to flicker.

    v17 separates acquisition and hold:
    - ENTER: a new target needs a stricter confidence and short confirmation.
    - HOLD: an already-confirmed locked target may continue at a lower confidence
      for a short time, as long as geometry validation still passed in main.py.
    - LOSS GRACE: missing frames do not reset confirmation immediately. They stop
      movement, but the same target can resume without rebuilding confirmation
      from zero if it reappears quickly and close enough.
    """

    require_confirmed_frames: int = 2
    min_head_conf_enter: float = 0.40
    min_head_conf_hold: float = 0.30
    high_conf_head: float = 0.82
    high_conf_confirmed_frames: int = 1
    max_target_jump_px: float = 120.0
    active_press_delay_frames: int = 1
    reset_on_no_head: bool = False
    max_control_distance_px: float = 0.0
    locked_target_grace_frames: int = 3
    locked_target_grace_ms: float = 70.0
    locked_target_max_drift_px: float = 40.0
    # v17.2: confirmation memory must expire. Without this, a long-gone
    # confirmed target could keep using the lower HOLD confidence threshold
    # when a new target appears much later.
    confirmed_memory_ms: float = 250.0

    # v17.8: if a confirmed, body-paired target drops for one or two frames,
    # continue using the last validated target briefly instead of toggling
    # movement_ready False. This converts detection flicker into a stable short
    # hold, but is bounded tightly to avoid ghost movement.
    allow_missing_target_hold_control: bool = True
    missing_target_hold_frames: int = 2
    missing_target_hold_ms: float = 35.0
    missing_target_hold_min_conf: float = 0.30
    # V17.8.10: missing-frame hold must not keep a false head-only map-light
    # target alive after the detector drops it. Default: only body-paired targets
    # get held continuity. Strong head-only hold is opt-in and tightly bounded.
    missing_target_hold_requires_body: bool = True
    missing_target_hold_allow_strong_head_only: bool = False
    missing_target_hold_head_only_min_conf: float = 0.82
    missing_target_hold_small_head_allowed: bool = False

    # v17.8: close-range sudden targets should not wait for the full ENTER
    # window when the geometry is already a valid head+body pair.
    instant_enter_enabled: bool = True
    instant_enter_center_dist_px: float = 135.0
    instant_enter_min_conf: float = 0.32
    instant_enter_requires_body: bool = True
    skip_active_delay_on_instant_target: bool = True

    # V17.8.15: reactive acquisition for sudden/fast-moving real targets.
    # This is a bounded shortcut: it applies only to body-paired, non-suspicious
    # candidates with sufficient head/body confidence and enough body height.
    # It keeps anti-map rules for tiny lights, short bodies and wide/squat props.
    reactive_fast_enter_enabled: bool = True
    reactive_fast_enter_min_conf: float = 0.70
    reactive_fast_enter_min_body_conf: float = 0.62
    reactive_fast_enter_center_dist_px: float = 155.0
    reactive_fast_enter_close_dist_px: float = 95.0
    reactive_fast_enter_confirm_frames: int = 2
    reactive_fast_enter_close_confirm_frames: int = 1
    reactive_fast_enter_min_body_height_px: float = 44.0

    # v17.8.3: the target-lock layer is the identity authority.  Logs showed
    # repeated movement_ready flicker after arrival because the temporal gate
    # rejected a same-lock, high-confidence head-body candidate as a jump > 40 px,
    # then accepted it one frame later.  Trust a "kept locked target" candidate
    # within this wider bound instead of turning movement off for one frame.
    trust_locked_target_jump: bool = True
    trusted_locked_jump_px: float = 95.0
    trusted_locked_min_conf: float = 0.45

    # v17.8.4: never turn one-frame target-lock disagreement into a motor on/off pulse.
    # If the lock manager still says this is the kept locked target but the temporal
    # point jumps, hold the last validated target instead of returning no target.
    hold_on_locked_jump: bool = True
    locked_jump_hold_frames: int = 4
    locked_jump_hold_ms: float = 90.0
    locked_jump_hold_min_conf: float = 0.25
    locked_jump_hold_max_px: float = 160.0

    # V17.8.21: do not convert same-lock body-paired measurement jumps into
    # a one-frame motor OFF pulse. When target_lock says the identity is still
    # the kept locked target, prefer the current raw measurement over a stale
    # held point. The old hold path could return a target already shifted by
    # ego-motion to the reticle center, producing the visible move-pause-move
    # cadence seen in logs.
    same_lock_jump_accept_enabled: bool = True
    same_lock_jump_accept_px: float = 190.0
    same_lock_jump_max_center_dist_px: float = 260.0
    same_lock_jump_center_worse_tolerance_px: float = 80.0
    same_lock_jump_min_conf: float = 0.42
    same_lock_jump_requires_body: bool = True

    # v17.8.5: motion-smooth same-lock target filtering.
    smooth_locked_target: bool = True
    locked_jitter_px: float = 2.0
    locked_jitter_radius_fraction: float = 0.10
    locked_jitter_alpha: float = 0.18
    locked_smooth_alpha: float = 0.55
    locked_slew_px_per_frame: float = 9.0
    locked_slew_radius_fraction: float = 0.55
    locked_snap_px: float = 42.0
    locked_snap_min_conf: float = 0.86

    # v17.8.6: if the same-lock measurement jumps far away from the last
    # committed control point, never slew an old point toward it while still
    # passing current head/body geometry. That produced poisoned control points
    # outside the current head box. Either rebase directly to the current raw
    # head center when it is reliable, or hold the old target with zero movement.
    locked_rebase_enabled: bool = True
    locked_rebase_px: float = 44.0
    locked_rebase_radius_fraction: float = 1.25
    locked_rebase_min_conf: float = 0.58
    locked_rebase_requires_body: bool = True
    # V17.8.14: large same-lock rebase must not jump across unrelated
    # left/right targets. Rebase only when the new measurement is not clearly
    # farther from the reticle than the last validated point, and cap the jump.
    locked_rebase_max_jump_px: float = 150.0
    locked_rebase_max_center_dist_px: float = 190.0
    locked_rebase_center_worse_tolerance_px: float = 18.0
    locked_smoothing_max_raw_lag_px: float = 3.0
    locked_smoothing_max_raw_lag_radius_fraction: float = 0.18

    # Head-only far targets may be real but need extra temporal confirmation.
    head_only_confirmed_frames: int = 2
    head_only_min_conf: float = 0.45

    # V17.8.11: anti-map temporal confirmation. Map lights/props can emit a
    # plausible head+body pair for 1-3 frames. Do not let small or squat pairs
    # drive HID until they persist across several frames. This is separate from
    # drawing/detection: boxes may still appear, but movement stays locked out.
    small_target_confirmed_frames: int = 7
    tiny_target_confirmed_frames: int = 10
    small_target_high_conf: float = 0.86
    small_target_high_conf_frames: int = 4
    small_target_area_px: float = 96.0
    small_target_max_dim_px: float = 18.0
    suspicious_body_height_px: float = 42.0
    suspicious_body_aspect: float = 0.82
    suspicious_target_confirmed_frames: int = 8


class ConfirmedHeadGate:
    """Anti flicker gate for HID movement.

    It never moves on a missing target. The grace state only preserves identity and
    confirmation memory so that a real target with one weak/missing frame does not
    have to be re-confirmed from scratch.
    """

    def __init__(self, cfg: ControlGateConfig):
        self.cfg = cfg
        self.reset("init")

    def reset(self, reason: str = "reset") -> None:
        self.confirm_count = 0
        self.last_point: Optional[Tuple[float, float]] = None
        self.last_valid_point: Optional[Tuple[float, float]] = None
        self.last_valid_target: Optional[TargetResult] = None
        self.last_conf = 0.0
        self.frames_since_active_press = 0
        self.missing_frames = 0
        self.last_seen_time = 0.0
        self.was_confirmed = False
        self.last_reason = reason

    def on_active_rising(self) -> None:
        self.reset("active key rising")
        self.frames_since_active_press = int(max(0, self.cfg.active_press_delay_frames))

    def _return_no_move(self, reason: str) -> tuple[bool, TargetResult, str]:
        self.last_reason = reason
        return False, TargetResult(False, reason=reason), reason

    def _clear_confirmed_memory(self) -> None:
        self.was_confirmed = False
        self.last_valid_point = None
        self.last_valid_target = None
        self.last_seen_time = 0.0

    def _has_recent_confirmed_memory(self) -> bool:
        if not self.was_confirmed or self.last_valid_point is None or self.last_seen_time <= 0:
            return False
        age_ms = (time.perf_counter() - self.last_seen_time) * 1000.0
        return age_ms <= float(self.cfg.confirmed_memory_ms)

    def _hard_reject(self, reason: str, *, clear_memory: bool = True) -> tuple[bool, TargetResult, str]:
        self.confirm_count = 0
        self.last_point = None
        self.last_conf = 0.0
        self.missing_frames = 0
        if clear_memory:
            self._clear_confirmed_memory()
        elif not self.was_confirmed:
            self.last_valid_point = None
        return self._return_no_move(reason)

    def _within_recent_lock_grace(self) -> bool:
        if not self._has_recent_confirmed_memory():
            return False
        if self.missing_frames > int(self.cfg.locked_target_grace_frames):
            return False
        age_ms = (time.perf_counter() - self.last_seen_time) * 1000.0
        return age_ms <= float(self.cfg.locked_target_grace_ms)


    def on_validation_result(self, valid: bool, target: TargetResult) -> None:
        """Synchronize temporal memory with downstream geometry validation.

        V17.8 lets the temporal gate run before geometry validation so it can
        provide a short held target during one-frame detection misses. That also
        means the gate must discard its newly-built confirmation if geometry later
        rejects the candidate. Otherwise an invalid head-like box could poison
        the hold memory.
        """
        if valid:
            return
        if target.found:
            self.confirm_count = 0
            self.last_point = None
            self.last_conf = 0.0
            self._clear_confirmed_memory()
            self.last_reason = "geometry validation rejected gated target"

    def apply_ego_motion(self, dx: float, dy: float, scaler: float = 2.7) -> None:
        """Shift held control target after HID-induced camera movement.

        The tracker already applies ego motion. V17.8 also holds the last validated
        target for one or two missing frames, so the held point must be shifted by
        the same ego compensation; otherwise the hold itself can become a small
        over-correction.
        """
        if not (dx or dy):
            return
        sx = float(dx) * float(scaler)
        sy = float(dy) * float(scaler)
        def shift_point(pt: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
            if pt is None:
                return None
            return (pt[0] - sx, pt[1] - sy)
        self.last_point = shift_point(self.last_point)
        self.last_valid_point = shift_point(self.last_valid_point)
        if self.last_valid_target is not None and self.last_valid_target.found:
            t = self.last_valid_target
            t.x -= sx
            t.y -= sy
            if t.head_box is not None:
                t.head_box = t.head_box.shifted(-sx, -sy)
            if t.body_box is not None:
                t.body_box = t.body_box.shifted(-sx, -sy)

    @staticmethod
    def _copy_target(target: TargetResult, reason_suffix: str = "") -> TargetResult:
        if not target.found:
            return TargetResult(False, reason=target.reason)
        reason = target.reason
        if reason_suffix:
            reason = f"{reason}; {reason_suffix}" if reason else reason_suffix
        return TargetResult(
            found=True,
            x=float(target.x),
            y=float(target.y),
            source=str(target.source),
            confidence=float(target.confidence),
            reason=reason,
            head_box=target.head_box,
            body_box=target.body_box,
        )

    @staticmethod
    def _head_radius(target: TargetResult) -> float:
        if target is None or not target.found or target.head_box is None:
            return 0.0
        return max(float(target.head_box.w), float(target.head_box.h)) * 0.5

    def _same_lock_hint(self, target: TargetResult) -> bool:
        reason = str(getattr(target, "reason", "") or "")
        return "kept locked target" in reason or "lock_id=" in reason or "grace prediction" in reason

    def _smooth_locked_control_target(self, raw_target: TargetResult, *, force: bool = False) -> TargetResult:
        """Denoise same-lock target x/y while keeping current boxes for geometry."""
        if not bool(getattr(self.cfg, "smooth_locked_target", True)):
            return raw_target
        if not raw_target.found or raw_target.source != "head" or raw_target.head_box is None:
            return raw_target
        if self.last_valid_point is None:
            return raw_target
        if not (force or self._same_lock_hint(raw_target) or self.was_confirmed):
            return raw_target
        lx, ly = self.last_valid_point
        rx, ry = float(raw_target.x), float(raw_target.y)
        dx, dy = rx - lx, ry - ly
        dist = math.hypot(dx, dy)
        if not math.isfinite(dist) or dist <= 1e-6:
            return self._copy_target(raw_target, "control smoothing unchanged")
        radius = self._head_radius(raw_target)
        jitter_px = max(float(getattr(self.cfg, "locked_jitter_px", 2.0)), radius * float(getattr(self.cfg, "locked_jitter_radius_fraction", 0.10)))
        slew_px = max(float(getattr(self.cfg, "locked_slew_px_per_frame", 9.0)), radius * float(getattr(self.cfg, "locked_slew_radius_fraction", 0.55)), 1.0)
        snap_px = max(slew_px, float(getattr(self.cfg, "locked_snap_px", 42.0)))
        snap_conf = float(getattr(self.cfg, "locked_snap_min_conf", 0.86))
        if dist > snap_px and float(raw_target.confidence or 0.0) >= snap_conf:
            alpha = min(0.90, max(0.65, float(getattr(self.cfg, "locked_smooth_alpha", 0.55)) + 0.20))
            nx, ny = lx + dx * alpha, ly + dy * alpha
            suffix = f"control smoothed strong lock: {dist:.1f}px alpha={alpha:.2f}"
        elif dist <= jitter_px:
            alpha = max(0.0, min(1.0, float(getattr(self.cfg, "locked_jitter_alpha", 0.18))))
            nx, ny = lx + dx * alpha, ly + dy * alpha
            suffix = f"control jitter-smoothed: {dist:.1f}px <= {jitter_px:.1f}px"
        else:
            alpha = max(0.0, min(1.0, float(getattr(self.cfg, "locked_smooth_alpha", 0.55))))
            step = min(dist * alpha, slew_px)
            nx, ny = lx + dx / dist * step, ly + dy / dist * step
            suffix = f"control slew-smoothed: {dist:.1f}px step={step:.1f}px"
        # Final safety: the control point must remain anchored to the current
        # detected head center. If smoothing would leave the point outside the
        # current head neighborhood, clamp it back toward raw. This prevents the
        # poisoned case seen in logs: raw head y≈58 but gated control y≈262.
        max_raw_lag = max(
            float(getattr(self.cfg, "locked_smoothing_max_raw_lag_px", 3.0)),
            radius * float(getattr(self.cfg, "locked_smoothing_max_raw_lag_radius_fraction", 0.18)),
        )
        lag_raw = math.hypot(nx - rx, ny - ry)
        if lag_raw > max_raw_lag > 0.0:
            nx = rx + (nx - rx) / max(lag_raw, 1e-6) * max_raw_lag
            ny = ry + (ny - ry) / max(lag_raw, 1e-6) * max_raw_lag
            suffix = f"{suffix}; raw-lag-clamped {lag_raw:.1f}->{max_raw_lag:.1f}"
        out = self._copy_target(raw_target, suffix)
        out.x, out.y = float(nx), float(ny)
        return out

    def _commit_valid_target(self, target: TargetResult, point: tuple[float, float], reason: str) -> TargetResult:
        self.was_confirmed = True
        self.last_point = (float(point[0]), float(point[1]))
        self.last_valid_point = (float(point[0]), float(point[1]))
        self.last_valid_target = self._copy_target(target)
        self.last_valid_target.x = float(point[0])
        self.last_valid_target.y = float(point[1])
        self.last_seen_time = time.perf_counter()
        self.last_reason = reason
        target.reason = f"{target.reason}; {reason}" if target.reason else reason
        return target

    def _can_hold_missing_target(self) -> bool:
        if not bool(self.cfg.allow_missing_target_hold_control):
            return False
        if not self._within_recent_lock_grace():
            return False
        if self.last_valid_target is None or not self.last_valid_target.found:
            return False
        if self.last_valid_target.source != "head" or self.last_valid_target.head_box is None:
            return False
        if self.last_valid_target.body_box is None:
            if bool(getattr(self.cfg, "missing_target_hold_requires_body", True)):
                if not bool(getattr(self.cfg, "missing_target_hold_allow_strong_head_only", False)):
                    return False
                if float(self.last_valid_target.confidence) < float(getattr(self.cfg, "missing_target_hold_head_only_min_conf", 0.82)):
                    return False
            if self.last_valid_target.head_box is not None:
                h = self.last_valid_target.head_box
                small_head = h.area <= 96.0 or max(float(h.w), float(h.h)) <= 18.0
                if small_head and not bool(getattr(self.cfg, "missing_target_hold_small_head_allowed", False)):
                    return False
        if float(self.last_valid_target.confidence) < float(self.cfg.missing_target_hold_min_conf):
            return False
        if self.missing_frames > int(self.cfg.missing_target_hold_frames):
            return False
        age_ms = (time.perf_counter() - self.last_seen_time) * 1000.0
        return age_ms <= float(self.cfg.missing_target_hold_ms)

    def _return_held_missing_target(self) -> tuple[bool, TargetResult, str]:
        held = self._copy_target(
            self.last_valid_target,
            f"gate held missing target {self.missing_frames}/{self.cfg.missing_target_hold_frames}"
        )
        self.last_reason = f"target missing; holding last validated target {self.missing_frames}/{self.cfg.missing_target_hold_frames}"
        return True, held, self.last_reason

    def _can_hold_locked_jump(self, raw_target: TargetResult, jump: float) -> bool:
        if not bool(getattr(self.cfg, "hold_on_locked_jump", True)):
            return False
        if self.last_valid_target is None or not self.last_valid_target.found:
            return False
        if self.last_valid_target.source != "head" or self.last_valid_target.head_box is None:
            return False
        if self.last_valid_target.body_box is None:
            return False
        if not self._has_recent_confirmed_memory():
            return False
        age_ms = (time.perf_counter() - self.last_seen_time) * 1000.0 if self.last_seen_time else 999999.0
        if age_ms > float(getattr(self.cfg, "locked_jump_hold_ms", 90.0)):
            return False
        if self.missing_frames > int(getattr(self.cfg, "locked_jump_hold_frames", 4)):
            return False
        if float(raw_target.confidence or 0.0) < float(getattr(self.cfg, "locked_jump_hold_min_conf", 0.25)):
            return False
        if float(jump) > float(getattr(self.cfg, "locked_jump_hold_max_px", 160.0)):
            return False
        reason = str(raw_target.reason or "")
        return ("kept locked target" in reason or "lock_id=" in reason) and raw_target.body_box is not None

    def _can_accept_same_lock_jump(self, raw_target: TargetResult, jump: float, dist_center: float) -> bool:
        """Accept reliable same-lock measurements instead of toggling movement off.

        This is deliberately narrower than a generic jump allowance: the target
        must still be a head target, must carry the same-lock hint from
        target_lock, and normally must have a body pair. It fixes the stutter
        pattern where ego-compensated historical points and current detector
        measurements differ by 60-100 px while still describing the same target.
        """
        if not bool(getattr(self.cfg, "same_lock_jump_accept_enabled", True)):
            return False
        if not raw_target.found or raw_target.source != "head" or raw_target.head_box is None:
            return False
        if bool(getattr(self.cfg, "same_lock_jump_requires_body", True)) and raw_target.body_box is None:
            return False
        if not self._same_lock_hint(raw_target):
            return False
        if float(raw_target.confidence or 0.0) < float(getattr(self.cfg, "same_lock_jump_min_conf", 0.42)):
            return False
        max_jump = float(getattr(self.cfg, "same_lock_jump_accept_px", 190.0))
        if max_jump > 0.0 and float(jump) > max_jump:
            return False
        max_center = float(getattr(self.cfg, "same_lock_jump_max_center_dist_px", 260.0))
        if max_center > 0.0 and float(dist_center) > max_center:
            return False
        if self.last_valid_point is not None:
            last_center_dist = math.hypot(float(self.last_valid_point[0]) - float(getattr(self, "_current_center_x", 0.0)), float(self.last_valid_point[1]) - float(getattr(self, "_current_center_y", 0.0)))
            tol = float(getattr(self.cfg, "same_lock_jump_center_worse_tolerance_px", 80.0))
            if float(dist_center) > last_center_dist + tol:
                return False
        return True

    def _return_accepted_same_lock_jump(self, jump: float, raw_target: TargetResult) -> tuple[bool, TargetResult, str]:
        # Commit the current raw detector point. Do not return the stale held
        # point; that was the source of the visible pause in V17.8.20.
        accepted = self._copy_target(raw_target, f"same-lock jump accepted raw: {jump:.1f}px")
        point = (float(accepted.x), float(accepted.y))
        self.confirm_count = max(self.confirm_count, 1)
        accepted = self._commit_valid_target(accepted, point, f"same-lock jump accepted raw: {jump:.1f}px")
        return True, accepted, self.last_reason

    def _locked_rebase_limit_px(self, target: TargetResult) -> float:
        radius = self._head_radius(target)
        return max(
            float(getattr(self.cfg, "locked_rebase_px", 44.0)),
            radius * float(getattr(self.cfg, "locked_rebase_radius_fraction", 1.25)),
        )

    def _can_rebase_locked_jump(self, raw_target: TargetResult, jump: float) -> bool:
        if not bool(getattr(self.cfg, "locked_rebase_enabled", True)):
            return False
        if not raw_target.found or raw_target.source != "head" or raw_target.head_box is None:
            return False
        if bool(getattr(self.cfg, "locked_rebase_requires_body", True)) and raw_target.body_box is None:
            return False
        if float(raw_target.confidence or 0.0) < float(getattr(self.cfg, "locked_rebase_min_conf", 0.58)):
            return False
        if jump < self._locked_rebase_limit_px(raw_target):
            return False
        max_jump = float(getattr(self.cfg, "locked_rebase_max_jump_px", 150.0))
        if max_jump > 0.0 and float(jump) > max_jump:
            return False
        max_center_dist = float(getattr(self.cfg, "locked_rebase_max_center_dist_px", 190.0))
        if max_center_dist > 0.0:
            # ROI center is supplied only to update(), so compare against the stable
            # reticle point implicitly: last_valid_point and raw point distances to
            # center are available through recent x/y only inside update. The caller
            # stores a temporary override before calling this method.
            raw_center_dist = float(getattr(self, "_current_raw_center_dist", 0.0) or 0.0)
            last_center_dist = float(getattr(self, "_current_last_center_dist", raw_center_dist) or raw_center_dist)
            if raw_center_dist > max_center_dist:
                return False
            tolerance = float(getattr(self.cfg, "locked_rebase_center_worse_tolerance_px", 18.0))
            if raw_center_dist > last_center_dist + tolerance:
                return False
        return self._same_lock_hint(raw_target)

    def _return_rebased_locked_target(self, jump: float, raw_target: TargetResult) -> tuple[bool, TargetResult, str]:
        # Commit the *current* head center, not a slewed old point.  The caller in
        # main.py clears motor residual when it sees this reason.
        rebased = self._copy_target(raw_target, f"control rebase snap: jump {jump:.1f}px")
        point = (float(rebased.x), float(rebased.y))
        self.confirm_count = max(self.confirm_count, 1)
        rebased = self._commit_valid_target(rebased, point, f"control rebase snap: jump {jump:.1f}px")
        return True, rebased, self.last_reason

    def _return_held_locked_jump(self, jump: float, raw_target: TargetResult | None = None) -> tuple[bool, TargetResult, str]:
        # V17.8.6: do not slew a stale control point toward a far-away current
        # detection. Holding is safer; held targets have zero movement gain by
        # configuration, so movement_ready can stay stable without pushing the lens.
        held = self._copy_target(self.last_valid_target, f"gate held locked jump {jump:.1f}")
        self.last_reason = f"locked target jump {jump:.1f}px; holding last validated target"
        return True, held, self.last_reason

    def _target_small_or_suspicious(self, raw_target: TargetResult) -> tuple[bool, bool, str]:
        """Return (small_like, suspicious_like, reason).

        This is a movement-only guard. It intentionally treats micro head/body
        pairs as suspicious even when model confidence is high, because map props
        often produce consistent but physically tiny boxes.
        """
        if not raw_target.found or raw_target.head_box is None:
            return False, False, ""
        h = raw_target.head_box
        b = raw_target.body_box
        small = (
            float(h.area) <= float(self.cfg.small_target_area_px)
            or max(float(h.w), float(h.h)) <= float(self.cfg.small_target_max_dim_px)
        )
        tiny = (
            float(h.area) <= float(self.cfg.small_target_area_px) * 0.35
            or max(float(h.w), float(h.h)) <= float(self.cfg.small_target_max_dim_px) * 0.55
        )
        suspicious = False
        reasons = []
        if small:
            reasons.append(f"small_head area={h.area:.1f} wh={h.w:.1f}x{h.h:.1f}")
        if tiny:
            suspicious = True
            reasons.append("tiny_head")
        if b is not None:
            body_ar = float(b.w) / max(float(b.h), 1e-6)
            if float(b.h) < float(self.cfg.suspicious_body_height_px):
                suspicious = True
                reasons.append(f"short_body h={b.h:.1f}")
            if body_ar > float(self.cfg.suspicious_body_aspect):
                suspicious = True
                reasons.append(f"wide_body ar={body_ar:.2f}")
        return small, suspicious, "; ".join(reasons)

    def _looks_like_instant_target(self, raw_target: TargetResult, center_x: float, center_y: float) -> bool:
        if not bool(getattr(self.cfg, "skip_active_delay_on_instant_target", True)):
            return False
        if not raw_target.found or raw_target.source != "head" or raw_target.head_box is None:
            return False
        if bool(self.cfg.instant_enter_requires_body) and raw_target.body_box is None:
            return False
        small_like, suspicious_like, _ = self._target_small_or_suspicious(raw_target)
        if small_like or suspicious_like:
            return False
        dist_center = math.hypot(float(raw_target.x) - center_x, float(raw_target.y) - center_y)
        return (
            dist_center <= float(self.cfg.instant_enter_center_dist_px)
            and float(raw_target.confidence) >= float(self.cfg.instant_enter_min_conf)
        )

    def _reactive_fast_enter_frames(
        self,
        raw_target: TargetResult,
        *,
        dist_center: float,
        suspicious_like: bool,
    ) -> int | None:
        """Return a reduced confirmation window for fast real targets.

        V17.8.14 preserved anti-map behavior by requiring 4-6 frames for many
        small/suspicious targets. Logs showed real sudden targets often first appear
        as a body-paired small head and then vanish/reappear; waiting 5 frames made
        the system feel late. This shortcut is intentionally strict: no body, short
        body, weak body confidence, or suspicious shape means no shortcut.
        """
        if not bool(getattr(self.cfg, "reactive_fast_enter_enabled", True)):
            return None
        if not raw_target.found or raw_target.source != "head" or raw_target.head_box is None:
            return None
        if raw_target.body_box is None:
            return None
        if suspicious_like:
            return None
        body = raw_target.body_box
        if float(body.h) < float(getattr(self.cfg, "reactive_fast_enter_min_body_height_px", 44.0)):
            return None
        if float(raw_target.confidence or 0.0) < float(getattr(self.cfg, "reactive_fast_enter_min_conf", 0.70)):
            return None
        if float(body.conf or 0.0) < float(getattr(self.cfg, "reactive_fast_enter_min_body_conf", 0.62)):
            return None
        if float(dist_center) > float(getattr(self.cfg, "reactive_fast_enter_center_dist_px", 155.0)):
            return None
        if float(dist_center) <= float(getattr(self.cfg, "reactive_fast_enter_close_dist_px", 95.0)):
            return max(1, int(getattr(self.cfg, "reactive_fast_enter_close_confirm_frames", 1)))
        return max(1, int(getattr(self.cfg, "reactive_fast_enter_confirm_frames", 2)))

    def update(
        self,
        raw_target: TargetResult,
        *,
        active: bool,
        center_x: float,
        center_y: float,
    ) -> tuple[bool, TargetResult, str]:
        if not active:
            self.reset("inactive")
            return False, TargetResult(False, reason="inactive"), "inactive"

        if self.frames_since_active_press > 0:
            if not self._looks_like_instant_target(raw_target, center_x, center_y):
                self.frames_since_active_press -= 1
                return self._return_no_move("waiting fresh frame after active key press")
            # Current frame already contains a close, body-paired head. Do not add an
            # avoidable one-frame delay; keep the rest of the gate/validation checks.
            self.frames_since_active_press = 0

        if not raw_target.found:
            self.missing_frames += 1
            if self._can_hold_missing_target():
                return self._return_held_missing_target()
            if self._within_recent_lock_grace() and not self.cfg.reset_on_no_head:
                return self._return_no_move(
                    f"target missing; holding gate memory {self.missing_frames}/{self.cfg.locked_target_grace_frames}"
                )
            return self._hard_reject("no raw target")

        if raw_target.source != "head" or raw_target.head_box is None:
            self.missing_frames += 1
            if self._within_recent_lock_grace() and not self.cfg.reset_on_no_head:
                return self._return_no_move(
                    f"raw target is {raw_target.source}; holding gate memory {self.missing_frames}/{self.cfg.locked_target_grace_frames}"
                )
            return self._hard_reject(f"raw target is {raw_target.source}; movement requires head")

        point = (float(raw_target.x), float(raw_target.y))
        conf = float(raw_target.confidence)
        self.missing_frames = 0

        dist_center = math.hypot(point[0] - center_x, point[1] - center_y)
        self._current_center_x = float(center_x)
        self._current_center_y = float(center_y)
        if self.cfg.max_control_distance_px and self.cfg.max_control_distance_px > 0:
            if dist_center > float(self.cfg.max_control_distance_px):
                return self._hard_reject(f"target too far from center: {dist_center:.1f}px")

        recent_hold = self._within_recent_lock_grace()
        hold_like = self._has_recent_confirmed_memory()
        threshold = float(self.cfg.min_head_conf_hold if hold_like else self.cfg.min_head_conf_enter)
        small_like, suspicious_like, suspicious_reason = self._target_small_or_suspicious(raw_target)
        instant_enter = (
            bool(self.cfg.instant_enter_enabled)
            and not hold_like
            and not small_like
            and not suspicious_like
            and dist_center <= float(self.cfg.instant_enter_center_dist_px)
            and conf >= float(self.cfg.instant_enter_min_conf)
            and (not bool(self.cfg.instant_enter_requires_body) or raw_target.body_box is not None)
        )
        if conf < threshold and not instant_enter:
            # If we already have a confirmed target, do not erase its identity because
            # one low-confidence frame is exactly the flicker seen in the v16 logs.
            if hold_like:
                self.last_reason = f"hold target head confidence too low: {conf:.3f} < {threshold:.3f}"
                return False, TargetResult(False, reason=self.last_reason), self.last_reason
            return self._hard_reject(f"head confidence too low for enter: {conf:.3f} < {threshold:.3f}")

        reference = self.last_point or self.last_valid_point
        if reference is not None:
            jump = math.hypot(point[0] - reference[0], point[1] - reference[1])
            max_jump = float(self.cfg.max_target_jump_px)
            # Same-lock large movement must not be filtered from an old point.
            # Either snap/rebase to the current raw head if reliable, or hold the
            # previous target without motor movement.
            if hold_like and self._same_lock_hint(raw_target) and jump > self._locked_rebase_limit_px(raw_target):
                self._current_raw_center_dist = dist_center
                if self.last_valid_point is not None:
                    self._current_last_center_dist = math.hypot(self.last_valid_point[0] - center_x, self.last_valid_point[1] - center_y)
                else:
                    self._current_last_center_dist = dist_center
                if self._can_accept_same_lock_jump(raw_target, jump, dist_center):
                    return self._return_accepted_same_lock_jump(jump, raw_target)
                if self._can_rebase_locked_jump(raw_target, jump):
                    return self._return_rebased_locked_target(jump, raw_target)
                if self._can_hold_locked_jump(raw_target, jump):
                    return self._return_held_locked_jump(jump, raw_target)
            # During hold grace, require a tighter drift. This prevents a different
            # target from inheriting the old target's hold state.
            if hold_like and recent_hold:
                max_jump = min(max_jump, float(self.cfg.locked_target_max_drift_px))
            if jump > max_jump:
                # V17.8.21: for a reliable same-lock, body-paired measurement,
                # accept the current raw detector point instead of returning False
                # or feeding a stale held point to the controller. This removes
                # the movement_ready True/False pulse visible in run logs.
                if self._can_accept_same_lock_jump(raw_target, jump, dist_center):
                    return self._return_accepted_same_lock_jump(jump, raw_target)
                if self._can_hold_locked_jump(raw_target, jump):
                    return self._return_held_locked_jump(jump, raw_target)
                trusted_locked = (
                    bool(getattr(self.cfg, "trust_locked_target_jump", True))
                    and raw_target.body_box is not None
                    and conf >= float(getattr(self.cfg, "trusted_locked_min_conf", 0.45))
                    and "kept locked target" in str(raw_target.reason or "")
                    and jump <= float(getattr(self.cfg, "trusted_locked_jump_px", 95.0))
                )
                if trusted_locked:
                    self.confirm_count = max(self.confirm_count, 1)
                    self.last_point = point
                    self.last_conf = conf
                    self.last_reason = f"trusted locked target jump: {jump:.1f}px <= {self.cfg.trusted_locked_jump_px:.1f}px"
                else:
                    self.confirm_count = 1
                    self.last_point = point
                    self.last_conf = conf
                    self._clear_confirmed_memory()
                    self.last_reason = f"target jump {jump:.1f}px > {max_jump:.1f}px; new candidate 1"
                    return False, TargetResult(False, reason=self.last_reason), self.last_reason
            self.confirm_count += 1
        else:
            self.confirm_count = 1

        self.last_point = point
        self.last_conf = conf

        normal_needed = max(1, int(self.cfg.require_confirmed_frames))
        high_needed = max(1, int(self.cfg.high_conf_confirmed_frames))
        needed = high_needed if conf >= float(self.cfg.high_conf_head) else normal_needed
        if instant_enter:
            needed = 1
        if small_like:
            if conf >= float(self.cfg.small_target_high_conf):
                needed = max(needed, int(self.cfg.small_target_high_conf_frames))
            else:
                needed = max(needed, int(self.cfg.small_target_confirmed_frames))
        if suspicious_like:
            needed = max(needed, int(self.cfg.suspicious_target_confirmed_frames))

        fast_needed = self._reactive_fast_enter_frames(
            raw_target,
            dist_center=dist_center,
            suspicious_like=suspicious_like,
        )
        if fast_needed is not None and not hold_like:
            needed = min(needed, fast_needed)

        if raw_target.body_box is None:
            needed = max(needed, int(getattr(self.cfg, "head_only_confirmed_frames", 2)))
            if conf < float(getattr(self.cfg, "head_only_min_conf", 0.45)):
                self.last_reason = f"head-only conf too low for stable control: {conf:.2f} < {getattr(self.cfg, 'head_only_min_conf', 0.45):.2f}"
                return False, TargetResult(False, reason=self.last_reason), self.last_reason

        # Already-confirmed target uses hysteresis: it does not need to rebuild the
        # full enter window when it reappears quickly with valid geometry.
        if hold_like and conf >= float(self.cfg.min_head_conf_hold):
            out_target = self._smooth_locked_control_target(raw_target)
            point = (float(out_target.x), float(out_target.y))
            self.confirm_count = max(self.confirm_count, needed)
            reason = f"holding confirmed head: conf={conf:.2f}, threshold={self.cfg.min_head_conf_hold:.2f}"
            out_target = self._commit_valid_target(out_target, point, reason)
            return True, out_target, self.last_reason

        if self.confirm_count < needed:
            extra = f", anti_map={suspicious_reason}" if (small_like or suspicious_like) and suspicious_reason else ""
            self.last_reason = f"confirming head: {self.confirm_count}/{needed}, conf={conf:.2f}{extra}"
            return False, TargetResult(False, reason=self.last_reason), self.last_reason

        out_target = raw_target
        if self.was_confirmed or self.last_valid_point is not None:
            out_target = self._smooth_locked_control_target(raw_target)
            point = (float(out_target.x), float(out_target.y))
        reason = (
            f"instant confirmed head: conf={conf:.2f}, dist={dist_center:.1f}"
            if instant_enter else
            (f"reactive confirmed head: {self.confirm_count}/{needed}, conf={conf:.2f}, dist={dist_center:.1f}"
             if fast_needed is not None and not hold_like else
             f"confirmed head: {self.confirm_count}/{needed}, conf={conf:.2f}")
        )
        out_target = self._commit_valid_target(out_target, point, reason)
        return True, out_target, self.last_reason
