from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .types import TargetResult


class KalmanFilter4D:
    """Numerically stable constant-velocity Kalman filter: x, y, vx, vy.

    V17.8.18 audit fixes over the old implementation:
    - dt-scaled white-acceleration process noise instead of a fixed Q each frame;
    - Joseph-form covariance update, plus symmetry/positive diagonal repair;
    - innovation gating and velocity clamping to avoid one bad box poisoning state;
    - explicit ego-motion compensation with covariance inflation.
    """

    def __init__(
        self,
        q: float = 0.06,
        r: float = 0.12,
        *,
        max_velocity: float = 3200.0,
        max_prediction_dt: float = 0.12,
        innovation_gate_px: float = 145.0,
        covariance_floor: float = 1e-6,
        covariance_ceiling: float = 1.0e4,
        ego_covariance_boost: float = 0.18,
    ):
        self.q = max(1e-9, float(q))
        self.r = max(1e-9, float(r))
        self.max_velocity = max(0.0, float(max_velocity))
        self.max_prediction_dt = max(0.001, float(max_prediction_dt))
        self.innovation_gate_px = max(0.0, float(innovation_gate_px))
        self.covariance_floor = max(1e-12, float(covariance_floor))
        self.covariance_ceiling = max(self.covariance_floor * 10.0, float(covariance_ceiling))
        self.ego_covariance_boost = max(0.0, float(ego_covariance_boost))
        self._I4 = np.eye(4, dtype=np.float64)
        self._H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self._Ht = self._H.T.copy()
        self._R2 = np.eye(2, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(4, dtype=np.float64)
        # Position is immediately observed; velocity is initially uncertain.
        self.P = np.diag([16.0, 16.0, 900.0, 900.0]).astype(np.float64)
        self.R = self._R2 * self.r
        self.initialized = False

    def _transition(self, dt: float) -> np.ndarray:
        F = self._I4.copy()
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        # Continuous white-acceleration model discretized for [x, y, vx, vy].
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.q
        return q * np.array(
            [
                [dt4 * 0.25, 0.0, dt3 * 0.5, 0.0],
                [0.0, dt4 * 0.25, 0.0, dt3 * 0.5],
                [dt3 * 0.5, 0.0, dt2, 0.0],
                [0.0, dt3 * 0.5, 0.0, dt2],
            ],
            dtype=np.float64,
        )

    def _repair_covariance(self) -> None:
        if not np.isfinite(self.P).all():
            self.P = np.diag([16.0, 16.0, 900.0, 900.0]).astype(np.float64)
            return
        self.P = 0.5 * (self.P + self.P.T)
        diag = np.diag(self.P).copy()
        diag = np.clip(diag, self.covariance_floor, self.covariance_ceiling)
        np.fill_diagonal(self.P, diag)

    def _clamp_velocity(self) -> None:
        if self.max_velocity <= 0.0:
            return
        vx, vy = float(self.state[2]), float(self.state[3])
        speed = math.hypot(vx, vy)
        if speed > self.max_velocity:
            scale = self.max_velocity / max(speed, 1e-9)
            self.state[2] = vx * scale
            self.state[3] = vy * scale

    def apply_ego_motion(self, dx: float, dy: float, scaler: float = 2.7) -> None:
        if self.initialized:
            sx = float(dx) * float(scaler)
            sy = float(dy) * float(scaler)
            self.state[0] -= sx
            self.state[1] -= sy
            if self.ego_covariance_boost > 0.0:
                boost = self.ego_covariance_boost * (abs(sx) + abs(sy) + 1.0)
                self.P[0, 0] = min(self.covariance_ceiling, self.P[0, 0] + boost)
                self.P[1, 1] = min(self.covariance_ceiling, self.P[1, 1] + boost)
                self._repair_covariance()

    def decay_velocity(self, decay_factor: float = 0.86) -> None:
        if self.initialized:
            d = max(0.0, min(1.0, float(decay_factor)))
            self.state[2] *= d
            self.state[3] *= d

    def predict(self, dt: float) -> Tuple[float, float]:
        if not self.initialized:
            return 0.0, 0.0
        dt = max(0.0, min(float(dt), self.max_prediction_dt))
        F = self._transition(dt)
        Q = self._process_noise(dt)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        self._clamp_velocity()
        self._repair_covariance()
        if not np.isfinite(self.state).all():
            self.reset()
            return 0.0, 0.0
        return float(self.state[0]), float(self.state[1])

    def update(self, mx: float, my: float) -> Tuple[float, float]:
        mx, my = float(mx), float(my)
        if not (math.isfinite(mx) and math.isfinite(my)):
            if self.initialized:
                return float(self.state[0]), float(self.state[1])
            return 0.0, 0.0
        if not self.initialized:
            self.state = np.array([mx, my, 0.0, 0.0], dtype=np.float64)
            self.P = np.diag([4.0, 4.0, 900.0, 900.0]).astype(np.float64)
            self.initialized = True
            return mx, my

        z = np.array([mx, my], dtype=np.float64)
        y = z - self._H @ self.state
        innovation_len = float(np.linalg.norm(y))
        if self.innovation_gate_px > 0.0 and innovation_len > self.innovation_gate_px:
            # Treat hard identity swaps / false detections as new measurements instead of
            # blending them into the current velocity estimate.
            self.state = np.array([mx, my, 0.0, 0.0], dtype=np.float64)
            self.P = np.diag([9.0, 9.0, 900.0, 900.0]).astype(np.float64)
            return mx, my

        PHt = self.P @ self._Ht
        S = self._H @ PHt + self.R
        try:
            K = PHt @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = PHt @ np.linalg.pinv(S)

        self.state = self.state + K @ y
        # Joseph stabilized update: keeps P symmetric positive semi-definite.
        I_KH = self._I4 - K @ self._H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self._clamp_velocity()
        self._repair_covariance()

        if not np.isfinite(self.state).all():
            self.reset()
            return mx, my
        return float(self.state[0]), float(self.state[1])


@dataclass
class EmaPointTracker:
    alpha: float = 0.45
    _point: Optional[Tuple[float, float]] = None
    lost_frames: int = 0
    max_lost_frames: int = 4

    def reset(self) -> None:
        self._point = None
        self.lost_frames = 0

    def apply_ego_motion(self, dx: float, dy: float, scaler: float = 2.7) -> None:
        if self._point is not None:
            x, y = self._point
            self._point = (x - float(dx) * float(scaler), y - float(dy) * float(scaler))

    def update(self, target: TargetResult, dt: float = 0.0) -> TargetResult:
        if not target.found:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.reset()
            return target
        self.lost_frames = 0
        if self._point is None:
            self._point = (target.x, target.y)
        else:
            px, py = self._point
            a = max(0.0, min(1.0, float(self.alpha)))
            self._point = (
                a * target.x + (1.0 - a) * px,
                a * target.y + (1.0 - a) * py,
            )
        target.x, target.y = self._point
        target.reason = f"{target.reason}; ema_alpha={self.alpha}"
        return target


@dataclass
class LegacyPointTracker:
    q: float = 0.06
    r: float = 0.12
    max_lost_frames: int = 4
    ego_scaler: float = 2.7
    hold_body_fallback_after_head_frames: int = 2
    kalman_max_velocity_px_s: float = 3200.0
    kalman_max_prediction_dt: float = 0.12
    kalman_innovation_gate_px: float = 145.0
    kalman_ego_covariance_boost: float = 0.18

    def __post_init__(self) -> None:
        self.kalman = KalmanFilter4D(
            self.q,
            self.r,
            max_velocity=self.kalman_max_velocity_px_s,
            max_prediction_dt=self.kalman_max_prediction_dt,
            innovation_gate_px=self.kalman_innovation_gate_px,
            ego_covariance_boost=self.kalman_ego_covariance_boost,
        )
        self.lost_frames = 0
        self.last_source: str = "none"
        self.head_lost_frames = 999

    def reset(self) -> None:
        self.kalman.reset()
        self.lost_frames = 0
        self.last_source = "none"
        self.head_lost_frames = 999

    def apply_ego_motion(self, dx: float, dy: float, scaler: Optional[float] = None) -> None:
        self.kalman.apply_ego_motion(dx, dy, self.ego_scaler if scaler is None else scaler)

    def _predicted_target(self, template: TargetResult, reason: str) -> TargetResult:
        if not self.kalman.initialized:
            return template
        template.x = float(self.kalman.state[0])
        template.y = float(self.kalman.state[1])
        template.reason = f"{template.reason}; {reason}"
        return template

    def update(self, target: TargetResult, dt: float = 0.0) -> TargetResult:
        if not target.found:
            self.lost_frames += 1
            self.head_lost_frames += 1
            if self.kalman.initialized and self.lost_frames <= self.max_lost_frames:
                self.kalman.predict(dt)
                self.kalman.decay_velocity()
            else:
                self.reset()
            return target

        # Hold the previous head point across a short body-only blink instead of
        # jumping vertically to a body fallback point.
        if (
            target.source == "body_fallback"
            and self.last_source == "head"
            and self.head_lost_frames <= self.hold_body_fallback_after_head_frames
            and self.kalman.initialized
        ):
            self.head_lost_frames += 1
            self.kalman.predict(dt)
            target = self._predicted_target(target, "held predicted head after transient head loss")
            return target

        self.lost_frames = 0
        if target.source == "head":
            self.head_lost_frames = 0
        else:
            self.head_lost_frames += 1

        if self.kalman.initialized:
            self.kalman.predict(dt)
        x, y = self.kalman.update(float(target.x), float(target.y))
        if not (math.isfinite(x) and math.isfinite(y)):
            self.reset()
            return target
        target.x, target.y = x, y
        target.reason = (
            f"{target.reason}; kalman_cv_joseph q={self.q} r={self.r} "
            f"gate={self.kalman_innovation_gate_px} vmax={self.kalman_max_velocity_px_s} ego={self.ego_scaler}"
        )
        self.last_source = target.source
        return target
