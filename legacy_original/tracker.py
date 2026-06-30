import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import Config
from kalman import KalmanFilter4D


@dataclass
class Target:
    x: float
    y: float
    area: float
    distance: float
    w: int
    h: int
    aspect_ratio: float
    circularity: float
    bbox_x: Optional[int] = None
    bbox_y: Optional[int] = None
    confidence: float = 1.0

    # OpenCV 头部估计调试信息；不参与追踪核心计算，供 debug 可视化和离线微调用。
    raw_head_x: Optional[float] = None
    raw_head_y: Optional[float] = None
    head_roi_x: Optional[int] = None
    head_roi_y: Optional[int] = None
    head_roi_w: Optional[int] = None
    head_roi_h: Optional[int] = None
    head_quality: float = 0.0
    head_pose_hint: str = "upright"


class TargetTracker:
    def __init__(self, cfg: Config):
        self.kalman = KalmanFilter4D(cfg.kalman_process_noise, cfg.kalman_measurement_noise)
        self.min_hits = cfg.temporal_min_hits
        self.max_lost = cfg.temporal_max_lost
        self.dist_limit = cfg.spatial_distance_limit
        self.smoke_ratio = cfg.anti_smoke_max_area_ratio
        self.smoke_min_area = cfg.anti_smoke_min_area
        self.static_enabled = cfg.static_filter_enabled
        self.static_max_frames = cfg.static_max_frames
        self.static_pos_thr = cfg.static_pos_threshold
        self.static_area_ratio = cfg.static_area_change_ratio
        self.area_priority_ratio = cfg.area_priority_ratio
        self.reject_cooldown = cfg.model_filter_reject_cooldown
        self.reject_radius = cfg.model_filter_reject_radius

        self.last_velocity = None
        self.direction_change_guard_enabled = cfg.direction_change_guard_enabled
        self.direction_change_threshold = cfg.direction_change_threshold
        self.direction_change_min_speed = cfg.direction_change_min_speed
        self.direction_change_bypass_dist = cfg.direction_change_bypass_dist
        self.active_target: Optional[Target] = None
        self.hits = 0
        self.lost_frames = 0
        self.last_area = 0.0
        self.target_acquired_time = 0.0
        self.static_counter = 0
        self.last_reject_reason: Optional[str] = None

        self._rejected_positions: List[Tuple[float, float, float]] = []
        self._last_cleanup = 0.0
        self._cleanup_interval = 0.5

    def _cleanup_rejected_positions(self, now: Optional[float] = None):
        now = time.perf_counter() if now is None else now
        if now - self._last_cleanup <= self._cleanup_interval:
            return
        self._rejected_positions = [
            (px, py, pt) for px, py, pt in self._rejected_positions
            if now - pt < self.reject_cooldown
        ]
        self._last_cleanup = now

    def add_rejected_position(self, x: float, y: float):
        """记录被模型拒绝的目标位置，短期内不再锁定该位置。"""
        now = time.perf_counter()
        self._cleanup_rejected_positions(now)
        self._rejected_positions.append((x, y, now))

    def _is_near_rejected(self, x: float, y: float) -> bool:
        """检查位置是否在最近被拒绝的位置附近。"""
        now = time.perf_counter()
        self._cleanup_rejected_positions(now)
        if not self._rejected_positions:
            return False
        r2 = self.reject_radius * self.reject_radius
        for px, py, pt in self._rejected_positions:
            if now - pt > self.reject_cooldown:
                continue
            dx = x - px
            dy = y - py
            if dx * dx + dy * dy < r2:
                return True
        return False

    def _select_best_by_area(self, detections: List[Target]) -> Optional[Target]:
        """只取面积最大的两个候选，避免每帧完整排序。"""
        if len(detections) <= 1:
            return None
        largest = None
        second = None
        for t in detections:
            if largest is None or t.area > largest.area:
                second = largest
                largest = t
            elif second is None or t.area > second.area:
                second = t
        if largest is not None and second is not None and second.area > 0:
            if (largest.area / second.area) >= self.area_priority_ratio:
                return largest
        return None

    def _pick_nearest_to_prediction(self, detections: List[Target], pred_x: float, pred_y: float,
                                    limit: float) -> Optional[Target]:
        best = None
        min_d2 = float("inf")
        limit2 = limit * limit
        for t in detections:
            dx = t.x - pred_x
            dy = t.y - pred_y
            d2 = dx * dx + dy * dy
            if d2 < limit2 and d2 < min_d2:
                min_d2 = d2
                best = t
        return best

    def update(self, detections: List[Target], cx: float, cy: float, dt: float
               ) -> Tuple[float, float, float, float, float, float, bool, bool, Optional[Target]]:
        self.last_reject_reason = None
        now = time.perf_counter()
        pdx, pdy = self.kalman.predict(dt)
        pred_x = cx + pdx if self.kalman.initialized else cx
        pred_y = cy + pdy if self.kalman.initialized else cy

        if detections:
            detections = [t for t in detections if not self._is_near_rejected(t.x, t.y)]

        if not detections:
            self.lost_frames += 1
            if self.kalman.initialized and self.lost_frames <= self.max_lost:
                self.kalman.decay_velocity()
                return (pdx, pdy, 0.0, 0.0, self.kalman.state[2], self.kalman.state[3],
                        self.hits >= self.min_hits, False, None)
            self.reset()
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, None

        best = None
        if self.kalman.initialized and self.active_target is not None and self.lost_frames <= 2:
            best = self._pick_nearest_to_prediction(detections, pred_x, pred_y, self.dist_limit)
            if best is None:
                best = self._pick_nearest_to_prediction(
                    detections, self.active_target.x, self.active_target.y, self.dist_limit * 1.5
                )

        if best is None:
            best = self._select_best_by_area(detections)
            if best is None:
                best = min(detections, key=lambda t: t.distance)

        if math.hypot(best.x - pred_x, best.y - pred_y) > self.dist_limit:
            self.reset()
            best = min(detections, key=lambda t: t.distance)

        if self.kalman.initialized and self.active_target is not None:
            vx = self.kalman.state[2]
            vy = self.kalman.state[3]
            if self.direction_change_guard_enabled and self.last_velocity is not None:
                last_vx, last_vy = self.last_velocity
                dot = last_vx * vx + last_vy * vy
                last_speed = math.hypot(last_vx, last_vy)
                cur_speed = math.hypot(vx, vy)
                mags = last_speed * cur_speed
                if (best.distance > self.direction_change_bypass_dist and
                        last_speed >= self.direction_change_min_speed and
                        cur_speed >= self.direction_change_min_speed and
                        mags > 0.01 and
                        (dot / mags) < self.direction_change_threshold):
                    self.last_reject_reason = "direction_change"
                    self.lost_frames += 2
                    if self.lost_frames <= self.max_lost:
                        self.kalman.decay_velocity()
                        return (pdx, pdy, 0.0, 0.0,
                                self.kalman.state[2], self.kalman.state[3],
                                self.hits >= self.min_hits, False, None)
                    self.reset()
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, None
            self.last_velocity = (vx, vy)

        # 烟雾抑制：Config 会保证 smoke_min_area 不超过可检测面积区间。
        if (self.hits > 0 and self.last_area > 0 and
                best.area > self.last_area * self.smoke_ratio and
                best.area > self.smoke_min_area):
            self.last_reject_reason = "smoke_area_jump"
            self.lost_frames += 1
            if self.lost_frames <= self.max_lost:
                self.kalman.decay_velocity()
                return (pdx, pdy, 0.0, 0.0, self.kalman.state[2], self.kalman.state[3],
                        self.hits >= self.min_hits, False, None)
            self.reset()
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, None

        if self.static_enabled and self.active_target is not None:
            pos_change = math.hypot(best.x - self.active_target.x, best.y - self.active_target.y)
            area_change = abs(best.area - self.active_target.area) / max(self.active_target.area, 1.0)
            if pos_change <= self.static_pos_thr and area_change <= self.static_area_ratio:
                self.static_counter += 1
                if self.static_counter >= self.static_max_frames:
                    self.last_reject_reason = "static"
                    self.reset()
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, None
            else:
                self.static_counter = 0
        else:
            self.static_counter = 0

        self.active_target = best
        self.last_area = best.area
        self.lost_frames = 0
        if self.hits == 0:
            self.target_acquired_time = now
        self.hits += 1

        raw_ex = best.x - cx
        raw_ey = best.y - cy
        ex, ey = self.kalman.update(raw_ex, raw_ey)
        return (ex, ey, raw_ex, raw_ey, self.kalman.state[2], self.kalman.state[3],
                self.hits >= self.min_hits, True, best)

    def reset(self):
        self.kalman.reset()
        self.active_target = None
        self.hits = 0
        self.lost_frames = 0
        self.last_area = 0.0
        self.target_acquired_time = 0.0
        self.static_counter = 0
        self.last_velocity = None
