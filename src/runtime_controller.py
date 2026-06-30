from __future__ import annotations

import ctypes
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .leonardo_driver import LeonardoMouseDriver
from .log_utils import log


@dataclass
class ControlConfig:
    enabled: bool = False
    mode: str = "leonardo"  # none | leonardo
    port: str = "auto"
    baud: int = 115200

    # Old-project-compatible motion parameters.
    # Effective movement = error * sensitivity_scaler * gain_axis.
    gain_x: float = 1.0
    gain_y: float = 1.0
    sensitivity_scaler: float = 0.70
    sensitivity_boost_close: float = 1.12
    close_range_threshold: float = 130.0
    min_kinetic_speed: float = 0.0

    # V17.8.18: explicit PID layer with anti-windup. The old controller was
    # effectively P + residual integration. This keeps the same behavior by
    # default while making the control law bounded, resettable and auditable.
    pid_enabled: bool = True
    pid_kp: float = 1.0
    pid_ki: float = 0.006
    pid_kd: float = 0.004
    pid_integral_limit: float = 28.0
    pid_derivative_smoothing: float = 0.70
    pid_output_limit: float = 0.0
    pid_integral_deadband_px: float = 7.0
    pid_reset_jump_px: float = 95.0
    pid_dt_min: float = 0.001
    pid_dt_max: float = 0.08

    # v17.7: lateral fast-target compensation is still enabled, but it is now
    # measurement-noise aware. V17.6 used raw frame-to-frame error velocity; when
    # detector boxes jittered or target identity briefly swapped, that velocity was
    # amplified into high-frequency camera shake. The fields below keep the
    # feature active while constraining it to physically plausible, same-target
    # movement.
    velocity_lead_ms: float = 24.0
    velocity_lead_max_px: float = 16.0
    error_velocity_smoothing: float = 0.82
    velocity_reset_jump_px: float = 45.0
    max_error_velocity_px_s: float = 1800.0
    velocity_lead_error_fraction: float = 0.35
    velocity_lead_min_error_px: float = 8.0
    max_residual_total: float = 64.0

    # V17.8.7: detector publishes at inference FPS, while HID drains at a much
    # higher motor-loop rate.  Adding a whole correction at once creates a
    # burst-pause-burst packet pattern.  Smooth injection spreads a new visual
    # correction across a short tick window so motion remains continuous.
    smooth_residual_injection: bool = True
    residual_injection_min_ticks: int = 4
    residual_injection_max_ticks: int = 28
    residual_injection_interval_fraction: float = 0.85

    # V17.8.7: final packet shaping.  This sits after target lock, residual
    # calculation and overshoot guard.  It limits packet-to-packet acceleration,
    # brakes on near-center direction reversals, and keeps the uncommitted
    # residual in the controller so convergence is not lost.
    natural_motion_enabled: bool = True
    natural_motion_alpha: float = 0.46
    natural_motion_max_delta: float = 2.4
    natural_motion_close_delta: float = 0.85
    natural_motion_close_px: float = 34.0
    natural_motion_zero_cross_brake: bool = True

    # V17.8.20: continuous pull bridge. Real detections may briefly miss for one
    # inference frame while the target is moving or while the screen is being
    # corrected. Older builds immediately invalidated the motor target and reset
    # the packet profile, which creates a visible move-pause-move cadence. These
    # fields keep only the already-validated residual draining for a very short
    # gap; no new correction is invented during the gap.
    continuous_motion_profile_hold: bool = True
    continuous_motion_profile_hold_ms: float = 45.0
    continuous_motion_profile_decay: float = 0.82
    no_target_soft_hold_enabled: bool = True
    no_target_soft_hold_ms: float = 65.0

    # v17.8 re-audit: a target is published by the vision loop every inference
    # cycle, not every motor tick. A fixed 25ms stale timeout was too close to
    # real 60-100 FPS inference jitter, so the motor could clear a target between
    # two valid detections and create visible stutter. Use an adaptive freshness
    # timeout based on the recent submit interval.
    adaptive_stale_target: bool = True
    stale_target_min_seconds: float = 0.045
    stale_target_max_seconds: float = 0.120
    stale_target_interval_multiplier: float = 3.0

    # If the user presses the active key while a just-published target is already
    # available, consume it instead of always discarding the first frame. This keeps
    # the anti-ghost protection but removes an avoidable one-frame reaction delay.
    active_press_accept_recent_ms: float = 22.0

    # Held/predicted targets are useful to bridge one-frame detector misses, but
    # they must not receive full velocity lead or full movement gain.
    held_target_sensitivity_scale: float = 0.35
    held_target_disable_lead: bool = True

    # Convert residual into HID integer motion with less latency than int()
    # truncation. V17.8.1 makes this conditional on current target error so it
    # cannot generate 1px jitter near the center.
    micro_step_threshold: float = 0.95
    micro_step_min_error_px: float = 8.0

    # Explicit no-overshoot guard: movement must converge toward current error,
    # not pass it and pull back. These limits are axis-aware and are applied
    # before residual injection and again when residual is drained into HID.
    overshoot_guard_enabled: bool = True
    overshoot_error_fraction: float = 0.42
    residual_error_fraction: float = 0.58
    drain_error_fraction: float = 0.45

    # v17.8.3: final-settle lock.  Logs showed that after the crosshair had
    # already reached the head center, detector/track jitter kept publishing
    # 1-5 px alternating errors.  The motor treated those as real corrections,
    # creating the observed shake-after-arrival.  Once inside the settle window,
    # the controller freezes residual motion until a real drift persists outside
    # a larger exit radius.  This is hysteresis, not a disabled feature.
    settle_lock_enabled: bool = True
    settle_enter_px: float = 2.2
    settle_enter_frames: int = 1
    settle_exit_px: float = 6.0
    settle_hard_exit_px: float = 18.0
    settle_release_frames: int = 2
    settle_min_conf: float = 0.30
    # v17.8.4: when scoped, the apparent head box is larger and the detector center
    # jitters by more pixels.  A fixed 6px settle exit releases too easily and causes
    # shake. Scale the settle window by current head-box radius while capping it.
    settle_target_radius_enabled: bool = True
    settle_enter_radius_fraction: float = 0.10
    settle_exit_radius_fraction: float = 0.55
    settle_hard_exit_radius_fraction: float = 1.15
    settle_exit_max_px: float = 18.0
    settle_hard_exit_max_px: float = 34.0

    # v17.8: do not turn small remaining detector noise near the center into
    # visible micro-shake. This is damping, not disabling; large errors keep full
    # response and velocity lead.
    near_center_damping_px: float = 12.0
    near_center_damping_scale: float = 0.45

    # v13 separates three limits:
    # - max_submit_error_px: clamps the visual error before it becomes motion;
    # - max_residual_add_per_frame: prevents a one-frame false positive from creating a jump;
    # - max_move/max_step: HID packet clamp.
    max_submit_error_px: float = 180.0
    max_residual_add_per_frame: float = 24.0
    max_step: int = 34
    max_move: int = 34

    # Center deadzone. V17.7 keeps aiming responsive, but restores enough
    # fine-zone damping to avoid micro-oscillation around the target center.
    deadzone: float = 3.4
    fine_deadzone: float = 1.8
    residual_epsilon: float = 0.18
    invert_y: bool = False

    only_when_active: bool = True
    active_key: str = "shift"
    toggle_key: str = "f8"
    quit_key: str = "f10"

    control_loop_hz: int = 1000
    stale_target_seconds: float = 0.16
    reset_residual_on_direction_change: bool = True
    suppress_reverse_inside_deadzone: bool = True
    log_interval_seconds: float = 1.0

    # Ego-motion compensation. Every successful HID move is accumulated and consumed by
    # the vision loop so the tracker understands that the camera/crosshair moved.
    ego_scaler: float = 2.7

    # Anti ghost-move gates.
    require_head_for_movement: bool = True
    allow_body_fallback_control: bool = False
    clear_residual_on_no_target: bool = True
    clear_residual_on_active_press: bool = True

    # V17.8.22: audited fire gate. Older builds clicked every motor tick when
    # fire_enabled was true, the target was inside fire_radius, and the shaped
    # movement packet was zero. That could spam CLICK at control_loop_hz and did
    # not distinguish a fresh stable target from a stale/held/predicted target.
    # The fields below implement hysteresis, cooldown, target freshness,
    # confidence gating, optional held-target blocking, and one-shot arming.
    fire_enabled: bool = False
    fire_radius: float = 4.0
    fire_exit_radius: float = 7.0
    fire_rearm_radius: float = 9.0
    fire_cooldown_ms: float = 165.0
    fire_min_conf: float = 0.50
    fire_stable_frames: int = 2
    fire_max_target_age_ms: float = 90.0
    fire_allow_held_target: bool = False
    fire_repeat_while_in_radius: bool = False
    fire_reset_on_active_release: bool = True
    fire_log_events: bool = False
    # V17.8.23: integrity checks shared with the motion pipeline.
    # The fire gate should only trigger after a fresh, stable, non-held target is
    # actually settled and no residual motion debt remains.  These guards prevent
    # fire_enabled from racing the movement loop or reacting to one-frame target jitter.
    fire_require_zero_motion: bool = True
    fire_max_motion_debt_px: float = 0.90
    fire_min_time_after_move_ms: float = 22.0
    fire_stable_error_delta_px: float = 2.8
    fire_block_during_settle_release: bool = True
    # V17.8.24: repeat/held-target integrity. Repeating inside radius should not
    # consume the same detector publication forever, and held/predicted targets
    # need a stricter confidence/age gate if explicitly allowed by the config.
    fire_repeat_requires_fresh_detection: bool = True
    fire_min_repeat_seq_delta: int = 1
    fire_held_target_min_conf: float = 0.72
    fire_held_target_max_age_ms: float = 45.0
    fire_block_on_stale_gate: bool = True

    def __post_init__(self) -> None:
        self.fire_radius = max(0.0, float(self.fire_radius))
        self.fire_exit_radius = max(self.fire_radius, float(self.fire_exit_radius))
        self.fire_rearm_radius = max(self.fire_exit_radius, float(self.fire_rearm_radius))
        self.fire_cooldown_ms = max(0.0, float(self.fire_cooldown_ms))
        self.fire_min_conf = max(0.0, min(1.0, float(self.fire_min_conf)))
        self.fire_stable_frames = max(1, int(self.fire_stable_frames))
        self.fire_max_target_age_ms = max(1.0, float(self.fire_max_target_age_ms))
        self.fire_max_motion_debt_px = max(0.0, float(self.fire_max_motion_debt_px))
        self.fire_min_time_after_move_ms = max(0.0, float(self.fire_min_time_after_move_ms))
        self.fire_stable_error_delta_px = max(0.0, float(self.fire_stable_error_delta_px))
        self.fire_min_repeat_seq_delta = max(1, int(self.fire_min_repeat_seq_delta))
        self.fire_held_target_min_conf = max(self.fire_min_conf, min(1.0, float(self.fire_held_target_min_conf)))
        self.fire_held_target_max_age_ms = max(1.0, min(self.fire_max_target_age_ms, float(self.fire_held_target_max_age_ms)))

    # v17.3: avoid a 2-second realtime-loop stall when F8 first enables control.
    # The Leonardo serial connection can be opened in the background while the
    # detector keeps running. Movement is still suppressed until the driver is ready.
    preconnect_driver: bool = True
    driver_connect_in_background: bool = True


class PIDAxis:
    """Bounded one-axis PID controller with derivative filtering and anti-windup."""

    def __init__(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0
        self.derivative = 0.0
        self.initialized = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0
        self.derivative = 0.0
        self.initialized = False

    def update(self, error: float, now: float, cfg: ControlConfig, *, fine_dead: float = 0.0) -> float:
        error = float(error)
        if not math.isfinite(error):
            self.reset()
            return 0.0
        kp = max(0.0, float(getattr(cfg, "pid_kp", 1.0)))
        ki = max(0.0, float(getattr(cfg, "pid_ki", 0.0)))
        kd = max(0.0, float(getattr(cfg, "pid_kd", 0.0)))
        dt_min = max(1e-4, float(getattr(cfg, "pid_dt_min", 0.001)))
        dt_max = max(dt_min, float(getattr(cfg, "pid_dt_max", 0.08)))
        reset_jump = max(0.0, float(getattr(cfg, "pid_reset_jump_px", 95.0)))
        if not self.initialized:
            self.prev_error = error
            self.prev_time = now
            self.derivative = 0.0
            self.integral = 0.0
            self.initialized = True
        dt = max(dt_min, min(float(now - self.prev_time), dt_max)) if self.prev_time > 0.0 else dt_min
        jump = abs(error - self.prev_error)
        if reset_jump > 0.0 and jump > reset_jump:
            # Target identity swap or bad measurement: do not carry old I/D state.
            self.integral = 0.0
            self.derivative = 0.0
            self.prev_error = error
            self.prev_time = now
            return kp * error

        raw_d = (error - self.prev_error) / dt
        a = max(0.0, min(1.0, float(getattr(cfg, "pid_derivative_smoothing", 0.70))))
        self.derivative = self.derivative * a + raw_d * (1.0 - a)

        deadband = max(float(fine_dead), max(0.0, float(getattr(cfg, "pid_integral_deadband_px", 7.0))))
        i_limit = max(0.0, float(getattr(cfg, "pid_integral_limit", 28.0)))
        if ki > 0.0 and abs(error) > deadband:
            self.integral += error * dt
            if i_limit > 0.0:
                self.integral = max(-i_limit, min(i_limit, self.integral))
        else:
            # Bleed integral near center so it cannot create endpoint oscillation.
            self.integral *= 0.72
            if abs(self.integral) < 1e-4:
                self.integral = 0.0

        out = kp * error + ki * self.integral + kd * self.derivative
        out_limit = max(0.0, float(getattr(cfg, "pid_output_limit", 0.0)))
        if out_limit <= 0.0:
            out_limit = max(1.0, float(getattr(cfg, "max_submit_error_px", 180.0)))
        out = max(-out_limit, min(out_limit, out))
        # Do not let I/D reverse the required direction. Braking is allowed only by
        # shrinking magnitude; direction flips are handled by residual reset logic.
        if abs(error) > max(float(fine_dead), 1e-6) and out != 0.0 and math.copysign(1.0, out) != math.copysign(1.0, error):
            out = 0.0
        self.prev_error = error
        self.prev_time = now
        return out


class RuntimeController:
    """Stable high-rate residual motor controller.

    Main fixes over v8:
    - uses old-project movement formula instead of a too-aggressive double-gain path;
    - uses a real center deadzone to stop end-point oscillation;
    - records sent mouse deltas for Kalman ego-motion compensation;
    - clears residual on key release / target loss / direction flip near center.
    """

    _VK = {
        "shift": 0x10,
        "lshift": 0xA0,
        "rshift": 0xA1,
        "ctrl": 0x11,
        "lctrl": 0xA2,
        "rctrl": 0xA3,
        "alt": 0x12,
        "lalt": 0xA4,
        "ralt": 0xA5,
        "f8": 0x77,
        "f10": 0x79,
        "mouse1": 0x01,
        "mouse2": 0x02,
        "space": 0x20,
    }

    def __init__(self, cfg: ControlConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self._driver: Optional[LeonardoMouseDriver] = None
        self._driver_lock = threading.RLock()
        self._connect_thread: Optional[threading.Thread] = None
        self._driver_connecting = False
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        self._toggle_latch = False

        self._state_lock = threading.Lock()
        self._target_seq = 0
        self._target_time = 0.0
        self._target_valid = False
        self._ex = 0.0
        self._ey = 0.0
        self._distance = 0.0
        self._confidence = 0.0
        self._target_held = False
        self._target_radius = 0.0
        self._last_submit_time = 0.0
        self._submit_interval_ema = 0.0

        self._error_lock = threading.Lock()
        self._last_error_x = 0.0
        self._last_error_y = 0.0
        self._last_error_time = 0.0
        self._error_vx = 0.0
        self._error_vy = 0.0

        self._pid_x = PIDAxis()
        self._pid_y = PIDAxis()

        self._residual_lock = threading.RLock()
        self._residual_x = 0.0
        self._residual_y = 0.0
        self._pending_add_x = 0.0
        self._pending_add_y = 0.0
        self._pending_ticks = 0
        self._motion_profile_x = 0.0
        self._motion_profile_y = 0.0
        self._motion_round_x = 0.0
        self._motion_round_y = 0.0
        self._last_profile_zero_time = 0.0
        self._last_seq_consumed = -1
        self._last_warn = 0.0
        self._active_was_down = False

        self._fire_lock = threading.RLock()
        self._fire_last_click_time = 0.0
        self._fire_last_click_seq = -1
        self._fire_last_click_error = 999999.0
        self._fire_candidate_seq = -1
        self._fire_stable_frames = 0
        self._fire_armed = True
        self._fire_last_reason = "init"
        self._fire_candidate_ex = 0.0
        self._fire_candidate_ey = 0.0
        self._fire_last_move_time = 0.0

        self._settled_lock = threading.RLock()
        self._settled = False
        self._settle_enter_count = 0
        self._settle_release_count = 0

        self._ego_lock = threading.Lock()
        self._ego_dx = 0.0
        self._ego_dy = 0.0

        if cfg.mode not in ("none", "leonardo"):
            raise ValueError(f"Unknown control.mode: {cfg.mode}")
        if cfg.mode == "none":
            log("control mode none: detection only", "WARN")
        else:
            if cfg.preconnect_driver:
                log("control mode leonardo: serial device preconnect is enabled", "INFO")
            else:
                log("control mode leonardo: serial device is connected on demand", "INFO")
        log(f"control initial state: {'ON' if self.enabled else 'OFF'}; {cfg.toggle_key}=toggle, {cfg.quit_key}=quit", "INFO")
        self.start()
        if cfg.mode == "leonardo" and cfg.preconnect_driver:
            self._start_driver_connect_async(reason="startup preconnect")

    @staticmethod
    def _now() -> float:
        return time.perf_counter()

    @classmethod
    def _vk_code(cls, key: str) -> Optional[int]:
        k = str(key or "").strip().lower()
        if k.startswith("0x"):
            try:
                return int(k, 16)
            except ValueError:
                return None
        if len(k) == 1:
            return ord(k.upper())
        return cls._VK.get(k)

    @classmethod
    def _key_down_fast(cls, key: str) -> bool:
        code = cls._vk_code(key)
        if code is not None:
            try:
                return bool(ctypes.windll.user32.GetAsyncKeyState(code) & 0x8000)
            except Exception:
                pass
        try:
            import keyboard
            return keyboard.is_pressed(key)
        except Exception:
            return False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._motor_loop, name="stable-motor", daemon=True)
        self._thread.start()

    def _driver_ready_locked(self) -> bool:
        return bool(self._driver is not None and self._driver.initialized)

    def driver_status(self) -> str:
        if self.cfg.mode != "leonardo":
            return "disabled"
        with self._driver_lock:
            if self._driver_ready_locked():
                return "ready"
            if self._driver_connecting:
                return "connecting"
            return "not_ready"

    def _connect_driver_sync(self, reason: str = "sync request") -> bool:
        if self.cfg.mode != "leonardo" or self._closed:
            return False
        with self._driver_lock:
            if self._driver_ready_locked():
                return True
            if self._driver_connecting and threading.current_thread() is not self._connect_thread:
                return False
            self._driver_connecting = True
            try:
                log(f"Leonardo connect begin: {reason}", "INFO")
                # Recreate a failed driver rather than relying on a stale serial object.
                if self._driver is not None and not self._driver.initialized:
                    try:
                        self._driver.close()
                    except Exception:
                        pass
                    self._driver = None
                if self._driver is None:
                    self._driver = LeonardoMouseDriver(self.cfg.port, self.cfg.baud)
                return bool(self._driver and self._driver.initialized)
            finally:
                self._driver_connecting = False

    def _start_driver_connect_async(self, reason: str = "async request") -> None:
        if self.cfg.mode != "leonardo" or self._closed:
            return
        with self._driver_lock:
            if self._driver_ready_locked() or self._driver_connecting:
                return
            self._driver_connecting = True

        def _worker() -> None:
            try:
                log(f"Leonardo async connect begin: {reason}", "INFO")
                with self._driver_lock:
                    if not self._driver_ready_locked() and not self._closed:
                        if self._driver is not None and not self._driver.initialized:
                            try:
                                self._driver.close()
                            except Exception:
                                pass
                            self._driver = None
                        if self._driver is None:
                            self._driver = LeonardoMouseDriver(self.cfg.port, self.cfg.baud)
            finally:
                with self._driver_lock:
                    self._driver_connecting = False
                log(f"Leonardo async connect end: status={self.driver_status()}", "INFO")

        self._connect_thread = threading.Thread(target=_worker, name="leonardo-preconnect", daemon=True)
        self._connect_thread.start()

    def _ensure_driver(self) -> bool:
        if self.cfg.mode != "leonardo":
            return False
        with self._driver_lock:
            if self._driver_ready_locked():
                return True
        if self.cfg.driver_connect_in_background:
            self._start_driver_connect_async(reason="runtime ensure")
            return False
        return self._connect_driver_sync(reason="runtime ensure")

    def close(self) -> None:
        self._closed = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._connect_thread and self._connect_thread.is_alive():
            self._connect_thread.join(timeout=0.25)
        with self._driver_lock:
            if self._driver:
                try:
                    self._driver.release_left_click()
                except Exception:
                    pass
                self._driver.close()
                self._driver = None

    def poll_hotkeys(self) -> bool:
        if self._key_down_fast(self.cfg.toggle_key):
            if not self._toggle_latch:
                self.enabled = not self.enabled
                self.clear_residual()
                log(f"control toggle: {'ON' if self.enabled else 'OFF'}", "WARN")
                if self.enabled:
                    if self.cfg.driver_connect_in_background:
                        self._start_driver_connect_async(reason="control toggled on")
                    else:
                        self._ensure_driver()
            self._toggle_latch = True
        else:
            self._toggle_latch = False

        if self._key_down_fast(self.cfg.quit_key):
            log("quit hotkey received", "WARN")
            return False
        return True

    def is_active(self) -> bool:
        if not self.enabled:
            return False
        if not self.cfg.only_when_active:
            return True
        return self._key_down_fast(self.cfg.active_key)

    def clear_residual(self) -> None:
        with self._residual_lock:
            self._residual_x = 0.0
            self._residual_y = 0.0
            self._pending_add_x = 0.0
            self._pending_add_y = 0.0
            self._pending_ticks = 0
            self._last_seq_consumed = -1
        self._reset_motion_profile()
        self._reset_pid()

    def _reset_motion_profile(self) -> None:
        with self._residual_lock:
            self._motion_profile_x = 0.0
            self._motion_profile_y = 0.0
            self._motion_round_x = 0.0
            self._motion_round_y = 0.0
            self._last_profile_zero_time = 0.0

    def _has_residual_motion_debt_locked(self) -> bool:
        eps = max(float(getattr(self.cfg, "residual_epsilon", 0.18)), 0.05)
        return (
            abs(self._residual_x) >= eps
            or abs(self._residual_y) >= eps
            or abs(self._pending_add_x) >= eps
            or abs(self._pending_add_y) >= eps
            or self._pending_ticks > 0
        )

    def _reset_settle_lock(self) -> None:
        with self._settled_lock:
            self._settled = False
            self._settle_enter_count = 0
            self._settle_release_count = 0

    def _settle_consume_if_needed(self, seq: int, err_len: float, conf: float, held: bool, fine_dead: float, target_radius: float = 0.0) -> bool:
        if not bool(getattr(self.cfg, "settle_lock_enabled", True)):
            return False
        if held:
            return False
        if not math.isfinite(err_len):
            return False
        if float(conf or 0.0) < float(getattr(self.cfg, "settle_min_conf", 0.30)):
            return False
        enter_px = max(fine_dead, float(getattr(self.cfg, "settle_enter_px", 2.2)))
        exit_px = max(enter_px, float(getattr(self.cfg, "settle_exit_px", 6.0)))
        hard_exit_px = max(exit_px, float(getattr(self.cfg, "settle_hard_exit_px", 18.0)))
        if bool(getattr(self.cfg, "settle_target_radius_enabled", True)) and target_radius > 0.0:
            enter_px = max(enter_px, float(target_radius) * float(getattr(self.cfg, "settle_enter_radius_fraction", 0.10)))
            exit_px = max(exit_px, float(target_radius) * float(getattr(self.cfg, "settle_exit_radius_fraction", 0.55)))
            hard_exit_px = max(hard_exit_px, float(target_radius) * float(getattr(self.cfg, "settle_hard_exit_radius_fraction", 1.15)))
            exit_px = min(exit_px, float(getattr(self.cfg, "settle_exit_max_px", 18.0)))
            hard_exit_px = min(hard_exit_px, float(getattr(self.cfg, "settle_hard_exit_max_px", 34.0)))
            hard_exit_px = max(hard_exit_px, exit_px + 1.0)
        enter_frames = max(1, int(getattr(self.cfg, "settle_enter_frames", 1)))
        release_frames = max(1, int(getattr(self.cfg, "settle_release_frames", 2)))
        with self._settled_lock:
            if self._settled:
                if err_len >= hard_exit_px:
                    self._settled = False
                    self._settle_enter_count = 0
                    self._settle_release_count = 0
                    return False
                if err_len <= exit_px:
                    self._settle_release_count = 0
                    with self._residual_lock:
                        self._residual_x = 0.0
                        self._residual_y = 0.0
                        self._last_seq_consumed = int(seq)
                    return True
                self._settle_release_count += 1
                if self._settle_release_count < release_frames:
                    with self._residual_lock:
                        self._residual_x = 0.0
                        self._residual_y = 0.0
                        self._last_seq_consumed = int(seq)
                    return True
                self._settled = False
                self._settle_enter_count = 0
                self._settle_release_count = 0
                return False
            if err_len <= enter_px:
                self._settle_enter_count += 1
                if self._settle_enter_count >= enter_frames:
                    self._settled = True
                    self._settle_release_count = 0
                    with self._residual_lock:
                        self._residual_x = 0.0
                        self._residual_y = 0.0
                        self._last_seq_consumed = int(seq)
                    return True
            else:
                self._settle_enter_count = 0
                self._settle_release_count = 0
        return False

    def _reset_error_velocity(self) -> None:
        with self._error_lock:
            self._last_error_x = 0.0
            self._last_error_y = 0.0
            self._last_error_time = 0.0
            self._error_vx = 0.0
            self._error_vy = 0.0
        self._reset_pid()

    def _reset_pid(self) -> None:
        self._pid_x.reset()
        self._pid_y.reset()

    def _mark_wait_for_new_target(self, current_seq: int | None = None) -> None:
        """Ignore already-published targets until the vision loop publishes a new one.

        This prevents the classic ghost-move bug: the user presses Shift after a target
        disappeared, but the motor loop still consumes the last valid target snapshot.
        """
        if current_seq is None:
            with self._state_lock:
                current_seq = self._target_seq
        with self._residual_lock:
            self._last_seq_consumed = int(current_seq)

    def submit(
        self,
        target_x: float,
        target_y: float,
        center_x: float,
        center_y: float,
        distance: float = 0.0,
        confidence: float = 0.0,
        valid: bool = True,
        held: bool = False,
        target_radius: float = 0.0,
    ) -> None:
        ex = float(target_x - center_x)
        ey = float(target_y - center_y)
        if self.cfg.invert_y:
            ey = -ey
        max_err = float(getattr(self.cfg, "max_submit_error_px", 0.0) or 0.0)
        err_len = math.hypot(ex, ey)
        if max_err > 0 and err_len > max_err:
            scale = max_err / max(err_len, 1e-6)
            ex *= scale
            ey *= scale
        now = self._now()
        with self._state_lock:
            if self._target_time > 0.0:
                dt = max(1e-3, min(now - self._target_time, 0.25))
                if self._submit_interval_ema <= 0.0:
                    self._submit_interval_ema = dt
                else:
                    self._submit_interval_ema = self._submit_interval_ema * 0.85 + dt * 0.15
            self._last_submit_time = now
            self._target_seq += 1
            self._target_time = now
            self._target_valid = bool(valid and math.isfinite(ex) and math.isfinite(ey))
            self._ex = ex
            self._ey = ey
            self._distance = float(distance or 0.0)
            self._confidence = float(confidence or 0.0)
            self._target_held = bool(held)
            self._target_radius = float(target_radius or 0.0)

    def clear_target(self, *, soft: bool = False, active: bool = False) -> None:
        # Clearing a target must usually stop leftover residual motion immediately.
        # V17.8.20 adds a short soft-clear path for active, already-validated targets:
        # if the detector drops one frame, keep draining the current residual/pending
        # motion instead of resetting the motor to zero. This removes move-pause-move
        # stutter without inventing a new target.
        now = self._now()
        if soft and active and bool(getattr(self.cfg, "no_target_soft_hold_enabled", True)):
            hold_s = max(0.0, float(getattr(self.cfg, "no_target_soft_hold_ms", 65.0))) / 1000.0
            with self._state_lock:
                # Use the last genuine submit time, not the refreshed motor timestamp;
                # otherwise repeated no-target frames could extend the hold forever.
                age = now - self._last_submit_time if self._target_valid and self._last_submit_time > 0.0 else 999.0
                if self._target_valid and age <= hold_s:
                    # Refresh timestamp so the motor loop does not trip stale-target
                    # clearing during a tiny visual dropout. Sequence is unchanged,
                    # so no new residual is added; only existing residual drains.
                    self._target_time = now
                    self._target_held = True
                    return
        with self._state_lock:
            self._target_seq += 1
            self._target_time = now
            self._target_valid = False
            self._ex = 0.0
            self._ey = 0.0
            self._distance = 0.0
            self._confidence = 0.0
            self._target_held = False
            self._target_radius = 0.0
            current_seq = self._target_seq
        if self.cfg.clear_residual_on_no_target:
            self.clear_residual()
            self._reset_error_velocity()
            self._mark_wait_for_new_target(current_seq)

    def apply(self, target_x: float, target_y: float, center_x: float, center_y: float) -> None:
        self.submit(target_x, target_y, center_x, center_y, valid=True)

    def consume_ego_delta(self) -> Tuple[int, int]:
        with self._ego_lock:
            dx = int(round(self._ego_dx))
            dy = int(round(self._ego_dy))
            self._ego_dx -= dx
            self._ego_dy -= dy
            return dx, dy

    def _record_ego_delta(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        with self._ego_lock:
            self._ego_dx += float(dx)
            self._ego_dy += float(dy)

    def _snapshot_target(self) -> Tuple[int, float, bool, float, float, float, float, bool, float, float]:
        with self._state_lock:
            return (
                self._target_seq,
                self._target_time,
                self._target_valid,
                self._ex,
                self._ey,
                self._distance,
                self._confidence,
                self._target_held,
                self._submit_interval_ema,
                self._target_radius,
            )

    def _effective_stale_seconds(self, submit_interval_ema: float) -> float:
        base = max(0.0, float(self.cfg.stale_target_seconds))
        if not bool(getattr(self.cfg, "adaptive_stale_target", True)):
            return base
        min_s = max(base, float(getattr(self.cfg, "stale_target_min_seconds", 0.045)))
        max_s = max(min_s, float(getattr(self.cfg, "stale_target_max_seconds", 0.120)))
        mult = max(1.0, float(getattr(self.cfg, "stale_target_interval_multiplier", 3.0)))
        if submit_interval_ema > 0.0:
            return max(min_s, min(max_s, float(submit_interval_ema) * mult))
        return min_s

    @staticmethod
    def _same_sign_or_zero(a: float, b: float) -> bool:
        if abs(a) < 1e-9 or abs(b) < 1e-9:
            return True
        return (a > 0) == (b > 0)

    @staticmethod
    def _limit_vector(x: float, y: float, limit: float) -> Tuple[float, float]:
        limit = float(limit or 0.0)
        if limit <= 0:
            return x, y
        length = math.hypot(x, y)
        if length > limit:
            scale = limit / max(length, 1e-6)
            return x * scale, y * scale
        return x, y

    def _safe_axis_lead(self, err: float, lead: float, dead: float) -> float:
        """Clamp velocity lead so prediction cannot flip or dominate the real error.

        V17.6 allowed `err + lead` to cross zero. With noisy detection boxes this
        creates a right-left-right feedback loop. The corrected rule is:
        - no lead in the fine center zone;
        - same-direction lead is capped to a fraction of current error;
        - opposite-direction lead may reduce movement, but never reverse it.
        """
        err = float(err)
        lead = float(lead)
        if not (math.isfinite(err) and math.isfinite(lead)):
            return 0.0
        min_err = max(float(self.cfg.velocity_lead_min_error_px), dead * 1.5)
        if abs(err) < min_err:
            return 0.0
        frac = max(0.0, min(1.0, float(self.cfg.velocity_lead_error_fraction)))
        same_dir_cap = abs(err) * frac
        opposite_cap = abs(err) * 0.70
        if self._same_sign_or_zero(err, lead):
            return math.copysign(min(abs(lead), same_dir_cap), lead)
        # Allow lead to brake an overtake, but not to reverse the required move.
        return math.copysign(min(abs(lead), opposite_cap), lead)

    def _add_target_error_once(self, seq: int, ex: float, ey: float, dist: float, conf: float = 0.0, *, held: bool = False, target_radius: float = 0.0, submit_interval_ema: float = 0.0) -> None:
        err_len = math.hypot(ex, ey)
        dead = max(0.0, float(self.cfg.deadzone))
        fine_dead = max(0.0, float(getattr(self.cfg, "fine_deadzone", dead)))

        if self._settle_consume_if_needed(seq, err_len, conf, held, fine_dead, target_radius):
            return

        now = self._now()
        with self._error_lock:
            velocity_valid = False
            if self._last_error_time > 0.0:
                dt = max(1e-3, min(now - self._last_error_time, 0.20))
                jump = math.hypot(ex - self._last_error_x, ey - self._last_error_y)
                raw_vx = (ex - self._last_error_x) / dt
                raw_vy = (ey - self._last_error_y) / dt
                raw_speed = math.hypot(raw_vx, raw_vy)
                # Detector jitter, target swaps and one-frame lock predictions can create
                # impossible velocities. Those must reset the lead estimator instead of
                # feeding it, otherwise the controller turns measurement noise into shake.
                if (jump <= float(self.cfg.velocity_reset_jump_px) and
                        raw_speed <= float(self.cfg.max_error_velocity_px_s)):
                    a = max(0.0, min(1.0, float(self.cfg.error_velocity_smoothing)))
                    self._error_vx = self._error_vx * a + raw_vx * (1.0 - a)
                    self._error_vy = self._error_vy * a + raw_vy * (1.0 - a)
                    velocity_valid = True
                else:
                    self._error_vx = 0.0
                    self._error_vy = 0.0
            else:
                self._error_vx = 0.0
                self._error_vy = 0.0
            self._last_error_x = ex
            self._last_error_y = ey
            self._last_error_time = now
            evx = self._error_vx if velocity_valid else 0.0
            evy = self._error_vy if velocity_valid else 0.0

        if held and bool(getattr(self.cfg, "held_target_disable_lead", True)):
            evx = 0.0
            evy = 0.0
        lead_s = max(0.0, float(getattr(self.cfg, "velocity_lead_ms", 0.0))) / 1000.0
        lead_x = evx * lead_s
        lead_y = evy * lead_s
        lead_max = max(0.0, float(getattr(self.cfg, "velocity_lead_max_px", 0.0)))
        lead_x, lead_y = self._limit_vector(lead_x, lead_y, lead_max)
        lead_x = self._safe_axis_lead(ex, lead_x, dead)
        lead_y = self._safe_axis_lead(ey, lead_y, dead)

        move_ex = ex + lead_x
        move_ey = ey + lead_y
        # Absolute safety: predicted error may reduce the original error, but it must
        # not invert the movement direction on either axis.
        if abs(ex) > dead and not self._same_sign_or_zero(ex, move_ex):
            move_ex = math.copysign(max(abs(ex) * 0.25, dead), ex)
        if abs(ey) > dead and not self._same_sign_or_zero(ey, move_ey):
            move_ey = math.copysign(max(abs(ey) * 0.25, dead), ey)

        effective_sens = float(self.cfg.sensitivity_scaler)
        if dist > 0 and dist < float(self.cfg.close_range_threshold):
            effective_sens *= float(self.cfg.sensitivity_boost_close)
        if held:
            effective_sens *= max(0.0, min(1.0, float(getattr(self.cfg, "held_target_sensitivity_scale", 0.35))))

        if bool(getattr(self.cfg, "pid_enabled", True)):
            pid_ex = self._pid_x.update(move_ex, now, self.cfg, fine_dead=fine_dead)
            pid_ey = self._pid_y.update(move_ey, now, self.cfg, fine_dead=fine_dead)
        else:
            pid_ex = move_ex
            pid_ey = move_ey

        total_x = pid_ex * effective_sens * float(self.cfg.gain_x)
        total_y = pid_ey * effective_sens * float(self.cfg.gain_y)

        damp_px = max(0.0, float(getattr(self.cfg, "near_center_damping_px", 0.0) or 0.0))
        if damp_px > 0.0 and err_len < damp_px:
            base = max(0.0, min(1.0, float(getattr(self.cfg, "near_center_damping_scale", 0.45))))
            # Smoothly ramp from base damping near the deadzone to full response at damp_px.
            ramp = max(0.0, min(1.0, (err_len - dead) / max(damp_px - dead, 1e-6)))
            damp = base + (1.0 - base) * ramp
            total_x *= damp
            total_y *= damp

        max_add = float(getattr(self.cfg, "max_residual_add_per_frame", 0.0) or 0.0)
        total_len = math.hypot(total_x, total_y)
        if max_add > 0 and total_len > max_add:
            scale = max_add / max(total_len, 1e-6)
            total_x *= scale
            total_y *= scale

        min_speed = max(0.0, float(self.cfg.min_kinetic_speed))
        if abs(ex) > dead * 1.5 and 0.0 < abs(total_x) < min_speed:
            total_x = math.copysign(min_speed, total_x)
        if abs(ey) > dead * 1.5 and 0.0 < abs(total_y) < min_speed:
            total_y = math.copysign(min_speed, total_y)

        if abs(ex) <= fine_dead:
            total_x = 0.0
        if abs(ey) <= fine_dead:
            total_y = 0.0

        if bool(getattr(self.cfg, "overshoot_guard_enabled", True)):
            frac = max(0.05, min(1.0, float(getattr(self.cfg, "overshoot_error_fraction", 0.42))))
            if abs(ex) > fine_dead:
                cap_x = abs(ex) * frac
                if not self._same_sign_or_zero(total_x, ex):
                    total_x = 0.0
                elif abs(total_x) > cap_x:
                    total_x = math.copysign(cap_x, total_x)
            else:
                total_x = 0.0
            if abs(ey) > fine_dead:
                cap_y = abs(ey) * frac
                if not self._same_sign_or_zero(total_y, ey):
                    total_y = 0.0
                elif abs(total_y) > cap_y:
                    total_y = math.copysign(cap_y, total_y)
            else:
                total_y = 0.0

        if self.cfg.suppress_reverse_inside_deadzone:
            if abs(ex) <= dead * 2.0 and abs(total_x) <= max(min_speed, self.cfg.residual_epsilon):
                total_x = 0.0
            if abs(ey) <= dead * 2.0 and abs(total_y) <= max(min_speed, self.cfg.residual_epsilon):
                total_y = 0.0

        with self._residual_lock:
            if seq == self._last_seq_consumed:
                return
            self._last_seq_consumed = seq

            # V17.8.2 tested hotfix: do not stop at the coarse deadzone.
            # The previous condition used `err_len <= dead`, so with deadzone≈2.6
            # the controller could intentionally settle 1.5-2.6 px to the left/right
            # of the current head center.  That matches the observed side-offset
            # landing.  Use the fine deadzone as the true final-stop threshold; the
            # coarse deadzone remains useful for velocity/lead damping only.
            if err_len <= fine_dead:
                self._residual_x = 0.0
                self._residual_y = 0.0
                self._pending_add_x = 0.0
                self._pending_add_y = 0.0
                self._pending_ticks = 0
                return

            if self.cfg.reset_residual_on_direction_change:
                # v17.6 keeps the direction-reset behavior, but the velocity lead means
                # a real fast target can reverse without accumulating old residual.
                if (self._residual_x > 0 and total_x < 0) or (self._residual_x < 0 and total_x > 0):
                    self._residual_x = 0.0
                    self._pending_add_x = 0.0
                if (self._residual_y > 0 and total_y < 0) or (self._residual_y < 0 and total_y > 0):
                    self._residual_y = 0.0
                    self._pending_add_y = 0.0

            if total_x == 0.0:
                self._residual_x = 0.0
                self._pending_add_x = 0.0
            if total_y == 0.0:
                self._residual_y = 0.0
                self._pending_add_y = 0.0

            if bool(getattr(self.cfg, "smooth_residual_injection", True)):
                hz = max(60, int(getattr(self.cfg, "control_loop_hz", 1000)))
                if submit_interval_ema and submit_interval_ema > 0:
                    base_ticks = int(float(submit_interval_ema) * hz * float(getattr(self.cfg, "residual_injection_interval_fraction", 0.85)))
                else:
                    # Before EMA exists, avoid a one-frame lump. 120 FPS inference corresponds to about 8 ticks.
                    base_ticks = int(hz / 120.0)
                min_ticks = max(1, int(getattr(self.cfg, "residual_injection_min_ticks", 4)))
                max_ticks = max(min_ticks, int(getattr(self.cfg, "residual_injection_max_ticks", 28)))
                ticks = max(min_ticks, min(max_ticks, base_ticks))
                if ticks <= 1:
                    self._residual_x += total_x
                    self._residual_y += total_y
                    self._pending_add_x = 0.0
                    self._pending_add_y = 0.0
                    self._pending_ticks = 0
                else:
                    # Replace stale pending correction with the newest detector frame.
                    self._residual_x += total_x / ticks
                    self._residual_y += total_y / ticks
                    self._pending_add_x = total_x * (ticks - 1) / ticks
                    self._pending_add_y = total_y * (ticks - 1) / ticks
                    self._pending_ticks = ticks - 1
            else:
                self._residual_x += total_x
                self._residual_y += total_y
                self._pending_add_x = 0.0
                self._pending_add_y = 0.0
                self._pending_ticks = 0
            max_total = float(getattr(self.cfg, "max_residual_total", 0.0) or 0.0)
            if max_total > 0:
                self._residual_x, self._residual_y = self._limit_vector(self._residual_x, self._residual_y, max_total)

            if bool(getattr(self.cfg, "overshoot_guard_enabled", True)):
                rfrac = max(0.05, min(1.0, float(getattr(self.cfg, "residual_error_fraction", 0.58))))
                if abs(ex) <= fine_dead or not self._same_sign_or_zero(self._residual_x, ex):
                    self._residual_x = 0.0
                    self._pending_add_x = 0.0
                else:
                    rcap_x = abs(ex) * rfrac
                    if abs(self._residual_x) > rcap_x:
                        self._residual_x = math.copysign(rcap_x, self._residual_x)
                if abs(ey) <= fine_dead or not self._same_sign_or_zero(self._residual_y, ey):
                    self._residual_y = 0.0
                    self._pending_add_y = 0.0
                else:
                    rcap_y = abs(ey) * rfrac
                    if abs(self._residual_y) > rcap_y:
                        self._residual_y = math.copysign(rcap_y, self._residual_y)

    def _axis_allows_motion(self, residual: float, err: float, fine_dead: float) -> bool:
        """Return whether residual is still allowed to move toward target.

        This is the final overshoot brake. It prevents stale residual from being
        drained after the current error has crossed zero or fallen into the fine
        deadzone. Without this, an old residual can carry the camera past the
        target and then force a visible correction back.
        """
        if abs(residual) < float(self.cfg.residual_epsilon):
            return False
        if bool(getattr(self.cfg, "overshoot_guard_enabled", True)):
            if abs(err) <= fine_dead:
                return False
            if not self._same_sign_or_zero(residual, err):
                return False
        return True

    def _axis_step_from_residual(self, residual: float, err: float, limit: int, fine_dead: float) -> int:
        if not self._axis_allows_motion(residual, err, fine_dead):
            return 0
        threshold = max(float(self.cfg.residual_epsilon), float(getattr(self.cfg, "micro_step_threshold", 0.95)))
        min_error_for_micro = max(fine_dead * 2.0, float(getattr(self.cfg, "micro_step_min_error_px", 8.0)))
        if abs(residual) < 1.0:
            if abs(err) >= min_error_for_micro and abs(residual) >= threshold:
                step = int(math.copysign(1, residual))
            else:
                return 0
        else:
            step = int(residual)
        if bool(getattr(self.cfg, "overshoot_guard_enabled", True)):
            frac = max(0.05, min(1.0, float(getattr(self.cfg, "drain_error_fraction", 0.45))))
            axis_cap = max(1, int(abs(err) * frac))
            step = max(-axis_cap, min(axis_cap, step))
        return max(-limit, min(limit, step))

    def _inject_pending_residual_tick(self) -> None:
        if not bool(getattr(self.cfg, "smooth_residual_injection", True)):
            return
        with self._residual_lock:
            if self._pending_ticks <= 0:
                return
            dx = self._pending_add_x / max(1, self._pending_ticks)
            dy = self._pending_add_y / max(1, self._pending_ticks)
            self._residual_x += dx
            self._residual_y += dy
            self._pending_add_x -= dx
            self._pending_add_y -= dy
            self._pending_ticks -= 1
            max_total = float(getattr(self.cfg, "max_residual_total", 0.0) or 0.0)
            if max_total > 0:
                self._residual_x, self._residual_y = self._limit_vector(self._residual_x, self._residual_y, max_total)

    def _drain_residual(self, ex: float, ey: float, *, held: bool = False) -> Tuple[int, int]:
        limit = max(1, min(127, int(self.cfg.max_move or self.cfg.max_step)))
        fine_dead = max(0.0, float(getattr(self.cfg, "fine_deadzone", self.cfg.deadzone)))
        self._inject_pending_residual_tick()

        with self._residual_lock:
            if bool(getattr(self.cfg, "overshoot_guard_enabled", True)):
                if not self._axis_allows_motion(self._residual_x, ex, fine_dead):
                    self._residual_x = 0.0
                    self._pending_add_x = 0.0
                if not self._axis_allows_motion(self._residual_y, ey, fine_dead):
                    self._residual_y = 0.0
                    self._pending_add_y = 0.0
                if self._pending_add_x == 0.0 and self._pending_add_y == 0.0:
                    self._pending_ticks = 0
            mx = self._axis_step_from_residual(self._residual_x, ex, limit, fine_dead)
            my = self._axis_step_from_residual(self._residual_y, ey, limit, fine_dead)
            if mx == 0 and abs(self._residual_x) < self.cfg.residual_epsilon:
                self._residual_x = 0.0
            if my == 0 and abs(self._residual_y) < self.cfg.residual_epsilon:
                self._residual_y = 0.0
            return mx, my

    def _shape_motion_output(self, mx: int, my: int, ex: float, ey: float, *, held: bool = False) -> Tuple[int, int]:
        """Apply packet-level smoothing to the final HID output.

        This layer is deliberately after the residual / overshoot guard. It does
        not create extra movement. It only reduces abrupt packet-to-packet
        acceleration and prevents near-center opposite-direction twitches. The
        residual is committed using the actual sent packet, so unsent motion stays
        available for later ticks instead of being lost.
        """
        if not bool(getattr(self.cfg, "natural_motion_enabled", True)):
            return mx, my
        raw_x = float(mx)
        raw_y = float(my)
        err_len = math.hypot(float(ex), float(ey))
        if raw_x == 0.0 and raw_y == 0.0:
            if bool(getattr(self.cfg, "continuous_motion_profile_hold", True)):
                hold_ms = max(0.0, float(getattr(self.cfg, "continuous_motion_profile_hold_ms", 45.0)))
                decay = max(0.0, min(1.0, float(getattr(self.cfg, "continuous_motion_profile_decay", 0.82))))
                now = self._now()
                with self._residual_lock:
                    has_debt = self._has_residual_motion_debt_locked()
                    recently_zero = (self._last_profile_zero_time > 0.0 and (now - self._last_profile_zero_time) * 1000.0 <= hold_ms)
                    if has_debt or recently_zero:
                        # Do not hard-reset acceleration state between two valid residual
                        # packets. Decay it so the next non-zero packet resumes smoothly.
                        self._motion_profile_x *= decay
                        self._motion_profile_y *= decay
                        self._motion_round_x *= decay
                        self._motion_round_y *= decay
                        self._last_profile_zero_time = now
                        return 0, 0
            self._reset_motion_profile()
            return 0, 0

        close_px = max(1.0, float(getattr(self.cfg, "natural_motion_close_px", 34.0)))
        close_delta = max(0.25, float(getattr(self.cfg, "natural_motion_close_delta", 0.85)))
        far_delta = max(close_delta, float(getattr(self.cfg, "natural_motion_max_delta", 2.4)))
        ramp = max(0.0, min(1.0, (err_len - close_px * 0.35) / max(close_px * 0.65, 1e-6)))
        delta_cap = close_delta + (far_delta - close_delta) * ramp
        if held:
            delta_cap = min(delta_cap, close_delta)

        alpha = max(0.05, min(1.0, float(getattr(self.cfg, "natural_motion_alpha", 0.46))))
        if held:
            alpha *= 0.65

        with self._residual_lock:
            px, py = self._motion_profile_x, self._motion_profile_y

            def shape_axis(raw: float, prev: float, err: float) -> float:
                if bool(getattr(self.cfg, "natural_motion_zero_cross_brake", True)):
                    if raw != 0.0 and prev != 0.0 and not self._same_sign_or_zero(raw, prev):
                        if abs(err) <= max(float(getattr(self.cfg, "fine_deadzone", self.cfg.deadzone)) * 2.0, close_px):
                            return 0.0
                desired = prev + (raw - prev) * alpha
                change = desired - prev
                if change > delta_cap:
                    desired = prev + delta_cap
                elif change < -delta_cap:
                    desired = prev - delta_cap
                return desired

            sx = shape_axis(raw_x, px, float(ex))
            sy = shape_axis(raw_y, py, float(ey))

            # Final convergence guard: never emit a packet that moves away from
            # the current target on either axis.
            if sx != 0.0 and not self._same_sign_or_zero(sx, ex):
                sx = 0.0
            if sy != 0.0 and not self._same_sign_or_zero(sy, ey):
                sy = 0.0

            self._motion_profile_x = sx
            self._motion_profile_y = sy

            ox_f = sx + self._motion_round_x
            oy_f = sy + self._motion_round_y
            ox = int(round(ox_f))
            oy = int(round(oy_f))

            if ox != 0 and not self._same_sign_or_zero(ox, ex):
                ox = 0
                self._motion_round_x = 0.0
            else:
                self._motion_round_x = ox_f - ox
            if oy != 0 and not self._same_sign_or_zero(oy, ey):
                oy = 0
                self._motion_round_y = 0.0
            else:
                self._motion_round_y = oy_f - oy

            limit = max(1, min(127, int(self.cfg.max_move or self.cfg.max_step)))
            ox = max(-limit, min(limit, ox))
            oy = max(-limit, min(limit, oy))
            if ox == 0 and oy == 0 and abs(raw_x) + abs(raw_y) > 0:
                min_micro_err = max(float(getattr(self.cfg, "micro_step_min_error_px", 8.0)), float(getattr(self.cfg, "deadzone", 0.0)) * 2.0)
                if err_len >= min_micro_err:
                    if abs(raw_x) >= abs(raw_y) and raw_x != 0.0 and self._same_sign_or_zero(raw_x, ex):
                        ox = int(math.copysign(1, raw_x))
                    elif raw_y != 0.0 and self._same_sign_or_zero(raw_y, ey):
                        oy = int(math.copysign(1, raw_y))
            if ox != 0 or oy != 0:
                self._last_profile_zero_time = 0.0
            return ox, oy

    def _commit_sent_move(self, mx: int, my: int) -> None:
        with self._residual_lock:
            self._residual_x -= mx
            self._residual_y -= my

    def _reset_fire_state(self, reason: str = "reset") -> None:
        with self._fire_lock:
            self._fire_candidate_seq = -1
            self._fire_stable_frames = 0
            self._fire_armed = True
            self._fire_last_reason = reason
            self._fire_candidate_ex = 0.0
            self._fire_candidate_ey = 0.0

    def _fire_motion_debt_px(self) -> float:
        with self._residual_lock:
            return math.hypot(
                float(self._residual_x) + float(self._pending_add_x),
                float(self._residual_y) + float(self._pending_add_y),
            )

    def _fire_settle_releasing(self) -> bool:
        with self._settled_lock:
            return bool(self._settled and self._settle_release_count > 0)

    def _set_fire_reason(self, reason: str) -> None:
        with self._fire_lock:
            self._fire_last_reason = str(reason)

    def _fire_gate_ready(
        self,
        *,
        seq: int,
        now: float,
        ts: float,
        ex: float,
        ey: float,
        conf: float,
        held: bool,
        motion_zero: bool = True,
    ) -> tuple[bool, str]:
        if not bool(getattr(self.cfg, "fire_enabled", False)):
            self._set_fire_reason("disabled")
            return False, "disabled"

        if bool(getattr(self.cfg, "fire_require_zero_motion", True)) and not bool(motion_zero):
            self._set_fire_reason("motion still active")
            return False, "motion still active"

        err = math.hypot(float(ex), float(ey))
        if not math.isfinite(err):
            self._reset_fire_state("invalid error")
            return False, "invalid error"

        fire_radius = max(0.0, float(getattr(self.cfg, "fire_radius", 4.0)))
        exit_radius = max(fire_radius, float(getattr(self.cfg, "fire_exit_radius", fire_radius + 3.0)))
        rearm_radius = max(exit_radius, float(getattr(self.cfg, "fire_rearm_radius", exit_radius + 2.0)))
        if err > exit_radius:
            with self._fire_lock:
                self._fire_candidate_seq = -1
                self._fire_stable_frames = 0
                self._fire_candidate_ex = 0.0
                self._fire_candidate_ey = 0.0
                if err >= rearm_radius:
                    self._fire_armed = True
                self._fire_last_reason = "outside radius"
            return False, "outside radius"

        if err > fire_radius:
            self._set_fire_reason("inside exit hysteresis only")
            return False, "inside exit hysteresis only"

        max_age_ms = max(1.0, float(getattr(self.cfg, "fire_max_target_age_ms", 90.0)))
        target_age_ms = (now - float(ts)) * 1000.0 if ts > 0.0 else 999999.0
        if target_age_ms > max_age_ms:
            self._set_fire_reason("target too stale")
            return False, "target too stale"

        if bool(held):
            if not bool(getattr(self.cfg, "fire_allow_held_target", False)):
                self._set_fire_reason("held target blocked")
                return False, "held target blocked"
            held_min_conf = float(getattr(self.cfg, "fire_held_target_min_conf", max(0.72, float(getattr(self.cfg, "fire_min_conf", 0.50)))))
            held_max_age_ms = max(1.0, float(getattr(self.cfg, "fire_held_target_max_age_ms", min(45.0, max_age_ms))))
            if float(conf) < held_min_conf:
                self._set_fire_reason("held confidence below fire_held_target_min_conf")
                return False, "held confidence below fire_held_target_min_conf"
            if target_age_ms > held_max_age_ms:
                self._set_fire_reason("held target too stale")
                return False, "held target too stale"

        if float(conf) < float(getattr(self.cfg, "fire_min_conf", 0.50)):
            self._set_fire_reason("confidence below fire_min_conf")
            return False, "confidence below fire_min_conf"

        max_debt = max(0.0, float(getattr(self.cfg, "fire_max_motion_debt_px", 0.90)))
        if self._fire_motion_debt_px() > max_debt:
            self._set_fire_reason("motion debt not drained")
            return False, "motion debt not drained"

        min_after_move_ms = max(0.0, float(getattr(self.cfg, "fire_min_time_after_move_ms", 22.0)))
        if min_after_move_ms > 0.0 and self._fire_last_move_time > 0.0:
            if (now - self._fire_last_move_time) * 1000.0 < min_after_move_ms:
                self._set_fire_reason("too soon after movement")
                return False, "too soon after movement"

        if bool(getattr(self.cfg, "fire_block_during_settle_release", True)) and self._fire_settle_releasing():
            self._set_fire_reason("settle release in progress")
            return False, "settle release in progress"

        required_frames = max(1, int(getattr(self.cfg, "fire_stable_frames", 2)))
        stable_delta = max(0.0, float(getattr(self.cfg, "fire_stable_error_delta_px", 2.8)))
        with self._fire_lock:
            if seq != self._fire_candidate_seq:
                if self._fire_candidate_seq < 0:
                    self._fire_stable_frames = 1
                else:
                    delta = math.hypot(float(ex) - self._fire_candidate_ex, float(ey) - self._fire_candidate_ey)
                    if delta <= stable_delta:
                        self._fire_stable_frames += 1
                    else:
                        self._fire_stable_frames = 1
                self._fire_candidate_seq = int(seq)
                self._fire_candidate_ex = float(ex)
                self._fire_candidate_ey = float(ey)
            if self._fire_stable_frames < required_frames:
                self._fire_last_reason = "waiting stable frames"
                return False, "waiting stable frames"

            cooldown_s = max(0.0, float(getattr(self.cfg, "fire_cooldown_ms", 165.0))) / 1000.0
            if cooldown_s > 0.0 and now - self._fire_last_click_time < cooldown_s:
                self._fire_last_reason = "cooldown"
                return False, "cooldown"

            repeat = bool(getattr(self.cfg, "fire_repeat_while_in_radius", False))
            if not self._fire_armed and not repeat:
                self._fire_last_reason = "one-shot disarmed"
                return False, "one-shot disarmed"

            if repeat and bool(getattr(self.cfg, "fire_repeat_requires_fresh_detection", True)):
                min_seq_delta = max(1, int(getattr(self.cfg, "fire_min_repeat_seq_delta", 1)))
                if self._fire_last_click_seq >= 0 and int(seq) - int(self._fire_last_click_seq) < min_seq_delta:
                    self._fire_last_reason = "waiting fresh detection"
                    return False, "waiting fresh detection"

            self._fire_last_reason = "ready"
            return True, "ready"

    def _try_fire(
        self,
        *,
        seq: int,
        ts: float,
        ex: float,
        ey: float,
        conf: float,
        held: bool,
        motion_zero: bool = True,
    ) -> None:
        now = self._now()
        ready, reason = self._fire_gate_ready(seq=seq, now=now, ts=ts, ex=ex, ey=ey, conf=conf, held=held, motion_zero=motion_zero)
        if not ready:
            return
        if not self._ensure_driver():
            return
        with self._fire_lock:
            # Re-check cooldown after any driver wait/reconnect path.
            now = self._now()
            cooldown_s = max(0.0, float(getattr(self.cfg, "fire_cooldown_ms", 165.0))) / 1000.0
            if cooldown_s > 0.0 and now - self._fire_last_click_time < cooldown_s:
                return
            if self._driver and self._driver.click_left():
                self._fire_last_click_time = now
                self._fire_last_click_seq = int(seq)
                self._fire_last_click_error = math.hypot(float(ex), float(ey))
                self._fire_armed = False
                if bool(getattr(self.cfg, "fire_log_events", False)):
                    log(f"FIRE click: seq={seq}, err={self._fire_last_click_error:.2f}, conf={float(conf):.3f}, held={held}, repeat={bool(getattr(self.cfg, 'fire_repeat_while_in_radius', False))}", "INFO")

    def _motor_loop(self) -> None:
        hz = max(60, int(self.cfg.control_loop_hz))
        interval = 1.0 / float(hz)
        next_t = self._now()
        while not self._closed:
            next_t += interval
            sleep_for = next_t - self._now()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = self._now()

            active = self.is_active()
            if not self.enabled or not active:
                self._active_was_down = False
                self.clear_residual()
                self._reset_settle_lock()
                if bool(getattr(self.cfg, "fire_reset_on_active_release", True)):
                    self._reset_fire_state("inactive")
                continue

            seq, ts, valid, ex, ey, dist, conf, held, submit_interval_ema, target_radius = self._snapshot_target()

            # On the rising edge of the active key, discard any target that was
            # published before the key press. Movement must be based on a fresh frame.
            if active and not self._active_was_down:
                self._active_was_down = True
                if self.cfg.clear_residual_on_active_press:
                    self.clear_residual()
                    self._reset_settle_lock()
                    self._reset_error_velocity()
                    recent_ms = max(0.0, float(getattr(self.cfg, "active_press_accept_recent_ms", 22.0)))
                    target_age = self._now() - ts if valid and ts > 0.0 else 999.0
                    if not (valid and target_age <= recent_ms / 1000.0):
                        self._mark_wait_for_new_target(seq)
                        continue

            if (not valid) or (self._now() - ts > self._effective_stale_seconds(submit_interval_ema)):
                self.clear_residual()
                self._reset_error_velocity()
                self._reset_fire_state("no valid fresh target")
                self._mark_wait_for_new_target(seq)
                continue

            self._add_target_error_once(seq, ex, ey, dist, conf, held=held, target_radius=target_radius, submit_interval_ema=submit_interval_ema)
            mx, my = self._drain_residual(ex, ey, held=held)
            mx, my = self._shape_motion_output(mx, my, ex, ey, held=held)
            if mx == 0 and my == 0:
                self._try_fire(seq=seq, ts=ts, ex=ex, ey=ey, conf=conf, held=held)
                continue

            if not self._ensure_driver():
                now = self._now()
                if now - self._last_warn > float(self.cfg.log_interval_seconds):
                    log("Leonardo is not ready; movement suppressed", "WARN")
                    self._last_warn = now
                self.clear_residual()
                continue

            if self._driver and self._driver.move(mx, my):
                self._commit_sent_move(mx, my)
                self._record_ego_delta(mx, my)
                self._fire_last_move_time = self._now()
            elif self._driver:
                # Do not let stale residual build up behind a failed serial write.
                self.clear_residual()
