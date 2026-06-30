from dataclasses import dataclass, field
from typing import List
import yaml
from log_utils import log


@dataclass
class Config:
    run_mode: str = "runtime"
    benchmark_frames: int = 0
    control_enabled: bool = True
    roi_width: int = 400
    roi_height: int = 400
    target_fps: int = 144
    leonardo_port: str = "auto"
    leonardo_baud: int = 115200
    color_lower: List[int] = field(default_factory=lambda: [138, 125, 105])
    color_upper: List[int] = field(default_factory=lambda: [162, 255, 255])
    min_contour_area: float = 8
    max_contour_area: float = 10000
    morph_kernel_width: int = 9
    morph_kernel_height: int = 17
    morph_iterations: int = 1
    filter_solidity_max: float = 1.0
    filter_solidity_min: float = 0.0
    filter_extent_max: float = 1.0
    filter_aspect_min: float = 0.1
    filter_circularity_max: float = 1.0
    filter_rectangularity_min: float = 0.0
    filter_max_width: int = 600
    filter_max_height: int = 600
    filter_min_height: int = 2
    close_target_rescue_enabled: bool = True
    close_target_center_zone_radius: int = 80
    close_target_max_area_ratio: float = 0.35
    max_head_estimation_candidates: int = 8
    anti_smoke_max_area_ratio: float = 20.0
    anti_smoke_min_area: float = 50000
    edge_margin: int = 2
    temporal_min_hits: int = 1
    temporal_max_lost: int = 10
    spatial_distance_limit: float = 500
    direction_change_guard_enabled: bool = True
    direction_change_threshold: float = -0.2
    direction_change_min_speed: float = 30.0
    direction_change_bypass_dist: float = 120.0
    area_priority_ratio: float = 1.8
    dead_zone: float = 0.5
    max_move: int = 127
    sensitivity_scaler: float = 0.75
    sensitivity_boost_close: float = 1.3
    close_range_threshold: float = 100
    min_kinetic_speed: float = 1.0
    ego_scaler: float = 2.7
    # OpenCV 头部/上部目标点估计
    head_search_ratio: float = 0.38
    head_search_ratio_wide: float = 0.48
    head_min_search_px: int = 8
    head_row_min_pixels: int = 2
    head_row_min_density: float = 0.08
    head_row_gap_tolerance: int = 2
    head_row_weight_power: float = 1.35
    head_vertical_decay: float = 0.018
    head_y_band_position: float = 0.42
    head_offset_y_ratio: float = -0.06
    head_offset_y_min: float = -2.0
    head_offset_y_max: float = -10.0
    head_wide_width_ratio_threshold: float = 0.75
    head_wide_offset_scale: float = 0.65
    show_head_roi: bool = True
    show_aim_point: bool = True
    targeting_offset_x: int = 0
    targeting_offset_y: int = 0
    aim_key: str = "0xA0"
    trigger_enabled: bool = True
    trigger_tolerance: float = 3.0
    trigger_delay_first_shot: float = 0.040
    trigger_burst_interval: float = 0.088
    trigger_max_velocity_px_per_frame: float = 10.0
    ignore_falling_speed_px_per_frame: float = 14.0
    burst_shots_limit: int = 2
    burst_cooldown: float = 0.35
    rcs_pull_down_pixels: int = 6
    kalman_process_noise: float = 0.06
    kalman_measurement_noise: float = 0.12
    show_debug: bool = True
    show_target_info: bool = True
    show_prediction_vector: bool = True
    debug_show_mask: bool = False
    debug_draw_stride: int = 2
    debug_max_draw_candidates: int = 12
    prediction_vector_dt: float = 0.04
    motion_enabled: bool = False
    motion_diff_threshold: int = 10
    motion_dilate_kernel: int = 5
    static_filter_enabled: bool = False
    static_max_frames: int = 20
    static_pos_threshold: float = 3.0
    static_area_change_ratio: float = 0.08
    dynamic_offset_enabled: bool = True
    dynamic_offset_near_y: float = -8
    dynamic_offset_far_y: float = -1
    dynamic_offset_near_dist: float = 50
    dynamic_offset_far_dist: float = 300

    # 神经网络相关
    model_path: str = "fire_model.onnx"
    fire_threshold: float = 0.7
    learning_enabled: bool = True
    img_size: int = 48
    model_input_channels: int = 4  # 1=旧mask模型；3=RGB；4=RGB+mask新版模型

    # 模型过滤配置
    model_filter_threshold: float = 0.5
    model_filter_consecutive: int = 2
    model_filter_reject_cooldown: float = 0.5
    model_filter_cache_ttl: float = 0.05
    model_filter_reject_radius: float = 40.0
    model_inference_in_main: bool = True

    # 训练优化配置
    focal_loss_gamma: float = 2.0
    augment_training_data: bool = True
    quantize_onnx: bool = True
    auto_save_hard_negatives: bool = True

    # 自动训练
    auto_train_enabled: bool = False
    train_sample_threshold: int = 50
    train_check_interval: int = 30
    max_samples_per_class: int = 5000
    train_epochs: int = 10
    train_batch_size: int = 32

    def __post_init__(self):
        self._normalize()

    def _normalize(self):
        """只做安全归一化，不删除配置项。"""
        self.roi_width = max(64, int(self.roi_width))
        self.roi_height = max(64, int(self.roi_height))
        self.target_fps = max(1, int(self.target_fps))
        self.leonardo_baud = max(1, int(self.leonardo_baud))
        self.run_mode = str(self.run_mode or "runtime").lower()
        if self.run_mode not in ("debug", "benchmark", "runtime"):
            log(f"run_mode={self.run_mode} 不支持，已改为 runtime", "WARN")
            self.run_mode = "runtime"
        self.benchmark_frames = max(0, int(self.benchmark_frames))

        self.color_lower = [max(0, min(255, int(x))) for x in list(self.color_lower)[:3]]
        self.color_upper = [max(0, min(255, int(x))) for x in list(self.color_upper)[:3]]
        while len(self.color_lower) < 3:
            self.color_lower.append(0)
        while len(self.color_upper) < 3:
            self.color_upper.append(255)

        self.min_contour_area = max(0.0, float(self.min_contour_area))
        self.max_contour_area = max(self.min_contour_area, float(self.max_contour_area))
        self.filter_max_width = max(1, int(self.filter_max_width))
        self.filter_max_height = max(1, int(self.filter_max_height))
        self.filter_min_height = max(1, int(self.filter_min_height))
        self.close_target_center_zone_radius = max(0, int(self.close_target_center_zone_radius))
        self.close_target_max_area_ratio = max(0.0, min(1.0, float(self.close_target_max_area_ratio)))
        self.max_head_estimation_candidates = max(1, int(self.max_head_estimation_candidates))
        self.morph_kernel_width = max(1, int(self.morph_kernel_width))
        self.morph_kernel_height = max(1, int(self.morph_kernel_height))
        self.morph_iterations = max(0, int(self.morph_iterations))
        self.filter_solidity_min = max(0.0, min(1.0, float(self.filter_solidity_min)))
        self.filter_solidity_max = max(self.filter_solidity_min, min(1.0, float(self.filter_solidity_max)))
        self.filter_extent_max = max(0.0, min(1.0, float(self.filter_extent_max)))
        self.filter_rectangularity_min = max(0.0, min(1.0, float(self.filter_rectangularity_min)))

        self.direction_change_threshold = max(-1.0, min(1.0, float(self.direction_change_threshold)))
        self.direction_change_min_speed = max(0.0, float(self.direction_change_min_speed))
        self.direction_change_bypass_dist = max(0.0, float(self.direction_change_bypass_dist))

        self.model_filter_threshold = max(0.0, min(1.0, float(self.model_filter_threshold)))
        self.fire_threshold = max(self.model_filter_threshold, min(1.0, float(self.fire_threshold)))
        self.model_filter_consecutive = max(1, int(self.model_filter_consecutive))
        self.model_filter_cache_ttl = max(0.0, float(self.model_filter_cache_ttl))
        self.model_filter_reject_cooldown = max(0.0, float(self.model_filter_reject_cooldown))
        self.model_filter_reject_radius = max(1.0, float(self.model_filter_reject_radius))

        self.max_move = max(1, min(127, int(self.max_move)))
        self.dead_zone = max(0.0, float(self.dead_zone))
        self.min_kinetic_speed = max(0.0, float(self.min_kinetic_speed))
        self.debug_draw_stride = max(1, int(self.debug_draw_stride))
        self.debug_max_draw_candidates = max(0, int(self.debug_max_draw_candidates))

        self.head_search_ratio = max(0.05, min(0.90, float(self.head_search_ratio)))
        self.head_search_ratio_wide = max(self.head_search_ratio, min(0.95, float(self.head_search_ratio_wide)))
        self.head_min_search_px = max(1, int(self.head_min_search_px))
        self.head_row_min_pixels = max(1, int(self.head_row_min_pixels))
        self.head_row_min_density = max(0.0, min(0.8, float(self.head_row_min_density)))
        self.head_row_gap_tolerance = max(0, int(self.head_row_gap_tolerance))
        self.head_row_weight_power = max(0.1, float(self.head_row_weight_power))
        self.head_vertical_decay = max(0.0, float(self.head_vertical_decay))
        self.head_y_band_position = max(0.0, min(1.0, float(self.head_y_band_position)))
        self.head_offset_y_ratio = float(self.head_offset_y_ratio)
        # 两个值都允许为负；归一化时不强行改变语义，估计函数会取 min/max 做夹逼。
        self.head_offset_y_min = float(self.head_offset_y_min)
        self.head_offset_y_max = float(self.head_offset_y_max)
        self.head_wide_width_ratio_threshold = max(0.1, float(self.head_wide_width_ratio_threshold))
        self.head_wide_offset_scale = max(0.0, min(2.0, float(self.head_wide_offset_scale)))

        self.trigger_tolerance = max(0.0, float(self.trigger_tolerance))
        self.burst_shots_limit = max(1, int(self.burst_shots_limit))
        self.img_size = max(8, int(self.img_size))
        self.model_input_channels = int(getattr(self, "model_input_channels", 4) or 4)
        if self.model_input_channels not in (1, 3, 4):
            log(f"model_input_channels={self.model_input_channels} 不支持，已自动改为 4", "WARN")
            self.model_input_channels = 4

        self.train_sample_threshold = max(1, int(self.train_sample_threshold))
        self.train_check_interval = max(5, int(self.train_check_interval))
        self.max_samples_per_class = max(0, int(self.max_samples_per_class))
        self.train_epochs = max(1, int(self.train_epochs))
        self.train_batch_size = max(1, int(self.train_batch_size))

        # 原配置中 anti_smoke_min_area 可能大于 max_contour_area，导致反烟雾分支永远不可达。
        # 这里不删除反烟雾功能，而是把阈值压回可检测轮廓面积范围内。
        self.anti_smoke_min_area = max(0.0, float(self.anti_smoke_min_area))
        if self.anti_smoke_min_area > self.max_contour_area:
            adjusted = max(self.min_contour_area, self.max_contour_area * 0.80)
            log(
                f"anti_smoke_min_area={self.anti_smoke_min_area:g} 大于 max_contour_area={self.max_contour_area:g}，"
                f"已自动调整为 {adjusted:g}，避免反烟雾逻辑不可达",
                "WARN",
            )
            self.anti_smoke_min_area = adjusted

        if self.dynamic_offset_far_dist <= self.dynamic_offset_near_dist:
            self.dynamic_offset_far_dist = self.dynamic_offset_near_dist + 1

    @classmethod
    def from_yaml(cls, path="config.yaml"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            log("config.yaml 加载失败，使用默认配置", "WARN")
            return cls()

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(k for k in data.keys() if k not in field_names)
        if unknown:
            log(f"config.yaml 中存在未识别配置项，已忽略: {', '.join(unknown)}", "WARN")

        filtered = {k: v for k, v in data.items() if k in field_names}
        for list_field in ["color_lower", "color_upper"]:
            if list_field in filtered and isinstance(filtered[list_field], list):
                filtered[list_field] = [int(x) for x in filtered[list_field]]
        return cls(**filtered)
