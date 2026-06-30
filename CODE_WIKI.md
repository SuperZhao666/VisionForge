# VisionForge Code Wiki

> 本文档为 VisionForge 项目的结构化代码知识库，涵盖项目整体架构、主要模块职责、关键类与函数说明、依赖关系及运行方式。
> 对应代码版本：`v17.8.32_gpu_runtime_scroll_log_fix`

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈与依赖关系](#2-技术栈与依赖关系)
3. [项目整体架构](#3-项目整体架构)
4. [目录结构](#4-目录结构)
5. [核心模块职责详解](#5-核心模块职责详解)
6. [关键类与函数说明](#6-关键类与函数说明)
7. [数据流与处理流程](#7-数据流与处理流程)
8. [配置系统](#8-配置系统)
9. [运行方式](#9-运行方式)
10. [构建与打包](#10-构建与打包)
11. [许可证系统](#11-许可证系统)
12. [硬件固件协议](#12-硬件固件协议)
13. [工具脚本一览](#13-工具脚本一览)

---

## 1. 项目概述

VisionForge 是一款基于 Python 的 **AI 视觉目标检测与硬件级鼠标控制工具**，采用 YOLO 模型（ONNX Runtime 推理）进行实时目标识别，配合 Arduino Leonardo 开发板实现 USB HID 级别的鼠标移动与点击控制。

### 核心能力

| 能力 | 实现方式 |
|------|----------|
| 毫秒级推理 | ONNX Runtime + TensorRT/CUDA/CPU 多后端，单帧推理 < 1ms |
| 精准目标检测 | YOLO 模型，区分 `body`(0)/`head`(1) 两类目标 |
| 智能目标跟踪 | 自研 4D 卡尔曼滤波 + EMA 平滑 + 自身运动补偿 |
| 稳定目标锁定 | 多帧确认 + 滞后阈值 + 丢失预测的时序门控 |
| 硬件级控制 | Arduino Leonardo HID 串口协议，相对移动 ±127 像素 |
| 高速控制环 | 默认 1000Hz 残差电机控制环 + PID + 防过冲保护 |
| 离线授权 | RSA-2048 (PKCS#1 v1.5 + SHA-256) 签名验证 + 机器绑定 |

### 设计哲学

- **多层防误检**：检测过滤 → 目标锁定 → 几何校验 → 时序门控 → 运动校验，每层都有独立的几何/置信度/时序约束。地图灯光、边缘贴边、微小误检框即使被画出，也会被后续层级阻止驱动 HID。
- **身份与位置解耦**：`target_lock` 决定"锁定哪个目标"，`tracker` 决定"控制点在哪"，两者职责分离。
- **延迟优先**：屏幕采集使用"最新帧优先"策略主动丢弃旧帧；推理与控制解耦。
- **可追溯**：所有运行行为以时间戳文件日志形式落盘，便于离线分析调优。

> ⚠️ 项目仅供技术研究与学习交流，严禁用于违法违规用途。

---

## 2. 技术栈与依赖关系

### 2.1 语言与运行环境

- **Python 3.10+**（主语言）
- **C++（Arduino）**：Leonardo HID 固件
- **目标系统**：Windows 10/11 (64-bit)（核心控制功能依赖 Windows API）
- **推荐 GPU**：NVIDIA GPU + CUDA 12.x + cuDNN 9.x + TensorRT（可选）

### 2.2 Python 依赖矩阵

项目按用途拆分多个 requirements 文件：

| 文件 | 用途 | 关键依赖 |
|------|------|----------|
| [requirements.txt](file:///workspace/requirements.txt) | GPU 桌面运行 | `onnxruntime-gpu==1.20.1`, `opencv-python`, `numpy`, `pyyaml`, `pyserial`, `keyboard`, `dxcam`, `mss`, `psutil`, `customtkinter`, `pillow`, `pyinstaller`, `requests` |
| [requirements-cpu.txt](file:///workspace/requirements-cpu.txt) | CPU 回退运行 | 将 `onnxruntime-gpu` 换为 `onnxruntime==1.20.1`，去掉 GUI 依赖 |
| [requirements-cn.txt](file:///workspace/requirements-cn.txt) | 国内镜像安装 | 同 CPU 版（搭配清华镜像源） |
| [requirements-exe.txt](file:///workspace/requirements-exe.txt) | EXE 打包构建 | 同 GPU 版，含 `pyinstaller`/`customtkinter` |
| [requirements-protection.txt](file:///workspace/requirements-protection.txt) | 保护型构建 | `nuitka`, `ordered-set`, `zstandard`（可选 `pyarmor`） |
| [requirements-owner.txt](file:///workspace/requirements-owner.txt) | 作者密钥生成 | `cryptography>=42.0.0`（生成 RSA 密钥对） |

### 2.3 关键第三方库职责

| 库 | 用途 |
|----|------|
| `onnxruntime-gpu` | ONNX 模型高性能推理（TensorRT/CUDA/CPU 后端） |
| `opencv-python` | 图像读写、绘制、颜色空间转换 |
| `numpy` | 张量运算、卡尔曼滤波矩阵操作 |
| `dxcam` | Windows DXGI 桌面高速流式截图（主后端） |
| `mss` | 跨平台截图（dxcam 不可用时的回退后端） |
| `pyserial` | 与 Arduino Leonardo 的串口通信 |
| `keyboard` | 全局热键检测（F8 切换/F10 退出/激活键） |
| `customtkinter` | 现代化深色主题桌面 GUI |
| `pillow` | GUI 图像资源处理 |
| `pyyaml` | 配置文件读写 |
| `psutil` | 系统资源诊断 |
| `pyinstaller` / `nuitka` | 打包为 Windows 单文件 EXE |

### 2.4 内部模块依赖图

```
                       ┌──────────────────────────────────────┐
                       │            app_gui.py                 │  桌面 GUI 入口
                       │  (CustomTkinter + 许可证 + 诊断)       │
                       └───────────────┬──────────────────────┘
                                       │ 调用 realtime_main.main()
                                       ▼
                       ┌──────────────────────────────────────┐
                       │              main.py                  │  实时核心编排
                       │   (load_cfg / run_screen / run_image) │
                       └───────────────┬──────────────────────┘
                                       │
        ┌──────────────────┬───────────┼───────────┬──────────────────┐
        ▼                  ▼           ▼           ▼                  ▼
 ┌─────────────┐   ┌──────────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐
 │ screen_     │   │ onnx_yolo_   │ │ tracker  │ │ control_   │ │ runtime_     │
 │ capture     │   │ detector     │ │          │ │ gate       │ │ controller   │
 │ +frame_     │   │ +detection_  │ │ (Kalman) │ │ (时序门)    │ │ (PID+电机环) │
 │  pipeline   │   │  filter      │ │          │ │            │ │              │
 └─────────────┘   │ +target_     │ └──────────┘ └────────────┘ └──────┬───────┘
                   │  selector    │                                       │
                   │ +target_lock │                                       ▼
                   │ +target_     │                              ┌──────────────┐
                   │  validation  │                              │ leonardo_    │
                   └──────────────┘                              │ driver       │
                                                                 │ (串口 HID)   │
                                                                 └──────────────┘
基础层: types.py(数据结构) · app_paths.py(路径/DLL) · log_utils.py(日志) ·
        profiler.py(性能) · offline_license.py(授权)
```

---

## 3. 项目整体架构

VisionForge 采用**分层流水线架构**，自上而下分为四层：

### 3.1 应用层（Application Layer）

- [app_gui.py](file:///workspace/app_gui.py)：CustomTkinter 桌面 GUI，用户主入口。包含 6 个页面（总览/授权/启动/调节/环境/更新），通过守护线程在进程内启动实时核心（避免重复 PyInstaller 解压）。
- [main.py](file:///workspace/main.py)：实时核心编排器。负责配置加载、模块装配、实时循环调度、日志统计。

### 3.2 视觉流水线层（Vision Pipeline Layer）

按数据流向依次：

1. **屏幕采集** — [src/screen_capture.py](file:///workspace/src/screen_capture.py) + [src/frame_pipeline.py](file:///workspace/src/frame_pipeline.py)：DXGI/MSS 截取中心 ROI，后台线程持续刷新最新帧。
2. **目标检测** — [src/onnx_yolo_detector.py](file:///workspace/src/onnx_yolo_detector.py)：ONNX YOLO 推理，输出 `DetectionBox` 列表。
3. **几何过滤** — [src/detection_filter.py](file:///workspace/src/detection_filter.py)：基于面积/宽高/长宽比/头身配对/边缘的几何过滤，抑制地图误检。
4. **目标选择** — [src/target_selector.py](file:///workspace/src/target_selector.py)：从多框中选出单个控制点（优先头部，回退身体）。
5. **目标锁定** — [src/target_lock.py](file:///workspace/src/target_lock.py)：时序身份管理，防止目标抖动切换，桥接短时丢失。
6. **轨迹平滑** — [src/tracker.py](file:///workspace/src/tracker.py)：卡尔曼/EMA 平滑 + 自身运动补偿。

### 3.3 控制门控层（Control Gating Layer）

7. **时序门控** — [src/control_gate.py](file:///workspace/src/control_gate.py)：确认帧计数、滞后阈值、丢失宽限、即时/反应式快速进入。
8. **运动校验** — [src/target_validation.py](file:///workspace/src/target_validation.py)：HID 移动前的最终几何/置信度校验（比检测层更严格）。

### 3.4 执行层（Execution Layer）

9. **运行时控制器** — [src/runtime_controller.py](file:///workspace/src/runtime_controller.py)：1000Hz 残差电机控制环，PID + 速度前馈 + 防过冲 + 收敛锁 + 自动扳机门控。
10. **硬件驱动** — [src/leonardo_driver.py](file:///workspace/src/leonardo_driver.py)：串口封装，发送 5 字节指令包到 Arduino Leonardo。

### 3.5 基础设施层（Infrastructure Layer）

- [src/types.py](file:///workspace/src/types.py)：共享数据结构（`DetectionBox`、`TargetResult`）。
- [src/app_paths.py](file:///workspace/src/app_paths.py)：跨环境路径解析、DLL 搜索路径注入。
- [src/log_utils.py](file:///workspace/src/log_utils.py)：线程安全文件优先日志。
- [src/profiler.py](file:///workspace/src/profiler.py)：滚动窗口性能采样。
- [src/offline_license.py](file:///workspace/src/offline_license.py)：离线 RSA 授权验证。

### 3.6 线程模型

| 线程 | 职责 | 所在模块 |
|------|------|----------|
| 主线程 | GUI 事件循环 / 实时循环 | `app_gui.py` / `main.py` |
| LatestFrameReader | 持续刷新最新帧 | `frame_pipeline.py` |
| stable-motor | 1000Hz 残差电机控制环 | `runtime_controller.py` |
| Leonardo 后台预连接 | 启动时异步连接硬件 | `runtime_controller.py` |
| Leonardo 自动重连 | 串口异常时后台重连 | `leonardo_driver.py` |
| GUI 诊断工作线程 | 异步环境检测 | `app_gui.py` |

---

## 4. 目录结构

```
VisionForge/
├── app_gui.py                 # 桌面 GUI 主入口（EXE 入口点）
├── main.py                    # 实时核心编排器
├── config.yaml                # 运行配置（用户可编辑）
├── requirements*.txt          # 多套依赖清单
├── LICENSE / README.md
│
├── src/                       # 核心源代码（视觉 + 控制 + 基础设施）
│   ├── __init__.py
│   ├── types.py               # 共享数据结构
│   ├── app_paths.py           # 路径/DLL/运行时布局
│   ├── log_utils.py           # 日志设施
│   ├── profiler.py            # 性能采样
│   ├── screen_capture.py      # 屏幕截图（dxcam/mss）
│   ├── frame_pipeline.py      # 最新帧后台读取
│   ├── onnx_yolo_detector.py  # ONNX YOLO 检测器
│   ├── detection_filter.py    # 几何过滤
│   ├── target_selector.py     # 目标选择
│   ├── target_lock.py         # 目标锁定（时序身份）
│   ├── target_validation.py   # 运动校验
│   ├── tracker.py             # 卡尔曼/EMA 平滑
│   ├── control_gate.py        # 时序控制门
│   ├── runtime_controller.py  # 运行时控制器（电机环 + PID + 扳机）
│   ├── leonardo_driver.py     # Leonardo 串口驱动
│   └── offline_license.py     # 离线 RSA 授权
│
├── tools/                     # 工具脚本（开发/构建/诊断）
│   ├── config_tuner_gui.py    # 浏览器高级参数调优
│   ├── env_diagnostics.py     # 环境诊断
│   ├── license_keygen.py      # 授权卡密生成（作者专用）
│   ├── inspect_onnx.py        # ONNX 模型检查
│   ├── benchmark_onnx.py      # 推理性能基准
│   ├── check_gpu_provider.py  # GPU Provider 验证
│   ├── collect_cuda_dlls.py   # CUDA DLL 收集（打包用）
│   ├── audit_build_readiness.py  # 构建前审计
│   └── test_image.py          # 单图流水线测试
│
├── scripts/                   # Windows 运行/构建脚本（.bat/.ps1/.sh）
│   ├── run_desktop_gui.bat
│   ├── run_realtime_control.bat
│   ├── run_realtime_preview*.bat
│   ├── run_config_tuner_gui*.bat
│   ├── run_test_image.bat
│   ├── run_env_diagnostics_gui.bat
│   ├── diagnose_windows.bat
│   ├── setup_windows*.bat / setup_linux.sh
│   ├── build_windows_exe.bat / build_windows_onefile_exe.bat
│   ├── build_protected_onefile_nuitka.bat
│   ├── generate_license_key.bat
│   └── open_driver_cuda_links.bat
│
├── packaging/                 # PyInstaller 打包规格
│   ├── VisionForge_ONEFILE.spec         # 当前激活的单文件规格
│   ├── V17_8_27_Runtime_GUI_ONEFILE.spec  # 兼容包装
│   └── V17_8_25_Runtime_GUI.spec        # 旧目录式（已弃用）
│
├── firmware/                  # Arduino 固件
│   └── leonardo_mouse_hid/
│       └── leonardo_mouse_hid.ino
│
├── vendor_models/             # 预训练 ONNX 模型
│   ├── valorant_320_v11n.onnx
│   └── README.txt
│
├── assets/                    # 图标/演示视频
├── keys/                      # 授权密钥存放（运行时生成）
├── logs/                      # 运行日志（运行时生成）
├── config_backups/            # 配置备份（调优时生成）
└── legacy_original/           # 原始版本代码（存档参考，不参与运行）
```

---

## 5. 核心模块职责详解

### 5.1 视觉检测链

#### `onnx_yolo_detector.py` — ONNX YOLO 检测器

- **职责**：加载 Ultralytics YOLO 导出的 ONNX 模型，执行预处理（letterbox）/推理/后处理（解码 + NMS），返回 `DetectionBox` 列表。
- **后端策略**：优先 `TensorrtExecutionProvider` → `CUDAExecutionProvider` → `CPUExecutionProvider`；TensorRT 初始化失败自动回退到 CUDA。
- **兼容性**：自动识别两种 ONNX 输出格式（带/不带 objectness，NMS 预处理输出），通过列数启发式判断。
- **关键方法**：`predict(image_bgr)`、`predict_with_profile(image_bgr)` 返回 `(boxes, {pre_ms, infer_ms, post_ms})`。

#### `detection_filter.py` — 几何过滤

- **职责**：在 NMS 之后、目标选择之前，基于几何特征过滤误检框。这是反地图灯光误检的第一道防线。
- **双档配置**：常规目标用严格档，小/远目标（`small_head_area_px`/`small_head_max_dim_px` 判定）用宽松档。
- **核心校验**：头身配对（IoU/距离/包含关系）、头身面积/宽高比窗口、身体必须在头部下方延伸、边缘贴边拒绝、V17.8.12 失效保护（短身体或远离中心时要求高置信度）。
- **关键函数**：`filter_detections_by_geometry(boxes, cfg, *, center, frame_shape)` 返回 `(filtered_boxes, FilterStats)`。

#### `target_selector.py` — 目标选择

- **职责**：从过滤后的多框中选出**单个控制点**。
- **策略**：优先选头部中心（按置信度→距中心距离→面积排序）；无合格头部时回退到身体上 18% 处（颈部/上胸部位置）作为 `body_fallback`。
- **头身配对**：`_match_body` 优先选包含头部中心的身体框，避免误关联不相关的高置信度身体。

#### `target_lock.py` — 目标锁定（时序身份）

- **职责**：维持"锁定哪个目标"的身份连续性，防止多目标间抖动切换，桥接短时检测丢失。
- **核心机制**：
  - **锁定匹配**：基于距离 + 头部 IoU + 身体 IoU 的多因子匹配；身体 IoU 单独不足以继承锁。
  - **切换仲裁**：锁定中切换需多帧确认（`switch_confirm_frames`），且新目标必须在中心距离/置信度/总分上同时占优。
  - **丢失预测**：丢失后 ≤`predict_lost_frames` 帧且 ≤`predict_lost_ms` 毫秒内，基于最后速度做航位推算。
  - **速度学习**：EMA 平滑速度，跳变/超速时清零以防预测中毒。
- **关键类**：`TargetLockManager`，主入口 `select(boxes, selector, center, *, active)`。

#### `tracker.py` — 轨迹平滑

- **职责**：对锁定后的控制点 `(x, y)` 做数值平滑与预测。
- **算法**：
  - `KalmanFilter4D`：4 维常速卡尔曼（x, y, vx, vy），Joseph 形式协方差更新防止数值失稳，新息门控识别身份切换并重置。
  - `EmaPointTracker`：指数移动平均回退方案。
  - `LegacyPointTracker`：编排器，封装卡尔曼 + 丢失保持 + 身体回退保持 + 自身运动补偿（`apply_ego_motion` 膨胀位置协方差）。

### 5.2 控制门控链

#### `control_gate.py` — 时序控制门

- **职责**：在目标送到控制器之前，决定其"是否足够可靠可驱动 HID 移动"。分离**进入**（严格）与**保持**（宽松带滞后）。
- **核心机制**：
  - **确认帧计数**：目标需连续 `require_confirmed_frames` 帧达标；高置信度/即时/反应式快速进入可减少所需帧数。
  - **滞后阈值**：进入用 `min_head_conf_enter`，保持用更低的 `min_head_conf_hold`。
  - **确认记忆**：`confirmed_memory_ms`（默认 250ms）内曾确认的目标降低后续进入门槛。
  - **丢失保持**：短时丢失（`missing_target_hold_frames`/`ms`）继续输出保持点。
  - **同锁跳变接受**（V17.8.21）：可靠的同锁测量直接接受当前原始点，避免移动-暂停-移动卡顿。
  - **反地图时序确认**：小/矮/可疑形状需更多帧（`small_target_confirmed_frames`/`tiny_target_confirmed_frames`）才允许驱动。
  - **控制点平滑**：对锁定目标的 `(x, y)` 做抖动带 + 摆率限制 + 高置信度大跳变吸附，且控制点不得滞后当前头部框过远。
- **主入口**：`update(raw_target, *, active, center_x, center_y)` 返回 `(movement_ready, target, reason)`。

#### `target_validation.py` — 运动校验

- **职责**：HID 移动前的**最终安全校验**，比检测层更严格（"可画出但不可移动"）。
- **校验内容**：头部置信度、头/身面积、身体宽高/长宽比、头身比例窗口、边缘拒绝、头身配对、身体向下延伸、小/微头专用阈值。
- **关键函数**：`validate_movement_target(target, cfg, *, center_x, center_y)` 返回 `(ok, reason)`。

### 5.3 执行链

#### `runtime_controller.py` — 运行时控制器

- **职责**：将视觉发布的目标误差转换为 HID 鼠标增量，运行 1000Hz 残差电机控制环，并管理自动扳机门控。整个项目的运动控制大脑。
- **核心子模块**：
  - **`PIDAxis`**：单轴 PID，带抗积分饱和、积分死区、微分低通、方向锁（输出不得翻转误差符号）、目标跳变重置。
  - **残差系统**：将每帧推理误差分配到多个电机 tick 平滑注入，避免"爆发-暂停-爆发"。
  - **速度前馈**：EMA 估计目标速度，带跳变/超速门控，前馈幅度受限于误差方向同向占比。
  - **防过冲保护**：移动量上限为 `|error| * overshoot_error_fraction`，残差上限为 `|error| * residual_error_fraction`，禁止符号翻转。
  - **收敛锁（settle lock）**：进入/退出/硬退出三级滞后半径，冻结近中心残差。
  - **自然运动整形**：α-EMA + 加速度上限 + 过零制动 + 连续运动轮廓保持，使移动更拟人。
  - **自动扳机门控**：`fire_radius`/`fire_exit_radius`/`fire_rearm_radius` 三级滞后 + 冷却 + 稳定帧 + 新鲜度 + 运动债 + 移动后延迟完整性检查。
  - **自适应陈旧超时**：基于提交间隔 EMA 动态计算陈旧判定阈值。
- **主循环** `_motor_loop()`：定频调度；激活键上升沿丢弃按下前目标（防幽灵）；陈旧/无效目标清零；否则 `_add_target_error_once` → `_drain_residual` → `_shape_motion_output`；`(0,0)` 输出尝试扳机，否则 `_driver.move(mx,my)`。

#### `leonardo_driver.py` — Leonardo 串口驱动

- **职责**：封装与 Arduino Leonardo 的串口通信，发送 5 字节指令包，自动检测端口与异常重连。
- **指令包格式**：`[0xAA, cmd, dx, dy, checksum]`，checksum = `(cmd+dx+dy) & 0xFF`。
- **端口检测**：扫描 `2341:8036`/`2341:0036` (Leonardo VID:PID) 或描述含 "leonardo"/"arduino"。
- **重连**：串口异常时标记未初始化，1 秒节流的后台重连线程。
- **限幅**：HID 增量限制在 ±127（单字节有符号）。

### 5.4 基础设施

#### `app_paths.py` — 路径与 DLL 管理

- **职责**：跨环境（源码运行 / PyInstaller / Nuitka 冻结）解析配置、日志、模型、DLL、许可证路径；创建运行时目录布局；**在导入 onnxruntime 之前注入 CUDA/cuDNN/TensorRT/ORT DLL 搜索路径**（这是 v17.8.32 修复 GPU Provider 回退的关键）。
- **目录分离**：只读资源（`resource_root`）与可写用户数据（`user_data_dir` = `%LOCALAPPDATA%\VisionForge`）分离；支持从旧版 `V17_8_RUNTIME_GUI` 迁移配置与许可证。
- **关键函数**：`configure_dll_search_path()`、`ensure_runtime_layout()`、`apply_runtime_overrides(cfg)`。

#### `log_utils.py` — 日志设施

- **职责**：线程安全、文件优先的日志。默认时间戳 `.txt` 文件（`logs/run_YYYYMMDD_HHMMSS.txt`），控制台默认静默（v16 起）。
- **特性**：RLock 保护；每调用可覆盖控制台开关；文件写入失败回退 stderr，**永不中断实时控制**。
- **关键函数**：`init_logging(cfg)`、`log(msg, level)`、`log_kv(title, mapping)`、`log_block(title, text)`、`log_exception()`。

#### `offline_license.py` — 离线授权

- **职责**：RSA-2048 (PKCS#1 v1.5 + SHA-256) 离线许可证验证，机器绑定，套餐/有效期校验。客户端只内嵌公钥。
- **机器码**：`socket.gethostname()` + `uuid.getnode()` + Windows `wmic` 硬件序列号，加盐 SHA-256 取前 32 位十六进制。
- **RSA 验证**：纯 Python `pow(sig, e, n)` + 手动 PKCS#1 v1.5 填充校验，无 `cryptography` 依赖。
- **关键函数**：`validate_license_text(key_text)` 返回 `LicenseStatus`；`load_license()`、`save_license(key_text)`、`machine_code()`。

---

## 6. 关键类与函数说明

### 6.1 数据结构（[src/types.py](file:///workspace/src/types.py)）

#### `DetectionBox`（frozen dataclass）

不可变检测框，全链路共享。

| 字段/属性 | 类型 | 说明 |
|-----------|------|------|
| `cls_id` | `int` | 类别 ID（0=body, 1=head） |
| `cls_name` | `str` | 类别名 |
| `conf` | `float` | 置信度 |
| `x1, y1, x2, y2` | `float` | 左上/右下坐标 |
| `w`, `h`, `area` | `float` (property) | 宽/高/面积 |
| `center` | `tuple[float,float]` (property) | 中心点 |
| `shifted(dx,dy)` | `DetectionBox` | 平移后的新框 |
| `to_dict()` | `dict` | 序列化 |

#### `TargetResult`（mutable dataclass）

可变目标结果，下游可就地改写 `x/y/reason`。

| 字段 | 类型 | 说明 |
|------|------|------|
| `found` | `bool` | 是否找到目标 |
| `x, y` | `float` | 控制点坐标 |
| `source` | `str` | 来源：`head`/`body_fallback`/`none` |
| `confidence` | `float` | 置信度 |
| `reason` | `str` | 决策原因链（分号分隔） |
| `head_box`, `body_box` | `Optional[DetectionBox]` | 关联的头/身框 |

### 6.2 检测器（[src/onnx_yolo_detector.py](file:///workspace/src/onnx_yolo_detector.py)）

```python
class OnnxYoloDetector:
    def __init__(self, model_path, imgsz=320, conf=0.25, iou=0.70,
                 class_names=None, providers=None, max_candidates=300, require_gpu=True)
    def predict(self, image_bgr) -> list[DetectionBox]
    def predict_with_profile(self, image_bgr) -> tuple[list[DetectionBox], dict[str,float]]
```

- `_preprocess`：letterbox + BGR→RGB + /255 + CHW + 灰度填充 114。
- `_postprocess`：解码 + 置信度过滤 + unletterbox + 裁剪 + top-K + 按类 NMS。
- `_decode_rows`：按列数分支解码（6 列 / 5+nc 列 / 4+nc 列）。

### 6.3 控制器（[src/runtime_controller.py](file:///workspace/src/runtime_controller.py)）

```python
@dataclass
class ControlConfig:  # ~250 个调优字段
    enabled: bool; mode: str; port: str; baud: int
    gain_x, gain_y: float; sensitivity_scaler: float
    pid_enabled: bool; pid_kp, pid_ki, pid_kd: float
    pid_integral_limit, pid_integral_deadband_px, pid_reset_jump_px: float
    velocity_lead_ms, velocity_lead_max_px: float
    natural_motion_enabled: bool; natural_motion_alpha, natural_motion_max_delta: float
    settle_lock_enabled: bool; settle_enter_px, settle_exit_px, settle_hard_exit_px: float
    fire_enabled: bool; fire_radius, fire_exit_radius, fire_rearm_radius: float
    fire_cooldown_ms, fire_min_conf, fire_stable_frames: ...
    max_move, max_step, deadzone, fine_deadzone: ...
    active_key, toggle_key, quit_key: str
    # ... 其余字段见源码

class PIDAxis:
    def update(self, error, now, cfg, *, fine_dead=0.0) -> float  # 带抗饱和的方向锁 PID

class RuntimeController:
    def __init__(self, cfg: ControlConfig)
    def submit(self, target_x, target_y, center_x, center_y, *,
               distance=0, confidence=0, valid=True, held=False, target_radius=0)
    def clear_target(self, *, soft=False, active=False)
    def clear_residual(self)
    def consume_ego_delta(self) -> tuple[int, int]  # 自身运动补偿
    def poll_hotkeys(self) -> bool  # F8 切换/F10 退出
    def is_active(self) -> bool     # 激活键状态
    def driver_status(self) -> str  # disabled/ready/connecting/not_ready
    def close(self)
    # 内部: _motor_loop() 1000Hz 主循环
```

### 6.4 控制门（[src/control_gate.py](file:///workspace/src/control_gate.py)）

```python
@dataclass
class ControlGateConfig:  # ENTER/HOLD 滞后、确认帧、丢失保持、即时/反应式进入、同锁跳变、反地图时序
    min_head_conf_enter, min_head_conf_hold, high_conf_head: float
    require_confirmed_frames, high_conf_confirmed_frames: int
    missing_target_hold_frames, missing_target_hold_ms, missing_target_hold_min_conf: ...
    instant_enter_enabled, instant_enter_center_dist_px, instant_enter_min_conf: ...
    reactive_fast_enter_enabled, reactive_fast_enter_min_conf, ...
    same_lock_jump_accept_enabled, same_lock_jump_accept_px, ...
    small_target_confirmed_frames, tiny_target_confirmed_frames, suspicious_body_height_px: ...

class ConfirmedHeadGate:
    def update(self, raw_target, *, active, center_x, center_y)
        -> tuple[bool, TargetResult, str]  # (movement_ready, target, reason)
    def on_active_rising(self)
    def on_validation_result(self, valid, target)
    def apply_ego_motion(self, dx, dy, scaler=2.7)
```

### 6.5 目标锁定（[src/target_lock.py](file:///workspace/src/target_lock.py)）

```python
@dataclass
class TargetLockConfig:
    hold_lost_frames, hold_lost_seconds: ...
    match_max_distance_px, hard_match_max_distance_px, body_iou_match_max_distance_px: float
    min_lock_head_conf, head_without_body_lock_conf: float
    allow_switch_while_locked: bool
    switch_confirm_frames, switch_match_px, switch_center_advantage_px: ...
    predict_lost_target, predict_lost_frames, predict_lost_ms, predict_lost_min_conf: ...
    max_lock_velocity_px_s, max_velocity_update_jump_px: float

class TargetLockManager:
    def select(self, boxes, selector, center, *, active=True) -> TargetResult
    def on_active_rising(self)
    def reset(self, reason="reset")
    # 属性: last_reason, lost_frames
```

### 6.6 卡尔曼滤波（[src/tracker.py](file:///workspace/src/tracker.py)）

```python
class KalmanFilter4D:  # 4 维常速卡尔曼 (x, y, vx, vy)
    def __init__(self, q=0.06, r=0.12, *, max_velocity=3200,
                 max_prediction_dt=0.12, innovation_gate_px=145,
                 covariance_floor=1e-6, covariance_ceiling=1e4,
                 ego_covariance_boost=0.18)
    def predict(self, dt) -> tuple[float, float]
    def update(self, mx, my) -> tuple[float, float]   # Joseph 形式 + 新息门控
    def apply_ego_motion(self, dx, dy, scaler=2.7)    # 膨胀位置协方差
    def decay_velocity(self, decay_factor=0.86)
    def reset(self)

class LegacyPointTracker:  # 编排器
    def update(self, target: TargetResult, dt: float) -> TargetResult
    def apply_ego_motion(self, dx, dy, scaler=None)
    def reset(self)
```

### 6.7 Leonardo 驱动（[src/leonardo_driver.py](file:///workspace/src/leonardo_driver.py)）

```python
class LeonardoMouseDriver:
    HEADER = 0xAA; CMD_MOVE=0x01; CMD_PRESS=0x02; CMD_RELEASE=0x03
    CMD_CLICK=0x04; CMD_MOVE_PRESS=0x05; CMD_MOVE_RELEASE=0x06; CMD_HEARTBEAT=0xFF
    def __init__(self, port="auto", baud=115200)
    def move(self, dx, dy) -> bool          # 限幅 ±127
    def press_left_click(self) -> bool
    def release_left_click(self) -> bool
    def click_left(self) -> bool
    def close(self)
```

---

## 7. 数据流与处理流程

### 7.1 单帧实时循环（[main.py](file:///workspace/main.py) `run_screen`）

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. 采集帧 (profiler: capture)                                        │
│    LatestFrameReader.get_latest() → CapturedFrame(frame, seq, ts)    │
│    丢弃 seq 未变的旧帧 (drop_stale_frames)                            │
├─────────────────────────────────────────────────────────────────────┤
│ 2. 自身运动补偿 (profiler: ego)                                       │
│    controller.consume_ego_delta() → (ego_dx, ego_dy)                 │
│    tracker.apply_ego_motion() / gate.apply_ego_motion()              │
├─────────────────────────────────────────────────────────────────────┤
│ 3. 推理 + 几何过滤 (profiler: infer_total)                            │
│    detector.predict_with_profile(frame) → (boxes, det_profile)       │
│    filter_detections_by_geometry(boxes, filter_cfg, center)          │
├─────────────────────────────────────────────────────────────────────┤
│ 4. 选择 + 锁定 + 平滑 + 门控 (profiler: select_gate)                  │
│    active_now = controller.is_active()                               │
│    raw_target = target_lock.select(boxes, selector, center, active)  │
│    target = tracker.update(raw_target, dt_frame)                     │
│    target = _correct_control_point(raw, target, cfg, center)  # 限滞后│
│    gate_ok, gated_target, gate_reason = gate.update(target, ...)     │
│    base_ok, base_reason = target_allowed_for_control(...)            │
│    control_ok = gate_ok and base_ok and gated_target.found           │
├─────────────────────────────────────────────────────────────────────┤
│ 5. 提交到控制器                                                       │
│    if control_ok and source=="head":                                 │
│        controller.submit(global_x, global_y, global_center, ...)     │
│    else:                                                             │
│        controller.clear_target(soft=True, active=active_now)         │
├─────────────────────────────────────────────────────────────────────┤
│ 6. 绘制 (profiler: draw, 可选/限频)                                   │
│    _maybe_draw(...) → cv2.imshow                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 7. 日志/统计 (每 print_every 帧 STATUS, 每 rolling_summary_every_sec) │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 控制器内部电机环（[src/runtime_controller.py](file:///workspace/src/runtime_controller.py) `_motor_loop`，1000Hz）

```
loop @ control_loop_hz:
  ├─ poll_hotkeys() → F8 切换 / F10 退出
  ├─ 激活键上升沿 → 丢弃按下前目标 (active_press_accept_recent_ms 内例外)
  ├─ 快照目标状态 (_snapshot_target)
  ├─ if 目标陈旧/无效: clear_target(soft)
  ├─ else:
  │   ├─ _add_target_error_once: settle 消费 + 速度估计 + 前馈 + PID +
  │   │                        灵敏度缩放 + 近中心阻尼 + 防过冲 + 残差注入
  │   ├─ _drain_residual: 整数步进 + 微步逻辑
  │   └─ _shape_motion_output: α-EMA + 加速度上限 + 过零制动 + 收敛保护
  ├─ if (mx,my)==(0,0): _try_fire()  # 自动扳机门控
  ├─ else: driver.move(mx, my)
  ├─ 成功 → 提交残差 + 记录自身运动增量 + 更新扳机最后移动时间
  └─ 失败 → 清零残差
```

### 7.3 GPU Provider 加载时序（关键启动流程）

```
app_gui.run_realtime_from_gui() / main.main()
  └─ app_paths.configure_dll_search_path()   # 必须在 import onnxruntime 前
       └─ 扫描 runtime_dlls/ / CUDA_PATH / Program Files / site-packages
       └─ os.add_dll_directory() + PATH 前置
  └─ import onnxruntime
       └─ ort.preload_dlls(cuda, cudnn, msvc)  # ORT ≥1.21
  └─ OnnxYoloDetector.__init__()
       └─ providers = [Tensorrt, CUDA, CPU] ∩ available
       └─ InferenceSession(model, providers)
       └─ TensorRT 失败 → 回退 CUDA
       └─ require_gpu 且无 GPU Provider → RuntimeError
```

---

## 8. 配置系统

### 8.1 配置文件

- [config.yaml](file:///workspace/config.yaml)：用户可编辑的运行配置（当前版本 `v17.8.32_gpu_runtime_scroll_log_fix`）。
- `config.default_v17_8_32.yaml`：默认配置种子（打包进 EXE，首次运行时复制到用户数据目录）。

### 8.2 配置节结构

| 顶层节 | 职责 | 关键字段示例 |
|--------|------|-------------|
| `model` | 模型与推理 | `path`, `imgsz`, `conf`, `iou`, `providers`, `require_gpu`, `classes` |
| `capture` | 屏幕采集 | `source`, `backend`, `roi_width/height`, `target_fps`, `max_reused_frames` |
| `detection_filter` | 几何过滤 | `min_head_conf`, `head_only_min_conf`, `small_head_area_px`, 头身比例窗口 |
| `selection` | 目标选择 | `prefer_head`, `fallback_to_body`, `body_fallback_y_ratio`, `head_conf` |
| `target_lock` | 目标锁定 | `hold_lost_frames`, `match_max_distance_px`, `switch_confirm_frames`, `predict_lost_*` |
| `tracking` | 轨迹平滑 | `method` (legacy_kalman/ema), `kalman_*`, `control_lag_clamp_enabled` |
| `control` | 运行控制（最大节） | `enabled`, `mode`, PID, 残差, 自然运动, 收敛锁, 扳机门控, 热键 |
| `runtime` | 运行时 | `threaded_capture`, `drop_stale_frames`, `infer_fps_limit`, `warmup_inference_frames` |
| `visual` | 可视化 | `show_window`, `window_name`, `draw_every`, `max_fps` |
| `logging` | 日志 | `console`, `file`, `log_dir`, `print_every`, `profile`, `rolling_summary_every_sec` |

### 8.3 配置加载与覆盖

```python
# main.py
cfg = load_cfg(args.config)        # yaml.safe_load + apply_runtime_overrides
# 命令行覆盖: --control on/off, --visual on/off, --profile on/off,
#             --threaded-capture on/off, --capture-backend, --console-log, --source
```

`apply_runtime_overrides(cfg)` 注入 `model.path` 和 `logging.log_dir`，不覆盖用户已设值。

### 8.4 调优界面

项目提供**两套调优界面**：

| 界面 | 入口 | 面向 | 参数范围 |
|------|------|------|----------|
| 桌面 GUI "调节"页 | `app_gui.py` | 普通用户 | ~40 个友好参数（`FRIENDLY_PARAMS`），4 个预设 |
| 浏览器高级调优 | `tools/config_tuner_gui.py` | 高级用户/开发者 | ~80 个完整参数（`PARAM_SPECS`），7 个预设，含日志分析器 |

浏览器调优器（`127.0.0.1:8765`）提供 REST API（`GET/POST /api/config`、`/api/raw_yaml`、`/api/analyze_latest_log`），保存前自动备份，原子写入。

---

## 9. 运行方式

### 9.1 环境准备

```bash
git clone https://github.com/SuperZhao666/VisionForge.git
cd VisionForge
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # GPU
# 或 pip install -r requirements-cpu.txt  # CPU 回退
```

GPU 加速需额外安装：NVIDIA 驱动 + CUDA 12.x + cuDNN 9.x +（可选）TensorRT。

### 9.2 三种运行模式

| 模式 | 命令 | 说明 |
|------|------|------|
| 桌面 GUI（推荐） | `scripts\run_desktop_gui.bat` | 完整 GUI，含授权/调优/诊断 |
| 实时控制 | `scripts\run_realtime_control.bat` | `python main.py --source screen --control on --visual off --profile on` |
| 预览模式 | `scripts\run_realtime_preview.bat` | 显示检测窗口，不控制硬件 |

### 9.3 实时控制热键

| 键 | 功能 |
|----|------|
| `LShift`（默认 `active_key`） | 按住激活控制（`only_when_active: true` 时） |
| `F8`（默认 `toggle_key`） | 切换控制开关 |
| `F10`（默认 `quit_key`） | 退出 |

### 9.4 命令行参数（[main.py](file:///workspace/main.py)）

```
--config              配置文件路径 (默认 config.yaml)
--source              screen / image / video
--control             on / off / config (是否启用硬件控制)
--visual              on / off / config (是否显示检测窗口)
--profile             on / off / config (性能采样)
--threaded-capture    on / off / config (后台采集线程)
--capture-backend     config / dxcam / dxcam_auto / mss
--console-log         on / off / config
```

### 9.5 硬件配置（实时控制必需）

- **开发板**：Arduino Leonardo / Pro Micro (ATmega32U4)
- **固件烧录**：Arduino IDE 打开 [firmware/leonardo_mouse_hid/leonardo_mouse_hid.ino](file:///workspace/firmware/leonardo_mouse_hid/leonardo_mouse_hid.ino) 上传
- **连接**：USB 数据线，波特率 115200，端口自动检测

---

## 10. 构建与打包

### 10.1 单文件 EXE 构建（主流程）

[scripts/build_windows_onefile_exe.bat](file:///workspace/scripts/build_windows_onefile_exe.bat) 执行：

```
1. 确保 .venv + 升级 pip/setuptools/wheel
2. pip install -r requirements-exe.txt
3. python tools/collect_cuda_dlls.py        # 收集 CUDA/cuDNN/ORT DLL 到 runtime_dlls/
4. python tools/audit_build_readiness.py    # 构建前审计（编译/命名冲突/资源）
5. python app_gui.py --self-test            # 冒烟测试
6. PyInstaller --clean --noconfirm packaging/VisionForge_ONEFILE.spec
   → dist/VisionForge.exe
```

### 10.2 PyInstaller 规格（[packaging/VisionForge_ONEFILE.spec](file:///workspace/packaging/VisionForge_ONEFILE.spec)）

- **入口**：`app_gui.py`，输出 `VisionForge.exe`，`console=False`，UPX 压缩
- **数据文件**：默认配置、用户配置、ONNX 模型、图标、`runtime_dlls/`、`customtkinter` 数据
- **二进制**：`onnxruntime`/`cv2`/`numpy` 的动态库
- **隐藏导入**：`src.offline_license`、`src.app_paths`、`tools.*` + 各依赖子模块
- **排除**：matplotlib/scipy/pandas/torch/tensorflow（精简体积）

### 10.3 保护型构建（可选）

[scripts/build_protected_onefile_nuitka.bat](file:///workspace/scripts/build_protected_onefile_nuitka.bat) 使用 Nuitka + zstandard 进行更强混淆打包（可选 pyarmor）。

### 10.4 构建工具

| 工具 | 职责 |
|------|------|
| [tools/collect_cuda_dlls.py](file:///workspace/tools/collect_cuda_dlls.py) | 收集 GPU 运行时 DLL（`REQUIRED_FOR_GPU`: ORT CUDA provider DLL） |
| [tools/audit_build_readiness.py](file:///workspace/tools/audit_build_readiness.py) | 字节编译 + GUI 类命名冲突检查 + 资源存在性检查 |
| [tools/check_gpu_provider.py](file:///workspace/tools/check_gpu_provider.py) | 端到端验证 GPU Provider 真正激活（`--strict` 可作 CI 门控） |
| [tools/inspect_onnx.py](file:///workspace/tools/inspect_onnx.py) | 打印 ONNX 模型输入输出张量与可用 Provider |
| [tools/benchmark_onnx.py](file:///workspace/tools/benchmark_onnx.py) | 推理延迟基准（mean/p50/p90/p99） |

---

## 11. 许可证系统

### 11.1 架构

```
作者侧 (owner)                          客户侧 (client)
┌─────────────────────┐                ┌─────────────────────────┐
│ tools/              │  签发卡密        │ src/offline_license.py  │
│  license_keygen.py  │ ──────────────→ │  (内嵌公钥)             │
│  (持有私钥)         │   VFG-<b64>.<b64>│  validate_license_text  │
└─────────────────────┘                └─────────────────────────┘
```

- **算法**：RSA-2048 + PKCS#1 v1.5 + SHA-256
- **密钥管理**：私钥存于 `owner_secrets/`（绝不打包进 EXE）；公钥以 `(n, e)` 数字形式硬编码进 `src/offline_license.py`
- **机器绑定**：可选 `hwid_hash`（机器码 SHA-256 前 32 位）
- **套餐**：`day`(1天) / `week`(7天) / `month`(31天) / `permanent`(永久)

### 11.2 卡密格式

```
VFG-<base64url(canonical_json_payload)>.<base64url(rsa_signature)>
```

payload 含：`version`, `product`, `license_id`, `plan`, `issued_at`, `expires_at`, `hwid_hash`(可选), `note`。

### 11.3 验证流程（`validate_license_text`）

1. 去前缀 + 按 `.` 分割为 payload + 签名
2. JSON 解码 payload
3. 产品 ID 校验（`VISIONFORGE` / `V17_8_RUNTIME_GUI`）
4. RSA 签名校验（`pow(sig, e, n)` + PKCS#1 v1.5 填充重建比对）
5. 机器绑定校验（若有 `hwid_hash`）
6. 有效期解析与过期检查
7. 返回 `LicenseStatus(valid, reason, plan, days_left, ...)`

### 11.4 桌面 GUI 集成

[app_gui.py](file:///workspace/app_gui.py) 的"授权"页：复制机器码 → 输入卡密 → `save_license()` → `refresh_license_status()`。`start_realtime` 受 `license_status.valid` 门控。

---

## 12. 硬件固件协议

### 12.1 固件（[firmware/leonardo_mouse_hid/leonardo_mouse_hid.ino](file:///workspace/firmware/leonardo_mouse_hid/leonardo_mouse_hid.ino)）

Arduino Leonardo / Pro Micro (ATmega32U4) 作为 USB HID 鼠标，通过串口接收指令。

### 12.2 串口协议

- **波特率**：115200
- **数据包**：`[0xAA, cmd, dx, dy, checksum]`，5 字节
- **校验和**：`checksum = (cmd + dx + dy) & 0xFF`
- **dx/dy**：int8_t，范围 -127 ~ 127（相对移动）
- **响应**：`0xBB` 成功，`0xEE` 失败

### 12.3 指令列表

| 指令 | 值 | 说明 |
|------|----|------|
| MOVE | `0x01` | 相对移动 |
| PRESS | `0x02` | 按下左键 |
| RELEASE | `0x03` | 释放左键 |
| CLICK | `0x04` | 单击左键 |
| MOVE+PRESS | `0x05` | 移动并按下 |
| MOVE+RELEASE | `0x06` | 移动并释放 |
| HEARTBEAT | `0xFF` | 心跳，返回 `0xBB` |

### 12.4 安全机制

- **2 秒自动释放**：`FAILSAFE_RELEASE_MS = 2000`，超时未收到指令自动释放左键，防止卡死。
- **校验和验证**：每包校验，错误返回 `0xEE`。
- **移动限幅**：`Mouse.move` 前限幅 ±127。

---

## 13. 工具脚本一览

### 13.1 运行脚本（[scripts/](file:///workspace/scripts/)）

| 脚本 | 用途 |
|------|------|
| `run_desktop_gui.bat` | 桌面 GUI 主启动（建 venv + 装依赖 + 跑 `app_gui.py`） |
| `run_realtime_control.bat` | 实时控制模式（`--control on --visual off`） |
| `run_realtime_control_debug_window.bat` | 实时控制 + 调试窗口 |
| `run_realtime_preview.bat` / `run_realtime_preview_mss.bat` | 预览模式（dxcam / mss 后端） |
| `run_config_tuner_gui.bat` / `run_config_tuner_gui_no_browser.bat` | 浏览器调优器 |
| `run_test_image.bat` | 单图流水线测试 |
| `run_env_diagnostics_gui.bat` | 环境诊断 GUI |
| `diagnose_windows.bat` | 5 步 CLI 诊断（Python/ORT/nvidia-smi/模型/GPU） |
| `setup_windows.bat` / `setup_windows_cn.bat` / `setup_linux.sh` | 环境安装 |
| `build_windows_exe.bat` / `build_windows_onefile_exe.bat` | EXE 构建 |
| `build_protected_onefile_nuitka.bat` | Nuitka 保护型构建 |
| `generate_license_key.bat` | 授权卡密生成（作者） |
| `open_driver_cuda_links.bat` | 打开驱动/CUDA 下载链接 |

### 13.2 工具脚本（[tools/](file:///workspace/tools/)）

| 脚本 | 用途 | 关键函数/类 |
|------|------|------------|
| [config_tuner_gui.py](file:///workspace/tools/config_tuner_gui.py) | 浏览器高级参数调优 | `PARAM_SPECS`, `PRESETS`, `TunerState`, `Handler`, `analyze_log` |
| [env_diagnostics.py](file:///workspace/tools/env_diagnostics.py) | 环境诊断 | `DiagnosticItem`, `collect_diagnostics`, `summarize`, `_check_cuda_provider` |
| [license_keygen.py](file:///workspace/tools/license_keygen.py) | 卡密生成（作者专用） | `init_keys`, `make_payload`, `sign_payload` |
| [inspect_onnx.py](file:///workspace/tools/inspect_onnx.py) | ONNX 模型检查 | `parse_providers`, `main` |
| [benchmark_onnx.py](file:///workspace/tools/benchmark_onnx.py) | 推理基准 | `main`（warmup + loops 统计） |
| [check_gpu_provider.py](file:///workspace/tools/check_gpu_provider.py) | GPU Provider 验证 | `main`（`--strict` 门控） |
| [collect_cuda_dlls.py](file:///workspace/tools/collect_cuda_dlls.py) | CUDA DLL 收集 | `PATTERNS`, `candidate_dirs`, `ensure_gpu_package` |
| [audit_build_readiness.py](file:///workspace/tools/audit_build_readiness.py) | 构建前审计 | `compile_check`, `collision_check`, `required_assets_check` |
| [test_image.py](file:///workspace/tools/test_image.py) | 单图端到端测试 | `main`（检测→选择→平滑→绘制→保存） |

### 13.3 遗留代码（[legacy_original/](file:///workspace/legacy_original/)）

原始版本存档（`benchmark.py`、`detector.py`、`kalman.py`、`tracker.py`、`main.py` 等），不参与运行，仅供历史参考。

---

## 附录：版本与兼容性说明

- **当前版本**：`v17.8.32_gpu_runtime_scroll_log_fix`（主要修复 ONNX Runtime 导入前未注入 DLL 搜索路径导致 GPU Provider 回退的问题）
- **测试环境**：Windows 11 + RTX 3050 (4GB) + 驱动 591.74 + CUDA 13.1 + Python 3.10+
- **模型切换**：将训练好的 ONNX 模型放入 `vendor_models/`，修改 `config.yaml` 的 `model.path` 与 `model.imgsz` 即可切换游戏（默认 Valorant 320×320）
- **类别约定**：全链路统一 `0=body`, `1=head`

> 本 Wiki 基于源码静态分析生成，反映代码当前磁盘状态。具体调优参数含义请配合 [tools/config_tuner_gui.py](file:///workspace/tools/config_tuner_gui.py) 的 `PARAM_SPECS` 中每个参数的 `description`/`when_to_tune`/`bigger_effect`/`smaller_effect`/`danger` 字段查阅。
