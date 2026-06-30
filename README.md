# VisionForge

<div align="center">
  <img src="assets/app_icon.png" alt="VisionForge Logo" width="128" height="128">
  <h1>⚡ VisionForge</h1>
  <p>AI 自瞄吸附 · 自动扳机 · 射击辅助工具</p>
  
  [![GitHub Stars](https://img.shields.io/github/stars/SuperZhao666/VisionForge?style=social)](https://github.com/SuperZhao666/VisionForge)
  [![GitHub License](https://img.shields.io/github/license/SuperZhao666/VisionForge)](LICENSE)
  [![GitHub Issues](https://img.shields.io/github/issues/SuperZhao666/VisionForge)](https://github.com/SuperZhao666/VisionForge/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/SuperZhao666/VisionForge)](https://github.com/SuperZhao666/VisionForge/pulls)
  
  [![QQ Group](https://img.shields.io/badge/QQ%20Group-1044257667-blue)](https://qm.qq.com/cgi-bin/qm/qr?k=U97K77g4eE9pU97K77g4eE9p&jump_from=webapi)
  [![Telegram](https://img.shields.io/badge/Telegram-Join%20Group-blue)](https://t.me/+OdV-iEskI1MzNDll)
</div>

---

## 📋 项目简介

VisionForge 是一款基于 Python 的 **AI 自瞄吸附与自动扳机射击辅助工具**，采用 YOLO 模型进行目标识别，支持毫秒级推理速度（单帧推理时间低于 1 毫秒）。通过硬件级鼠标控制（Arduino Leonardo），实现自动瞄准、自动扳机、目标跟踪等功能。

**⚠️ 重要声明**：本项目仅供技术研究和学习交流使用，严禁用于任何违法违规行为（如破坏游戏公平性、违反游戏服务条款等）。作者不承担任何因滥用本项目而产生的法律责任。

### 🎯 核心功能

项目集成了多种先进的 AI 功能，用于技术研究与学习交流：

| 功能 | 描述 |
|------|------|
| **自动瞄准（Auto Aim）** | 通过 YOLO 模型实时检测目标位置，结合 Kalman/EKF 滤波算法进行平滑跟踪，实现精准的目标锁定 |
| **自动扳机（Auto Trigger）** | 当检测到目标进入指定区域时，自动触发鼠标点击操作，实现快速响应 |
| **目标跟踪（Target Tracking）** | 基于扩展卡尔曼滤波（EKF）的目标跟踪算法，预测目标运动轨迹，补偿系统延迟 |
| **区域过滤（Region Filter）** | 支持自定义检测区域，只在指定范围内进行目标识别和跟踪 |
| **多目标选择（Multi-Target Selection）** | 智能选择最近/最优目标，支持优先级配置和目标切换策略 |
| **参数调优（Parameter Tuning）** | 可视化参数调整界面，实时预览检测效果，灵活配置各项参数 |

### 🎮 游戏支持

当前默认提供 **Valorant（无畏契约）** 预训练模型（320×320 分辨率），开箱即可使用。通过更换不同的训练模型，本引擎同样支持多种游戏：

- ✅ **Valorant（无畏契约）** - 默认支持，内置预训练模型
- 🎯 **CS2** - 更换对应模型即可支持
- 🎯 **三角洲行动** - 更换对应模型即可支持
- 🎯 **其他 FPS 游戏** - 只需训练或导入适配的 YOLO 模型

**🔄 模型切换**：只需将训练好的 ONNX 模型文件放入 `vendor_models/` 目录，并修改配置文件中的模型路径即可切换到不同游戏。

**⚠️ 重要声明**：本项目仅供技术研究和学习交流使用，严禁用于任何违法违规行为。作者不承担任何因滥用本项目而产生的法律责任。

### ✨ 核心特性

- **🚀 毫秒级推理**：基于 ONNX Runtime，支持 TensorRT/CUDA/CPU 多后端加速，单帧推理时间低于 1 毫秒
- **🎯 精准目标检测**：YOLO 模型集成，支持多种目标类别识别与定位
- **🔄 智能跟踪算法**：EKF/Kalman 滤波，实现平滑稳定的目标跟踪与锁定
- **⚡ 实时控制**：基于 Arduino Leonardo 开发板实现硬件级鼠标控制，支持低延迟移动和点击操作
- **🖥️ 原生 GUI**：Windows 原生桌面应用，中文友好界面，参数可视化调整
- **🛠️ 参数调优**：可视化参数调整器，实时预览推理效果
- **📊 环境诊断**：自动检测 CUDA/cuDNN/TensorRT 环境配置与兼容性
- **🔒 模块化设计**：清晰的架构设计，易于扩展和二次开发

### 📁 项目结构

```
VisionForge/
├── src/              # 核心源代码
│   ├── onnx_yolo_detector.py   # YOLO 检测器
│   ├── target_lock.py          # 目标锁定模块
│   ├── tracker.py              # 跟踪器
│   ├── control_gate.py         # 控制门限
│   ├── runtime_controller.py   # 运行时控制器
│   └── leonardo_driver.py      # Leonardo 硬件驱动
├── tools/            # 工具脚本
│   ├── config_tuner_gui.py     # 参数调优 GUI
│   └── env_diagnostics.py      # 环境诊断工具
├── scripts/          # 运行脚本
├── vendor_models/    # 预训练模型（内置 Valorant 320×320）
├── assets/           # 资源文件
└── firmware/         # 硬件固件
    └── leonardo_mouse_hid/     # Arduino Leonardo HID 固件
```

---

## 🚀 快速开始

### 🔧 环境要求

- **操作系统**：Windows 10/11 (64-bit)
- **Python**：3.10+
- **NVIDIA GPU**（推荐，支持 TensorRT/CUDA 加速）
- **CUDA**：12.x + cuDNN 9.x（GPU 加速必需）

### 📋 测试环境

本项目仅在以下环境进行过测试验证：

| 项目 | 版本/型号 |
|------|-----------|
| 操作系统 | Windows 11 专业版 (10.0.26100) |
| GPU | NVIDIA GeForce RTX 3050 (4GB) |
| NVIDIA 驱动 | 591.74 |
| CUDA | 13.1 |
| Python | 3.10+ |

> ⚠️ **注意**：以上为开发者测试环境，其他配置可能存在兼容性问题，欢迎反馈测试结果！

### 🛠️ 驱动安装

要使用 GPU 加速功能，需要安装以下依赖：

#### 1. NVIDIA 显卡驱动

从 [NVIDIA 官网](https://www.nvidia.com/Download/index.aspx) 下载并安装最新驱动，或使用 GeForce Experience 自动更新。

#### 2. CUDA Toolkit

下载地址：[CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-toolkit-archive)

安装步骤：
1. 下载对应版本的 CUDA Toolkit（推荐 12.1 或 12.2）
2. 运行安装程序，选择"Custom"安装
3. 确保勾选"CUDA Runtime"、"CUDA Samples"等组件
4. 安装完成后，确认系统环境变量 `CUDA_PATH` 已设置

#### 3. cuDNN

下载地址：[cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

安装步骤：
1. 下载与 CUDA 版本匹配的 cuDNN（如 CUDA 12.1 对应 cuDNN 9.x）
2. 解压压缩包
3. 将解压后的 `bin`、`include`、`lib` 目录下的文件复制到 CUDA 安装目录对应文件夹中
   - `bin/*.dll` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\`
   - `include/*.h` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\`
   - `lib/x64/*.lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\`

#### 4. TensorRT（可选，推荐）

下载地址：[TensorRT](https://developer.nvidia.com/tensorrt)

安装步骤：
1. 下载与 CUDA/cuDNN 版本匹配的 TensorRT
2. 解压压缩包
3. 将 `lib` 目录添加到系统环境变量 `PATH` 中

#### 5. ONNX Runtime GPU

项目依赖 `onnxruntime-gpu`，安装命令：

```bash
pip install onnxruntime-gpu==1.20.1
```

> 💡 **提示**：如果安装失败或 GPU 不可用，可改用 CPU 版本：
> ```bash
> pip install onnxruntime==1.20.1
> ```

### 📥 安装步骤

```bash
# 克隆仓库
git clone https://github.com/SuperZhao666/VisionForge.git
cd VisionForge

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### ▶️ 运行方式

#### 方式一：桌面 GUI（推荐）

```bash
scripts\run_desktop_gui.bat
```

#### 方式二：实时控制

```bash
scripts\run_realtime_control.bat
```

#### 方式三：预览模式

```bash
scripts\run_realtime_preview.bat
```

### 🔧 硬件配置（实时控制模式）

若要使用实时控制功能（自动鼠标移动），需要配置 Arduino Leonardo 开发板：

#### 硬件要求

- **Arduino Leonardo** / **Pro Micro (ATmega32U4)** / 兼容板
- USB 数据线

#### 购买渠道

开发板可以在各大网购平台购买，价格非常实惠（约 **30 元左右**）：

- 🛒 **淘宝** - 搜索"Arduino Leonardo"或"Pro Micro"
- 🛒 **拼多多** - 搜索"Leonardo R3 开发板"，性价比更高
- 🛒 **京东** - 搜索"Arduino Leonardo"，品质有保障

**推荐型号**：Leonardo R3 开发板（基于 ATMEGA32U4 芯片）

#### 固件烧录

1. 打开 Arduino IDE，加载 `firmware/leonardo_mouse_hid/leonardo_mouse_hid.ino`
2. 选择正确的开发板和端口
3. 上传固件到开发板

#### 串口协议

固件使用自定义串口协议与 PC 通信（波特率 115200）：

```
数据包格式：[0xAA, cmd, dx, dy, checksum]
checksum = (cmd + dx + dy) & 0xFF
dx/dy: int8_t，相对移动范围 -127~127

命令列表：
- 0x01: 移动鼠标
- 0x02: 按下左键
- 0x03: 释放左键
- 0x04: 点击左键
- 0x05: 移动并按下左键
- 0x06: 移动并释放左键
- 0xFF: 心跳检测，返回 0xBB
```

#### 安全机制

- **2秒自动释放**：若超过 2 秒未收到指令，自动释放左键，防止卡死
- **校验和验证**：每个数据包都包含校验和，确保传输完整性

---

## 📦 发布版本

我们提供预编译的 Windows 单文件可执行程序，无需安装 Python 环境即可运行。

### 📥 下载地址

最新发布版本请访问 [GitHub Releases](https://github.com/SuperZhao666/VisionForge/releases)

### 📝 更新日志

详细的更新记录请查看项目发布页面。

---

## 🤝 贡献指南

欢迎各种形式的贡献！无论是代码提交、问题报告还是功能建议，我们都非常感谢。

### 🐛 报告问题

如果您发现任何 Bug 或有改进建议，请在 [Issues](https://github.com/SuperZhao666/VisionForge/issues) 页面提交。

提交问题时请包含：
- 问题描述
- 复现步骤
- 环境信息（Windows 版本、GPU 型号、Python 版本）
- 错误日志（如有）

### 🔧 代码贡献

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建 Pull Request

---

## 💬 社区交流

加入我们的社区，获取最新动态和技术支持！

| 平台 | 链接 |
|------|------|
| QQ 群 | 1044257667 |
| Telegram | [https://t.me/+OdV-iEskI1MzNDll](https://t.me/+OdV-iEskI1MzNDll) |

### 🎁 免费体验

加入社区即可获得 **免费的一周试用许可**，欢迎大家来测试！

---

## 📜 许可证

本项目采用 **MIT License** 开源许可证，详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢以下开源项目和技术：

- [ONNX Runtime](https://onnxruntime.ai/) - 高性能推理引擎
- [YOLO](https://github.com/ultralytics/yolov5) - 目标检测模型
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [NumPy](https://numpy.org/) - 数值计算库
- [PyInstaller](https://pyinstaller.org/) - 打包工具

---

<div align="center">
  <p>⭐ 如果这个项目对您有帮助，请给我们一个 Star！</p>
</div>