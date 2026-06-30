# VisionForge

<div align="center">
  <img src="assets/app_icon.png" alt="VisionForge Logo" width="128" height="128">
  <h1>⚡ VisionForge</h1>
  <p>AI Auto Aim · Auto Trigger · Shooting Assistant</p>
  
  [![GitHub Stars](https://img.shields.io/github/stars/SuperZhao666/VisionForge?style=social)](https://github.com/SuperZhao666/VisionForge)
  [![GitHub License](https://img.shields.io/github/license/SuperZhao666/VisionForge)](LICENSE)
  [![GitHub Issues](https://img.shields.io/github/issues/SuperZhao666/VisionForge)](https://github.com/SuperZhao666/VisionForge/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/SuperZhao666/VisionForge)](https://github.com/SuperZhao666/VisionForge/pulls)
  
  [![QQ Group](https://img.shields.io/badge/QQ%20Group-1044257667-blue)](https://qm.qq.com/cgi-bin/qm/qr?k=U97K77g4eE9pU97K77g4eE9p&jump_from=webapi)
  [![Telegram](https://img.shields.io/badge/Telegram-Join%20Group-blue)](https://t.me/+OdV-iEskI1MzNDll)
  
  [中文文档](README.md) | **English**
</div>

---

## 📋 Overview

VisionForge is a **AI Auto Aim & Auto Trigger Shooting Assistant** built with Python, using YOLO models for target detection with millisecond-level inference speed (single frame inference time < 1ms). Through hardware-level mouse control (Arduino Leonardo), it achieves auto aim, auto trigger, target tracking, and more.

### 🎬 Demo Video

https://github.com/SuperZhao666/VisionForge/assets/assets/demo.mp4

<video src="https://raw.githubusercontent.com/SuperZhao666/VisionForge/main/assets/demo.mp4" width="800" controls autoplay loop muted></video>

📺 **More Demo Videos**:
- [VisionForge Valorant AI Game Assistant - Long-term Stable Version](https://b23.tv/6I90Dpq) (Bilibili)
- [VisionForge High-end AI Game Assistant Tool](https://v.douyin.com/qcqneULylOs/) (Douyin/TikTok)

**⚠️ Disclaimer**: This project is for technical research and educational purposes only. Do NOT use it for any illegal activities (e.g., disrupting game fairness, violating game Terms of Service). The author is NOT responsible for any consequences resulting from misuse.

### 🎯 Core Features

| Feature | Description |
|---------|-------------|
| **Auto Aim** | Real-time target detection via YOLO model, combined with Kalman/EKF filtering for smooth tracking and precise target locking |
| **Auto Trigger** | Automatically triggers mouse click when target enters specified region, achieving rapid response |
| **Target Tracking** | Extended Kalman Filter (EKF) based tracking algorithm, predicts target trajectory and compensates system latency |
| **Region Filter** | Custom detection regions, target recognition and tracking only within specified areas |
| **Multi-Target Selection** | Intelligent selection of nearest/optimal target, supports priority configuration and target switching strategies |
| **Parameter Tuning** | Visual parameter adjustment interface, real-time preview of detection results, flexible configuration |

### 🎮 Game Support

Currently provides **Valorant** pre-trained model (320×320 resolution) by default. By replacing training models, this engine supports multiple games:

- ✅ **Valorant** - Default support, built-in pre-trained model
- 🎯 **CS2** - Replace corresponding model
- 🎯 **Delta Force** - Replace corresponding model
- 🎯 **Other FPS Games** - Train or import compatible YOLO models

**🔄 Model Switching**: Simply place trained ONNX model files in `vendor_models/` directory and modify model path in config file.

### ✨ Key Highlights

- **🚀 Millisecond Inference**: ONNX Runtime backend, supports TensorRT/CUDA/CPU acceleration, < 1ms per frame
- **🎯 Precise Detection**: YOLO model integration, multi-class target recognition and localization
- **🔄 Smart Tracking**: EKF/Kalman filtering for smooth, stable target tracking and locking
- **⚡ Real-time Control**: Arduino Leonardo hardware-level mouse control, low-latency movement and clicks
- **🖥️ Native GUI**: Windows native desktop app, user-friendly interface, visual parameter adjustment
- **🛠️ Parameter Tuning**: Visual tuner for real-time inference preview
- **📊 Environment Diagnostics**: Auto-detect CUDA/cuDNN/TensorRT configuration and compatibility
- **🔒 Modular Design**: Clean architecture, easy to extend and customize

---

## 🚀 Quick Start

### 🔧 Requirements

- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.10+
- **NVIDIA GPU** (recommended, TensorRT/CUDA acceleration)
- **CUDA**: 12.x + cuDNN 9.x (required for GPU acceleration)

### 📋 Test Environment

This project has only been tested on the following configuration:

| Component | Version/Model |
|-----------|---------------|
| OS | Windows 11 Pro (10.0.26100) |
| GPU | NVIDIA GeForce RTX 3050 (4GB) |
| NVIDIA Driver | 591.74 |
| CUDA | 13.1 |
| Python | 3.10+ |

> ⚠️ **Note**: Above is developer's test environment. Other configurations may have compatibility issues. Feedback welcome!

### 🛠️ Driver Installation

For GPU acceleration, install the following dependencies:

#### 1. NVIDIA Graphics Driver

Download from [NVIDIA Official Site](https://www.nvidia.com/Download/index.aspx) or use GeForce Experience.

#### 2. CUDA Toolkit

Download: [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-toolkit-archive)

Steps:
1. Download CUDA Toolkit (recommend 12.1 or 12.2)
2. Run installer, select "Custom" installation
3. Ensure "CUDA Runtime", "CUDA Samples" components are checked
4. After installation, verify `CUDA_PATH` environment variable

#### 3. cuDNN

Download: [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

Steps:
1. Download cuDNN matching CUDA version (e.g., CUDA 12.1 → cuDNN 9.x)
2. Extract archive
3. Copy files from `bin`, `include`, `lib` directories to CUDA installation directory:
   - `bin/*.dll` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\`
   - `include/*.h` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\`
   - `lib/x64/*.lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\`

#### 4. TensorRT (Optional, Recommended)

Download: [TensorRT](https://developer.nvidia.com/tensorrt)

Steps:
1. Download TensorRT matching CUDA/cuDNN version
2. Extract archive
3. Add `lib` directory to system `PATH` environment variable

#### 5. ONNX Runtime GPU

Project depends on `onnxruntime-gpu`:

```bash
pip install onnxruntime-gpu==1.20.1
```

> 💡 **Tip**: If installation fails or GPU unavailable, use CPU version:
> ```bash
> pip install onnxruntime==1.20.1
> ```

### 📥 Installation

```bash
# Clone repository
git clone https://github.com/SuperZhao666/VisionForge.git
cd VisionForge

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### ▶️ Usage

#### Option 1: Desktop GUI (Recommended)

```bash
scripts\run_desktop_gui.bat
```

#### Option 2: Real-time Control

```bash
scripts\run_realtime_control.bat
```

#### Option 3: Preview Mode

```bash
scripts\run_realtime_preview.bat
```

### 🔧 Hardware Configuration (Real-time Control)

For real-time control (auto mouse movement), configure Arduino Leonardo:

#### Hardware Requirements

- **Arduino Leonardo** / **Pro Micro (ATmega32U4)** / Compatible board
- USB cable

#### Purchase Channels

Available on major online platforms (~30 CNY):

- 🛒 **Taobao** - Search "Arduino Leonardo" or "Pro Micro"
- 🛒 **JD.com** - Search "Arduino Leonardo", quality guaranteed
- 🛒 **AliExpress/Amazon** - International shipping available

**Recommended**: Leonardo R3 board (ATMEGA32U4 chip)

#### Firmware Upload

1. Open Arduino IDE, load `firmware/leonardo_mouse_hid/leonardo_mouse_hid.ino`
2. Select correct board and port
3. Upload firmware

#### Serial Protocol

Custom serial protocol (baud rate 115200):

```
Packet format: [0xAA, cmd, dx, dy, checksum]
checksum = (cmd + dx + dy) & 0xFF
dx/dy: int8_t, range -127~127

Commands:
- 0x01: Move mouse
- 0x02: Press left button
- 0x03: Release left button
- 0x04: Click left button
- 0x05: Move + Press left button
- 0x06: Move + Release left button
- 0xFF: Heartbeat, returns 0xBB
```

#### Safety Mechanisms

- **2-second auto release**: Releases left button if no command received for 2 seconds
- **Checksum validation**: Every packet includes checksum for data integrity

---

## 📦 Releases

Pre-compiled Windows executables available, no Python environment required.

### 📥 Download

Visit [GitHub Releases](https://github.com/SuperZhao666/VisionForge/releases)

---

## 🤝 Contributing

Contributions welcome! Code submissions, bug reports, feature suggestions all appreciated.

### 🐛 Bug Reports

Submit issues at [GitHub Issues](https://github.com/SuperZhao666/VisionForge/issues)

Include:
- Problem description
- Reproduction steps
- Environment info (Windows version, GPU model, Python version)
- Error logs (if available)

### 🔧 Code Contributions

1. Fork this repository
2. Create feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add some feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Create Pull Request

---

## 💬 Community

Join our community for updates and support!

| Platform | Link |
|----------|------|
| QQ Group | 1044257667 |
| Telegram | [https://t.me/+OdV-iEskI1MzNDll](https://t.me/+OdV-iEskI1MzNDll) |

### 🎁 Free Trial

Join community for **free 1-week trial license**!

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

Thanks to:

- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference engine
- [YOLO](https://github.com/ultralytics/yolov5) - Object detection model
- [OpenCV](https://opencv.org/) - Computer vision library
- [NumPy](https://numpy.org/) - Numerical computing
- [PyInstaller](https://pyinstaller.org/) - Packaging tool

---

<div align="center">
  <p>⭐ If this project helps you, please give us a Star!</p>
</div>