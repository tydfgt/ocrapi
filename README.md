# ⚡ ocrplus — 焊机参数 OCR 智能识别系统

> **ESP32 拍照 → Jetson Orin Nano OCR → 结构化焊机参数** | PP-OCRv5 Mobile | FastAPI + C++ | GPU FP32 ~5.5s/图

<p align="center">
  <a href="https://tydfgt.github.io/ocrapi"><img src="https://img.shields.io/badge/GitHub%20Pages-在线文档-blue" alt="pages"></a>
  <img src="https://img.shields.io/badge/platform-Jetson%20Orin%20Nano-green" alt="platform">
  <img src="https://img.shields.io/badge/JetPack-6.0-blue" alt="jetpack">
  <img src="https://img.shields.io/badge/CUDA-12.6.68-brightgreen" alt="cuda">
  <img src="https://img.shields.io/badge/TensorRT-10.3.0.30-orange" alt="tensorrt">
  <img src="https://img.shields.io/badge/license-Apache%202.0-lightgrey" alt="license">
</p>

> 📖 **GitHub Pages**: [https://tydfgt.github.io/ocrapi](https://tydfgt.github.io/ocrapi)

---

## 项目简介

**焊机参数 OCR 智能识别系统** — 在焊接作业中，通过 ESP32 + 摄像头实时采集焊机表头图像，WiFi 回传至 Jetson Orin Nano 边端 OCR API，自动识别电流、电压、焊接速度等参数并结构化输出。

```
ESP32-CAM → WiFi → HTTP POST /ocr → Jetson Orin Nano → JSON 焊机参数
```

基于 **PaddleOCR PP-OCRv5 Mobile** 模型，在 **NVIDIA Jetson Orin Nano Super (8GB)** 上实现 C++ 高性能推理。

- **推理引擎**: Paddle Inference C++ (GPU FP32 / TensorRT FP16)
- **模型**: PP-OCRv5 Mobile — 检测 4.8MB + 识别 17MB（超轻量）
- **API 服务**: FastAPI + uvicorn，端口 8899
- **前端**: 纯 HTML5 暗色主题，拖拽/粘贴上传，硬件监控面板
- **部署教程**: 完整从零部署指南，含 TensorRT 源码修改

---

## 硬件平台

| 项目 | 参数 |
|------|------|
| 设备 | NVIDIA Jetson Orin Nano Super (P3767-0005) |
| CPU | 6核 ARM Cortex-A78AE @ 1.5GHz |
| GPU | Ampere GA10B, 1024 CUDA cores, SM 8.7 |
| 内存 | 8GB LPDDR5（CPU/GPU 统一内存） |
| 存储 | 238GB NVMe SSD |

---

## 软件栈

| 组件 | 版本 |
|------|------|
| JetPack / L4T | 6.0 / R36.5.0 |
| CUDA | 12.6.68 |
| cuDNN | 9.3.0 |
| TensorRT | 10.3.0.30 |
| OpenCV | 4.10.0（自编译 CUDA 加速） |
| GCC | 11.4.0 |
| CMake | 3.26.4 |
| Paddle Inference | 3.0.0 (JetPack 5.1.2 预编译) |
| Python | 3.10.12 |
| FastAPI | 0.136.3 |

---

## 性能基准

> 测试图片: 612×408 PNG, 含 12 行中文

| 模式 | 平均耗时 | 说明 |
|------|---------|------|
| GPU FP32 (Paddle) | **~5.55s** | 默认模式，最稳定 |
| GPU FP16 (Paddle) | ~5.61s | 与 FP32 基本持平 |
| TRT FP16 | ~5.60s | Paddle TRT 8.5 预编译 vs 系统 TRT 10.3 版本差异 |
| CPU | ~18s | 无 MKL/MKLDNN，不推荐 |

---

## 快速开始

### 1. 环境准备

```bash
# 激活 Python 虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 一键启动（自动激活 venv）
bash start.sh

# 或手动启动
source venv/bin/activate && python server.py
```

服务启动后访问: **http://[设备IP]:8899**

### 3. API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | Web 前端界面 |
| `POST` | `/ocr` | 上传图片文件，返回 OCR 结果 |
| `POST` | `/ocr_base64` | Base64 图片识别 |
| `GET` | `/status` | 实时硬件状态（GPU/CPU/RAM/温度/功耗） |
| `GET` | `/health` | 健康检查 |

### 4. ESP32 端调用

ESP32-CAM 拍照后通过 HTTP 上传到 OCR API：

```cpp
// ESP32 Arduino 示例
HTTPClient http;
http.begin("http://192.168.3.110:8899/ocr");
http.addHeader("Content-Type", "image/jpeg");

camera_fb_t *fb = esp_camera_fb_get();
int code = http.POST(fb->buf, fb->len);
esp_camera_fb_return(fb);

String result = http.getString();  // JSON OCR 结果
```

### 5. 压力测试

```bash
# 10 并发，总共 50 个请求
python stress_test.py 10 50

# 默认: 5 并发，20 个请求
python stress_test.py
```

---

## 项目结构

```
ocrcplus/
├── server.py                  # FastAPI 主服务 (310行)
├── start.sh                   # 一键启动脚本
├── stress_test.py             # 并发压力测试工具
├── requirements.txt           # Python 依赖
├── .gitignore
├── docs/
│   └── index.html             # GitHub Pages 项目主页
├── static/
│   └── index.html             # Web 前端 (暗色主题 + 硬件监控)
├── models/
│   ├── PP-OCRv5_mobile_det_infer/   # 文本检测模型 (4.8MB)
│   ├── PP-OCRv5_mobile_rec_infer/   # 文本识别模型 (17MB)
│   └── test.png                     # 测试图片
├── output/                    # OCR 结果输出目录
├── PaddleOCR/                 # PaddleOCR 源码
│   └── deploy/cpp_infer/
│       ├── src/               # C++ 推理源码 (含 TRT 修改)
│       ├── build/             # 标准构建 (Paddle GPU)
│       ├── build_trt/         # TRT 构建
│       └── cli.cc             # 命令行入口
├── paddle_inference_install_dir/  # Paddle 推理库
├── 软著模板/                   # 软件著作权申请材料
│   ├── 01-软件著作权登记申请表.md
│   ├── 02-软件说明书（用户手册）.md
│   ├── 03-源代码文档说明.md
│   ├── 04-软件设计说明书.md
│   ├── 05-申请材料清单.md
│   ├── extract_sources.py     # 源代码提取脚本
│   ├── screenshots/           # 截图素材
│   └── latex/                 # LaTeX 版本
└── Jetson_Orin_Nano_PaddleOCR_C++_部署教程.md  # 完整部署指南
```

---

## C++ 推理命令行

```bash
# GPU FP32 (默认)
./build/ppocr \
  --image_dir=../models/test.png \
  --device=gpu \
  --det_model_dir=../models/PP-OCRv5_mobile_det_infer \
  --rec_model_dir=../models/PP-OCRv5_mobile_rec_infer \
  --text_detection_model_name=PP-OCRv5_mobile_det

# TensorRT FP16
./build_trt/ppocr \
  --image_dir=../models/test.png \
  --device=gpu \
  --run_mode=trt_fp16 \
  --det_model_dir=../models/PP-OCRv5_mobile_det_infer \
  --rec_model_dir=../models/PP-OCRv5_mobile_rec_infer \
  --text_detection_model_name=PP-OCRv5_mobile_det
```

---

## TensorRT 支持

在 PaddleOCR C++ 推理源码中新增了 TensorRT 运行模式：

| 文件 | 修改内容 |
|------|---------|
| `pp_option.h` | 新增 `paddle_trt_fp32` / `paddle_trt_fp16` 枚举 |
| `base_predictor.cc` | 添加 `trt_fp16` → `SetRunMode("paddle_trt_fp16")` 映射 |
| `static_infer.cc` | 添加 `EnableTensorRtEngine()` 配置（1GB workspace, 动态 batch） |
| `args.cc` | 更新 `--run_mode` 帮助文本 |

详细教程见: [Jetson_Orin_Nano_PaddleOCR_C++_部署教程.md](Jetson_Orin_Nano_PaddleOCR_C++_部署教程.md)

---

## 软著申请

本项目包含完整的 **中国软件著作权申请材料**（Markdown + LaTeX），见 [`软著模板/`](软著模板/) 目录。

---

## License

本项目基于 [Apache License 2.0](LICENSE)。PaddleOCR 模型和推理库版权归百度所有。

---

*最后更新: 2026年6月2日*
