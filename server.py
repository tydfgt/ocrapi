"""
PaddleOCR FastAPI Server - Jetson Orin Nano
端口: 8899
用法: source venv/bin/activate && python server.py
"""
import base64
import json
import os
import re
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

BASE_DIR = Path(__file__).parent
PPOCR_BIN = str(BASE_DIR / "PaddleOCR/deploy/cpp_infer/build/ppocr")
MODELS_DIR = str(BASE_DIR / "models")
OUTPUT_DIR = str(BASE_DIR / "output")
STATIC_DIR = BASE_DIR / "static"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ---- jtop 全局实例 (后台持续采集) ----
_jtop = None
_jtop_stats_cache = {}
_jtop_lock = threading.Lock()

def _jtop_thread():
    """后台线程持续从 jtop 读取 Jetson 状态"""
    global _jtop, _jtop_stats_cache
    try:
        from jtop import jtop
        with jtop() as jetson:
            _jtop = jetson
            while True:
                time.sleep(1)
                with _jtop_lock:
                    _jtop_stats_cache = jetson.stats.copy() if jetson.stats else {}
    except Exception as e:
        print(f"⚠️ jtop 后台线程异常: {e}")

# ---- 机器状态采集 ----
DEVICE_INFO = {
    "model": "NVIDIA Jetson Orin Nano Super",
    "module": "P3767-0005",
    "soc": "tegra234",
    "cpu": "6× ARM Cortex-A78AE",
    "gpu": "Ampere GA10B | 1024 CUDA cores | SM 8.7",
    "memory": "8GB LPDDR5 (CPU/GPU 统一内存)",
    "l4t": "",
    "jetpack": "6.0",
    "cuda": "12.6",
    "cudnn": "9.3",
    "tensorrt": "10.3",
    "opencv": "4.10.0 (CUDA)",
}

def _read_l4t():
    try: return Path("/etc/nv_tegra_release").read_text().strip().split("\n")[0].replace("# ", "")
    except: return "N/A"

# ---- FastAPI lifespan: 启动 jtop + 预热 ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动 jtop 后台线程
    t = threading.Thread(target=_jtop_thread, daemon=True, name="jtop")
    t.start()
    time.sleep(2)  # 等 jtop 采集第一帧
    print("✅ jtop 状态采集已启动")

    # 预热 GPU
    warmup_img = BASE_DIR / "models" / "test.png"
    if warmup_img.exists():
        print("🔥 GPU 预热中（首次推理，约 8s）...")
        try:
            run_ocr(str(warmup_img), "warmup")
            print("✅ GPU 预热完成！模型已加载，后续请求 ~5.5s/图")
        except Exception as e:
            print(f"⚠️ 预热失败: {e}")

    yield  # 服务运行中
    # shutdown 清理
    print("🛑 服务关闭")

app = FastAPI(title="PaddleOCR API", version="1.0.0", docs_url="/docs", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/status")
async def status():
    """机器状态 + API 信息（来自 jtop）"""
    DEVICE_INFO["l4t"] = _read_l4t()

    with _jtop_lock:
        s = _jtop_stats_cache.copy() if _jtop_stats_cache else {}

    # 解析 jtop 数据
    cpu_cores = []
    for i in range(1, 7):
        v = s.get(f"CPU{i}")
        if v is not None:
            cpu_cores.append(v)
    cpu_percent = round(sum(cpu_cores) / len(cpu_cores), 1) if cpu_cores else 0
    gpu_percent = round(s.get("GPU", 0), 1)
    ram_frac = s.get("RAM", 0)
    swap_frac = s.get("SWAP", 0)
    ram_total_mb = 7607  # Orin Nano 8GB
    ram_used_mb = round(ram_total_mb * ram_frac)
    swap_total_mb = 11264  # 11GB
    swap_used_mb = round(swap_total_mb * swap_frac)

    temps = {}
    for k, v in s.items():
        if k.startswith("Temp "):
            temps[k.replace("Temp ", "")] = round(v, 1)

    power = {}
    for k, v in s.items():
        if k.startswith("Power "):
            power[k.replace("Power ", "")] = round(v, 1)

    disk = psutil.disk_usage("/")

    return {
        "device": DEVICE_INFO,
        "realtime": {
            "cpu_percent": cpu_percent,
            "cpu_cores": cpu_cores,
            "gpu_percent": gpu_percent,
            "ram_used_mb": ram_used_mb,
            "ram_total_mb": ram_total_mb,
            "ram_percent": round(ram_frac * 100, 1),
            "swap_used_mb": swap_used_mb,
            "swap_total_mb": swap_total_mb,
            "swap_percent": round(swap_frac * 100, 1),
            "disk_used_gb": round(disk.used / (1024**3), 1),
            "disk_total_gb": round(disk.total / (1024**3), 1),
            "disk_percent": round(disk.percent, 1),
            "temps_c": temps,
            "power_mw": power,
            "fan_pwm": s.get("Fan pwmfan0", "N/A"),
            "jetson_clocks": s.get("jetson_clocks", "N/A"),
            "nvp_model": s.get("nvp model", "N/A"),
            "_source": "jtop",
        },
        "api": {
            "ppocr_bin": PPOCR_BIN,
            "det_model": "PP-OCRv5_mobile_det",
            "rec_model": "PP-OCRv5_mobile_rec",
            "device": "gpu",
            "precision": "fp32",
            "avg_time_s": 5.55,
        },
    }


def run_ocr(image_path: str, output_name: str) -> dict:
    """调用 C++ ppocr 二进制执行 OCR"""
    t0 = time.time()
    cmd = [
        PPOCR_BIN, "ocr",
        "--input", image_path,
        "--text_detection_model_dir", f"{MODELS_DIR}/PP-OCRv5_mobile_det_infer",
        "--text_detection_model_name", "PP-OCRv5_mobile_det",
        "--text_recognition_model_dir", f"{MODELS_DIR}/PP-OCRv5_mobile_rec_infer",
        "--text_recognition_model_name", "PP-OCRv5_mobile_rec",
        "--use_doc_orientation_classify", "false",
        "--use_doc_unwarping", "false",
        "--use_textline_orientation", "false",
        "--device", "gpu",
        "--save_path", OUTPUT_DIR,
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        f"{BASE_DIR}/paddle_inference_install_dir/paddle/lib:"
        f"{env.get('LD_LIBRARY_PATH', '')}"
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
    t1 = time.time()

    # 查找生成的 JSON 文件
    json_files = sorted(
        Path(OUTPUT_DIR).glob("*_res.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    img_files = sorted(
        Path(OUTPUT_DIR).glob("*_res_img.png"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )

    ocr_data = {}
    vis_base64 = ""

    if json_files:
        with open(json_files[0], "r") as f:
            ocr_data = json.load(f)

    if img_files:
        with open(img_files[0], "rb") as f:
            vis_base64 = base64.b64encode(f.read()).decode()

    texts = ocr_data.get("rec_texts", [])
    scores = ocr_data.get("rec_scores", [])

    return {
        "texts": texts,
        "scores": scores,
        "count": len(texts),
        "time_s": round(t1 - t0, 3),
        "json": ocr_data,
        "vis_base64": vis_base64,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """前端页面"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>PaddleOCR API Running</h1><p>Please visit <a href='/docs'>/docs</a></p>"


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    OCR 识别接口
    POST /ocr  上传图片，返回识别结果
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "只支持图片文件"}, status_code=400)

    # 保存上传的图片
    suffix = Path(file.filename).suffix or ".png"
    tmp_path = Path(OUTPUT_DIR) / f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    content = await file.read()
    tmp_path.write_bytes(content)

    try:
        output_name = tmp_path.stem
        result = run_ocr(str(tmp_path), output_name)
        return JSONResponse(result)
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "推理超时 (60s)"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # 清理临时文件
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/ocr_base64")
async def ocr_base64(data: dict):
    """
    Base64 图片 OCR 接口
    POST /ocr_base64  {"image": "base64字符串"}
    """
    try:
        img_b64 = data.get("image", "")
        if not img_b64:
            return JSONResponse({"error": "缺少 image 字段"}, status_code=400)

        img_bytes = base64.b64decode(img_b64)
        tmp_path = Path(OUTPUT_DIR) / f"b64_{uuid.uuid4().hex[:8]}.png"
        tmp_path.write_bytes(img_bytes)

        try:
            result = run_ocr(str(tmp_path), tmp_path.stem)
            return JSONResponse(result)
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "ppocr_bin": PPOCR_BIN, "models": MODELS_DIR}


if __name__ == "__main__":
    import uvicorn
    print(f"🚀 PaddleOCR API 启动: http://0.0.0.0:8899")
    print(f"📄 API 文档: http://0.0.0.0:8899/docs")
    uvicorn.run(app, host="0.0.0.0", port=8899, log_level="info")
