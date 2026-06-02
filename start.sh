#!/bin/bash
# PaddleOCR API 一键启动脚本
# 用法: ./start.sh

cd "$(dirname "$0")"

# 激活虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ venv 不存在，请先创建: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate

# 检查依赖
python3 -c "import fastapi, uvicorn, psutil" 2>/dev/null || {
    echo "📦 安装依赖..."
    pip install fastapi uvicorn python-multipart Pillow aiofiles psutil jetson-stats -q
}

echo "🚀 启动 PaddleOCR API (端口 8899)..."
echo "📄 前端页面: http://$(hostname -I | awk '{print $1}'):8899"
echo "📄 API 文档: http://$(hostname -I | awk '{print $1}'):8899/docs"
echo ""

python server.py
