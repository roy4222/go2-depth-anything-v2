#!/bin/bash
# Perception Server V2 啟動腳本
# YOLO-World + Depth Anything V2 融合感知服務

echo "🚀 啟動 Perception Server V2..."
echo "📍 位置: $(pwd)"

# 啟動 conda 環境
source ~/miniconda3/bin/activate depth-v2

# 啟動服務
echo "🌐 Port: 8001"
uvicorn perception_server_v2:app --host 0.0.0.0 --port 8001
