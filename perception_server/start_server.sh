#!/bin/bash
# Perception Server 啟動腳本

echo "🚀 啟動 Perception Server..."

# 確認在正確目錄
cd "$(dirname "$0")"

# 啟動 uvicorn
uvicorn perception_server:app --host 0.0.0.0 --port 8000 --reload

echo "💡 Server 已停止"
