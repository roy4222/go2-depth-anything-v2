"""
Perception Server V2 - YOLO-World + Depth Anything V2 融合
用途: 接收圖片，回傳物件偵測 + 深度估計 + 導航指令

API 端點:
- /perceive     - 深度估計與避障建議 (原有)
- /find_object  - 物件偵測 + 距離估計 + 導航指令 (新增)
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import sys
import os
import time

# 加入 Depth Anything V2 metric_depth 路徑
DA3_ROOT = os.path.expanduser("~/Depth_Anything_V2/Depth-Anything-V2")
sys.path.insert(0, os.path.join(DA3_ROOT, 'metric_depth'))
from depth_anything_v2.dpt import DepthAnythingV2

# YOLO-World
from ultralytics import YOLO

app = FastAPI(
    title="Perception Server V2",
    description="YOLO-World + DA3 融合感知 API - 為 Go2 機器狗提供物件偵測與導航",
    version="2.0.0"
)

# ======================
# 全域模型物件
# ======================
depth_model = None
yolo_model = None
device = None

# ======================
# 校正參數 (Go2 廣角鏡頭)
# ======================
SCALE_FACTOR = 0.60      # DA3 距離校正
IMAGE_WIDTH = 640        # 畫面寬度
LEFT_THRESHOLD = 213     # 左側區域 (< 1/3)
RIGHT_THRESHOLD = 427    # 右側區域 (> 2/3)


@app.on_event("startup")
async def load_models():
    """伺服器啟動時載入模型"""
    global depth_model, yolo_model, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 裝置: {device}")
    
    # === 載入 Depth Anything V2 ===
    print("🔄 正在載入 Depth Anything V2...")
    depth_model = DepthAnythingV2(
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        max_depth=20.0
    )
    
    checkpoint_path = os.path.join(
        DA3_ROOT, 
        'checkpoints', 
        'depth_anything_v2_metric_hypersim_vitl.pth'
    )
    
    if os.path.exists(checkpoint_path):
        depth_model.load_state_dict(
            torch.load(checkpoint_path, map_location=device), 
            strict=False
        )
        depth_model = depth_model.to(device).eval()
        print("✅ DA3 模型載入完成")
    else:
        print(f"⚠️ DA3 權重不存在: {checkpoint_path}")
    
    # === 載入 YOLO-World ===
    print("🔄 正在載入 YOLO-World...")
    yolo_path = os.path.join(os.path.dirname(__file__), 'yolov8l-worldv2.pt')
    
    if not os.path.exists(yolo_path):
        # 備用路徑
        yolo_path = "/home/roy422/Yoloworld/yolov8l-worldv2.pt"
    
    yolo_model = YOLO(yolo_path)
    yolo_model.to(device)
    yolo_model.set_classes(["person", "chair", "table", "obstacle"])  # 預設
    print("✅ YOLO-World 模型載入完成")
    
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"🚀 所有模型就緒 (VRAM: {vram:.2f} GB)")


def get_direction(cx: float) -> tuple[str, float]:
    """
    根據物件中心點判斷方向與角速度
    
    Returns:
        (direction, angular_z)
    """
    if cx < LEFT_THRESHOLD:
        return "左側", 0.3
    elif cx > RIGHT_THRESHOLD:
        return "右側", -0.3
    else:
        return "正前方", 0.0


def generate_cmd_vel(direction: str, distance: float, angular_z: float) -> dict:
    """
    產生 cmd_vel 指令
    
    距離越遠 linear.x 越大，太近則停止
    """
    if distance < 0.3:
        # 太近，停止
        return {"linear_x": 0.0, "angular_z": 0.0}
    elif distance < 1.0:
        # 靠近中，慢速
        return {"linear_x": 0.1, "angular_z": angular_z}
    else:
        # 正常距離
        return {"linear_x": 0.2, "angular_z": angular_z}


# ======================
# API: 健康檢查
# ======================
@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": "2.0.0",
        "models": {
            "depth": depth_model is not None,
            "yolo": yolo_model is not None
        },
        "device": str(device)
    }


# ======================
# API: 物件搜尋 + 深度 + 導航
# ======================
@app.post("/find_object")
async def find_object(
    image: UploadFile = File(...),
    target: str = Form(default="water bottle"),
    conf: float = Form(default=0.25)
):
    """
    融合 YOLO + DA3 的物件搜尋 API
    
    Args:
        image: 上傳的圖片
        target: 搜尋目標 (可用逗號分隔多個)
        conf: 信心度閾值
        
    Returns:
        JSON 包含偵測結果、距離、方向、cmd_vel
    """
    if yolo_model is None or depth_model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "模型尚未載入完成"}
        )
    
    start_time = time.time()
    
    try:
        # 1. 讀取圖片
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        raw_image = np.array(pil_image)
        
        # 2. 設定 YOLO 目標類別
        targets = [t.strip() for t in target.split(",")]
        yolo_model.set_classes(targets)
        
        # 3. YOLO 偵測
        yolo_start = time.time()
        results = yolo_model(pil_image, conf=conf)
        yolo_time = (time.time() - yolo_start) * 1000
        
        # 4. DA3 深度估計
        depth_start = time.time()
        with torch.no_grad():
            depth_map = depth_model.infer_image(raw_image)
        depth_time = (time.time() - depth_start) * 1000
        
        # 5. 融合結果
        response = {
            "found": False,
            "target": target,
            "results": [],
            "timing_ms": {
                "yolo": round(yolo_time, 1),
                "depth": round(depth_time, 1),
                "total": round((time.time() - start_time) * 1000, 1)
            }
        }
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # 座標
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # 類別與信心度
                cls_id = int(box.cls)
                label = yolo_model.names[cls_id]
                score = float(box.conf)
                
                # 從深度圖取得距離 (注意: depth_map[y, x] 不是 [x, y])
                if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                    raw_distance = float(depth_map[cy, cx])
                    distance = raw_distance * SCALE_FACTOR
                else:
                    distance = -1  # 無法取得
                
                # 方向與導航指令
                direction, angular_z = get_direction(cx)
                cmd_vel = generate_cmd_vel(direction, distance, angular_z)
                
                response["results"].append({
                    "label": label,
                    "confidence": round(score, 3),
                    "distance_m": round(distance, 2),
                    "direction": direction,
                    "center": [cx, cy],
                    "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                    "cmd_vel": cmd_vel
                })
            
            if len(response["results"]) > 0:
                response["found"] = True
                # 按距離排序，最近的優先
                response["results"].sort(key=lambda x: x["distance_m"])
        
        return response
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"處理失敗: {str(e)}"}
        )


# ======================
# API: 簡化版 (只回最近的物件)
# ======================
@app.post("/find_object/summary")
async def find_object_summary(
    image: UploadFile = File(...),
    target: str = Form(default="water bottle"),
    conf: float = Form(default=0.25)
):
    """回傳最近一個物件的摘要（給 LLM 用）"""
    result = await find_object(image, target, conf)
    
    if isinstance(result, JSONResponse):
        return result
    
    if not result["found"]:
        return {"message": f"找不到 {target}"}
    
    nearest = result["results"][0]
    return {
        "found": True,
        "label": nearest["label"],
        "distance_m": nearest["distance_m"],
        "direction": nearest["direction"],
        "cmd_vel": nearest["cmd_vel"],
        "message": f"發現 {nearest['label']} 在{nearest['direction']}，距離 {nearest['distance_m']:.1f} 公尺"
    }


# ======================
# API: 原有深度分析 (保留相容性)
# ======================
def analyze_depth_regions(depth_map: np.ndarray) -> dict:
    h, w = depth_map.shape
    left_region = depth_map[:, :w//3]
    center_region = depth_map[:, w//3:2*w//3]
    right_region = depth_map[:, 2*w//3:]
    
    left_dist = float(np.median(left_region)) * SCALE_FACTOR
    center_dist = float(np.median(center_region)) * SCALE_FACTOR
    right_dist = float(np.median(right_region)) * SCALE_FACTOR
    
    center_lower = depth_map[h//2:, w//3:2*w//3]
    center_front = float(np.percentile(center_lower, 25)) * SCALE_FACTOR
    
    return {
        "left_m": round(left_dist, 2),
        "center_m": round(center_dist, 2),
        "right_m": round(right_dist, 2),
        "front_obstacle_m": round(center_front, 2),
    }


@app.post("/perceive")
async def perceive(
    image: UploadFile = File(...),
    obstacle_threshold: float = 1.0
):
    """原有的深度感知 API (保留相容性)"""
    if depth_model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "模型尚未載入完成"}
        )
    
    start_time = time.time()
    
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        raw_image = np.array(pil_image)
        
        with torch.no_grad():
            depth_map = depth_model.infer_image(raw_image)
        
        regions = analyze_depth_regions(depth_map)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            **regions,
            "inference_ms": round(elapsed_ms, 1)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"處理失敗: {str(e)}"}
        )


# ======================
# 直接運行 (開發用)
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
