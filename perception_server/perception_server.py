"""
Perception Server - FastAPI + Depth Anything V2
用途: 接收圖片，回傳深度估計與避障建議
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import sys
import os
import time

# 加入 Depth Anything V2 metric_depth 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'metric_depth'))
from depth_anything_v2.dpt import DepthAnythingV2

app = FastAPI(
    title="Perception Server",
    description="DA3 深度估計 API - 為 Go2 機器狗提供避障感知",
    version="1.0.0"
)

# 全域模型物件
model = None
device = None

# 距離校正係數 (TODO: 用 Go2 實拍校正)
SCALE_FACTOR = 1.0


@app.on_event("startup")
async def load_models():
    """伺服器啟動時載入模型"""
    global model, device
    
    print("🔄 正在載入 Depth Anything V2 模型...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  裝置: {device}")
    
    # 初始化模型
    model = DepthAnythingV2(
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        max_depth=20.0
    )
    
    # 載入權重
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'checkpoints', 
        'depth_anything_v2_metric_hypersim_vitl.pth'
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 找不到權重檔案: {checkpoint_path}")
        print("  請先下載 metric depth 權重！")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model = model.to(device).eval()
    
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"✅ DA3 模型載入完成 (VRAM: {vram:.2f} GB)")


def analyze_depth_regions(depth_map: np.ndarray) -> dict:
    """
    分析深度圖的左/中/右區域
    
    Args:
        depth_map: 深度圖 (H, W)，單位為公尺
        
    Returns:
        區域分析結果 dict
    """
    h, w = depth_map.shape
    
    # 分割為左中右三區（各佔畫面 1/3）
    left_region = depth_map[:, :w//3]
    center_region = depth_map[:, w//3:2*w//3]
    right_region = depth_map[:, 2*w//3:]
    
    # 取各區域的中位數距離（比平均值更穩健）
    left_dist = float(np.median(left_region)) * SCALE_FACTOR
    center_dist = float(np.median(center_region)) * SCALE_FACTOR
    right_dist = float(np.median(right_region)) * SCALE_FACTOR
    
    # 取中央下半部（更準確反映前方障礙物）
    center_lower = depth_map[h//2:, w//3:2*w//3]
    center_front = float(np.percentile(center_lower, 25)) * SCALE_FACTOR  # 取較近的距離
    
    return {
        "left_m": round(left_dist, 2),
        "center_m": round(center_dist, 2),
        "right_m": round(right_dist, 2),
        "front_obstacle_m": round(center_front, 2),
        "min_m": round(float(np.min(depth_map)) * SCALE_FACTOR, 2),
        "max_m": round(float(np.max(depth_map)) * SCALE_FACTOR, 2)
    }


def generate_suggestion(regions: dict, obstacle_threshold: float = 1.0) -> str:
    """
    根據區域分析產生避障建議
    
    Args:
        regions: 區域分析結果
        obstacle_threshold: 障礙物警戒距離（公尺）
        
    Returns:
        避障建議字串
    """
    front = regions["front_obstacle_m"]
    left = regions["left_m"]
    right = regions["right_m"]
    
    if front > obstacle_threshold:
        return f"✅ 前方暢通（{front:.1f}m），可安全前進"
    
    # 前方有障礙，判斷繞行方向
    if right > left and right > obstacle_threshold:
        return f"⚠️ 正前方 {front:.1f}m 有障礙，建議向右繞行（右側 {right:.1f}m 較空曠）"
    elif left > obstacle_threshold:
        return f"⚠️ 正前方 {front:.1f}m 有障礙，建議向左繞行（左側 {left:.1f}m 較空曠）"
    else:
        return f"🛑 三面受阻（前 {front:.1f}m / 左 {left:.1f}m / 右 {right:.1f}m），建議後退或停止"


@app.get("/")
async def root():
    """健康檢查端點"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }


@app.post("/perceive")
async def perceive(
    image: UploadFile = File(...),
    obstacle_threshold: float = 1.0
):
    """
    接收圖片，回傳深度分析與避障建議
    
    Args:
        image: 上傳的圖片檔案
        obstacle_threshold: 障礙物警戒距離（公尺），預設 1.0m
        
    Returns:
        JSON 包含區域距離與避障建議
    """
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "模型尚未載入完成，請稍候"}
        )
    
    start_time = time.time()
    
    try:
        # 1. 讀取圖片
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        raw_image = np.array(pil_image)
        
        # 2. 推論深度
        with torch.no_grad():
            depth_map = model.infer_image(raw_image)
        
        # 3. 區域分析
        regions = analyze_depth_regions(depth_map)
        
        # 4. 產生建議
        suggestion = generate_suggestion(regions, obstacle_threshold)
        
        # 5. 計算耗時
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            **regions,
            "suggestion": suggestion,
            "inference_ms": round(elapsed_ms, 1),
            "image_size": f"{pil_image.width}x{pil_image.height}"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"處理失敗: {str(e)}"}
        )


@app.post("/perceive/summary")
async def perceive_summary(image: UploadFile = File(...)):
    """
    簡化版：只回傳純文字感知摘要（給 LLM 使用）
    
    Returns:
        純文字感知摘要
    """
    result = await perceive(image)
    
    if isinstance(result, JSONResponse):
        return result
    
    summary = f"""[環境感知摘要]
- 左側: {result['left_m']}m
- 正前方: {result['front_obstacle_m']}m
- 右側: {result['right_m']}m
{result['suggestion']}"""
    
    return {"summary": summary, "inference_ms": result["inference_ms"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
