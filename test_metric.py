import sys
import os
import time # 新增 time 模組

# 強制指向 metric_depth 資料夾
sys.path.insert(0, 'metric_depth')

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# ================= 設定區 =================
IMAGE_PATH = "assets/examples/test3.png"
OUTPUT_DIR = "outputs"
SCENE_TYPE = 'indoor' 
# =========================================

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"錯誤：找不到圖片 {IMAGE_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename_base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    output_viz_path = os.path.join(OUTPUT_DIR, f"{filename_base}_{SCENE_TYPE}_metric.png")
    output_npy_path = os.path.join(OUTPUT_DIR, f"{filename_base}_{SCENE_TYPE}_metric.npy")

    print(f"正在初始化 Metric Depth ({SCENE_TYPE})...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if SCENE_TYPE == 'indoor':
        dataset = 'hypersim'
        max_depth = 20.0 
    else: 
        dataset = 'vkitti'
        max_depth = 80.0 
        
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    model = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': max_depth})
    
    checkpoint_path = f'checkpoints/depth_anything_v2_metric_{dataset}_vitl.pth'
    if not os.path.exists(checkpoint_path):
        print(f"錯誤：找不到權重 {checkpoint_path}")
        return

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    except Exception as e:
        print("載入權重時遇到輕微不匹配，嘗試使用 strict=False...")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)

    model = model.to(DEVICE).eval()
    
    raw_img = cv2.imread(IMAGE_PATH)

    # --- 計時開始：推論 ---
    print(f"正在計算真實距離 (圖片: {os.path.basename(IMAGE_PATH)})...")
    t_start_infer = time.time()
    
    depth = model.infer_image(raw_img)
    
    t_end_infer = time.time()
    infer_duration = t_end_infer - t_start_infer
    # ---------------------
    
    H, W = depth.shape
    center_dist = depth[H // 2, W // 2]
    
    print("=" * 40)
    print(f"【真實距離數據 (單位: 公尺)】")
    print(f"  - 最近距離: {depth.min():.2f} m")
    print(f"  - 最遠距離: {depth.max():.2f} m")
    print(f"  - 中心點距離: {center_dist:.2f} m")
    print("-" * 40)
    print(f"【效能測試】")
    print(f"  - 推論時間 (GPU): {infer_duration:.4f} 秒")
    print("=" * 40)

    np.save(output_npy_path, depth)
    
    # --- 計時開始：繪圖與存檔 ---
    print("正在生成可視化圖片...")
    t_start_plot = time.time()

    plt.figure(figsize=(10, 8))
    viz_max = depth.max() if depth.max() < max_depth else max_depth
    plt.imshow(depth, cmap='inferno_r', vmax=viz_max) 
    plt.colorbar(label='Depth (Meters)')
    plt.title(f'Metric: {filename_base} ({SCENE_TYPE})')
    plt.savefig(output_viz_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    t_end_plot = time.time()
    plot_duration = t_end_plot - t_start_plot
    # -------------------------
    
    print(f"[完成] 圖片已存為: {output_viz_path}")
    print(f"  - 繪圖存檔耗時:   {plot_duration:.4f} 秒")
    print(f"  - 總流程耗時:     {infer_duration + plot_duration:.4f} 秒")

if __name__ == '__main__':
    main()