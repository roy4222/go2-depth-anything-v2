import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time  # 新增 time 模組
from depth_anything_v2.dpt import DepthAnythingV2

# ================= 設定區 =================
IMAGE_PATH = "assets/examples/test3.png" 
OUTPUT_DIR = "outputs"
ENCODER = 'vitl' 
# =========================================

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"錯誤：找不到圖片 {IMAGE_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename_base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    output_viz_path = os.path.join(OUTPUT_DIR, f"{filename_base}_viz.png")
    output_npy_path = os.path.join(OUTPUT_DIR, f"{filename_base}_raw.npy")

    print(f"正在初始化... (圖片: {os.path.basename(IMAGE_PATH)})")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    model = DepthAnythingV2(**model_configs[ENCODER])
    checkpoint_file = f'checkpoints/depth_anything_v2_{ENCODER}.pth'
    
    if not os.path.exists(checkpoint_file):
        print(f"錯誤：找不到模型 {checkpoint_file}")
        return

    model.load_state_dict(torch.load(checkpoint_file, map_location='cpu'))
    model = model.to(DEVICE).eval()

    # --- 讀取圖片 ---
    raw_img = cv2.imread(IMAGE_PATH)

    # --- 計時開始：推論 ---
    print("正在計算深度 (Inference)...")
    t_start_infer = time.time()  # 開始計時
    
    depth = model.infer_image(raw_img)
    
    t_end_infer = time.time()    # 結束計時
    infer_duration = t_end_infer - t_start_infer
    # ---------------------

    # 顯示數據
    H, W = depth.shape
    center_val = depth[H // 2, W // 2]
    
    print("=" * 40)
    print(f"【深度圖數據 - Relative Depth】")
    print(f"  - 圖片尺寸: {W} x {H}")
    print(f"  - 最大值: {depth.max():.4f}")
    print(f"  - 最小值: {depth.min():.4f}")
    print("-" * 40)
    print(f"【效能測試】")
    print(f"  - 推論時間 (GPU): {infer_duration:.4f} 秒")
    print("=" * 40)

    # 儲存 NPY
    np.save(output_npy_path, depth)

    # --- 計時開始：繪圖與存檔 ---
    print("正在生成圖片並存檔...")
    t_start_plot = time.time() # 開始計時

    plt.figure(figsize=(10, 8))
    plt.imshow(depth, cmap='inferno')
    plt.axis('off')
    plt.title(f'Depth Map: {filename_base}')
    plt.savefig(output_viz_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    t_end_plot = time.time()   # 結束計時
    plot_duration = t_end_plot - t_start_plot
    # -------------------------

    print(f"[完成] 圖片已存為: {output_viz_path}")
    print(f"  - 繪圖存檔耗時:   {plot_duration:.4f} 秒")
    print(f"  - 總流程耗時:     {infer_duration + plot_duration:.4f} 秒")

if __name__ == '__main__':
    main()