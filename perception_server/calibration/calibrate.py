#!/usr/bin/env python3
"""
Go2 距離校正工具
用途: 分析校正圖片，計算 SCALE_FACTOR
"""

import requests
import sys
import os
from pathlib import Path

# Perception Server URL
SERVER_URL = "http://localhost:8000"


def analyze_image(image_path: str) -> dict:
    """送圖片到 Perception Server 分析"""
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{SERVER_URL}/perceive",
            files={"image": f}
        )
    return response.json()


def calibrate(calibration_data: list[tuple[float, str]]) -> float:
    """
    計算校正係數
    
    Args:
        calibration_data: [(實際距離, 圖片路徑), ...]
        
    Returns:
        SCALE_FACTOR
    """
    print("\n📊 校正分析結果")
    print("=" * 50)
    
    scale_factors = []
    
    for actual_dist, image_path in calibration_data:
        if not os.path.exists(image_path):
            print(f"⚠️  找不到: {image_path}")
            continue
            
        result = analyze_image(image_path)
        
        if "error" in result:
            print(f"❌ 分析失敗: {result['error']}")
            continue
        
        da3_dist = result["front_obstacle_m"]
        error_pct = abs(da3_dist - actual_dist) / actual_dist * 100
        scale = actual_dist / da3_dist
        scale_factors.append(scale)
        
        print(f"\n📷 {os.path.basename(image_path)}")
        print(f"   實際距離: {actual_dist:.1f} m")
        print(f"   DA3 輸出: {da3_dist:.2f} m")
        print(f"   誤差: {error_pct:.1f}%")
        print(f"   校正係數: {scale:.3f}")
    
    if not scale_factors:
        print("\n❌ 沒有有效的校正數據！")
        return 1.0
    
    avg_scale = sum(scale_factors) / len(scale_factors)
    
    print("\n" + "=" * 50)
    print(f"✅ 平均校正係數 (SCALE_FACTOR): {avg_scale:.4f}")
    print("\n📝 請更新 perception_server.py:")
    print(f"   SCALE_FACTOR = {avg_scale:.4f}")
    
    return avg_scale


def main():
    print("🔧 Go2 距離校正工具")
    print("=" * 50)
    
    # 檢查 Perception Server 連線
    try:
        resp = requests.get(f"{SERVER_URL}/")
        if resp.json().get("status") != "ok":
            print("❌ Perception Server 未正常運行")
            sys.exit(1)
        print("✅ Perception Server 連線成功")
    except Exception as e:
        print(f"❌ 無法連接 Perception Server: {e}")
        sys.exit(1)
    
    # 預設校正圖路徑
    calibration_dir = Path(__file__).parent
    calibration_data = [
        (1.0, str(calibration_dir / "calibration_1m.jpg")),
        (2.0, str(calibration_dir / "calibration_2m.jpg")),
        (3.0, str(calibration_dir / "calibration_3m.jpg")),
    ]
    
    # 檢查圖片是否存在
    existing = [(d, p) for d, p in calibration_data if os.path.exists(p)]
    
    if not existing:
        print("\n⚠️  找不到校正圖片！請先拍攝：")
        print("   - calibration_1m.jpg")
        print("   - calibration_2m.jpg")
        print("   - calibration_3m.jpg")
        print("\n📖 參考 README.md 拍攝流程")
        sys.exit(0)
    
    print(f"\n📷 找到 {len(existing)} 張校正圖片")
    
    # 執行校正
    scale_factor = calibrate(existing)
    
    # 儲存結果
    result_file = calibration_dir / "calibration_result.txt"
    with open(result_file, 'w') as f:
        f.write(f"SCALE_FACTOR = {scale_factor:.4f}\n")
        f.write(f"校正時間: {__import__('datetime').datetime.now()}\n")
    
    print(f"\n💾 結果已儲存至: {result_file}")


if __name__ == "__main__":
    main()
