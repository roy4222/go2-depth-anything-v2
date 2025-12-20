#!/usr/bin/env python3
"""距離校正驗證腳本"""

import requests
from pathlib import Path

SERVER_URL = "http://localhost:8001"
EXAMPLES_DIR = Path("~/Depth_Anything_V2/Depth-Anything-V2/assets/examples").expanduser()

TEST_SET = [
    (0.15, "15cm.png", "chair"),
    (0.30, "30cm.png", "chair"),
    (0.45, "45cm.png", "chair"),
    (1.00, "1m.png", "chair"),
    (2.00, "2m.png", "chair"),
]

def test_calibration():
    print("=== 距離校正驗證 ===\n")
    results = []

    for actual_dist, filename, target in TEST_SET:
        image_path = EXAMPLES_DIR / filename

        if not image_path.exists():
            print(f"⚠️  找不到: {filename}")
            continue

        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{SERVER_URL}/find_object",
                files={"image": f},
                data={"target": target, "use_fallback": "true"}
            )

        result = response.json()

        if result.get("found"):
            estimated = result["results"][0]["distance_m"]
            error = abs(estimated - actual_dist)
            error_pct = error / actual_dist * 100
            stage = result.get("detection_stage", "N/A")

            status = "✅" if error_pct < 20 else "⚠️"
            print(f"{status} {filename:12} | 實際: {actual_dist:5.2f}m | 估計: {estimated:5.2f}m | "
                  f"誤差: {error_pct:5.1f}% | 階段: {stage}")

            results.append(error_pct)
        else:
            print(f"❌ {filename:12} | 實際: {actual_dist:5.2f}m | 未偵測到")

    if results:
        avg_error = sum(results) / len(results)
        pass_count = sum(1 for e in results if e < 20)
        print(f"\n平均誤差: {avg_error:.1f}%")
        print(f"通過率 (<20% 誤差): {pass_count}/{len(results)}")

if __name__ == "__main__":
    test_calibration()
