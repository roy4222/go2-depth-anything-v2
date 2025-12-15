# Go2 距離校正指南

## 📋 校正步驟

### 準備工具
- [ ] 捲尺
- [ ] 膠帶（標記地板）
- [ ] 測試物體（紙箱/椅子，高度 > 30cm）

### 拍攝流程

1. **標記地板距離**
   - 從 Go2 起點標記 1m / 2m / 3m 位置

2. **放置物體並拍照**
   ```bash
   # 在 Mac VM 執行
   ros2 service call /capture_snapshot std_srvs/srv/Trigger
   
   # 複製到本機
   scp roy422@192.168.1.200:/tmp/snapshot_latest.jpg ./calibration_1m.jpg
   ```

3. **上傳至 GPU 伺服器分析**
   ```bash
   curl -X POST "http://localhost:8000/perceive" \
     -F "image=@calibration_1m.jpg" | python3 -m json.tool
   ```

4. **記錄結果**
   ```
   實際距離: 1.0m
   DA3 輸出: ___m
   誤差: ___%
   ```

## 📊 校正記錄表

| 實際距離 | DA3 front_obstacle_m | 誤差 | 備註 |
|---------|---------------------|------|------|
| 1.0 m   |                     |      |      |
| 2.0 m   |                     |      |      |
| 3.0 m   |                     |      |      |

## 🔧 計算 SCALE_FACTOR

```python
# 校正係數 = 實際距離 / DA3 輸出
scale_1m = 1.0 / da3_output_1m
scale_2m = 2.0 / da3_output_2m
scale_3m = 3.0 / da3_output_3m

SCALE_FACTOR = (scale_1m + scale_2m + scale_3m) / 3
```
