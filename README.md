<div align="center">
<h1>Depth Anything V2</h1>

[**Lihe Yang**](https://liheyoung.github.io/)<sup>1</sup> · [**Bingyi Kang**](https://bingykang.github.io/)<sup>2&dagger;</sup> · [**Zilong Huang**](http://speedinghzl.github.io/)<sup>2</sup>
<br>
[**Zhen Zhao**](http://zhaozhen.me/) · [**Xiaogang Xu**](https://xiaogang00.github.io/) · [**Jiashi Feng**](https://sites.google.com/site/jshfeng/)<sup>2</sup> · [**Hengshuang Zhao**](https://hszhao.github.io/)<sup>1*</sup>

<sup>1</sup>HKU&emsp;&emsp;&emsp;<sup>2</sup>TikTok
<br>
&dagger;project lead&emsp;*corresponding author

<a href="https://arxiv.org/abs/2406.09414"><img src='https://img.shields.io/badge/arXiv-Depth Anything V2-red' alt='Paper PDF'></a>
<a href='https://depth-anything-v2.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything V2-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-V2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href='https://huggingface.co/datasets/depth-anything/DA-2K'><img src='https://img.shields.io/badge/Benchmark-DA--2K-yellow' alt='Benchmark'></a>
</div>

This work presents Depth Anything V2. It significantly outperforms [V1](https://github.com/LiheYoung/Depth-Anything) in fine-grained details and robustness. Compared with SD-based models, it enjoys faster inference speed, fewer parameters, and higher depth accuracy.

![teaser](assets/teaser.png)


## News
- **2025-01-22:** [Video Depth Anything](https://videodepthanything.github.io) has been released. It generates consistent depth maps for super-long videos (e.g., over 5 minutes).
- **2024-12-22:** [Prompt Depth Anything](https://promptda.github.io/) has been released. It supports 4K resolution metric depth estimation when low-res LiDAR is used to prompt the DA models.
- **2024-07-06:** Depth Anything V2 is supported in [Transformers](https://github.com/huggingface/transformers/). See the [instructions](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2) for convenient usage.
- **2024-06-25:** Depth Anything is integrated into [Apple Core ML Models](https://developer.apple.com/machine-learning/models/). See the instructions ([V1](https://huggingface.co/apple/coreml-depth-anything-small), [V2](https://huggingface.co/apple/coreml-depth-anything-v2-small)) for usage.
- **2024-06-22:** We release [smaller metric depth models](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models) based on Depth-Anything-V2-Small and Base.
- **2024-06-20:** Our repository and project page are flagged by GitHub and removed from the public for 6 days. Sorry for the inconvenience.
- **2024-06-14:** Paper, project page, code, models, demo, and benchmark are all released.


## Pre-trained Models

We provide **four models** of varying scales for robust relative depth estimation:

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |
| Depth-Anything-V2-Giant | 1.3B | Coming soon |

---

## 功能詳細說明 (Detailed Features)

本專案提供完整的深度估計解決方案，以下列出所有可用功能：

### 🎯 核心功能一覽

| 功能 | 腳本 | 說明 |
|------|------|------|
| **圖片深度估計** | `run.py` | 對單張或多張圖片進行相對深度估計 |
| **影片深度估計** | `run_video.py` | 對影片逐幀進行深度估計，輸出深度影片 |
| **公制深度估計** | `metric_depth/run.py` | 輸出真實距離（公尺），適用於室內/室外場景 |
| **點雲生成** | `metric_depth/depth_to_pointcloud.py` | 將 2D 圖片轉換為 3D 點雲 (PLY 格式) |
| **互動式 Demo** | `app.py` | 基於 Gradio 的網頁介面，支援即時預覽 |
| **相對深度測試** | `test_depth.py` | 快速測試相對深度推論，包含效能計時 |
| **公制深度測試** | `test_metric.py` | 快速測試公制深度推論，輸出實際距離值 |

---

### 📷 1. 圖片相對深度估計 (`run.py`)

從單張圖片預測每個像素的相對深度值，輸出視覺化深度圖。

**輸入格式：**
- 單張圖片檔案 (`.jpg`, `.png` 等)
- 圖片資料夾
- 包含圖片路徑的 `.txt` 文字檔

**輸出格式：**
- 彩色深度圖（使用 Spectral_r 色表）
- 灰階深度圖（使用 `--grayscale` 選項）
- 可選擇只輸出深度圖或原圖+深度圖並排

**參數說明：**

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--encoder` | 模型大小：`vits`, `vitb`, `vitl`, `vitg` | `vitl` |
| `--img-path` | 輸入圖片路徑 | (必填) |
| `--outdir` | 輸出目錄 | `./vis_depth` |
| `--input-size` | 推論時的輸入尺寸 | `518` |
| `--pred-only` | 僅輸出深度圖 | `False` |
| `--grayscale` | 輸出灰階深度圖 | `False` |

---

### 🎬 2. 影片深度估計 (`run_video.py`)

對影片進行逐幀深度估計，生成深度視覺化影片。

**輸入格式：**
- 單個影片檔案 (`.mp4`, `.avi` 等)
- 影片資料夾
- 包含影片路徑的 `.txt` 文字檔

**輸出格式：**
- `.mp4` 格式的深度影片
- 可選擇原影片+深度影片左右並排，或僅深度影片

**參數說明：**

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--encoder` | 模型大小 | `vitl` |
| `--video-path` | 輸入影片路徑 | (必填) |
| `--outdir` | 輸出目錄 | `./vis_video_depth` |
| `--input-size` | 推論時的輸入尺寸 | `518` |
| `--pred-only` | 僅輸出深度影片 | `False` |
| `--grayscale` | 輸出灰階深度 | `False` |

---

### 📏 3. 公制深度估計 (`metric_depth/`)

與相對深度不同，公制深度估計可輸出真實的物理距離（單位：公尺）。

**場景類型：**

| 場景 | 資料集 | 最大深度 | 適用情境 |
|------|--------|----------|----------|
| 室內 | Hypersim | 20 公尺 | 房間、辦公室、建築內部 |
| 室外 | Virtual KITTI | 80 公尺 | 街道、道路、戶外環境 |

**輸出格式：**
- 深度視覺化圖片 (PNG)
- 原始深度數據 (NumPy `.npy` 格式)

**快速測試腳本 `test_metric.py` 輸出範例：**
```
【真實距離數據 (單位: 公尺)】
  - 最近距離: 1.23 m
  - 最遠距離: 15.67 m
  - 中心點距離: 8.45 m
```

---

### 🌐 4. 點雲生成 (`metric_depth/depth_to_pointcloud.py`)

將 2D 圖片轉換為 3D 點雲，可用於 3D 視覺化和建模。

**輸出格式：**
- `.ply` 點雲檔案（包含顏色資訊）
- 可使用 Open3D、MeshLab 等工具檢視

**參數說明：**

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--encoder` | 模型大小 | `vitl` |
| `--load-from` | 模型權重路徑 | (必填) |
| `--max-depth` | 最大深度值 | `20` |
| `--img-path` | 輸入圖片路徑 | (必填) |
| `--outdir` | 輸出目錄 | `./vis_pointcloud` |
| `--focal-length-x` | X 軸焦距 | `470.4` |
| `--focal-length-y` | Y 軸焦距 | `470.4` |

---

### 🖥️ 5. Gradio 互動式 Demo (`app.py`)

提供基於網頁的互動式深度估計界面。

**功能特色：**
- 🖼️ 上傳圖片即時預覽深度圖
- 🎚️ 滑桿比較原圖與深度圖
- 📥 下載灰階深度圖 (PNG)
- 📥 下載 16-bit 原始深度數據 (可視為視差圖)

---

### 🧪 6. 測試腳本

#### `test_depth.py` - 相對深度測試
- 快速測試單張圖片的相對深度推論
- 輸出深度圖統計資訊（最大值、最小值）
- 輸出推論時間效能測試結果
- 儲存 NPY 原始數據和視覺化圖片

#### `test_metric.py` - 公制深度測試
- 快速測試公制深度推論
- 支援室內 (indoor) 和室外 (outdoor) 場景切換
- 輸出真實距離數據（單位：公尺）
- 包含效能計時功能

---

### 🔧 支援的硬體平台

| 平台 | 支援狀態 |
|------|----------|
| NVIDIA GPU (CUDA) | ✅ 完整支援 |
| Apple Silicon (MPS) | ✅ 支援 |
| CPU | ✅ 支援（較慢） |

---

### 📦 模型規格比較

| 模型 | 參數量 | 推論速度 | 精確度 | 適用場景 |
|------|--------|----------|--------|----------|
| Small (vits) | 24.8M | 最快 | 一般 | 即時應用、邊緣裝置 |
| Base (vitb) | 97.5M | 快 | 良好 | 平衡效能與精度 |
| Large (vitl) | 335.3M | 中等 | 優秀 | 高品質深度估計 |
| Giant (vitg) | 1.3B | 較慢 | 最佳 | 研究、最高品質需求 |

---

## Usage

### Prepraration

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
```

Download the checkpoints listed [here](#pre-trained-models) and put them under the `checkpoints` directory.

### Use our models
```python
import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
```

If you do not want to clone this repository, you can also load our models through [Transformers](https://github.com/huggingface/transformers/). Below is a simple code snippet. Please refer to the [official page](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2) for more details.

- Note 1: Make sure you can connect to Hugging Face and have installed the latest Transformers.
- Note 2: Due to the [upsampling difference](https://github.com/huggingface/transformers/pull/31522#issuecomment-2184123463) between OpenCV (we used) and Pillow (HF used), predictions may differ slightly. So you are more recommended to use our models through the way introduced above.
```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open('your/image/path')
depth = pipe(image)["depth"]
```

### Running script on *images*

```bash
python run.py \
  --encoder <vits | vitb | vitl | vitg> \
  --img-path <path> --outdir <outdir> \
  [--input-size <size>] [--pred-only] [--grayscale]
```
Options:
- `--img-path`: You can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.
- `--input-size` (optional): By default, we use input size `518` for model inference. ***You can increase the size for even more fine-grained results.***
- `--pred-only` (optional): Only save the predicted depth map, without raw image.
- `--grayscale` (optional): Save the grayscale depth map, without applying color palette.

For example:
```bash
python run.py --encoder vitl --img-path assets/examples --outdir depth_vis
```

### Running script on *videos*

```bash
python run_video.py \
  --encoder <vits | vitb | vitl | vitg> \
  --video-path assets/examples_video --outdir video_depth_vis \
  [--input-size <size>] [--pred-only] [--grayscale]
```

***Our larger model has better temporal consistency on videos.***

### Gradio demo

To use our gradio demo locally:

```bash
python app.py
```

You can also try our [online demo](https://huggingface.co/spaces/Depth-Anything/Depth-Anything-V2).

***Note: Compared to V1, we have made a minor modification to the DINOv2-DPT architecture (originating from this [issue](https://github.com/LiheYoung/Depth-Anything/issues/81)).*** In V1, we *unintentionally* used features from the last four layers of DINOv2 for decoding. In V2, we use [intermediate features](https://github.com/DepthAnything/Depth-Anything-V2/blob/2cbc36a8ce2cec41d38ee51153f112e87c8e42d8/depth_anything_v2/dpt.py#L164-L169) instead. Although this modification did not improve details or accuracy, we decided to follow this common practice.


## Fine-tuned to Metric Depth Estimation

Please refer to [metric depth estimation](./metric_depth).


## DA-2K Evaluation Benchmark

Please refer to [DA-2K benchmark](./DA-2K.md).


## Community Support

**We sincerely appreciate all the community support for our Depth Anything series. Thank you a lot!**

- Apple Core ML:
    - https://developer.apple.com/machine-learning/models
    - https://huggingface.co/apple/coreml-depth-anything-v2-small
    - https://huggingface.co/apple/coreml-depth-anything-small
- Transformers:
    - https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2
    - https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything
- TensorRT:
    - https://github.com/spacewalk01/depth-anything-tensorrt
    - https://github.com/zhujiajian98/Depth-Anythingv2-TensorRT-python
- ONNX: https://github.com/fabio-sim/Depth-Anything-ONNX
- ComfyUI: https://github.com/kijai/ComfyUI-DepthAnythingV2
- Transformers.js (real-time depth in web): https://huggingface.co/spaces/Xenova/webgpu-realtime-depth-estimation
- Android:
  - https://github.com/shubham0204/Depth-Anything-Android
  - https://github.com/FeiGeChuanShu/ncnn-android-depth_anything


## Acknowledgement

We are sincerely grateful to the awesome Hugging Face team ([@Pedro Cuenca](https://huggingface.co/pcuenq), [@Niels Rogge](https://huggingface.co/nielsr), [@Merve Noyan](https://huggingface.co/merve), [@Amy Roberts](https://huggingface.co/amyeroberts), et al.) for their huge efforts in supporting our models in Transformers and Apple Core ML.

We also thank the [DINOv2](https://github.com/facebookresearch/dinov2) team for contributing such impressive models to our community.


## LICENSE

Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
