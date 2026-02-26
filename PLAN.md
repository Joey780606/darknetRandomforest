# 用原版 Darknet 實現人臉特徵點偵測 — 方法一詳細實作方案

## Context

主管要求使用 [pjreddie/darknet](https://github.com/pjreddie/darknet) 來做人臉特徵點訓練。
採用方法：**把每個特徵點當成一個「微型物件」來偵測**，使用原版 Darknet，不修改原始碼。
訓練完的模型用於攝影機即時人臉特徵點偵測，推論階段用 Python（OpenCV DNN）。

---

## 專案目錄結構

```
darknetHumanFeature/
├── darknet/                    ← clone pjreddie/darknet 原始碼（用於訓練）
├── data/
│   ├── images/
│   │   ├── train/              ← 訓練圖片
│   │   └── val/                ← 驗證圖片
│   ├── labels/
│   │   ├── train/              ← 訓練標註 (.txt, YOLO格式)
│   │   └── val/                ← 驗證標註
│   ├── train.txt               ← 訓練圖片路徑清單
│   └── val.txt                 ← 驗證圖片路徑清單
├── cfg/
│   ├── face_landmark.data      ← 資料路徑設定
│   ├── face_landmark.names     ← 類別名稱
│   └── yolov3-face-landmark.cfg ← 網路架構設定
├── weights/                    ← 訓練產出的權重檔
├── scripts/
│   ├── convert_300w_to_yolo.py ← 將 300-W 資料集轉換為 YOLO 格式
│   └── generate_train_val.py   ← 產生 train.txt / val.txt
├── tools/
│   └── landmark_annotator.py   ← 標註小工具：點擊圖片上 5 個點，自動產生 .txt 標註檔
├── inference.py                ← Python 推論程式（攝影機即時偵測）
└── PLAN.md                     ← 本文件
```

---

## Step 1: 環境建置

### 1a. 編譯 Darknet（Windows）
```bash
git clone https://github.com/pjreddie/darknet.git
cd darknet
# Windows 建議使用 AlexeyAB/darknet fork，對 Windows 支援更好：
# git clone https://github.com/AlexeyAB/darknet.git
# 用 Visual Studio 或 CMake 編譯
```
> 注意：pjreddie/darknet 原版對 Windows 支援較差，建議用 [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)（完全相容原版，且有 Windows 編譯指南）

### 1b. Python 環境
```bash
pip install opencv-python numpy
```

---

## Step 2: 準備訓練資料

### 2a. 資料集選擇
使用 **300-W** 資料集（300 Faces In-the-Wild）：
- 包含 68 個人臉特徵點座標
- 我們從中選取 5 個關鍵點：左眼中心、右眼中心、鼻尖、左嘴角、右嘴角
- 下載來源：https://ibug.doc.ic.ac.uk/resources/300-W/

### 2b. 定義偵測的特徵點類別（5 類）
```
class 0: left_eye      ← 左眼中心（68點中的第 37-42 點平均）
class 1: right_eye     ← 右眼中心（68點中的第 43-48 點平均）
class 2: nose_tip      ← 鼻尖（68點中的第 31 點）
class 3: left_mouth    ← 左嘴角（68點中的第 49 點）
class 4: right_mouth   ← 右嘴角（68點中的第 55 點）
```

### 2c. 轉換腳本 `scripts/convert_300w_to_yolo.py`
功能：讀取 300-W 的 .pts 標註檔，提取 5 個關鍵點，轉換為 YOLO 格式的 .txt 標註檔
- 每個特徵點以極小的 bounding box 表示（寬高約為圖片尺寸的 2%~3%）
- 輸出格式：`<class_id> <x_center> <y_center> <width> <height>`

### 2d. 產生路徑清單
`scripts/generate_train_val.py`：掃描圖片目錄，依 80/20 比例產生 `train.txt` 和 `val.txt`

---

## Step 3: 建立 Darknet 設定檔

### 3a. `cfg/face_landmark.names`
```
left_eye
right_eye
nose_tip
left_mouth
right_mouth
```

### 3b. `cfg/face_landmark.data`
```
classes = 5
train = data/train.txt
valid = data/val.txt
names = cfg/face_landmark.names
backup = weights/
```

### 3c. `cfg/yolov3-face-landmark.cfg`
基於 `yolov3.cfg` 修改以下關鍵參數：
- `batch = 64`
- `subdivisions = 16`（依 GPU 記憶體調整）
- `width = 416`, `height = 416`
- 每個 `[yolo]` 層：`classes = 5`
- 每個 `[yolo]` 層前面的 `[convolutional]` 層：`filters = 30`（計算公式：`(classes + 5) * 3 = (5+5)*3 = 30`）
- `max_batches = 10000`（5 類 × 2000）
- `steps = 8000,9000`

---

## Step 4: 訓練

### 4a. 下載預訓練權重
```bash
# 下載 darknet53.conv.74 作為初始權重（遷移學習）
wget https://pjreddie.com/media/files/darknet53.conv.74
```

### 4b. 執行訓練
```bash
./darknet detector train cfg/face_landmark.data cfg/yolov3-face-landmark.cfg darknet53.conv.74
```
訓練完成後，權重檔會存在 `weights/` 目錄下。

---

## Step 5: Python 推論程式 `inference.py`

使用 `cv2.dnn.readNetFromDarknet()` 載入訓練好的模型：
1. 開啟攝影機
2. 每個 frame 送入 YOLO 模型偵測
3. 取得每個偵測結果的 bounding box 中心點作為特徵點座標
4. 在畫面上繪製特徵點
5. 即時顯示

---

## Step 6: 驗證

1. 用攝影機即時測試，確認 5 個特徵點能正確標示在人臉上
2. 測量 FPS（目標：15+ FPS）
3. 測試不同角度、光線、距離下的偵測穩定性

---

## Step 7: 標註小工具 `tools/landmark_annotator.py`

用於日後自己新增訓練資料。功能：
1. 指定一個圖片資料夾，逐張載入圖片
2. 畫面上依序提示點擊：左眼 → 右眼 → 鼻尖 → 左嘴角 → 右嘴角
3. 每點擊一個點，在畫面上即時繪製標記和名稱
4. 5 個點都點完後，自動產生同名的 `.txt` 標註檔（YOLO 格式）
5. 按 `r` 可重新標註當前圖片，按 `n` 跳到下一張，按 `q` 結束

### 再訓練流程
```
1. 用 landmark_annotator.py 標註新圖片 → 產生 .txt 標註檔
2. 把新圖片和標註放進 data/images/train/ 和 data/labels/train/
3. 更新 train.txt（加入新圖片路徑）
4. 用上次的權重繼續訓練：
   ./darknet detector train cfg/face_landmark.data cfg/yolov3-face-landmark.cfg weights/yolov3-face-landmark_last.weights
```

---

## 需要你準備的資料

| 資料 | 來源 | 說明 |
|------|------|------|
| 300-W 資料集 | [ibug.doc.ic.ac.uk](https://ibug.doc.ic.ac.uk/resources/300-W/) | 需下載圖片 + .pts 標註檔 |
| GPU（建議） | 自備 | 有 NVIDIA GPU 訓練快很多；CPU 也能訓練但很慢 |
| 攝影機 | 自備 | 用於即時偵測推論 |

> 其餘所有設定檔、轉換腳本、推論程式，我都可以幫你產生。

---

## 實作清單

1. `scripts/convert_300w_to_yolo.py` — 資料集格式轉換
2. `scripts/generate_train_val.py` — 產生訓練/驗證路徑清單
3. `cfg/face_landmark.names` — 類別名稱
4. `cfg/face_landmark.data` — 訓練設定
5. `cfg/yolov3-face-landmark.cfg` — 網路架構（基於 yolov3.cfg 修改）
6. `inference.py` — Python 攝影機即時推論程式
7. `tools/landmark_annotator.py` — 標註小工具（點擊 5 點自動產生標註檔）

---

## 注意事項

- **微小物件偵測的限制**：YOLO 對極小物件的偵測能力較弱，特徵點的 bounding box 設定（2%~3% 圖片尺寸）需要實驗調整
- **建議先用 YOLOv3-tiny**：如果 GPU 資源有限，可先用 tiny 版本快速驗證可行性
- **Windows 編譯**：強烈建議使用 AlexeyAB/darknet fork，它提供 Visual Studio solution 檔，在 Windows 上編譯更方便
