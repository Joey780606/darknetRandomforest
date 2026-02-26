# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

結合 YOLOv3/Darknet 人臉特徵點偵測與 Random Forest 分類器的人臉辨識系統。
使用預訓練的 YOLOv3 權重偵測 5 個 facial landmarks（left_eye, right_eye, nose_tip, left_mouth, right_mouth），
開啟 webcam，若能偵測到人臉，用 Random Forest 分類器判斷是誰；
若判斷不到，讓使用者輸入名字，從 webcam 定期抓 frame 做分類器訓練。

## Architecture

```
webcam frame
    ↓
inference.py (detect_landmarks)     ← YOLOv3/Darknet weights
    ↓ landmarks dict {class_id: {center:(x,y), confidence:float}}
PersonClassifier.ExtractFeatures()  ← 10 pairwise Euclidean 距離特徵
    ↓ np.ndarray shape (10,)
RandomForestClassifier.predict()    ← sklearn
    ↓
辨識結果 / 學習樣本收集 → SaveModel → models/rf_classifier.pkl
```

**特徵設計**: 5 個特徵點的 C(5,2)=10 組 pairwise Euclidean 距離，以左眼↔右眼距離正規化，對臉部位置與大小具不變性。

**YOLOv3 參考資料**: https://github.com/AlexeyAB/darknet
**Random Forest 來源**: 純 numpy/Python 自行實作（`_RandomForest` class，NEstimators=50），不依賴 scikit-learn / scipy

## Code Specification

1. 開發語言: Python 3.13
2. UI library: customtkinter
3. 註解請用中文
4. Variable Naming: CamelCase is used consistently
5. Error Handling: All API calls must include a try-catch block

## Design Decisions / Constraints

1. 程式開啟時自動開啟 Webcam；程式關閉前先關掉 Webcam。
2. YOLO 推論在 UI thread 的 `after()` callback 中執行（無額外 worker thread）。
3. `_UpdateWebcamView`（30ms）不執行 YOLO，改從 `_LastLandmarks` 快取繪製，避免 UI 卡頓。
4. 辨識採多數決（10 秒內所有推論結果投票），學習採時間窗口收集（60 秒）。
5. 模型儲存完整歷史訓練資料（AllFeatures + AllLabels），支援增量學習新增人物。
6. 辨識功能需至少 2 個不同人的訓練資料才能啟用（`CanDetect()` 判斷）。

## UI Design（4 rows）

**Row 0 — 辨識功能列**
- `btnDetectName`（Button）: 文字為 "Detect face"。按下後改成 "Stop detect face"，每 200ms 執行一次 YOLO+RF 推論，持續 10 秒。結束後以多數決顯示辨識結果，找不到則顯示 dialog。
- `lblDetectName`（Label）: 顯示辨識出的姓名，或 "Not found"。

**Row 1 — 學習功能列**
- `tblMyName`（Entry）: 供操作者輸入姓名。
- `btnLearn`（Button）: 文字為 "Learning"。按下後每 500ms 抓一次 frame，若偵測到 ≥3 個 facial landmarks 則加入學習樣本，持續 60 秒後自動訓練並儲存模型。

**Row 2 — Webcam 畫面**
- `WebcamCanvas`（CTkLabel）: 顯示 webcam 即時畫面，overlay 顯示偵測到的 landmarks（彩色圓點+名稱）。

**Row 3 — 剩餘時間列**
- `lblRemain`（Label）: 內容為 "Remaining study seconds: XX"，學習期間顯示倒數秒數。

## Commands

### 安裝依賴
```bash
pip install customtkinter pillow opencv-python numpy
```
（不需要 scikit-learn / scipy，Random Forest 以純 numpy 實作，避免 Windows Application Control 封鎖 scipy DLL 的問題）

### 執行主程式
```bash
cd randomForest
python main.py
```

### 首次使用流程
1. 啟動程式，`btnDetectName` 初始為 disabled（尚無模型）
2. 在 `tblMyName` 輸入名字 A → 按 `btnLearn` → 等待 60 秒
3. 換人，輸入名字 B → 按 `btnLearn` → 等待 60 秒（訓練完後辨識按鈕自動啟用）
4. 按 `btnDetectName` → 等待 10 秒 → 查看辨識結果（不需重啟程式）

## File Structure

```
randomForest/
├── main.py                             # 主應用程式（UI + RF 分類器 + webcam 管理）
├── inference.py                        # YOLO 推論函式（由 main.py import 重用）
├── cfg/
│   ├── face_landmark.names             # 5 個 landmark 類別名稱
│   └── yolov3-face-landmark.cfg        # YOLOv3 網路組態
├── weights/
│   └── yolov3-face-landmark_final.weights  # 預訓練權重
└── models/
    └── rf_classifier.pkl               # 訓練後的 RF 模型（首次學習後自動產生）
```

## Key Classes and Functions

### `main.py`

| Class / 函式 | 說明 |
|-------------|------|
| `WebcamManager` | daemon thread 持續讀取 webcam，Lock 保護 frame buffer |
| `PersonClassifier` | RF 特徵抽取、訓練、預測、模型持久化 |
| `PersonClassifier.ExtractFeatures(Landmarks, W, H)` | 計算 10 個 pairwise 距離，以 left_eye↔right_eye 正規化 |
| `PersonClassifier.Train()` | 合併歷史資料重新訓練，需 ≥2 類別 |
| `PersonClassifier.SaveModel()` | joblib 儲存 Classifier + LabelEncoder + AllFeatures + AllLabels |
| `MainApp._UpdateWebcamView()` | 每 30ms 更新 UI 畫面（不執行 YOLO） |
| `MainApp._DetectionTick()` | 每 200ms 執行 YOLO+RF 推論（辨識模式） |
| `MainApp._LearningTick()` | 每 500ms 收集樣本 + 倒數（學習模式） |

### `inference.py`（被 main.py import）

| 函式 | 說明 |
|------|------|
| `load_class_names(names_path)` | 讀取 .names 檔，回傳 list[str] |
| `load_network(cfg_path, weights_path)` | 載入 Darknet 模型，回傳 (net, output_layer_names) |
| `detect_landmarks(net, layers, frame, size, conf, nms)` | YOLO 推論，回傳 `{class_id: {"center": (x,y), "confidence": float}}` |
| `draw_landmarks(frame, landmarks, class_names)` | 在 frame 上繪製彩色圓點與標籤 |

## Known Limitations

- YOLOv3 在 CPU 上推論速度約 0.5~2 FPS，學習/辨識期間 UI 可能有短暫延遲。建議使用 CUDA 版 OpenCV 搭配 NVIDIA GPU。
- Pairwise 距離特徵對臉部旋轉（roll）不完全不變，正面朝向鏡頭效果最佳。
- Random Forest 需要至少 2 個不同人的訓練資料才能進行辨識。
