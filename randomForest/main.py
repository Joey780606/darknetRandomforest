"""
main.py

結合 YOLOv3 人臉特徵點偵測與 Random Forest 的人臉辨識應用程式。
使用 CustomTkinter 建立 GUI，可透過 webcam 學習並辨識不同人的身份。

使用方式：
    python main.py

依賴套件：
    pip install customtkinter pillow opencv-python numpy
    （不需要 scikit-learn / scipy，Random Forest 以純 numpy 實作）
"""

import os
import pickle
import threading
import time
from collections import Counter

import cv2
import numpy as np
import customtkinter
from PIL import Image, ImageTk
import tkinter.messagebox as MsgBox

from inference import load_class_names, load_network, detect_landmarks, draw_landmarks


# ==============================================================================
# 純 numpy 實作：標籤編碼器
# ==============================================================================
class _LabelEncoder:
    """簡易標籤編碼器，取代 sklearn.preprocessing.LabelEncoder。"""

    def __init__(self):
        self.classes_    = []
        self._ClassToIdx = {}

    def fit_transform(self, Labels: list) -> np.ndarray:
        """配適並轉換標籤為整數索引。"""
        self.classes_    = sorted(list(set(Labels)))
        self._ClassToIdx = {C: I for I, C in enumerate(self.classes_)}
        return np.array([self._ClassToIdx[L] for L in Labels], dtype=np.int32)

    def transform(self, Labels: list) -> np.ndarray:
        """將標籤轉換為整數索引。"""
        return np.array([self._ClassToIdx[L] for L in Labels], dtype=np.int32)

    def inverse_transform(self, Indices) -> list:
        """將整數索引還原為標籤。"""
        return [self.classes_[int(I)] for I in Indices]


# ==============================================================================
# 純 numpy 實作：決策樹（Gini 不純度）
# ==============================================================================
class _DecisionTree:
    """簡易決策樹分類器，使用 Gini 不純度，支援隨機特徵子集。"""

    def __init__(self, MaxDepth: int = 10, MinSamplesSplit: int = 2, NFeatures: int = None):
        self._MaxDepth        = MaxDepth
        self._MinSamplesSplit = MinSamplesSplit
        self._NFeatures       = NFeatures   # None = 使用全部特徵
        self._Tree            = None
        self._NClasses        = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """訓練決策樹。"""
        self._NClasses = int(np.max(y)) + 1
        NFeat = self._NFeatures or X.shape[1]
        self._Tree = self._BuildTree(X, y, Depth=0, NFeat=NFeat)

    def _Gini(self, y: np.ndarray) -> float:
        """計算 Gini 不純度。"""
        if len(y) == 0:
            return 0.0
        Counts = np.bincount(y, minlength=self._NClasses)
        Probs  = Counts / len(y)
        return float(1.0 - np.sum(Probs ** 2))

    def _BestSplit(self, X: np.ndarray, y: np.ndarray, NFeat: int) -> tuple:
        """找出最佳分裂點（最大化 Gini 增益）。"""
        BestGain    = -1.0
        BestFeature = 0
        BestThresh  = 0.0
        ParentGini  = self._Gini(y)
        NSamples    = len(y)

        # 隨機選取特徵子集
        FeatureIdx = np.random.choice(X.shape[1], size=NFeat, replace=False)

        for FIdx in FeatureIdx:
            Thresholds = np.unique(X[:, FIdx])
            for Thresh in Thresholds:
                LeftMask  = X[:, FIdx] <= Thresh
                RightMask = ~LeftMask
                if LeftMask.sum() == 0 or RightMask.sum() == 0:
                    continue
                Gain = ParentGini - (
                    LeftMask.sum()  / NSamples * self._Gini(y[LeftMask]) +
                    RightMask.sum() / NSamples * self._Gini(y[RightMask])
                )
                if Gain > BestGain:
                    BestGain    = Gain
                    BestFeature = int(FIdx)
                    BestThresh  = float(Thresh)

        return BestFeature, BestThresh, BestGain

    def _BuildTree(self, X: np.ndarray, y: np.ndarray, Depth: int, NFeat: int) -> dict:
        """遞迴建立決策樹節點。"""
        # 終止條件：超過最大深度、樣本太少、或所有樣本同類
        if (Depth >= self._MaxDepth or
                len(y) < self._MinSamplesSplit or
                len(np.unique(y)) == 1):
            Counts = np.bincount(y, minlength=self._NClasses)
            return {"Leaf": True, "Probs": Counts / Counts.sum()}

        FIdx, Thresh, Gain = self._BestSplit(X, y, NFeat)
        if Gain <= 0:
            Counts = np.bincount(y, minlength=self._NClasses)
            return {"Leaf": True, "Probs": Counts / Counts.sum()}

        Mask  = X[:, FIdx] <= Thresh
        Left  = self._BuildTree(X[Mask],  y[Mask],  Depth + 1, NFeat)
        Right = self._BuildTree(X[~Mask], y[~Mask], Depth + 1, NFeat)
        return {"Leaf": False, "Feature": FIdx, "Thresh": Thresh,
                "Left": Left, "Right": Right}

    def _PredictOne(self, x: np.ndarray, Node: dict) -> np.ndarray:
        """對單一樣本遞迴預測機率。"""
        if Node["Leaf"]:
            return Node["Probs"]
        if x[Node["Feature"]] <= Node["Thresh"]:
            return self._PredictOne(x, Node["Left"])
        return self._PredictOne(x, Node["Right"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """回傳每個樣本的類別機率矩陣，shape (N, NClasses)。"""
        return np.array([self._PredictOne(x, self._Tree) for x in X])


# ==============================================================================
# 純 numpy 實作：Random Forest 分類器
# ==============================================================================
class _RandomForest:
    """Random Forest 分類器，純 numpy/Python 實作，無需 scipy 或 scikit-learn。"""

    def __init__(self, NEstimators: int = 50, MaxDepth: int = 10):
        self._NEstimators = NEstimators
        self._MaxDepth    = MaxDepth
        self._Trees       = []
        self._NClasses    = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """以 Bootstrap 抽樣訓練所有決策樹。"""
        self._NClasses = int(np.max(y)) + 1
        NSamples, NFeatures = X.shape
        # 每棵樹使用 sqrt(NFeatures) 個特徵（Random Forest 標準做法）
        NFeat = max(1, int(np.sqrt(NFeatures)))
        self._Trees = []

        for _ in range(self._NEstimators):
            # Bootstrap 抽樣（有放回）
            Indices = np.random.choice(NSamples, size=NSamples, replace=True)
            Tree = _DecisionTree(MaxDepth=self._MaxDepth, MinSamplesSplit=2, NFeatures=NFeat)
            Tree.fit(X[Indices], y[Indices])
            self._Trees.append(Tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """平均所有決策樹的機率預測，回傳 shape (N, NClasses)。"""
        AllProbs = np.array([Tree.predict_proba(X) for Tree in self._Trees])
        return AllProbs.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """回傳每個樣本的預測類別索引。"""
        return np.argmax(self.predict_proba(X), axis=1)

# --- 路徑常數 ---
CFG_PATH     = "cfg/yolov3-face-landmark.cfg"
WEIGHTS_PATH = "weights/yolov3-face-landmark_final.weights"
NAMES_PATH   = "cfg/face_landmark.names"
MODEL_PATH   = "models/rf_classifier.pkl"

# --- YOLO 推論常數 ---
YOLO_INPUT_SIZE = 416
CONF_THRESHOLD  = 0.5
NMS_THRESHOLD   = 0.4

# --- 應用程式常數 ---
DETECT_SECONDS  = 10     # 辨識模式持續秒數
LEARN_SECONDS   = 60     # 學習模式持續秒數
MIN_LANDMARKS   = 3      # 最少需要偵測到的特徵點數
FEATURE_DIM     = 10     # C(5,2) = 10 個 pairwise 距離
UI_REFRESH_MS   = 30     # webcam 畫面更新間隔（毫秒）
LEARN_TICK_MS   = 500    # 學習時每次抓 frame 的間隔（每秒 2 個樣本）
DETECT_TICK_MS  = 200    # 辨識時每次推論的間隔

# pairwise 索引對（共 10 組，C(5,2)）
# 0:(0,1) 1:(0,2) 2:(0,3) 3:(0,4)
# 4:(1,2) 5:(1,3) 6:(1,4)
# 7:(2,3) 8:(2,4) 9:(3,4)
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


# ==============================================================================
# Class: WebcamManager
# ==============================================================================
class WebcamManager:
    """管理 webcam 資源，以 daemon 背景執行緒持續讀取 frame。"""

    def __init__(self, CameraIndex: int = 0):
        self._CameraIndex  = CameraIndex
        self._Cap          = None
        self._Lock         = threading.Lock()
        self._LatestFrame  = None
        self._Running      = False
        self._Thread       = None

    def Open(self) -> bool:
        """開啟 webcam 並啟動背景擷取執行緒。"""
        try:
            self._Cap = cv2.VideoCapture(self._CameraIndex)
            if not self._Cap.isOpened():
                raise RuntimeError(f"無法開啟攝影機 {self._CameraIndex}")
            self._Running = True
            self._Thread  = threading.Thread(target=self._CaptureLoop, daemon=True)
            self._Thread.start()
            return True
        except Exception as Error:
            print(f"[WebcamManager] 開啟攝影機失敗：{Error}")
            return False

    def Close(self) -> None:
        """停止背景執行緒並釋放 VideoCapture 資源。"""
        self._Running = False
        if self._Thread is not None:
            self._Thread.join(timeout=2.0)
            self._Thread = None
        if self._Cap is not None:
            self._Cap.release()
            self._Cap = None

    def GetLatestFrame(self) -> tuple:
        """以執行緒安全方式取得最新 frame 的副本。"""
        with self._Lock:
            if self._LatestFrame is None:
                return False, None
            return True, self._LatestFrame.copy()

    def _CaptureLoop(self) -> None:
        """背景執行緒：持續從攝影機讀取 frame 並更新共用緩衝區。"""
        while self._Running:
            try:
                Ret, Frame = self._Cap.read()
                if Ret:
                    with self._Lock:
                        self._LatestFrame = Frame
                else:
                    time.sleep(0.01)
            except Exception as Error:
                print(f"[WebcamManager] 擷取 frame 失敗：{Error}")
                time.sleep(0.1)


# ==============================================================================
# Class: PersonClassifier
# ==============================================================================
class PersonClassifier:
    """包裝 RandomForestClassifier，提供特徵抽取、訓練、預測和模型持久化功能。"""

    def __init__(self):
        self._Classifier      = None      # _RandomForest 實例
        self._LabelEncoder    = None      # _LabelEncoder 實例
        self._IsReady         = False
        self._KnownPersons    = []
        # 歷史訓練資料（跨 session 累積）
        self._AllFeatures     = []   # list of np.ndarray, shape (10,)
        self._AllLabels       = []   # list of str
        # 本次學習的暫存資料
        self._PendingFeatures = []
        self._PendingLabels   = []

    def ExtractFeatures(self, Landmarks: dict, FrameWidth: int, FrameHeight: int) -> np.ndarray:
        """
        從偵測到的特徵點計算 10 維 pairwise Euclidean 距離特徵向量。
        以參考距離（左眼↔右眼距離）正規化，使特徵對縮放具不變性。
        缺少的 pair 填入 0。
        """
        # 取得各特徵點的像素座標
        Points = {}
        for ClassId, Info in Landmarks.items():
            Points[ClassId] = np.array(Info["center"], dtype=float)

        # 計算所有 pair 的 Euclidean 距離
        RawDists = np.zeros(FEATURE_DIM, dtype=np.float32)
        for Idx, (A, B) in enumerate(PAIRS):
            if A in Points and B in Points:
                RawDists[Idx] = float(np.linalg.norm(Points[A] - Points[B]))

        # 以參考距離正規化（scale-invariant）
        # 優先使用左眼↔右眼距離（PAIRS index 0），最穩定的基準線
        RefDist = RawDists[0]
        if RefDist < 1e-6:
            # 若兩眼都不在，改用所有非零距離中的最大值
            NonZero = RawDists[RawDists > 1e-6]
            RefDist = float(NonZero.max()) if len(NonZero) > 0 else 1.0

        NormDists = RawDists / RefDist
        return NormDists

    def AddSample(self, Features: np.ndarray, PersonName: str) -> None:
        """將本次學習的樣本加入暫存清單。"""
        self._PendingFeatures.append(Features.copy())
        self._PendingLabels.append(PersonName)

    def ClearPendingSamples(self) -> None:
        """清空本次學習的暫存樣本。"""
        self._PendingFeatures = []
        self._PendingLabels   = []

    def Train(self) -> bool:
        """
        合併歷史資料與本次新樣本，重新訓練 RandomForestClassifier。
        需要至少 2 個不同類別才能訓練。
        """
        try:
            if len(self._PendingFeatures) == 0:
                print("[PersonClassifier] 無新樣本，訓練取消。")
                return False

            # 合併歷史資料與本次新樣本
            AllFeatures = self._AllFeatures + self._PendingFeatures
            AllLabels   = self._AllLabels   + self._PendingLabels

            UniqueClasses = list(set(AllLabels))
            if len(UniqueClasses) < 2:
                print(f"[PersonClassifier] 只有 {len(UniqueClasses)} 個類別，需要至少 2 個。")
                return False

            X = np.array(AllFeatures, dtype=np.float32)
            Y = np.array(AllLabels)

            # 訓練標籤編碼
            self._LabelEncoder = _LabelEncoder()
            YEncoded = self._LabelEncoder.fit_transform(list(Y))

            # 訓練 Random Forest（純 numpy 實作，NEstimators=50 兼顧速度與準確度）
            self._Classifier = _RandomForest(NEstimators=50, MaxDepth=10)
            self._Classifier.fit(X, YEncoded)

            # 更新歷史資料
            self._AllFeatures  = AllFeatures
            self._AllLabels    = AllLabels
            self._KnownPersons = list(self._LabelEncoder.classes_)
            self._IsReady      = True

            print(f"[PersonClassifier] 訓練完成。已知人物：{self._KnownPersons}，樣本數：{len(AllFeatures)}")
            return True

        except Exception as Error:
            print(f"[PersonClassifier] 訓練失敗：{Error}")
            return False

    def Predict(self, Features: np.ndarray) -> tuple:
        """
        預測特徵向量對應的人物。
        回傳 (姓名, 機率) tuple。若無法預測，回傳 ('Not found', 0.0)。
        """
        try:
            if not self._IsReady or self._Classifier is None:
                return "Not found", 0.0

            Probs   = self._Classifier.predict_proba(Features.reshape(1, -1))[0]
            BestIdx = int(np.argmax(Probs))
            BestProb = float(Probs[BestIdx])
            BestName = self._LabelEncoder.inverse_transform([BestIdx])[0]
            return BestName, BestProb

        except Exception as Error:
            print(f"[PersonClassifier] 預測失敗：{Error}")
            return "Not found", 0.0

    def SaveModel(self) -> bool:
        """將分類器與完整歷史訓練資料儲存至 pkl 檔（使用標準 pickle）。"""
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            Payload = {
                "Classifier":   self._Classifier,
                "LabelEncoder": self._LabelEncoder,
                "AllFeatures":  self._AllFeatures,
                "AllLabels":    self._AllLabels,
            }
            with open(MODEL_PATH, "wb") as F:
                pickle.dump(Payload, F)
            print(f"[PersonClassifier] 模型已儲存至 {MODEL_PATH}")
            return True
        except Exception as Error:
            print(f"[PersonClassifier] 模型儲存失敗：{Error}")
            return False

    def LoadModel(self) -> bool:
        """從 pkl 檔載入分類器與歷史訓練資料（使用標準 pickle）。"""
        try:
            if not os.path.exists(MODEL_PATH):
                return False
            with open(MODEL_PATH, "rb") as F:
                Payload = pickle.load(F)
            self._Classifier   = Payload["Classifier"]
            self._LabelEncoder = Payload["LabelEncoder"]
            self._AllFeatures  = Payload["AllFeatures"]
            self._AllLabels    = Payload["AllLabels"]
            self._KnownPersons = list(self._LabelEncoder.classes_)
            self._IsReady      = True
            print(f"[PersonClassifier] 模型載入成功。已知人物：{self._KnownPersons}")
            return True
        except Exception as Error:
            print(f"[PersonClassifier] 模型載入失敗：{Error}")
            return False

    def CanDetect(self) -> bool:
        """判斷是否有足夠的訓練資料可以進行辨識（需要至少 2 個不同人）。"""
        return self._IsReady and len(self._KnownPersons) >= 2

    def GetKnownPersons(self) -> list:
        """回傳已知人物名單。"""
        return list(self._KnownPersons)


# ==============================================================================
# Class: MainApp
# ==============================================================================
class MainApp(customtkinter.CTk):
    """主應用程式視窗，整合 webcam、YOLO 推論與 Random Forest 辨識功能。"""

    def __init__(self):
        super().__init__()

        # 辨識模式狀態
        self._DetectActive      = False
        self._DetectStartTime   = 0.0
        self._DetectPredictions = []

        # 學習模式狀態
        self._LearnActive       = False
        self._LearnStartTime    = 0.0
        self._LearnName         = ""
        self._LearnRemainSecs   = 0

        # YOLO 推論狀態
        self._Net               = None
        self._OutputLayers      = []
        self._ClassNames        = []
        self._LastLandmarks     = {}   # 上次偵測結果快取（供 UI 繪製用）

        # UI 圖像參照（防止被 GC 回收）
        self._CurrentPhotoImage = None

        # 核心元件
        self._Webcam            = WebcamManager(CameraIndex=0)
        self._Classifier        = PersonClassifier()

        # 建立 UI
        self._BuildUI()

        # 初始化元件（開啟攝影機、載入 YOLO、載入分類器）
        self._InitComponents()

    # --------------------------------------------------------------------------
    # UI 建立
    # --------------------------------------------------------------------------
    def _BuildUI(self) -> None:
        """建立 4-row CustomTkinter UI 介面。"""
        self.title("人臉辨識系統")
        self.protocol("WM_DELETE_WINDOW", self._OnClose)
        self.resizable(True, True)

        # Row 0：辨識功能列
        Row0 = customtkinter.CTkFrame(self)
        Row0.pack(fill="x", padx=10, pady=(10, 5))

        self._BtnDetectName = customtkinter.CTkButton(
            Row0,
            text="Detect face",
            width=140,
            command=self._OnBtnDetectName
        )
        self._BtnDetectName.pack(side="left", padx=(5, 10), pady=5)

        self._LblDetectName = customtkinter.CTkLabel(
            Row0,
            text="",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        self._LblDetectName.pack(side="left", padx=5, pady=5)

        # Row 1：學習功能列
        Row1 = customtkinter.CTkFrame(self)
        Row1.pack(fill="x", padx=10, pady=5)

        self._TblMyName = customtkinter.CTkEntry(
            Row1,
            placeholder_text="輸入姓名",
            width=200
        )
        self._TblMyName.pack(side="left", padx=(5, 10), pady=5)

        self._BtnLearn = customtkinter.CTkButton(
            Row1,
            text="Learning",
            width=120,
            command=self._OnBtnLearn
        )
        self._BtnLearn.pack(side="left", padx=5, pady=5)

        # Row 2：Webcam 畫面
        Row2 = customtkinter.CTkFrame(self)
        Row2.pack(fill="both", expand=True, padx=10, pady=5)

        self._WebcamCanvas = customtkinter.CTkLabel(
            Row2,
            text="攝影機畫面載入中...",
            width=640,
            height=480
        )
        self._WebcamCanvas.pack(fill="both", expand=True)

        # Row 3：剩餘時間列
        Row3 = customtkinter.CTkFrame(self)
        Row3.pack(fill="x", padx=10, pady=(5, 10))

        self._LblRemain = customtkinter.CTkLabel(
            Row3,
            text="Remaining study seconds: --"
        )
        self._LblRemain.pack(side="left", padx=5, pady=5)

    # --------------------------------------------------------------------------
    # 初始化
    # --------------------------------------------------------------------------
    def _InitComponents(self) -> None:
        """初始化攝影機、YOLO 模型與分類器。"""
        # 1. 開啟攝影機
        CamOk = self._Webcam.Open()
        if not CamOk:
            MsgBox.showwarning(
                "攝影機錯誤",
                "無法開啟攝影機，請確認連線後重新啟動。\n應用程式將以無攝影機模式執行。"
            )

        # 2. 載入 YOLO 模型
        self._LoadYolo()

        # 3. 載入 Random Forest 分類器
        self._LoadClassifier()

        # 4. 根據分類器狀態決定辨識按鈕的初始狀態
        if not self._Classifier.CanDetect():
            self._BtnDetectName.configure(state="disabled")

        # 5. 啟動 webcam 畫面更新迴圈
        self.after(UI_REFRESH_MS, self._UpdateWebcamView)

    def _LoadYolo(self) -> None:
        """載入 YOLO 類別名稱與網路模型。"""
        try:
            self._ClassNames   = load_class_names(NAMES_PATH)
            self._Net, self._OutputLayers = load_network(CFG_PATH, WEIGHTS_PATH)
        except Exception as Error:
            print(f"[MainApp] YOLO 模型載入失敗：{Error}")
            MsgBox.showerror(
                "YOLO 載入錯誤",
                f"YOLO 模型載入失敗：{Error}\n請確認 cfg/ 與 weights/ 目錄下的檔案存在。"
            )
            self._Net = None

    def _LoadClassifier(self) -> None:
        """載入已存在的 Random Forest 分類器（若存在）。"""
        try:
            self._Classifier.LoadModel()
        except Exception as Error:
            # 第一次使用，模型不存在是正常情況，靜默處理
            print(f"[MainApp] 分類器載入：{Error}")

    # --------------------------------------------------------------------------
    # Webcam 畫面更新迴圈（每 30ms 執行一次）
    # --------------------------------------------------------------------------
    def _UpdateWebcamView(self) -> None:
        """
        從 WebcamManager 取得最新 frame，繪製快取的 landmarks，
        轉換為 tkinter PhotoImage 並顯示在 UI 中。
        """
        try:
            Ok, Frame = self._Webcam.GetLatestFrame()
            if Ok and Frame is not None:
                # 繪製上次偵測到的 landmarks（使用快取，不在此執行 YOLO）
                if self._LastLandmarks and self._ClassNames:
                    Frame = draw_landmarks(Frame, self._LastLandmarks, self._ClassNames)

                # 轉換 BGR → RGB → PIL Image → PhotoImage
                FrameRgb = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)
                Img = Image.fromarray(FrameRgb)

                # 縮放至 canvas 尺寸
                W = self._WebcamCanvas.winfo_width()
                H = self._WebcamCanvas.winfo_height()
                if W > 1 and H > 1:
                    Img = Img.resize((W, H), Image.LANCZOS)

                Photo = ImageTk.PhotoImage(Img)
                self._WebcamCanvas.configure(image=Photo, text="")
                self._CurrentPhotoImage = Photo   # 保持參照，防止 GC 回收

        except Exception as Error:
            print(f"[MainApp] 更新畫面失敗：{Error}")
        finally:
            # 持續排程下一次更新
            self.after(UI_REFRESH_MS, self._UpdateWebcamView)

    # --------------------------------------------------------------------------
    # 辨識功能
    # --------------------------------------------------------------------------
    def _OnBtnDetectName(self) -> None:
        """辨識按鈕點擊事件處理。"""
        if self._DetectActive:
            self._StopDetection()
        else:
            if not self._Classifier.CanDetect():
                MsgBox.showwarning(
                    "無法辨識",
                    "尚無足夠的訓練資料。\n請先以至少 2 個不同人的資料進行學習。"
                )
                return
            self._StartDetection()

    def _StartDetection(self) -> None:
        """開始辨識模式。"""
        self._DetectActive      = True
        self._DetectStartTime   = time.time()
        self._DetectPredictions = []
        self._BtnDetectName.configure(text="Stop detect face")
        self._LblDetectName.configure(text="辨識中...")
        # 停用學習按鈕（辨識期間不允許學習）
        self._BtnLearn.configure(state="disabled")
        # 啟動辨識 tick
        self.after(0, self._DetectionTick)

    def _StopDetection(self) -> None:
        """結束辨識模式，顯示辨識結果。"""
        self._DetectActive = False
        self._BtnDetectName.configure(text="Detect face")
        self._BtnLearn.configure(state="normal")

        if self._DetectPredictions:
            # 多數決：取出現次數最多的名字
            BestName = Counter(self._DetectPredictions).most_common(1)[0][0]
            self._LblDetectName.configure(text=BestName)
        else:
            self._LblDetectName.configure(text="Not found")
            MsgBox.showinfo("辨識結果", "找不到符合的人臉。\n請確認臉部在鏡頭範圍內，或先進行學習。")

    def _DetectionTick(self) -> None:
        """辨識模式每次推論的 tick（每 200ms 執行一次）。"""
        if not self._DetectActive:
            return

        Elapsed = time.time() - self._DetectStartTime
        if Elapsed >= DETECT_SECONDS:
            self._StopDetection()
            return

        try:
            Ok, Frame = self._Webcam.GetLatestFrame()
            if Ok and Frame is not None and self._Net is not None:
                # 執行 YOLO 推論
                Landmarks = detect_landmarks(
                    self._Net, self._OutputLayers, Frame,
                    YOLO_INPUT_SIZE, CONF_THRESHOLD, NMS_THRESHOLD
                )
                self._LastLandmarks = Landmarks   # 更新 UI 快取

                # 若偵測到足夠的特徵點，進行 RF 預測
                if len(Landmarks) >= MIN_LANDMARKS:
                    H, W = Frame.shape[:2]
                    Features = self._Classifier.ExtractFeatures(Landmarks, W, H)
                    Name, Prob = self._Classifier.Predict(Features)
                    if Name != "Not found":
                        self._DetectPredictions.append(Name)

        except Exception as Error:
            print(f"[MainApp] 辨識 tick 失敗：{Error}")

        # 排程下一次 tick
        self.after(DETECT_TICK_MS, self._DetectionTick)

    # --------------------------------------------------------------------------
    # 學習功能
    # --------------------------------------------------------------------------
    def _OnBtnLearn(self) -> None:
        """學習按鈕點擊事件處理。"""
        if self._LearnActive:
            return   # 學習進行中，忽略重複點擊

        PersonName = self._TblMyName.get().strip()
        if not PersonName:
            MsgBox.showwarning("缺少姓名", "請先在姓名欄位輸入要學習的姓名。")
            return

        self._StartLearning(PersonName)

    def _StartLearning(self, PersonName: str) -> None:
        """開始學習模式，收集指定人物的特徵樣本。"""
        self._LearnActive       = True
        self._LearnName         = PersonName
        self._LearnStartTime    = time.time()
        self._LearnRemainSecs   = LEARN_SECONDS
        self._Classifier.ClearPendingSamples()

        # 停用相關按鈕（學習期間不允許其他操作）
        self._BtnLearn.configure(state="disabled")
        self._BtnDetectName.configure(state="disabled")
        self._LblRemain.configure(text=f"Remaining study seconds: {self._LearnRemainSecs}")

        # 啟動學習 tick
        self.after(0, self._LearningTick)

    def _StopLearning(self) -> None:
        """結束學習模式，訓練並儲存模型。"""
        self._LearnActive = False
        self._LblRemain.configure(text="Remaining study seconds: --")
        self._BtnLearn.configure(state="normal")

        # 嘗試訓練模型
        TrainOk = self._Classifier.Train()

        if TrainOk:
            SaveOk = self._Classifier.SaveModel()
            if SaveOk:
                Msg = f"已完成 [{self._LearnName}] 的學習！\n已知人物：{', '.join(self._Classifier.GetKnownPersons())}"
            else:
                Msg = f"學習完成，但模型儲存失敗。\n（本次學習的資料仍可在本 session 使用）"
            MsgBox.showinfo("學習完成", Msg)
        else:
            MsgBox.showwarning(
                "學習失敗",
                "樣本不足或目前只有 1 個人的資料。\n辨識功能需要至少 2 個不同人的學習資料。"
            )

        # 根據分類器狀態決定辨識按鈕是否可用
        if self._Classifier.CanDetect():
            self._BtnDetectName.configure(state="normal")
        else:
            self._BtnDetectName.configure(state="disabled")

    def _LearningTick(self) -> None:
        """學習模式每次 tick（每 500ms 執行一次）：收集樣本並更新倒數計時。"""
        if not self._LearnActive:
            return

        Elapsed = time.time() - self._LearnStartTime
        Remain  = max(0, LEARN_SECONDS - int(Elapsed))
        self._LblRemain.configure(text=f"Remaining study seconds: {Remain}")

        if Elapsed >= LEARN_SECONDS:
            self._StopLearning()
            return

        try:
            Ok, Frame = self._Webcam.GetLatestFrame()
            if Ok and Frame is not None and self._Net is not None:
                # 執行 YOLO 推論
                Landmarks = detect_landmarks(
                    self._Net, self._OutputLayers, Frame,
                    YOLO_INPUT_SIZE, CONF_THRESHOLD, NMS_THRESHOLD
                )
                self._LastLandmarks = Landmarks   # 更新 UI 快取

                # 若偵測到足夠的特徵點，加入學習樣本
                if len(Landmarks) >= MIN_LANDMARKS:
                    H, W = Frame.shape[:2]
                    Features = self._Classifier.ExtractFeatures(Landmarks, W, H)
                    self._Classifier.AddSample(Features, self._LearnName)

        except Exception as Error:
            print(f"[MainApp] 學習 tick 失敗：{Error}")

        # 排程下一次 tick
        self.after(LEARN_TICK_MS, self._LearningTick)

    # --------------------------------------------------------------------------
    # 關閉處理
    # --------------------------------------------------------------------------
    def _OnClose(self) -> None:
        """程式關閉前，先停止所有活動並釋放攝影機資源。"""
        print("[MainApp] 程式關閉中...")
        self._DetectActive = False
        self._LearnActive  = False
        self._Webcam.Close()
        self.destroy()


# ==============================================================================
# 程式進入點
# ==============================================================================
if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")
    App = MainApp()
    App.mainloop()
