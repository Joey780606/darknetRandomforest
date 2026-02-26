"""
inference.py

使用 OpenCV DNN 載入 Darknet/YOLO 訓練好的模型，
透過攝影機即時偵測人臉 5 個特徵點（左眼、右眼、鼻尖、左嘴角、右嘴角）。

使用方式：
    python inference.py

可選參數：
    --cfg       YOLOv3 設定檔路徑（預設: cfg/yolov3-face-landmark.cfg）
    --weights   權重檔路徑（預設: weights/yolov3-face-landmark_final.weights）
    --names     類別名稱檔路徑（預設: cfg/face_landmark.names）
    --camera    攝影機編號（預設: 0）
    --conf      信心度閾值（預設: 0.5）
    --nms       NMS 閾值（預設: 0.4）
    --size      輸入圖片尺寸（預設: 416）
"""

import argparse
import cv2
import numpy as np

# 每個特徵點的顏色 (BGR)
LANDMARK_COLORS = {
    "left_eye": (0, 255, 0),       # 綠色
    "right_eye": (0, 255, 0),      # 綠色
    "nose_tip": (255, 0, 0),       # 藍色
    "left_mouth": (0, 0, 255),     # 紅色
    "right_mouth": (0, 0, 255),    # 紅色
}

# 預設顏色
DEFAULT_COLOR = (255, 255, 0)  # 青色


def load_class_names(names_path):
    """載入類別名稱。"""
    with open(names_path, "r") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names


def load_network(cfg_path, weights_path):
    """載入 Darknet 網路模型。"""
    print(f"載入模型...")
    print(f"  設定檔: {cfg_path}")
    print(f"  權重檔: {weights_path}")

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

    # 優先使用 CUDA GPU，否則使用 CPU
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("  使用 CUDA GPU 加速")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        print("  使用 CPU 運算")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 取得輸出層名稱
    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayers()
    output_layer_names = [layer_names[i - 1] for i in output_layers.flatten()]

    print(f"  模型載入完成！")
    return net, output_layer_names


def detect_landmarks(net, output_layer_names, frame, input_size, conf_threshold, nms_threshold):
    """偵測畫面中的人臉特徵點。"""
    h, w = frame.shape[:2]

    # 前處理
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (input_size, input_size),
                                  swapRB=True, crop=False)
    net.setInput(blob)

    # 推論
    outputs = net.forward(output_layer_names)

    # 解析結果
    class_ids = []
    confidences = []
    centers = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                # 取 bounding box 中心點作為特徵點座標
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                centers.append((center_x, center_y))

    # 對每個類別分別做 NMS
    landmarks = {}
    for cls_id in range(5):  # 5 個特徵點類別
        cls_indices = [i for i, c in enumerate(class_ids) if c == cls_id]
        if not cls_indices:
            continue

        cls_confidences = [confidences[i] for i in cls_indices]
        # 找信心度最高的偵測結果
        best_idx = cls_indices[np.argmax(cls_confidences)]
        landmarks[cls_id] = {
            "center": centers[best_idx],
            "confidence": confidences[best_idx],
        }

    return landmarks


def draw_landmarks(frame, landmarks, class_names):
    """在畫面上繪製特徵點。"""
    for cls_id, info in landmarks.items():
        cx, cy = info["center"]
        conf = info["confidence"]
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        color = LANDMARK_COLORS.get(name, DEFAULT_COLOR)

        # 繪製實心圓點
        cv2.circle(frame, (cx, cy), 5, color, -1)
        # 繪製外框圓
        cv2.circle(frame, (cx, cy), 8, color, 2)
        # 標示名稱和信心度
        label = f"{name} {conf:.2f}"
        cv2.putText(frame, label, (cx + 12, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="人臉特徵點即時偵測")
    parser.add_argument("--cfg", default="cfg/yolov3-face-landmark.cfg", help="YOLOv3 設定檔")
    parser.add_argument("--weights", default="weights/yolov3-face-landmark_final.weights", help="權重檔")
    parser.add_argument("--names", default="cfg/face_landmark.names", help="類別名稱檔")
    parser.add_argument("--camera", type=int, default=0, help="攝影機編號")
    parser.add_argument("--conf", type=float, default=0.5, help="信心度閾值")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS 閾值")
    parser.add_argument("--size", type=int, default=416, help="輸入圖片尺寸")
    args = parser.parse_args()

    # 載入類別名稱
    class_names = load_class_names(args.names)
    print(f"類別: {class_names}")

    # 載入網路
    net, output_layer_names = load_network(args.cfg, args.weights)

    # 開啟攝影機
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"錯誤：無法開啟攝影機 {args.camera}")
        return

    print(f"\n攝影機已開啟。按 'q' 結束。")

    frame_count = 0
    fps = 0
    prev_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機畫面")
            break

        # 偵測特徵點
        landmarks = detect_landmarks(net, output_layer_names, frame, args.size,
                                     args.conf, args.nms)

        # 繪製特徵點
        frame = draw_landmarks(frame, landmarks, class_names)

        # 計算 FPS
        frame_count += 1
        if frame_count >= 10:
            curr_time = cv2.getTickCount()
            elapsed = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = curr_time

        # 顯示 FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 顯示偵測到的特徵點數量
        cv2.putText(frame, f"Landmarks: {len(landmarks)}/5", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Landmark Detection (YOLO)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("程式結束。")


if __name__ == "__main__":
    main()
