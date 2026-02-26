"""
convert_300w_to_yolo.py

將 300-W 資料集的 .pts 標註檔轉換為 YOLO 格式的 .txt 標註檔。
從 68 個特徵點中提取 5 個關鍵點，以微型 bounding box 表示。

使用方式：
    python scripts/convert_300w_to_yolo.py --input_dir <300W資料集路徑> --output_images <輸出圖片目錄> --output_labels <輸出標註目錄>

範例：
    python scripts/convert_300w_to_yolo.py --input_dir C:/datasets/300W --output_images data/images/train --output_labels data/labels/train
"""

import argparse
import os
import shutil
import cv2
import numpy as np


# 300-W 68 點中，我們選取的 5 個關鍵點對應的索引（0-based）
# 參考: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
LANDMARK_MAPPING = {
    0: {"name": "left_eye", "indices": [36, 37, 38, 39, 40, 41]},    # 左眼 6 點取平均
    1: {"name": "right_eye", "indices": [42, 43, 44, 45, 46, 47]},   # 右眼 6 點取平均
    2: {"name": "nose_tip", "indices": [30]},                          # 鼻尖
    3: {"name": "left_mouth", "indices": [48]},                        # 左嘴角
    4: {"name": "right_mouth", "indices": [54]},                       # 右嘴角
}

# 微型 bounding box 的相對大小（相對於圖片尺寸的比例）
BBOX_RELATIVE_SIZE = 0.025


def parse_pts_file(pts_path):
    """解析 300-W 的 .pts 標註檔，回傳 68 個特徵點的座標列表。"""
    points = []
    with open(pts_path, "r") as f:
        lines = f.readlines()

    # .pts 格式：
    # version: 1
    # n_points: 68
    # {
    # x1 y1
    # x2 y2
    # ...
    # }
    in_points = False
    for line in lines:
        line = line.strip()
        if line == "{":
            in_points = True
            continue
        if line == "}":
            break
        if in_points:
            parts = line.split()
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
                points.append((x, y))

    return points


def extract_5_landmarks(points_68):
    """從 68 個特徵點中提取 5 個關鍵點。"""
    landmarks = {}
    for class_id, info in LANDMARK_MAPPING.items():
        indices = info["indices"]
        xs = [points_68[i][0] for i in indices]
        ys = [points_68[i][1] for i in indices]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        landmarks[class_id] = (cx, cy)
    return landmarks


def convert_to_yolo_format(landmarks, img_width, img_height):
    """將 5 個關鍵點轉換為 YOLO 格式的標註行。"""
    lines = []
    bbox_w = BBOX_RELATIVE_SIZE
    bbox_h = BBOX_RELATIVE_SIZE

    for class_id in sorted(landmarks.keys()):
        cx, cy = landmarks[class_id]
        # 轉換為相對座標 (0~1)
        x_center = cx / img_width
        y_center = cy / img_height

        # 確保座標在有效範圍內
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")

    return lines


def find_image_for_pts(pts_path):
    """根據 .pts 檔案路徑尋找對應的圖片檔案。"""
    base = os.path.splitext(pts_path)[0]
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        img_path = base + ext
        if os.path.exists(img_path):
            return img_path
    return None


def process_dataset(input_dir, output_images, output_labels):
    """處理整個 300-W 資料集。"""
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    pts_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".pts"):
                pts_files.append(os.path.join(root, f))

    print(f"找到 {len(pts_files)} 個 .pts 標註檔")

    success_count = 0
    skip_count = 0

    for pts_path in pts_files:
        # 找到對應的圖片
        img_path = find_image_for_pts(pts_path)
        if img_path is None:
            print(f"  跳過（找不到圖片）: {pts_path}")
            skip_count += 1
            continue

        # 讀取圖片取得尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"  跳過（無法讀取圖片）: {img_path}")
            skip_count += 1
            continue

        img_height, img_width = img.shape[:2]

        # 解析 .pts 檔案
        points_68 = parse_pts_file(pts_path)
        if len(points_68) != 68:
            print(f"  跳過（特徵點數量不是 68）: {pts_path} ({len(points_68)} 點)")
            skip_count += 1
            continue

        # 提取 5 個關鍵點
        landmarks = extract_5_landmarks(points_68)

        # 轉換為 YOLO 格式
        yolo_lines = convert_to_yolo_format(landmarks, img_width, img_height)

        # 輸出檔案名稱（使用圖片原始檔名）
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"

        # 複製圖片
        dst_img = os.path.join(output_images, img_filename)
        shutil.copy2(img_path, dst_img)

        # 寫入標註檔
        dst_label = os.path.join(output_labels, label_filename)
        with open(dst_label, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        success_count += 1

    print(f"\n轉換完成！")
    print(f"  成功: {success_count} 張")
    print(f"  跳過: {skip_count} 張")
    print(f"  圖片輸出: {output_images}")
    print(f"  標註輸出: {output_labels}")


def main():
    parser = argparse.ArgumentParser(description="將 300-W 資料集轉換為 YOLO 格式")
    parser.add_argument("--input_dir", required=True, help="300-W 資料集根目錄路徑")
    parser.add_argument("--output_images", default="data/images/train", help="輸出圖片目錄")
    parser.add_argument("--output_labels", default="data/labels/train", help="輸出標註目錄")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"錯誤：找不到輸入目錄 {args.input_dir}")
        return

    process_dataset(args.input_dir, args.output_images, args.output_labels)


if __name__ == "__main__":
    main()
