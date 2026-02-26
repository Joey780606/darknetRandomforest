"""
landmark_annotator.py

人臉特徵點標註小工具。
在圖片上依序點擊 5 個特徵點（左眼、右眼、鼻尖、左嘴角、右嘴角），
自動產生 YOLO 格式的 .txt 標註檔。

使用方式：
    python tools/landmark_annotator.py --input_dir <圖片資料夾> --output_dir <標註輸出資料夾>

範例：
    python tools/landmark_annotator.py --input_dir data/images/train --output_dir data/labels/train

操作說明：
    - 依序點擊: 左眼 → 右眼 → 鼻尖 → 左嘴角 → 右嘴角
    - 按 'r' : 重新標註當前圖片
    - 按 'n' : 跳過當前圖片（不儲存）
    - 按 's' : 儲存當前標註並跳到下一張
    - 按 'q' : 結束程式
    - 5 個點都點完後會自動儲存
"""

import argparse
import os
import cv2
import numpy as np

# 特徵點定義
LANDMARKS = [
    {"id": 0, "name": "left_eye", "color": (0, 255, 0)},
    {"id": 1, "name": "right_eye", "color": (0, 255, 0)},
    {"id": 2, "name": "nose_tip", "color": (255, 0, 0)},
    {"id": 3, "name": "left_mouth", "color": (0, 0, 255)},
    {"id": 4, "name": "right_mouth", "color": (0, 0, 255)},
]

BBOX_RELATIVE_SIZE = 0.025  # 微型 bounding box 相對大小

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 全域變數
clicked_points = []
current_frame = None
display_frame = None


def mouse_callback(event, x, y, flags, param):
    """滑鼠點擊回調函數。"""
    global clicked_points, display_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) >= 5:
            return

        clicked_points.append((x, y))
        idx = len(clicked_points) - 1
        landmark = LANDMARKS[idx]

        # 在畫面上繪製標記
        display_frame = current_frame.copy()
        draw_all_points(display_frame)


def draw_all_points(frame):
    """繪製所有已點擊的標記點。"""
    for i, (px, py) in enumerate(clicked_points):
        landmark = LANDMARKS[i]
        color = landmark["color"]
        name = landmark["name"]

        cv2.circle(frame, (px, py), 5, color, -1)
        cv2.circle(frame, (px, py), 8, color, 2)
        cv2.putText(frame, f"{i}: {name}", (px + 12, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 顯示下一個要點的提示
    if len(clicked_points) < 5:
        next_landmark = LANDMARKS[len(clicked_points)]
        hint = f"Please click: {next_landmark['name']} ({len(clicked_points)+1}/5)"
        cv2.putText(frame, hint, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Done! Auto-saving...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def save_annotation(img_path, output_dir, points, img_width, img_height):
    """儲存 YOLO 格式的標註檔。"""
    img_filename = os.path.basename(img_path)
    label_filename = os.path.splitext(img_filename)[0] + ".txt"
    label_path = os.path.join(output_dir, label_filename)

    lines = []
    for class_id, (px, py) in enumerate(points):
        x_center = px / img_width
        y_center = py / img_height
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {BBOX_RELATIVE_SIZE:.6f} {BBOX_RELATIVE_SIZE:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Saved: {label_path}")
    return label_path


def main():
    global clicked_points, current_frame, display_frame

    parser = argparse.ArgumentParser(description="人臉特徵點標註工具")
    parser.add_argument("--input_dir", required=True, help="圖片來源資料夾")
    parser.add_argument("--output_dir", required=True, help="標註輸出資料夾")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: cannot find input directory {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 掃描圖片
    images = sorted([
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])

    if not images:
        print(f"Error: no images found in {args.input_dir}")
        return

    # 檢查哪些已經標註過
    existing_labels = set()
    for f in os.listdir(args.output_dir):
        if f.endswith(".txt"):
            existing_labels.add(os.path.splitext(f)[0])

    unannotated = [
        img for img in images
        if os.path.splitext(img)[0] not in existing_labels
    ]

    print(f"Total images: {len(images)}")
    print(f"Already annotated: {len(images) - len(unannotated)}")
    print(f"Remaining: {len(unannotated)}")
    print()
    print("Controls:")
    print("  Click  : Mark landmark point")
    print("  'r'    : Reset current image")
    print("  'n'    : Skip current image")
    print("  's'    : Save and next")
    print("  'q'    : Quit")
    print()

    window_name = "Landmark Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    idx = 0
    annotated_count = 0

    while idx < len(unannotated):
        img_name = unannotated[idx]
        img_path = os.path.join(args.input_dir, img_name)

        # 載入圖片
        current_frame = cv2.imread(img_path)
        if current_frame is None:
            print(f"  Cannot read: {img_path}, skipping")
            idx += 1
            continue

        img_h, img_w = current_frame.shape[:2]
        clicked_points = []
        display_frame = current_frame.copy()

        # 顯示圖片資訊
        info = f"[{idx + 1}/{len(unannotated)}] {img_name} ({img_w}x{img_h})"
        print(f"\n{info}")
        cv2.putText(display_frame,
                    f"Please click: {LANDMARKS[0]['name']} (1/5)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        saved = False

        while True:
            # 在 title bar 顯示進度
            cv2.setWindowTitle(window_name, f"Landmark Annotator - {info}")
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(50) & 0xFF

            # 5 個點都點完，自動儲存
            if len(clicked_points) == 5 and not saved:
                save_annotation(img_path, args.output_dir, clicked_points, img_w, img_h)
                annotated_count += 1
                saved = True
                # 短暫顯示完成狀態
                cv2.waitKey(500)
                idx += 1
                break

            if key == ord("q"):
                print(f"\nAnnotation finished. Annotated this session: {annotated_count}")
                cv2.destroyAllWindows()
                return

            elif key == ord("r"):
                # 重新標註
                clicked_points = []
                display_frame = current_frame.copy()
                draw_all_points(display_frame)
                saved = False
                print("  Reset")

            elif key == ord("n"):
                # 跳過
                print("  Skipped")
                idx += 1
                break

            elif key == ord("s"):
                # 手動儲存（即使不到 5 個點也能存）
                if clicked_points:
                    save_annotation(img_path, args.output_dir, clicked_points, img_w, img_h)
                    annotated_count += 1
                idx += 1
                break

    print(f"\nAll done! Annotated this session: {annotated_count}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
